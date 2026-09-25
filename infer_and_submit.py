"""
infer_and_submit.py

매주 실행되는 라이브 추론 + 제출 스크립트.
- live.parquet 다운로드
- scaler/pca transform (fit 아님)
- lag1/lag2: 히스토리에서 7일 전 / 14일 전 날짜의 era 평균 사용 (학습 시 주간 era lag와 일치)
- 8개 base model 예측 -> rank-average (ensemble4)
- feature neutralization (proportion=0.3, pca_0~499 기준)
- napi.upload_predictions()
- 이번 라운드 pca 평균을 히스토리에 추가 저장 (이후 라운드를 위해)

20260922 수정: [2]~[9] 전체를 재시도 루프로 감쌈.
  - 외부 GH Actions cron이 라운드 창(12:16~13:25 UTC)을 4번 분산 트리거하고,
  - 각 job은 내부에서 최대 MAX_WAIT_MINUTES까지 스스로 재시도하며 버팀
  - 두 안전장치를 겹쳐서 late 확률을 최소화

20260925 수정: lag 시간 단위 train/serve 불일치 수정.
  - 기존: history_dates[-1], [-2] = 어제/그저께 라운드 (라운드가 화~토 매일이라 '일' 단위 lag)
  - 학습: era가 주 단위라 lag1 = 1주 전, lag2 = 2주 전
  - 변경: 오늘 기준 7일 전 / 14일 전에 가장 가까운 날짜(±LAG_TOLERANCE_DAYS)를 골라 사용
  - 히스토리 보관 개수 10 -> 20 (14일 전 값이 결측/실패일에도 남아있도록)
  - ※ era 정렬 버그 수정 후 재학습한 모델 파일과 반드시 "같이" 배포할 것

20260925 수정 (2):
  - 스킵 체크: "오늘 이미 제출"일 때만 즉시 종료, "새 라운드 아직 안 열림"이면 루프 안에서 대기
  - last_submitted_round.json에 date 필드 추가
  - xgboost params의 device를 cpu로 강제 (Colab GPU 학습 -> GH Actions CPU 추론)
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import numerapi
from scipy.stats import norm
from datetime import datetime, timezone, date

# ============================================
# 경로 설정
# ============================================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
HISTORY_PATH = os.path.join(REPO_ROOT, "era_pca_mean_history.json")
DATA_DIR = "/tmp/numerai_data"
os.makedirs(DATA_DIR, exist_ok=True)

NEUTRALIZATION_PROPORTION = 0.3
MODEL_ID_NAME = "esaa_maddox"

# lag 설정 (학습 때 era는 주 단위 -> lag1 = 7일 전, lag2 = 14일 전)
LAG1_DAYS = 7
LAG2_DAYS = 14
LAG_TOLERANCE_DAYS = 3   # 정확히 그 날짜가 없으면 ±3일 안에서 가장 가까운 날짜 사용
HISTORY_KEEP = 20        # 화~토 5회/주 기준 약 4주치

script_start_utc = datetime.now(timezone.utc)
print(f"[LOG] 스크립트 시작 시각 (UTC): {script_start_utc.isoformat()}", flush=True)

# ============================================
# [1] Numerai API 인증 (GitHub Secrets에서 읽음)
# ============================================
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]

napi = numerapi.NumerAPI(public_id=NUMERAI_PUBLIC_ID, secret_key=NUMERAI_SECRET_KEY)
MODEL_ID = napi.get_models()[MODEL_ID_NAME]
print(f"model_id: {MODEL_ID}")

import sys

LAST_SUBMIT_ROUND_PATH = "last_submitted_round.json"

today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

if os.path.exists(LAST_SUBMIT_ROUND_PATH):
    with open(LAST_SUBMIT_ROUND_PATH) as f:
        _last = json.load(f)
    last_submitted_round = _last.get("round")
    last_submitted_date = _last.get("date")  # 20260925 추가 (예전 파일엔 없음)
else:
    last_submitted_round = None
    last_submitted_date = None

# 20260925 수정: "오늘 이미 제출함"과 "새 라운드가 아직 안 열림"을 구분
#   - 예전 코드는 둘 다 round 번호가 같아서 무조건 종료 -> 12:05 트리거가 라운드 오픈(12:16)을
#     기다리지 않고 즉시 끝나버려서, job 내부 재시도 루프가 사실상 무용지물이었음
#   - 이제: 오늘 날짜로 이미 제출했으면 즉시 종료, 아니면 재시도 루프 안에서 새 라운드를 기다림
if last_submitted_date == today_str:
    print(f"[LOG] 오늘({today_str}) 라운드 {last_submitted_round} 이미 제출 완료 — 스킵하고 종료", flush=True)
    sys.exit(0)


class RoundNotOpenYet(Exception):
    pass


# ============================================
# [3] feature 변환 준비물 로드 (재시도 루프 밖 — 매번 다시 읽을 필요 없음)
# ============================================
with open(os.path.join(MODELS_DIR, "selected_features.json")) as f:
    selected_features = json.load(f)

scaler = joblib.load(os.path.join(MODELS_DIR, "fitted_scaler.joblib"))
pca = joblib.load(os.path.join(MODELS_DIR, "fitted_pca.joblib"))

n_components = pca.n_components_
pca_cols = [f"pca_{i}" for i in range(n_components)]
lag1_cols = [f"{c}_lag1" for c in pca_cols]
lag2_cols = [f"{c}_lag2" for c in pca_cols]
FEATURES = pca_cols + lag1_cols + lag2_cols  # 1500개, 학습 때와 동일한 순서


def clean_array(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def pick_lag_mean(history, today_date, days_back, n_components, tolerance=LAG_TOLERANCE_DAYS):
    """
    history: {"YYYY-MM-DD": [pca mean ...]}
    today_date 기준 days_back일 전에 가장 가까운 날짜의 값을 반환.
    ±tolerance일 안에 후보가 없으면 0 벡터 반환.
    동률이면 더 과거 날짜를 선택 (미래 정보가 섞이지 않도록 보수적으로).
    반환: (np.ndarray, 선택된 날짜 문자열 또는 None)
    """
    candidates = []
    for key in history.keys():
        try:
            d = date.fromisoformat(key)
        except ValueError:
            continue
        age = (today_date - d).days
        if age <= 0:
            continue  # 오늘/미래 값은 lag로 쓰지 않음
        diff = abs(age - days_back)
        if diff <= tolerance:
            candidates.append((diff, -age, key))  # diff 작은 순, 동률이면 age 큰(더 과거) 순

    if not candidates:
        return np.zeros(n_components, dtype=np.float32), None

    candidates.sort()
    chosen_key = candidates[0][2]
    return np.array(history[chosen_key], dtype=np.float32), chosen_key


# ============================================
# [2]~[9] 재시도 루프
# 다운로드부터 제출까지 전체를 감싸서, 라운드가 아직 안 열렸거나
# 일시적 오류가 나면 job 안에서 스스로 몇 번이고 다시 시도한다.
# 외부 cron 트리거가 여러 번 분산되어 있는 것과 겹쳐서 이중 안전장치가 됨.
# ============================================
MAX_WAIT_MINUTES = 18       # 이 job이 최대 이만큼(분)까지 자체 재시도
RETRY_INTERVAL_SECONDS = 120  # 재시도 사이 대기 시간 (2분)

pipeline_start = datetime.now(timezone.utc)
attempt_num = 0
pca_df = None
live_df = None
final_pred = None

while True:
    attempt_num += 1
    elapsed_minutes = (datetime.now(timezone.utc) - pipeline_start).total_seconds() / 60
    print(f"[LOG] === 시도 {attempt_num} 시작 (경과 {elapsed_minutes:.1f}분) ===", flush=True)

    try:
        # 새 라운드가 열렸는지 확인 (안 열렸으면 대기 후 재시도)
        current_round = napi.get_current_round()
        print(f"[LOG] 현재 라운드 번호: {current_round}", flush=True)
        if current_round == last_submitted_round:
            raise RoundNotOpenYet(f"라운드 {current_round}는 지난번에 제출한 라운드 — 새 라운드 아직 안 열림")

        # ============================================
        # [2] live.parquet 다운로드
        # ============================================
        LIVE_PATH = os.path.join(DATA_DIR, "live.parquet")
        napi.download_dataset("v5.2/live.parquet", LIVE_PATH)
        print("live.parquet 다운로드 완료", flush=True)

        # ============================================
        # [4] live 데이터 로드 및 pca 변환
        # ============================================
        live_df = pd.read_parquet(LIVE_PATH, columns=selected_features)

        if live_df.index.name is not None or "id" not in live_df.columns:
            live_df = live_df.reset_index()

        if "id" not in live_df.columns:
            id_df = pd.read_parquet(LIVE_PATH, columns=["id"])
            if id_df.index.name is not None or "id" not in id_df.columns:
                id_df = id_df.reset_index()
            live_df["id"] = id_df["id"].values

        print(f"live_df 컬럼 확인: id 포함 여부 = {'id' in live_df.columns}", flush=True)

        X = clean_array(live_df[selected_features].to_numpy(dtype=np.float32))
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled).astype(np.float32)

        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df["id"] = live_df["id"].values

        print(f"live pca 변환 완료: {pca_df.shape}", flush=True)

        # ============================================
        # [5] lag1/lag2 처리 - 7일 전 / 14일 전 (학습 시 주간 era lag와 동일한 간격)
        # ============================================
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH) as f:
                history = json.load(f)
        else:
            history = {}

        print(f"현재 히스토리에 쌓인 라운드 수: {len(history)}", flush=True)

        today_date = datetime.now(timezone.utc).date()
        lag1_mean, lag1_key = pick_lag_mean(history, today_date, LAG1_DAYS, n_components)
        lag2_mean, lag2_key = pick_lag_mean(history, today_date, LAG2_DAYS, n_components)

        print(f"[LOG] lag1 ({LAG1_DAYS}일 전 목표): {lag1_key if lag1_key else '후보 없음 -> 0으로 채움'}", flush=True)
        print(f"[LOG] lag2 ({LAG2_DAYS}일 전 목표): {lag2_key if lag2_key else '후보 없음 -> 0으로 채움'}", flush=True)

        for i, col in enumerate(lag1_cols):
            pca_df[col] = lag1_mean[i]
        for i, col in enumerate(lag2_cols):
            pca_df[col] = lag2_mean[i]

        X_full_live = pca_df[FEATURES].to_numpy(dtype=np.float32)
        print(f"최종 feature matrix: {X_full_live.shape}", flush=True)

        # ============================================
        # [6] 8개 모델 로드 및 예측
        # ============================================
        predictions = {}

        from catboost import CatBoostRegressor
        catboost_model = CatBoostRegressor()
        catboost_model.load_model(os.path.join(MODELS_DIR, "catboost_model.cbm"))
        predictions["catboost"] = catboost_model.predict(X_full_live)

        import xgboost as xgb
        with open(os.path.join(MODELS_DIR, "xgboost_params.json")) as f:
            xgb_params = json.load(f)
        # 20260925: Colab에서 GPU로 학습해 params에 device='cuda'가 저장돼 있음 -> GH Actions는 CPU라 강제로 cpu
        xgb_params["device"] = "cpu"
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
        predictions["xgboost"] = xgb_model.predict(X_full_live)

        import lightgbm as lgb
        lgb_booster = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lightgbm_model.txt"))
        predictions["lightgbm"] = lgb_booster.predict(X_full_live)

        rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))
        predictions["rf"] = rf_model.predict(X_full_live)

        et_model = joblib.load(os.path.join(MODELS_DIR, "et_model.joblib"))
        predictions["et"] = et_model.predict(X_full_live)

        ridge_model = joblib.load(os.path.join(MODELS_DIR, "ridge_model.joblib"))
        predictions["ridge"] = ridge_model.predict(X_full_live)

        elasticnet_model = joblib.load(os.path.join(MODELS_DIR, "elasticnet_model.joblib"))
        predictions["elasticnet"] = elasticnet_model.predict(X_full_live)

        print(f"8개 모델 예측 완료: {list(predictions.keys())}", flush=True)
        # 참고: mlp, tabnet은 지난번 팀 결정대로 성능 낮아서 제외 (8개 -> 실제로는 7개, ensemble4 구성 그대로 반영)

        # ============================================
        # [7] rank-average 앙상블 (ensemble4)
        # ============================================
        rank_preds = []
        for name, pred in predictions.items():
            rank_pct = pd.Series(pred).rank(pct=True, method="first").values
            rank_preds.append(rank_pct)

        ensemble_pred = np.mean(rank_preds, axis=0)
        print("rank-average 앙상블 완료", flush=True)

        # ============================================
        # [8] feature neutralization (proportion=0.3, pca_0~499 기준)
        # ============================================
        def neutralize(pred, features, proportion):
            if proportion == 0:
                return pd.Series(pred).rank(pct=True, method="first").values

            pred_rank = pd.Series(pred).rank(pct=True, method="first").clip(1e-6, 1 - 1e-6)
            pred_gauss = norm.ppf(pred_rank)

            exposures = np.linalg.lstsq(features, pred_gauss, rcond=None)[0]
            neutralized = pred_gauss - proportion * (features @ exposures)

            return pd.Series(neutralized).rank(pct=True, method="first").values

        pca_only_features = pca_df[pca_cols].to_numpy(dtype=np.float32)
        final_pred = neutralize(ensemble_pred, pca_only_features, NEUTRALIZATION_PROPORTION)

        submission_df = pd.DataFrame({
            "id": live_df["id"].values,
            "prediction": final_pred,
        })
        print(f"neutralize({NEUTRALIZATION_PROPORTION}) 적용 완료, submission shape: {submission_df.shape}", flush=True)
        print(f"prediction 범위: {submission_df['prediction'].min():.6f} ~ {submission_df['prediction'].max():.6f}", flush=True)

        # ============================================
        # [9] 제출
        # ============================================
        SUBMIT_PATH = os.path.join(DATA_DIR, "submission.csv")
        submission_df.to_csv(SUBMIT_PATH, index=False)

        before_upload = datetime.now(timezone.utc)
        print(f"[LOG] 업로드 시도 시작 시각 (UTC): {before_upload.isoformat()}", flush=True)

        napi.upload_predictions(SUBMIT_PATH, model_id=MODEL_ID)

        after_upload = datetime.now(timezone.utc)
        print(f"Numerai 제출 완료 (시도 {attempt_num})", flush=True)
        print(f"[LOG] 업로드 완료 시각 (UTC): {after_upload.isoformat()}", flush=True)

        with open(LAST_SUBMIT_ROUND_PATH, "w") as f:
            json.dump({"round": current_round, "date": today_str}, f)

        # 성공했으면 루프 탈출 — 이 아래 [10]으로 진행
        break

    except Exception as e:
        elapsed_minutes = (datetime.now(timezone.utc) - pipeline_start).total_seconds() / 60
        print(f"[LOG] 시도 {attempt_num} 실패 (경과 {elapsed_minutes:.1f}분): {e}", flush=True)

        if elapsed_minutes >= MAX_WAIT_MINUTES:
            print(f"[LOG] 최대 대기 시간({MAX_WAIT_MINUTES}분) 초과 — 이번 job은 포기, 다음 트리거에 넘김", flush=True)
            if isinstance(e, RoundNotOpenYet):
                sys.exit(0)  # 라운드가 안 열린 건 에러가 아니므로 빨간 X 없이 정상 종료
            raise

        print(f"[LOG] {RETRY_INTERVAL_SECONDS}초 대기 후 재시도", flush=True)
        time.sleep(RETRY_INTERVAL_SECONDS)


# ============================================
# [10] 이번 라운드 pca 평균을 히스토리에 추가 저장 (이후 라운드 lag용)
# 제출이 성공(break)했을 때만 이 지점에 도달함
# ============================================
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
this_week_pca_mean = pca_df[pca_cols].mean(axis=0).tolist()

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH) as f:
        history = json.load(f)
else:
    history = {}

history[today] = this_week_pca_mean

# 최근 HISTORY_KEEP개만 유지 (14일 전 값이 결측일에도 남아있도록 넉넉히)
if len(history) > HISTORY_KEEP:
    for old_date in sorted(history.keys())[:-HISTORY_KEEP]:
        del history[old_date]

with open(HISTORY_PATH, "w") as f:
    json.dump(history, f)

print(f"히스토리 저장 완료 ({today} 추가, 현재 {len(history)}개 보관 중)", flush=True)
