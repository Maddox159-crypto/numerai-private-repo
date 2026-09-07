"""
infer_and_submit.py

매주 실행되는 라이브 추론 + 제출 스크립트.
- live.parquet 다운로드
- scaler/pca transform (fit 아님)
- lag1/lag2: 히스토리 2주 이상 쌓이면 실제 값, 아니면 0으로 채움
- 8개 base model 예측 -> rank-average (ensemble4)
- feature neutralization (proportion=0.3, pca_0~499 기준)
- napi.upload_predictions()
- 이번 주 pca 평균을 히스토리에 추가 저장 (다음 주를 위해)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import numerapi
from scipy.stats import norm
from datetime import datetime, timezone

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

# ↓↓↓ 여기에 로그 추가(20260903에 추가함) ↓↓↓
script_start_utc = datetime.now(timezone.utc)
print(f"[LOG] 스크립트 시작 시각 (UTC): {script_start_utc.isoformat()}", flush=True)
# ↑↑↑ 여기까지 ↑↑↑

# ============================================
# [1] Numerai API 인증 (GitHub Secrets에서 읽음)
# ============================================
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]

napi = numerapi.NumerAPI(public_id=NUMERAI_PUBLIC_ID, secret_key=NUMERAI_SECRET_KEY)
MODEL_ID = napi.get_models()[MODEL_ID_NAME]
print(f"model_id: {MODEL_ID}")

import sys
import json

LAST_SUBMIT_ROUND_PATH = "last_submitted_round.json"

current_round = napi.get_current_round()
print(f"[LOG] 현재 라운드 번호: {current_round}", flush=True)

if os.path.exists(LAST_SUBMIT_ROUND_PATH):
    with open(LAST_SUBMIT_ROUND_PATH) as f:
        last_submitted_round = json.load(f).get("round")
else:
    last_submitted_round = None

if last_submitted_round == current_round:
    print(f"[LOG] 라운드 {current_round}는 이미 제출 완료됨 — 스킵하고 종료", flush=True)
    sys.exit(0)
# ============================================
# [2] live.parquet 다운로드
# ============================================
LIVE_PATH = os.path.join(DATA_DIR, "live.parquet")
napi.download_dataset("v5.2/live.parquet", LIVE_PATH)
print("live.parquet 다운로드 완료")


# ============================================
# [3] feature 변환 준비물 로드
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


# ============================================
# [4] live 데이터 로드 및 pca 변환
# ============================================
live_cols = ["id"] + selected_features
live_df = pd.read_parquet(LIVE_PATH, columns=selected_features)

# id가 index로 들어있는 경우 처리 (validation.parquet에서 겪었던 것과 동일한 문제)
if live_df.index.name is not None or "id" not in live_df.columns:
    live_df = live_df.reset_index()

if "id" not in live_df.columns:
    # 그래도 없으면 원본 parquet에서 id 컬럼을 명시적으로 다시 읽어옴
    id_df = pd.read_parquet(LIVE_PATH, columns=["id"])
    if id_df.index.name is not None or "id" not in id_df.columns:
        id_df = id_df.reset_index()
    live_df["id"] = id_df["id"].values

print(f"live_df 컬럼 확인: id 포함 여부 = {'id' in live_df.columns}")

X = clean_array(live_df[selected_features].to_numpy(dtype=np.float32))
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled).astype(np.float32)

pca_df = pd.DataFrame(X_pca, columns=pca_cols)
pca_df["id"] = live_df["id"].values

print(f"live pca 변환 완료: {pca_df.shape}")


# ============================================
# [5] lag1/lag2 처리 - 히스토리 2주 이상이면 실제 값, 아니면 0
# ============================================
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH) as f:
        history = json.load(f)
else:
    history = {}

history_dates = sorted(history.keys())  # 오래된 -> 최신 순
print(f"현재 히스토리에 쌓인 라운드 수: {len(history_dates)}")

if len(history_dates) >= 2:
    lag1_mean = np.array(history[history_dates[-1]], dtype=np.float32)  # 가장 최근
    lag2_mean = np.array(history[history_dates[-2]], dtype=np.float32)  # 그 전
    print("lag1/lag2: 실제 히스토리 값 사용")
elif len(history_dates) == 1:
    lag1_mean = np.array(history[history_dates[-1]], dtype=np.float32)
    lag2_mean = np.zeros(n_components, dtype=np.float32)
    print("lag1: 실제 값, lag2: 0 (히스토리 1주치만 존재)")
else:
    lag1_mean = np.zeros(n_components, dtype=np.float32)
    lag2_mean = np.zeros(n_components, dtype=np.float32)
    print("lag1/lag2: 0으로 채움 (히스토리 없음)")

# 모든 행에 동일한 lag 값(era 단위 평균이므로) broadcast
for i, col in enumerate(lag1_cols):
    pca_df[col] = lag1_mean[i]
for i, col in enumerate(lag2_cols):
    pca_df[col] = lag2_mean[i]

X_full_live = pca_df[FEATURES].to_numpy(dtype=np.float32)
print(f"최종 feature matrix: {X_full_live.shape}")


# ============================================
# [6] 8개 모델 로드 및 예측
# ============================================
predictions = {}

# --- CatBoost ---
from catboost import CatBoostRegressor
catboost_model = CatBoostRegressor()
catboost_model.load_model(os.path.join(MODELS_DIR, "catboost_model.cbm"))
predictions["catboost"] = catboost_model.predict(X_full_live)

# --- XGBoost (트리+파라미터 같이 복원) ---
import xgboost as xgb
with open(os.path.join(MODELS_DIR, "xgboost_params.json")) as f:
    xgb_params = json.load(f)
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
predictions["xgboost"] = xgb_model.predict(X_full_live)

# --- LightGBM ---
import lightgbm as lgb
lgb_booster = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lightgbm_model.txt"))
predictions["lightgbm"] = lgb_booster.predict(X_full_live)

# --- RandomForest ---
rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))
predictions["rf"] = rf_model.predict(X_full_live)

# --- ExtraTrees ---
et_model = joblib.load(os.path.join(MODELS_DIR, "et_model.joblib"))
predictions["et"] = et_model.predict(X_full_live)

# --- Ridge ---
ridge_model = joblib.load(os.path.join(MODELS_DIR, "ridge_model.joblib"))
predictions["ridge"] = ridge_model.predict(X_full_live)

# --- ElasticNet ---
elasticnet_model = joblib.load(os.path.join(MODELS_DIR, "elasticnet_model.joblib"))
predictions["elasticnet"] = elasticnet_model.predict(X_full_live)

print(f"8개 모델 예측 완료: {list(predictions.keys())}")
# 참고: mlp, tabnet은 지난번 팀 결정대로 성능 낮아서 제외 (8개 -> 실제로는 7개, ensemble4 구성 그대로 반영)


# ============================================
# [7] rank-average 앙상블 (ensemble4)
# ============================================
rank_preds = []
for name, pred in predictions.items():
    rank_pct = pd.Series(pred).rank(pct=True, method="first").values
    rank_preds.append(rank_pct)

ensemble_pred = np.mean(rank_preds, axis=0)
print("rank-average 앙상블 완료")


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
print(f"neutralize({NEUTRALIZATION_PROPORTION}) 적용 완료, submission shape: {submission_df.shape}")
print(f"prediction 범위: {submission_df['prediction'].min():.6f} ~ {submission_df['prediction'].max():.6f}")

# =========================20260903수정=======
# [9] 제출 (일시적 오류 대비 재시도 포함)
# ============================================
import time
from datetime import datetime, timezone  # 파일 맨 위에 이미 import했다면 생략 가능

SUBMIT_PATH = os.path.join(DATA_DIR, "submission.csv")
submission_df.to_csv(SUBMIT_PATH, index=False)

MAX_RETRIES = 4
RETRY_WAIT_SECONDS = 30

before_upload = datetime.now(timezone.utc)
print(f"[LOG] 업로드 시도 시작 시각 (UTC): {before_upload.isoformat()}", flush=True)

for attempt in range(1, MAX_RETRIES + 1):
    try:
        napi.upload_predictions(SUBMIT_PATH, model_id=MODEL_ID)
        after_upload = datetime.now(timezone.utc)
        print(f"Numerai 제출 완료 (시도 {attempt}/{MAX_RETRIES})")
        print(f"[LOG] 업로드 완료 시각 (UTC): {after_upload.isoformat()}", flush=True)
        with open(LAST_SUBMIT_ROUND_PATH, "w") as f:
            json.dump({"round": current_round}, f)
        break
    except Exception as e:
        print(f"제출 실패 (시도 {attempt}/{MAX_RETRIES}): {e}")
        if attempt == MAX_RETRIES:
            raise
        time.sleep(RETRY_WAIT_SECONDS)
# ============================================
# [10] 이번 주 pca 평균을 히스토리에 추가 저장 (다음 주 lag용)
# ============================================
from datetime import datetime, timezone

today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
this_week_pca_mean = pca_df[pca_cols].mean(axis=0).tolist()

history[today] = this_week_pca_mean

# 히스토리가 너무 커지지 않도록 최근 10주치만 유지
if len(history) > 10:
    for old_date in sorted(history.keys())[:-10]:
        del history[old_date]

with open(HISTORY_PATH, "w") as f:
    json.dump(history, f)

print(f"히스토리 저장 완료 ({today} 추가, 현재 {len(history)}주치 보관 중)")
