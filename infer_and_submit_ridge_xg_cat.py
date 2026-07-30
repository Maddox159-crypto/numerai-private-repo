"""
infer_and_submit_ridge_xg_cat.py

ensemble4 파이프라인과 완전히 독립적인 실험용 스크립트.
ridge + xgboost + catboost 3개 모델을 rank-average로 조합해서
별도 모델 슬롯(esaa)에 제출.

- neutralization은 적용하지 않음 (이 조합 자체의 순수 성능 확인 목적)
- 기존 infer_and_submit.py, 히스토리 파일 등과 완전히 분리되어 동작
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import numerapi
import time

# ============================================
# 경로 설정
# ============================================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DATA_DIR = "/tmp/numerai_data_ridge_xg_cat"
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_ID_NAME = "esaa"  # ensemble4(esaa_maddox)와 다른 슬롯 사용

# ============================================
# [1] Numerai API 인증
# ============================================
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]

napi = numerapi.NumerAPI(public_id=NUMERAI_PUBLIC_ID, secret_key=NUMERAI_SECRET_KEY)
MODEL_ID = napi.get_models()[MODEL_ID_NAME]
print(f"model_id ({MODEL_ID_NAME}): {MODEL_ID}")


# ============================================
# [2] live.parquet 다운로드 (ensemble4용과는 별개 경로에 저장)
# ============================================
LIVE_PATH = os.path.join(DATA_DIR, "live.parquet")
napi.download_dataset("v5.2/live.parquet", LIVE_PATH)
print("live.parquet 다운로드 완료")


# ============================================
# [3] feature 변환 준비물 로드 (모델 파일은 기존 것 재사용)
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
live_df = pd.read_parquet(LIVE_PATH, columns=selected_features)

if live_df.index.name is not None or "id" not in live_df.columns:
    live_df = live_df.reset_index()

if "id" not in live_df.columns:
    id_df = pd.read_parquet(LIVE_PATH, columns=["id"])
    if id_df.index.name is not None or "id" not in id_df.columns:
        id_df = id_df.reset_index()
    live_df["id"] = id_df["id"].values

X = clean_array(live_df[selected_features].to_numpy(dtype=np.float32))
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled).astype(np.float32)

pca_df = pd.DataFrame(X_pca, columns=pca_cols)
pca_df["id"] = live_df["id"].values

print(f"live pca 변환 완료: {pca_df.shape}")


# ============================================
# [5] lag1/lag2 - ensemble4 히스토리 파일을 그대로 재사용 (읽기 전용, 갱신 안 함)
# ============================================
HISTORY_PATH = os.path.join(REPO_ROOT, "era_pca_mean_history.json")

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH) as f:
        history = json.load(f)
else:
    history = {}

history_dates = sorted(history.keys())
print(f"참고할 히스토리 라운드 수: {len(history_dates)} (읽기 전용, 이 스크립트에서는 갱신 안 함)")

if len(history_dates) >= 2:
    lag1_mean = np.array(history[history_dates[-1]], dtype=np.float32)
    lag2_mean = np.array(history[history_dates[-2]], dtype=np.float32)
elif len(history_dates) == 1:
    lag1_mean = np.array(history[history_dates[-1]], dtype=np.float32)
    lag2_mean = np.zeros(n_components, dtype=np.float32)
else:
    lag1_mean = np.zeros(n_components, dtype=np.float32)
    lag2_mean = np.zeros(n_components, dtype=np.float32)

for i, col in enumerate(lag1_cols):
    pca_df[col] = lag1_mean[i]
for i, col in enumerate(lag2_cols):
    pca_df[col] = lag2_mean[i]

X_full_live = pca_df[FEATURES].to_numpy(dtype=np.float32)
print(f"최종 feature matrix: {X_full_live.shape}")


# ============================================
# [6] 3개 모델(ridge, xgboost, catboost) 로드 및 예측
# ============================================
predictions = {}

from catboost import CatBoostRegressor
catboost_model = CatBoostRegressor()
catboost_model.load_model(os.path.join(MODELS_DIR, "catboost_model.cbm"))
predictions["catboost"] = catboost_model.predict(X_full_live)

import xgboost as xgb
with open(os.path.join(MODELS_DIR, "xgboost_params.json")) as f:
    xgb_params = json.load(f)
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
predictions["xgboost"] = xgb_model.predict(X_full_live)

ridge_model = joblib.load(os.path.join(MODELS_DIR, "ridge_model.joblib"))
predictions["ridge"] = ridge_model.predict(X_full_live)

print(f"3개 모델 예측 완료: {list(predictions.keys())}")


# ============================================
# [7] rank-average 앙상블 (neutralize 없음 - 실험용 raw 성능 확인)
# ============================================
rank_preds = []
for name, pred in predictions.items():
    rank_pct = pd.Series(pred).rank(pct=True, method="first").values
    rank_preds.append(rank_pct)

final_pred = np.mean(rank_preds, axis=0)
print("rank-average 앙상블 완료 (neutralize 미적용)")

submission_df = pd.DataFrame({
    "id": live_df["id"].values,
    "prediction": final_pred,
})
print(f"submission shape: {submission_df.shape}")
print(f"prediction 범위: {submission_df['prediction'].min():.6f} ~ {submission_df['prediction'].max():.6f}")


# ============================================
# [8] 제출 (재시도 포함)
# ============================================
SUBMIT_PATH = os.path.join(DATA_DIR, "submission.csv")
submission_df.to_csv(SUBMIT_PATH, index=False)

MAX_RETRIES = 4
RETRY_WAIT_SECONDS = 30

for attempt in range(1, MAX_RETRIES + 1):
    try:
        napi.upload_predictions(SUBMIT_PATH, model_id=MODEL_ID)
        print(f"Numerai 제출 완료 (시도 {attempt}/{MAX_RETRIES})")
        break
    except Exception as e:
        print(f"제출 실패 (시도 {attempt}/{MAX_RETRIES}): {e}")
        if attempt == MAX_RETRIES:
            raise
        time.sleep(RETRY_WAIT_SECONDS)
