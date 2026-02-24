# -*- coding: utf-8 -*-
"""
1월 2일 발주용 수요예측 - Part 2: LightGBM 모델 학습 & WMAPE 검증
[에러 수정 이력]
- bin size 3255: CATEGORICAL_FEATURES에서 sku_name 제거, Dataset max_bin=255 명시
- left_count > 0: min_child_samples 너무 작을 때 GPU에서 노드 분할 실패
  → min_child_samples=200, max_depth=6으로 안정화 (num_leaves=127 유지)
- 과소예측: tweedie_variance_power=1.0, learning_rate=0.02, early_stopping=500
- 카테고리별 편향 보정 계수(bias_corr) → Part 3 예측값 후처리에 사용
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from demand_forecast_part1_features import (
    load_data, build_features, encode_and_clean,
    split_data, build_forecast_rows,
    wmape, FEATURE_COLS, TARGET_COL
)


# ══════════════════════════════════════════════════════════════
# 1. LightGBM 파라미터
# ══════════════════════════════════════════════════════════════
LGB_PARAMS = {
    # 'device'              : 'gpu',   # GPU: left_count 에러 반복 → CPU 사용
    'objective'              : 'tweedie',
    'tweedie_variance_power' : 1.0,
    'metric'                 : 'rmse',
    'verbosity'              : -1,
    'random_state'           : 42,
    'n_jobs'                 : -1,       # CPU 전체 코어 사용
    # ✅ GPU bin 제한
    'max_bin'                : 255,
    # ✅ 트리 구조 — GPU 안정성 우선
    'num_leaves'             : 127,     # CPU에서 안전하게 사용 가능
    'max_depth'              : 6,       # 8→6: 노드 분할 실패 방지
    'min_child_samples'      : 20,      # CPU 기본값
    'min_child_weight'       : 1e-3,
    # 학습
    'learning_rate'          : 0.02,
    'n_estimators'           : 10000,  # 충분한 학습 라운드 확보
    'subsample'              : 0.85,
    'subsample_freq'         : 1,
    'colsample_bytree'       : 0.85,
    'reg_alpha'              : 0.05,
    'reg_lambda'             : 0.05,
    'cat_smooth'             : 10,
}

# ✅ sku_name 제거: 고유값 3,300개 → bin 3,255 생성 → GPU 에러
CATEGORICAL_FEATURES = ['warehouse', 'm_cat', 'season_name']


# ══════════════════════════════════════════════════════════════
# 2. Optuna 하이퍼파라미터 튜닝
# ══════════════════════════════════════════════════════════════
def run_optuna(train: pd.DataFrame, valid: pd.DataFrame,
               n_trials: int = 50) -> dict:
    """
    Optuna로 LightGBM 하이퍼파라미터 최적화
    목적함수: 연휴 직후 검증셋 WMAPE 최소화
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_valid = valid[FEATURE_COLS]
    y_valid = valid[TARGET_COL]

    max_date = train['sales_date'].max()
    days_from_max = (max_date - train['sales_date']).dt.days
    sample_weight = np.exp(-days_from_max / 45)

    def objective(trial):
        params = {
            # 고정
            'objective'              : 'tweedie',
            'metric'                 : 'rmse',
            'verbosity'              : -1,
            'random_state'           : 42,
            'n_jobs'                 : -1,
            'max_bin'                : 255,
            # 탐색 (이전 best 근방 정밀 탐색)
            # best: num_leaves=221, lr=0.049, max_depth=10, min_child=110
            'tweedie_variance_power' : trial.suggest_float('tweedie_variance_power', 1.0, 1.5),
            'num_leaves'             : trial.suggest_int('num_leaves', 63, 255),
            'max_depth'              : trial.suggest_int('max_depth', 6, 12),
            'min_child_samples'      : trial.suggest_int('min_child_samples', 20, 200),
            'learning_rate'          : trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'subsample'              : trial.suggest_float('subsample', 0.6, 0.85),
            'colsample_bytree'       : trial.suggest_float('colsample_bytree', 0.65, 0.85),
            'reg_alpha'              : trial.suggest_float('reg_alpha', 1e-3, 0.1, log=True),
            'reg_lambda'             : trial.suggest_float('reg_lambda', 0.3, 1.0, log=True),
            'cat_smooth'             : trial.suggest_int('cat_smooth', 12, 20),
        }

        dtrain = lgb.Dataset(
            X_train, label=y_train,
            weight=sample_weight,
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
            params={'max_bin': 255}
        )
        dvalid = lgb.Dataset(
            X_valid, label=y_valid,
            reference=dtrain,
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
            params={'max_bin': 255}
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            callbacks=callbacks,
        )

        preds = np.maximum(model.predict(X_valid), 0)
        return wmape(y_valid.values, preds)

    print(f"\n🔍 Optuna 하이퍼파라미터 탐색 시작 ({n_trials} trials)...")
    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_wmape = study.best_value
    print(f"\n  ✅ Optuna 완료 | Best WMAPE: {best_wmape:.2f}%")
    print(f"  📋 Best params:")
    for k, v in best.items():
        print(f"     {k}: {v}")

    return best, best_wmape


# ══════════════════════════════════════════════════════════════
# 3. 모델 학습 (Optuna 결과 또는 기본 파라미터 사용)
# ══════════════════════════════════════════════════════════════
def train_model(train: pd.DataFrame, valid: pd.DataFrame,
                best_params: dict = None):
    print("=" * 60)
    print("🤖 LightGBM 모델 학습 시작...")

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_valid = valid[FEATURE_COLS]
    y_valid = valid[TARGET_COL]

    # Optuna 결과 파라미터 또는 기본값 사용
    if best_params is not None:
        params = {
            'objective'              : 'tweedie',
            'metric'                 : 'rmse',
            'verbosity'              : -1,
            'random_state'           : 42,
            'n_jobs'                 : -1,
            'max_bin'                : 255,
            **best_params,
        }
        print(f"  ✅ Optuna 최적 파라미터 사용")
    else:
        params = LGB_PARAMS
        print(f"  ℹ️  기본 파라미터 사용")

    # 시간 가중치: 최근 45일 강조
    max_date = train['sales_date'].max()
    days_from_max = (max_date - train['sales_date']).dt.days
    sample_weight = np.exp(-days_from_max / 45)   # 전체 학습 기준 최근 45일 강조

    # ✅ Dataset에도 max_bin=255 명시 (params와 독립적으로 bin 생성됨)
    dtrain = lgb.Dataset(
        X_train, label=y_train,
        weight=sample_weight,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
        params={'max_bin': 255}
    )
    dvalid = lgb.Dataset(
        X_valid, label=y_valid,
        reference=dtrain,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
        params={'max_bin': 255}
    )

    wmape_log = []

    def wmape_callback(env):
        if env.iteration % 100 == 0:
            pred = np.maximum(env.model.predict(X_valid), 0)
            wm   = wmape(y_valid.values, pred)
            wmape_log.append((env.iteration, wm))
            print(f"  [Round {env.iteration:4d}] WMAPE: {wm:.2f}%")

    callbacks = [
        lgb.early_stopping(stopping_rounds=500, verbose=False),
        lgb.log_evaluation(period=-1),
        wmape_callback,
    ]

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=10000,
        valid_sets=[dvalid],
        callbacks=callbacks,
    )

    print(f"\n  ✅ 학습 완료 | Best iteration: {model.best_iteration}")
    return model, wmape_log


# ══════════════════════════════════════════════════════════════
# 3. 검증 평가 + 카테고리별 편향 보정 계수 산출
# ══════════════════════════════════════════════════════════════
def evaluate(model, valid: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 검증 평가 (WMAPE)")

    preds = np.maximum(model.predict(valid[FEATURE_COLS]), 0)

    valid = valid.copy()
    valid['pred']    = preds
    valid['abs_err'] = np.abs(valid['qty'] - valid['pred'])

    # 전체 WMAPE
    total_wm = wmape(valid['qty'].values, valid['pred'].values)
    print(f"\n  🎯 전체 WMAPE      : {total_wm:.2f}%  (목표 <= 10%)")
    print(f"  {'✅ 목표 달성!' if total_wm <= 10 else '⚠️  추가 튜닝 필요'}")

    # 카테고리별 WMAPE
    grp = valid.groupby('m_cat').apply(
        lambda g: wmape(g['qty'].values, g['pred'].values)
    ).reset_index()
    grp.columns = ['m_cat', 'wmape']
    print(f"\n  📋 집계별 WMAPE (코드값 기준, Part 3에서 원본명 복원):")

    # 편향 보정 계수: 실제합 / 예측합 (카테고리별 체계적 과소예측 수치 보정)
    bias_corr = valid.groupby('m_cat').apply(
        lambda g: g['qty'].sum() / (g['pred'].sum() + 1e-9)
    ).reset_index()
    bias_corr.columns = ['m_cat', 'bias_correction']

    valid_corr = valid.merge(bias_corr, on='m_cat', how='left')
    valid_corr['pred_corrected'] = valid_corr['pred'] * valid_corr['bias_correction']
    corrected_wm = wmape(valid_corr['qty'].values, valid_corr['pred_corrected'].values)
    print(f"  🔧 편향 보정 후 WMAPE : {corrected_wm:.2f}%")

    return valid, total_wm, bias_corr


# ══════════════════════════════════════════════════════════════
# 4. 피처 중요도 출력
# ══════════════════════════════════════════════════════════════
def print_feature_importance(model, top_n: int = 20):
    imp_df = pd.DataFrame({
        'feature'   : model.feature_name(),
        'importance': model.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False).head(top_n)

    print(f"\n🔍 피처 중요도 Top {top_n}:")
    print("-" * 45)
    for _, row in imp_df.iterrows():
        bar = "█" * int(row['importance'] / imp_df['importance'].max() * 30)
        print(f"  {row['feature']:25s} {bar}")
    print("-" * 45)
    return imp_df


# ══════════════════════════════════════════════════════════════
# 5. 모델 저장
# ══════════════════════════════════════════════════════════════
def save_model(model, path: str = 'lgb_demand_model.pkl'):
    joblib.dump(model, path)
    model.save_model(path.replace('.pkl', '.txt'))
    print(f"\n💾 모델 저장: {path}, {path.replace('.pkl', '.txt')}")


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    df_raw  = load_data('sales_data.parquet')
    df_feat = build_features(df_raw.copy())
    df_enc  = encode_and_clean(df_feat)
    train, valid = split_data(df_enc)

    # Optuna 실행 (n_trials 조정 가능, 1trial당 약 20~30초)
    best_params, best_wmape = run_optuna(train, valid, n_trials=50)

    model, wmape_log = train_model(train, valid, best_params)
    valid_result, total_wmape, bias_corr = evaluate(model, valid)
    feat_imp = print_feature_importance(model, top_n=20)

    joblib.dump(bias_corr, 'bias_correction.pkl')
    print(f"  💾 편향 보정 계수 저장: bias_correction.pkl")
    save_model(model)

    print("\n✅ Part 2 완료")
    print(f"   최종 검증 WMAPE: {total_wmape:.2f}%")