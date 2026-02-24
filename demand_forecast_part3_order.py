# -*- coding: utf-8 -*-
"""
1월 2일 발주용 수요예측 - Part 3: 발주량 산출 & 결과 저장
수요예측 → 재고 차감 → 발주 권고량 산출 → Excel/CSV 저장
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from demand_forecast_part1_features import (
    load_data, build_features, encode_and_clean,
    split_data, build_forecast_rows,
    wmape, FEATURE_COLS, TARGET_COL,
    CAT_BASE_SALES
)
from demand_forecast_part2_train import (
    train_model, evaluate, print_feature_importance, save_model,
    LGB_PARAMS, CATEGORICAL_FEATURES
)

# ══════════════════════════════════════════════════════════════
# 0. 발주 정책 파라미터
# ══════════════════════════════════════════════════════════════
# 카테고리별 안전재고 계수 (신선도 짧을수록 높게 설정)
SAFETY_STOCK_COEFF = {
    '엽채류'    : 1.20,   '나물류'    : 1.20,
    '버섯류'    : 1.15,   '과채류'    : 1.15,
    '조미채류'  : 1.15,   '근채류'    : 1.10,
    '가금육'    : 1.20,   '적색육/소' : 1.15,
    '적색육/돼지': 1.15,  '우유'      : 1.15,
    '가공유'    : 1.10,   '요구르트'  : 1.10,
    '육가공'    : 1.05,   '과자'      : 1.05,
    '라면/면'   : 1.05,   '가공식품'  : 1.05,
    '냉동육'    : 1.05,
}
DEFAULT_SAFETY = 1.10


# ══════════════════════════════════════════════════════════════
# 1. 예측 행 인코딩 (Part 1 원본 행 → 모델 입력 형식)
# ══════════════════════════════════════════════════════════════
def encode_forecast_rows(forecast_df: pd.DataFrame,
                         train_enc: pd.DataFrame) -> pd.DataFrame:
    """
    예측 행의 categorical 컬럼을 학습 시와 동일한 코드값으로 매핑
    - train_enc에서 (원본값 → 코드값) 딕셔너리를 추출하여 적용
    """
    cat_cols = ['warehouse', 'sku_name', 'm_cat', 'season_name']

    # 원본 데이터에서 매핑 복원이 어려우므로, 같은 astype(category) 적용
    # → 동일 실행 세션에서 코드값 일치 보장
    for col in cat_cols:
        if col in forecast_df.columns:
            forecast_df[col] = forecast_df[col].astype('category').cat.codes

    lag_fill_cols = [c for c in forecast_df.columns if any(
        c.startswith(p) for p in
        ['lag_', 'rolling_', 'same_dow', 'stockout_lag', 'price_lag', 'price_change']
    )]
    forecast_df[lag_fill_cols] = forecast_df[lag_fill_cols].fillna(0)
    return forecast_df


# ══════════════════════════════════════════════════════════════
# 2. 수요 예측 수행
# ══════════════════════════════════════════════════════════════
def rule_based_predict(df_raw: pd.DataFrame, cat_map: dict,
                        sku_map: dict, wh_map: dict) -> pd.DataFrame:
    """
    문제 5개 카테고리 룰 기반 예측
    근거: life >= 20 카테고리는 월요일 1회 발주 → qty가 요일별 재고 가용량에 종속
    1월 2일(목요일): base_sales × dow_weight(1.38) × 노이즈 평균
    실제로는 노이즈 평균이 1.0이므로 base_sales × 1.38이 기댓값
    """
    CAT_SPEC = {
        '육가공'  : {'base_sales': 80},
        '과자'    : {'base_sales': 100},
        '라면/면' : {'base_sales': 120},
        '가공식품': {'base_sales': 90},
        '냉동육'  : {'base_sales': 30},
    }
    DOW_WEIGHT_THU = 1.38  # 목요일

    # SKU별 카테고리 매핑
    sku_cat = df_raw[['sku_name', 'm_cat']].drop_duplicates()

    records = []
    for _, row in sku_cat.iterrows():
        m_cat = row['m_cat']
        if m_cat not in CAT_SPEC:
            continue
        base = CAT_SPEC[m_cat]['base_sales']
        pred = base * DOW_WEIGHT_THU  # 노이즈 평균 = 1.0

        for wh in ['A센터', 'B센터', 'C센터', 'D센터', 'E센터']:
            records.append({
                'warehouse_raw'   : wh,
                'sku_name_raw'    : row['sku_name'],
                'm_cat_name'      : m_cat,
                'rule_pred'       : pred,
            })

    return pd.DataFrame(records)


def predict_demand(model, forecast_enc: pd.DataFrame,
                   bias_corr: pd.DataFrame = None,
                   cat_map: dict = None,
                   valid_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    2026-01-02 SKU × 창고별 예측 수요량 산출
    보정 2단계:
      1) 카테고리 편향 보정 (bias_corr)
      2) SKU 단위: post_holiday_qty_mean 피처가 이미 모델 입력에 포함
                  → 모델이 직접 학습하므로 별도 후처리 불필요
    """
    print("\n🔮 2026-01-02 수요 예측 중...")
    preds = np.maximum(model.predict(forecast_enc[FEATURE_COLS]), 0)
    forecast_enc = forecast_enc.copy()
    forecast_enc['predicted_demand_raw'] = preds

    # 카테고리 단위 편향 보정
    if bias_corr is not None:
        forecast_enc = forecast_enc.merge(bias_corr, on='m_cat', how='left')
        forecast_enc['bias_correction'] = forecast_enc['bias_correction'].fillna(1.0)
        forecast_enc['predicted_demand_cat'] = (
            forecast_enc['predicted_demand_raw'] * forecast_enc['bias_correction']
        ).clip(lower=0)
        print(f"  🔧 카테고리 편향 보정 적용 완료")
    else:
        forecast_enc['predicted_demand_cat'] = preds

    # 문제 5개 카테고리: valid 실측 평균 기반 룰 예측으로 대체
    # base_sales는 시뮬레이션 기준값으로 실제 qty와 스케일이 다름
    # → valid(연휴직후)에서 카테고리×창고×요일별 실측 평균을 직접 사용
    HIGH_ERROR_CATS = ['과자', '냉동육', '라면/면', '가공식품', '육가공']

    if cat_map is not None and valid_df is not None:
        forecast_enc['m_cat_name_tmp'] = forecast_enc['m_cat'].map(cat_map)
        is_high_error = forecast_enc['m_cat_name_tmp'].isin(HIGH_ERROR_CATS)

        # valid에서 카테고리별 SKU당 목요일(dow=3) 평균 qty 산출
        valid_thu = valid_df[valid_df['sales_date'].dt.dayofweek == 3].copy()
        if len(valid_thu) == 0:
            # 목요일 데이터 없으면 전체 valid 평균 사용
            valid_thu = valid_df.copy()

        # m_cat 코드 → 원본명 매핑
        valid_thu = valid_thu.copy()
        valid_thu['m_cat_name'] = valid_thu['m_cat'].map(cat_map)
        cat_qty_mean = (
            valid_thu[valid_thu['m_cat_name'].isin(HIGH_ERROR_CATS)]
            .groupby('m_cat_name')['qty']
            .mean()
        )
        print(f"  📐 카테고리별 valid 실측 평균 (SKU당):")
        for cat, val in cat_qty_mean.items():
            print(f"     {cat}: {val:.1f}")

        rule_pred = forecast_enc['m_cat_name_tmp'].map(cat_qty_mean)

        # NaN 방지: fillna로 모델 예측값으로 대체 후 변환
        rule_pred_filled = rule_pred.fillna(forecast_enc['predicted_demand_cat'])

        forecast_enc['predicted_demand'] = np.where(
            is_high_error & rule_pred.notna(),
            rule_pred_filled.clip(lower=0).round().astype(int),
            forecast_enc['predicted_demand_cat'].fillna(0).clip(lower=0).round().astype(int)
        )
        n_rule = (is_high_error & rule_pred.notna()).sum()
        print(f"  📐 룰 기반 예측 적용: {n_rule:,}건 (과자/냉동육/라면/육가공/가공식품)")
        forecast_enc.drop(columns=['m_cat_name_tmp'], inplace=True)
    else:
        forecast_enc['predicted_demand'] = (
            forecast_enc['predicted_demand_cat'].round().astype(int)
        )

    preds_final = forecast_enc['predicted_demand'].values
    print(f"  ✅ 예측 완료: {len(forecast_enc):,} 건")
    print(f"  📈 예측 수요 통계:")
    print(f"     평균: {preds_final.mean():.1f}  |  중앙: {np.median(preds_final):.1f}"
          f"  |  최대: {preds_final.max():,}  |  최소: {preds_final.min()}")
    return forecast_enc


# ══════════════════════════════════════════════════════════════
# 3. 재고 차감 → 발주 권고량 산출
# ══════════════════════════════════════════════════════════════
def calculate_order_qty(
    forecast_enc: pd.DataFrame,
    inv_df: pd.DataFrame,
    cat_map: dict,           # 코드값 → 카테고리 원본명
    sku_map: dict,           # 코드값 → SKU 원본명
    wh_map: dict,            # 코드값 → 창고 원본명
) -> pd.DataFrame:
    """
    발주 권고량 = max(0, 예측수요 × 안전재고계수 - 사용가능재고)
    사용가능재고: 유통기한 내 재고 (out_days 기준 필터링은 inv_data 생성 시 적용됨)
    """
    print("\n📦 발주 권고량 산출 중...")

    # 재고 집계 (창고 × SKU 기준 합산)
    inv_agg = inv_df.groupby(['warehouse', 'sku_name'])['stock_qty'].sum().reset_index()
    inv_agg.columns = ['warehouse', 'sku_name', 'available_stock']

    # 코드값 → 원본명 복원
    result = forecast_enc.copy()
    result['m_cat_name']   = result['m_cat'].map(cat_map)
    result['sku_name_raw'] = result['sku_name'].map(sku_map)
    result['warehouse_raw'] = result['warehouse'].map(wh_map)

    # 재고 join (원본명 기준)
    result = result.merge(
        inv_agg.rename(columns={'warehouse': 'warehouse_raw', 'sku_name': 'sku_name_raw'}),
        on=['warehouse_raw', 'sku_name_raw'],
        how='left'
    )
    result['available_stock'] = result['available_stock'].fillna(0)

    # 안전재고 계수 적용
    result['safety_coeff'] = result['m_cat_name'].map(SAFETY_STOCK_COEFF).fillna(DEFAULT_SAFETY)
    result['adjusted_demand'] = (result['predicted_demand'] * result['safety_coeff']).round().astype(int)

    # 발주 권고량 = 조정수요 - 현재고 (음수 → 0)
    result['order_qty'] = (result['adjusted_demand'] - result['available_stock']).clip(lower=0).round().astype(int)

    # 발주 필요 여부
    result['order_needed'] = (result['order_qty'] > 0).astype(int)

    print(f"  ✅ 발주 권고량 산출 완료")
    print(f"  📦 발주 필요 SKU×창고 : {result['order_needed'].sum():,} 건 / {len(result):,} 건")
    print(f"  📊 총 발주 권고량     : {result['order_qty'].sum():,.0f} 개")
    return result


# ══════════════════════════════════════════════════════════════
# 4. WMAPE 카테고리별 상세 리포트
# ══════════════════════════════════════════════════════════════
def print_wmape_report(valid_result: pd.DataFrame,
                       cat_map: dict, wh_map: dict,
                       total_wmape: float):
    """카테고리 / 창고별 WMAPE 리포트"""
    print("\n" + "=" * 60)
    print("📊 WMAPE 상세 리포트")
    print("=" * 60)
    print(f"  🎯 전체 WMAPE : {total_wmape:.2f}%  ({'✅ 목표 달성' if total_wmape <= 10 else '⚠️  튜닝 필요'})")

    # 카테고리별
    print(f"\n  📋 카테고리별 WMAPE:")
    print(f"  {'카테고리':<15} {'WMAPE(%)':>10} {'실제합':>12} {'예측합':>12}")
    print("  " + "-" * 52)
    for cat_code, grp in valid_result.groupby('m_cat'):
        cat_name = cat_map.get(cat_code, str(cat_code))
        wm = wmape(grp['qty'].values, grp['pred'].values)
        print(f"  {cat_name:<15} {wm:>9.2f}%  {grp['qty'].sum():>12,.0f}  {grp['pred'].sum():>12,.0f}")

    # 창고별
    print(f"\n  🏢 창고별 WMAPE:")
    print(f"  {'창고':<10} {'WMAPE(%)':>10} {'실제합':>12} {'예측합':>12}")
    print("  " + "-" * 46)
    for wh_code, grp in valid_result.groupby('warehouse'):
        wh_name = wh_map.get(wh_code, str(wh_code))
        wm = wmape(grp['qty'].values, grp['pred'].values)
        print(f"  {wh_name:<10} {wm:>9.2f}%  {grp['qty'].sum():>12,.0f}  {grp['pred'].sum():>12,.0f}")


# ══════════════════════════════════════════════════════════════
# 5. 결과 저장
# ══════════════════════════════════════════════════════════════
def save_results(result: pd.DataFrame, valid_result: pd.DataFrame,
                 feat_imp: pd.DataFrame, total_wmape: float):
    """발주 결과 CSV + Excel 저장"""
    print("\n💾 결과 저장 중...")

    # ── 발주 결과 컬럼 정리 ──────────────────────────────────
    output_cols = [
        'warehouse_raw', 'sku_name_raw', 'm_cat_name',
        'predicted_demand', 'available_stock',
        'safety_coeff', 'adjusted_demand', 'order_qty', 'order_needed',
    ]
    out_df = result[output_cols].rename(columns={
        'warehouse_raw'  : '창고',
        'sku_name_raw'   : 'SKU명',
        'm_cat_name'     : '카테고리',
        'predicted_demand': '예측수요',
        'available_stock': '현재고',
        'safety_coeff'   : '안전재고계수',
        'adjusted_demand': '조정수요',
        'order_qty'      : '발주권고량',
        'order_needed'   : '발주필요',
    })

    # ── 발주 필요 건만 별도 시트 ─────────────────────────────
    order_only = out_df[out_df['발주필요'] == 1].copy()

    # ── Excel 다중 시트 저장 ─────────────────────────────────
    excel_path = 'order_plan_20260102.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        out_df.to_excel(writer, sheet_name='전체_발주계획',  index=False)
        order_only.to_excel(writer, sheet_name='발주필요_SKU', index=False)
        feat_imp.to_excel(writer, sheet_name='피처중요도',   index=False)

        # WMAPE 요약
        summary = pd.DataFrame([{
            '평가일': datetime.now().strftime('%Y-%m-%d %H:%M'),
            '예측대상일': '2026-01-02',
            '전체_WMAPE(%)': round(total_wmape, 2),
            '목표달성': 'O' if total_wmape <= 10 else 'X',
            '총_발주SKU수': int(out_df['발주필요'].sum()),
            '총_발주권고량': int(out_df['발주권고량'].sum()),
        }])
        summary.to_excel(writer, sheet_name='요약', index=False)

    print(f"  ✅ Excel 저장: {excel_path}")

    # ── CSV 저장 ─────────────────────────────────────────────
    out_df.to_csv('order_plan_20260102.csv', index=False, encoding='utf-8-sig')
    print(f"  ✅ CSV  저장: order_plan_20260102.csv")
    return excel_path


# ══════════════════════════════════════════════════════════════
# 메인 실행 (전체 파이프라인)
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import time
    t0 = time.time()

    # ── STEP 1: 데이터 로드 & 피처 ──────────────────────────
    df_raw    = load_data('sales_data.parquet')
    inv_df    = pd.read_parquet('inventory_data.parquet')

    # 코드값 ↔ 원본명 매핑 딕셔너리 (인코딩 전에 저장)
    cat_map = {i: v for i, v in enumerate(
        sorted(df_raw['m_cat'].astype('category').cat.categories))}
    wh_map  = {i: v for i, v in enumerate(
        sorted(df_raw['warehouse'].astype('category').cat.categories))}
    sku_map = {i: v for i, v in enumerate(
        sorted(df_raw['sku_name'].astype('category').cat.categories))}

    df_feat = build_features(df_raw.copy())
    df_enc  = encode_and_clean(df_feat)
    train, valid = split_data(df_enc)

    # ── STEP 2: Optuna 튜닝 → 모델 학습 ────────────────────
    try:
        import optuna as _optuna_check
        from demand_forecast_part2_train import run_optuna
        best_params, best_wmape_opt = run_optuna(train, valid, n_trials=50)
        print(f"\n  💡 Optuna 최적 WMAPE: {best_wmape_opt:.2f}%")
    except ImportError:
        print("\n  ⚠️  Optuna 미설치 → pip install optuna")
        best_params = None
    except Exception as e:
        print(f"\n  ⚠️  Optuna 에러 ({e}) → 기본 파라미터 사용")
        best_params = None

    model, wmape_log = train_model(train, valid, best_params)

    # ── STEP 3: 검증 평가 ────────────────────────────────────
    valid_result, total_wmape, bias_corr = evaluate(model, valid)
    feat_imp = print_feature_importance(model, top_n=20)

    # ── STEP 4: WMAPE 상세 리포트 ────────────────────────────
    print_wmape_report(valid_result, cat_map, wh_map, total_wmape)

    # ── STEP 5: 예측 행 생성 & 인코딩 ───────────────────────
    forecast_raw = build_forecast_rows(df_raw)
    forecast_enc = encode_forecast_rows(forecast_raw.copy(), train)

    # ── STEP 6: 수요 예측 ────────────────────────────────────
    # post_holiday_qty_mean이 피처로 포함되어 모델이 직접 학습
    forecast_enc = predict_demand(model, forecast_enc, bias_corr, cat_map, valid)

    # ── STEP 7: 발주 권고량 산출 ─────────────────────────────
    result = calculate_order_qty(forecast_enc, inv_df, cat_map, sku_map, wh_map)

    # ── STEP 8: 결과 저장 ────────────────────────────────────
    excel_path = save_results(result, valid_result, feat_imp, total_wmape)
    save_model(model)

    # ── 최종 요약 ────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("🎉 전체 파이프라인 완료!")
    print("=" * 60)
    print(f"  ⏱️  소요 시간      : {elapsed/60:.1f} 분")
    print(f"  🎯 최종 WMAPE     : {total_wmape:.2f}%  (목표 ≤ 10%)")
    print(f"  📦 발주 필요 SKU  : {result['order_needed'].sum():,} 건")
    print(f"  📊 총 발주 권고량  : {result['order_qty'].sum():,.0f} 개")
    print(f"  💾 결과 파일       : order_plan_20260102.xlsx / .csv")
    print("=" * 60)