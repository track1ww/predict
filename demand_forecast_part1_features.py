# -*- coding: utf-8 -*-
"""
1월 2일 발주용 수요예측 - Part 1: 데이터 로드 & 피처 엔지니어링
모델  : LightGBM (테이블형 시계열 최적)
평가  : WMAPE = Σ|실제-예측| / Σ|실제| × 100  (목표 ≤ 10%)
예측일: 2026-01-02 (목요일)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 0. 상수 정의 (makedata.py 동일)
# ══════════════════════════════════════════════════════════════
ANALYSIS_DATE = datetime(2026, 1, 1)
FORECAST_DATE = datetime(2026, 1, 2)   # 예측 대상일

HOLIDAYS = {
    '2025-08-15': 2.0, '2025-09-27': 3.0, '2025-09-28': 4.5,
    '2025-09-29': 3.0, '2025-10-03': 1.5, '2025-10-09': 1.5,
    '2025-11-11': 6.0, '2025-12-25': 2.5
}

SEASON_EVENTS = [
    {'name': '여름휴가_성수기', 'start': '2025-07-15', 'end': '2025-08-20', 'weight': 1.3},
    {'name': '추석대목',        'start': '2025-09-20', 'end': '2025-09-29', 'weight': 1.8},
    {'name': '연말연시_피크',   'start': '2025-12-20', 'end': '2026-01-01', 'weight': 1.8},
]

PROMOTIONS = [
    {'name': '코리아세일페스타', 'start': '2025-11-01', 'end': '2025-11-30',
     'target_cats': ['적색육/소', '적색육/돼지', '가금육', '우유', '요구르트'], 'weight': 2.2},
    {'name': '홀리데이마켓',    'start': '2025-11-01', 'end': '2025-12-26',
     'target_cats': ['적색육/소', '적색육/돼지', '냉동육', '과자'], 'weight': 1.8},
    {'name': '컬리푸드페스타',  'start': '2025-12-18', 'end': '2025-12-29',
     'target_cats': ['가공식품', '라면/면', '육가공', '과자'], 'weight': 2.5},
]

CAT_BASE_SALES = {
    '엽채류': 180, '나물류': 160, '버섯류': 100, '과채류': 140,
    '조미채류': 170, '근채류': 120, '가금육': 55, '적색육/소': 50,
    '적색육/돼지': 70, '우유': 150, '가공유': 60, '요구르트': 70,
    '육가공': 80, '과자': 100, '라면/면': 120, '가공식품': 90, '냉동육': 30
}

DOW_WEIGHTS = {0: 1.18, 1: 1.09, 2: 1.08, 3: 1.38, 4: 0.81, 5: 0.60, 6: 0.85}


# ══════════════════════════════════════════════════════════════
# 1. 평가 지표 : WMAPE
# ══════════════════════════════════════════════════════════════
def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    WMAPE = Σ|A - P| / Σ|A| × 100
    - 판매량 많은 SKU에 자동으로 더 큰 가중치 부여
    - 소량 SKU의 이상치 영향 최소화
    - 목표: ≤ 10% (정확도 90% 이상)
    """
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(actual - predicted)) / denom * 100


# ══════════════════════════════════════════════════════════════
# 2. 데이터 로드
# ══════════════════════════════════════════════════════════════
def load_data(sales_path: str = 'sales_data.parquet') -> pd.DataFrame:
    print("=" * 60)
    print("📂 데이터 로드 중...")
    df = pd.read_parquet(sales_path)
    df['sales_date'] = pd.to_datetime(df['sales_date'])
    df = df.sort_values(['warehouse', 'sku_name', 'sales_date']).reset_index(drop=True)

    print(f"  ✅ 판매 레코드 : {len(df):,} 건")
    print(f"  📅 기간        : {df['sales_date'].min().date()} ~ {df['sales_date'].max().date()}")
    print(f"  🏢 창고        : {df['warehouse'].nunique()} 개")
    print(f"  📦 SKU         : {df['sku_name'].nunique():,} 개")
    print(f"  🗂️  카테고리    : {df['m_cat'].nunique()} 개")
    return df


# ══════════════════════════════════════════════════════════════
# 3. 피처 엔지니어링
# ══════════════════════════════════════════════════════════════
def get_event_features(date: datetime) -> dict:
    """날짜 → 이벤트 피처 딕셔너리"""
    d_str = date.strftime('%Y-%m-%d')
    holiday_w = HOLIDAYS.get(d_str, 1.0)

    season_w, season_name = 1.0, 'Normal'
    for ev in SEASON_EVENTS:
        if ev['start'] <= d_str <= ev['end']:
            if ev['weight'] > season_w:
                season_w    = ev['weight']
                season_name = ev['name']

    return {
        'holiday_weight' : holiday_w,
        'is_holiday'     : int(holiday_w > 1.0),
        'season_weight'  : season_w,
        'season_name'    : season_name,
        'is_season_event': int(season_w > 1.0),
    }


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """카테고리 × 날짜별 프로모션 피처 추가 (세분화)"""
    df['promo_weight']       = 1.0
    df['is_promo']           = 0
    df['is_korea_sale']      = 0   # 코리아세일페스타 (육류/유제품, x2.2)
    df['is_holiday_market']  = 0   # 홀리데이마켓 (육류/과자/냉동, x1.8)
    df['is_kurly_festa']     = 0   # 컬리푸드페스타 (가공/라면/육가공/과자, x2.5)
    df['promo_log_weight']   = 0.0 # log(promo_weight): 비선형 효과 포착

    promo_flags = {
        '코리아세일페스타': 'is_korea_sale',
        '홀리데이마켓'   : 'is_holiday_market',
        '컬리푸드페스타' : 'is_kurly_festa',
    }

    for promo in PROMOTIONS:
        mask = (
            (df['sales_date'].dt.strftime('%Y-%m-%d') >= promo['start']) &
            (df['sales_date'].dt.strftime('%Y-%m-%d') <= promo['end']) &
            (df['m_cat'].isin(promo['target_cats']))
        )
        df.loc[mask, 'promo_weight'] = df.loc[mask, 'promo_weight'].clip(lower=promo['weight'])
        df.loc[mask, 'is_promo']     = 1
        flag_col = promo_flags.get(promo['name'])
        if flag_col:
            df.loc[mask, flag_col] = 1

    df['promo_log_weight'] = np.log1p(df['promo_weight'] - 1.0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """전체 피처 생성 파이프라인"""
    print("\n🔧 피처 엔지니어링 시작...")

    # ── 3-1. 날짜 기본 피처 ──────────────────────────────────
    df['dayofweek']   = df['sales_date'].dt.dayofweek       # 0=월 ~ 6=일
    df['dow_weight']  = df['dayofweek'].map(DOW_WEIGHTS)
    df['month']       = df['sales_date'].dt.month
    df['day']         = df['sales_date'].dt.day
    df['weekofyear']  = df['sales_date'].dt.isocalendar().week.astype(int)
    df['is_weekday']  = (df['dayofweek'] < 5).astype(int)
    df['is_thursday'] = (df['dayofweek'] == 3).astype(int)  # 목(발주 피크)
    df['is_monday']   = (df['dayofweek'] == 0).astype(int)

    # ── 3-2. 이벤트 피처 ─────────────────────────────────────
    unique_dates = df['sales_date'].drop_duplicates().sort_values()
    ev_list = [{'sales_date': d, **get_event_features(d.to_pydatetime())} for d in unique_dates]
    ev_df   = pd.DataFrame(ev_list)
    df = df.merge(ev_df, on='sales_date', how='left')

    # ── 3-3. 프로모션 피처 ───────────────────────────────────
    df = add_promo_features(df)

    # ── 3-4. 복합 가중치 ─────────────────────────────────────
    df['total_weight'] = (
        df['dow_weight'] *
        df['holiday_weight'] *
        df['season_weight'] *
        df['promo_weight']
    )
    df['total_weight_log'] = np.log1p(df['total_weight'] - 1.0)  # 비선형 효과

    # ── 3-4b. 피크 이벤트 전/후 피처 (anticipation & hangover 효과) ──
    # 피크 직전 구매 급증(anticipation) + 피크 직후 급감(hangover) 반영
    df['days_to_chuseok']   = (pd.Timestamp('2025-09-28') - df['sales_date']).dt.days.clip(-60, 60)
    df['days_to_yearend']   = (pd.Timestamp('2025-12-31') - df['sales_date']).dt.days.clip(-60, 60)  # -30→-60: 1월2일(-2) 신호 포착
    df['days_to_bbaero']    = (pd.Timestamp('2025-11-11') - df['sales_date']).dt.days.clip(-30, 30)

    # ── 3-5. 카테고리 기본 판매량 ────────────────────────────
    df['cat_base_sales'] = df['m_cat'].map(CAT_BASE_SALES)

    # ── 3-6. Lag / Rolling 피처 (SKU × 창고 그룹) ────────────
    print("  ⏳ Lag 피처 생성 중 (약 1~2분 소요)...")
    key = ['warehouse', 'sku_name']
    df = df.sort_values(key + ['sales_date'])

    # Lag: 1, 3, 7, 14일 전 판매량
    for lag in [1, 3, 7, 14]:
        df[f'lag_{lag}'] = df.groupby(key)['qty'].shift(lag)

    # Rolling 평균 (shift(1) → 데이터 누수 방지)
    for window in [3, 7, 14, 28]:
        df[f'rolling_mean_{window}'] = (
            df.groupby(key)['qty']
              .shift(1)
              .groupby([df['warehouse'], df['sku_name']])
              .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    df['rolling_std_7'] = (
        df.groupby(key)['qty']
          .shift(1)
          .groupby([df['warehouse'], df['sku_name']])
          .transform(lambda x: x.rolling(7, min_periods=1).std())
    )

    df['rolling_sum_7'] = (
        df.groupby(key)['qty']
          .shift(1)
          .groupby([df['warehouse'], df['sku_name']])
          .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )

    # 같은 요일 기준 lag / 평균 (요일 패턴 포착)
    df['same_dow_last_week'] = df.groupby(key + ['dayofweek'])['qty'].shift(1)
    df['same_dow_4w_mean']   = (
        df.groupby(key + ['dayofweek'])['qty']
          .shift(1)
          .groupby([df['warehouse'], df['sku_name'], df['dayofweek']])
          .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )
    # 8주 요일 평균: 4주보다 안정적인 장기 요일 패턴
    df['same_dow_8w_mean']   = (
        df.groupby(key + ['dayofweek'])['qty']
          .shift(1)
          .groupby([df['warehouse'], df['sku_name'], df['dayofweek']])
          .transform(lambda x: x.rolling(8, min_periods=1).mean())
    )
    # 단기 추세: 최근 7일 평균 / 이전 7일 평균 (1이면 보합, >1이면 상승)
    roll7      = df.groupby(key)['qty'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    roll7_prev = df.groupby(key)['qty'].transform(lambda x: x.shift(8).rolling(7, min_periods=1).mean())
    df['qty_trend'] = (roll7 / (roll7_prev + 1e-9)).clip(0.5, 2.0)

    # ── 연휴 직후 전용 피처 ──────────────────────────────────
    # 연휴 직후 날짜(is_post_holiday)의 판매량만 추출하여 평균
    # → "이 SKU는 연휴 끝나고 첫날 평균 얼마나 팔리나" 직접 신호
    post_holiday_dates = pd.to_datetime([
        '2025-08-16', '2025-08-18', '2025-08-19', '2025-08-20',
        '2025-09-30', '2025-10-01', '2025-10-02',
        '2025-10-06', '2025-10-07', '2025-10-08',
        '2025-10-10', '2025-10-13', '2025-10-14',
    ])
    df['is_post_holiday'] = df['sales_date'].isin(post_holiday_dates).astype(int)

    # 연휴 직후 날짜의 판매량 평균 (SKU × 창고별)
    post_avg = (
        df[df['is_post_holiday'] == 1]
        .groupby(key)['qty']
        .mean()
        .reset_index()
        .rename(columns={'qty': 'post_holiday_qty_mean'})
    )
    df = df.merge(post_avg, on=key, how='left')
    df['post_holiday_qty_mean'] = df['post_holiday_qty_mean'].fillna(df['rolling_mean_28'])

    # ── 3-7. 가격 피처 ───────────────────────────────────────
    df['price_lag1']       = df.groupby(key)['price'].shift(1)
    df['price_change_pct'] = (df['price'] - df['price_lag1']) / (df['price_lag1'] + 1e-9) * 100

    # ── 3-8. 재고 부족 이력 ──────────────────────────────────
    df['stockout_lag1'] = df.groupby(key)['is_stockout'].shift(1)
    df['stockout_lag7'] = df.groupby(key)['is_stockout'].shift(7)

    print(f"  ✅ 피처 생성 완료: {df.shape[1]} 컬럼")
    return df


# ══════════════════════════════════════════════════════════════
# 4. 카테고리 인코딩 & 결측치 처리
# ══════════════════════════════════════════════════════════════
def encode_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Label Encoding + 결측치 처리"""
    cat_cols = ['warehouse', 'sku_name', 'm_cat', 'season_name']
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes  # LightGBM 내부 cat 처리 가능

    lag_fill_cols = [c for c in df.columns if any(
        c.startswith(p) for p in
        ['lag_', 'rolling_', 'same_dow', 'stockout_lag', 'price_lag', 'price_change']
    )]
    df[lag_fill_cols] = df[lag_fill_cols].fillna(0)
    return df


# ══════════════════════════════════════════════════════════════
# 5. 피처 컬럼 정의
# ══════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # 날짜
    'dayofweek', 'dow_weight', 'month', 'day', 'weekofyear',
    'is_weekday', 'is_thursday', 'is_monday',
    # 이벤트
    'holiday_weight', 'is_holiday', 'season_weight', 'is_season_event', 'season_name',
    # 프로모션 (세분화)
    'promo_weight', 'promo_log_weight', 'is_promo',
    'is_korea_sale', 'is_holiday_market', 'is_kurly_festa',
    # 복합 가중치
    'total_weight', 'total_weight_log',
    'days_to_chuseok', 'days_to_yearend', 'days_to_bbaero',
    # 카테고리 / ID
    'm_cat', 'cat_base_sales', 'warehouse', 'sku_name',
    # 가격
    'price', 'price_volatility', 'price_lag1', 'price_change_pct',
    # 재고 부족
    'stockout_lag1', 'stockout_lag7',
    # Lag
    'lag_1', 'lag_3', 'lag_7', 'lag_14',
    # Rolling
    'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
    'rolling_std_7', 'rolling_sum_7',
    # 요일 패턴
    'same_dow_last_week', 'same_dow_4w_mean', 'same_dow_8w_mean',
    # 단기 추세
    'qty_trend',
    # 연휴 직후 전용
    'is_post_holiday', 'post_holiday_qty_mean',
]

TARGET_COL = 'qty'


# ══════════════════════════════════════════════════════════════
# 6. Train / Valid 분할
# ══════════════════════════════════════════════════════════════
def split_data(df: pd.DataFrame):
    """
    [설계 원칙]
    - Train : 전체 기간 (7/1~12/31) — 연말연시 피크 패턴까지 모두 학습
    - Valid : "연휴 직후 + 프로모션 없음" 조건의 날짜만 추출
              → 1월2일(연말연시 직후, 프로모션 없음)과 동일 조건

    Valid 날짜 선정 기준:
      - 추석 직후  : 9/30~10/08 (추석 9/28 종료, 프로모션 없음)
      - 광복절 직후: 8/16~8/19  (광복절 8/15, 프로모션 없음)
      ※ 개천절(10/3), 한글날(10/9) 등 공휴일 당일은 제외
    """
    # Valid 날짜 정의 (학습에도 포함 — 시계열 in-sample 검증)
    # 연휴 직후 평상시 복귀 패턴을 얼마나 잘 포착하는지 측정
    post_holiday_dates = pd.to_datetime([
        # 광복절(8/15) 직후
        '2025-08-16', '2025-08-18', '2025-08-19', '2025-08-20',
        # 추석(9/28) 직후
        '2025-09-30', '2025-10-01', '2025-10-02',
        '2025-10-06', '2025-10-07', '2025-10-08',
        # 한글날(10/9) 직후
        '2025-10-10', '2025-10-13', '2025-10-14',
    ])

    # 전체 기간 학습
    train = df[df['sales_date'] <= pd.Timestamp('2025-12-31')].copy()
    # 연휴 직후 날짜만 검증
    valid = df[df['sales_date'].isin(post_holiday_dates)].copy()

    print(f"\n📊 데이터 분할:")
    print(f"  Train  : {train['sales_date'].min().date()} ~ {train['sales_date'].max().date()} | {len(train):,} 건")
    print(f"  Valid  : 연휴 직후 {len(post_holiday_dates)}일 | {len(valid):,} 건")
    print(f"  Valid 날짜: {sorted([d.strftime('%m/%d') for d in post_holiday_dates])}")
    return train, valid


# ══════════════════════════════════════════════════════════════
# 7. 예측 행 생성 (2026-01-02)
# ══════════════════════════════════════════════════════════════
def build_forecast_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    각 (warehouse, sku_name) 조합에 대해 2026-01-02 피처 행 생성
    lag 값은 원본(인코딩 전) 데이터에서 직접 계산
    """
    print("\n🗓️  2026-01-02 예측 행 생성 중...")

    forecast_ev   = get_event_features(FORECAST_DATE)
    promo_weight  = 1.0   # 2026-01-02: 모든 프로모션 종료
    is_promo      = 0
    dow_w         = DOW_WEIGHTS[FORECAST_DATE.weekday()]  # 목요일 = 1.38
    total_w       = dow_w * forecast_ev['holiday_weight'] * \
                    forecast_ev['season_weight'] * promo_weight

    records = []
    df_sorted = df_raw.sort_values(['warehouse', 'sku_name', 'sales_date'])

    for (wh, sku), grp in df_sorted.groupby(['warehouse', 'sku_name']):
        grp = grp.sort_values('sales_date')
        qty  = grp['qty'].values

        def lag_val(n):  return qty[-n]       if len(qty) >= n else 0
        def roll_mean(n): return qty[-n:].mean() if len(qty) >= 1 else 0
        def roll_std(n):  return qty[-n:].std()  if len(qty) >= n else 0
        def roll_sum(n):  return qty[-n:].sum()  if len(qty) >= 1 else 0

        # 같은 요일(목=3) 패턴
        dow_qty = grp[grp['sales_date'].dt.dayofweek == 3]['qty'].values
        same_dow_lw = dow_qty[-1]      if len(dow_qty) >= 1 else lag_val(7)
        same_dow_4w = dow_qty[-4:].mean() if len(dow_qty) >= 1 else lag_val(7)

        m_cat      = grp['m_cat'].iloc[-1]
        price_last = grp['price'].iloc[-1]
        price_prev = grp['price'].iloc[-2] if len(grp) >= 2 else price_last
        price_chg  = (price_last - price_prev) / (price_prev + 1e-9) * 100

        records.append({
            'sales_date'       : FORECAST_DATE,
            'warehouse'        : wh,
            'sku_name'         : sku,
            'm_cat'            : m_cat,
            # 날짜
            'dayofweek'        : FORECAST_DATE.weekday(),
            'dow_weight'       : dow_w,
            'month'            : FORECAST_DATE.month,
            'day'              : FORECAST_DATE.day,
            'weekofyear'       : int(FORECAST_DATE.isocalendar()[1]),
            'is_weekday'       : 1,
            'is_thursday'      : 1,
            'is_monday'        : 0,
            # 이벤트
            'holiday_weight'   : forecast_ev['holiday_weight'],
            'is_holiday'       : forecast_ev['is_holiday'],
            'season_weight'    : forecast_ev['season_weight'],
            'season_name'      : forecast_ev['season_name'],
            'is_season_event'  : forecast_ev['is_season_event'],
            # 프로모션 (세분화 — 2026-01-02는 모든 프로모션 종료)
            'promo_weight'        : promo_weight,
            'promo_log_weight'    : np.log1p(promo_weight - 1.0),  # = 0.0
            'is_promo'            : is_promo,
            'is_korea_sale'       : 0,
            'is_holiday_market'   : 0,
            'is_kurly_festa'      : 0,
            # 복합 가중치
            'total_weight'        : total_w,
            'total_weight_log'    : np.log1p(total_w - 1.0),
            # 이벤트 전/후 거리 (2026-01-02 기준)
            'days_to_chuseok'     : int(np.clip((pd.Timestamp('2025-09-28') - pd.Timestamp('2026-01-02')).days, -30, 30)),
            'days_to_yearend'     : int(np.clip((pd.Timestamp('2025-12-31') - pd.Timestamp('2026-01-02')).days, -30, 30)),
            'days_to_bbaero'      : int(np.clip((pd.Timestamp('2025-11-11') - pd.Timestamp('2026-01-02')).days, -10, 10)),
            # 카테고리
            'cat_base_sales'   : CAT_BASE_SALES.get(m_cat, 100),
            # 가격
            'price'            : price_last,
            'price_volatility' : grp['price_volatility'].iloc[-1],
            'price_lag1'       : price_prev,
            'price_change_pct' : price_chg,
            # 재고 부족
            'stockout_lag1'    : grp['is_stockout'].iloc[-1],
            'stockout_lag7'    : grp['is_stockout'].iloc[-7] if len(grp) >= 7 else 0,
            # Lag
            'lag_1'            : lag_val(1),
            'lag_3'            : lag_val(3),
            'lag_7'            : lag_val(7),
            'lag_14'           : lag_val(14),
            # Rolling
            'rolling_mean_3'   : roll_mean(3),
            'rolling_mean_7'   : roll_mean(7),
            'rolling_mean_14'  : roll_mean(14),
            'rolling_mean_28'  : roll_mean(28),
            'rolling_std_7'    : roll_std(7),
            'rolling_sum_7'    : roll_sum(7),
            # 요일 패턴
            'same_dow_last_week': same_dow_lw,
            'same_dow_4w_mean'  : same_dow_4w,
            'same_dow_8w_mean'  : dow_qty[-8:].mean() if len(dow_qty) >= 1 else lag_val(7),
            # 단기 추세: 연휴 직후는 평상시 회귀 → 1.0에 가까움
            'qty_trend'         : 1.0,
            # 연휴 직후 전용 (2026-01-02는 연말연시 직후 → is_post_holiday=1)
            'is_post_holiday'        : 1,
            'post_holiday_qty_mean'  : same_dow_4w,  # 근사값: 연휴직후 4주 요일평균
        })

    forecast_df = pd.DataFrame(records)
    print(f"  ✅ 예측 행: {len(forecast_df):,} 건 (창고 {forecast_df['warehouse'].nunique()}개 × SKU)")
    return forecast_df


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────
if __name__ == '__main__':
    df_raw  = load_data('sales_data.parquet')
    df_feat = build_features(df_raw.copy())
    df_enc  = encode_and_clean(df_feat)
    train, valid = split_data(df_enc)
    forecast_rows = build_forecast_rows(df_raw)
    print("\n✅ Part 1 완료 — Part 2(모델 학습)로 이동하세요.")