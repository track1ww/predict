![capsule-render api](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=발주에%20필요한%20예측&fontSize=90)

<div align="center">

<a href="https://www.python.org/" target="_blank"><img style="margin: 10px" src="https://profilinator.rishav.dev/skills-assets/python-original.svg" alt="Python" height="50" /></a>
<a href="https://github.com/microsoft/LightGBM" target="_blank"><img style="margin: 10px" src="https://img.shields.io/badge/LightGBM-2d6a4f?style=for-the-badge&logo=leaflet&logoColor=white" alt="LightGBM" height="35" /></a>
<a href="https://optuna.org/" target="_blank"><img style="margin: 10px" src="https://img.shields.io/badge/Optuna-4169E1?style=for-the-badge&logo=python&logoColor=white" alt="Optuna" height="35" /></a>
<a href="https://pandas.pydata.org/" target="_blank"><img style="margin: 10px" src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" height="35" /></a>

<br/>

[![WMAPE](https://img.shields.io/badge/WMAPE-8.33%25-brightgreen?style=flat-square&logo=checkmarx)](.)
[![Target](https://img.shields.io/badge/목표-≤%20WMAPE10%25-blue?style=flat-square)](.)
[![SKU](https://img.shields.io/badge/발주%20SKU-3%2C686건-orange?style=flat-square)](.)
[![Order](https://img.shields.io/badge/총%20발주권고량-313%2C508개-red?style=flat-square)](.)

</div>

---

## 📌 프로젝트 개요

> **2026년 1월 2일(목요일)** 연말연시 직후 수요를 예측하여 SKU × 창고별로 필요한 물량, 재고이관량을 산출하는 코드
- **예측 대상** : 5개 센터(A~E) × 3,300 SKU → 16,500건
- **평가 지표** : WMAPE ≤ 10% (Weighted Mean Absolute Percentage Error)
- **최종 결과** : WMAPE **8.33%** ✅ | 발주 필요 SKU **3,686건** | 총 발주량 **313,508개**

---

## 🗂️ 파이프라인 구조

```
sales_data.parquet
inventory_data.parquet
        │
        ▼
┌──────────────────────┐
│  Part 1 · Features   │  피처 엔지니어링
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Part 2 · Train      │  LightGBM 모델 학습 + Optuna 튜닝
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Part 3 · Order      │  발주 권고량 산출 & 결과 저장
└──────────┬───────────┘
           │
           ▼
  order_plan_20260102.xlsx / .csv
```

---

## 📂 파일별 상세 기능

### `demand_forecast_part1_features.py` — 데이터 로드 & 피처 엔지니어링

> 원본 판매 데이터를 불러와 모델 학습에 필요한 피처를 생성합니다.

| 함수 | 역할 |
|---|---|
| `load_data()` | Parquet 파일 로드, 기간/창고/SKU/카테고리 기본 통계 출력 |
| `get_event_features()` | 날짜별 공휴일 가중치, 시즌 이벤트(추석·연말·여름) 피처 생성 |
| `add_promo_features()` | 프로모션 3종 카테고리×날짜 교차 피처 |
| `build_features()` | 날짜 기본 → 이벤트 → 프로모션 → Lag/Rolling → 요일 패턴 → 연휴직후 전용 피처까지 **전체 파이프라인 실행** |
| `encode_and_clean()` | 카테고리 변수 Label Encoding, NaN 처리 |
| `split_data()` | Train: 전체(7/1~12/31) / Valid: 연휴 직후 13일(광복절·추석·한글날 직후) |
| `build_forecast_rows()` | 2026-01-02 예측용 입력 행 생성 (16,500건) |



### `demand_forecast_part2_train.py` — LightGBM 모델 학습 & WMAPE 검증

> **LightGBM (Tweedie 회귀)** 모델을 학습하고 Optuna로 하이퍼파라미터를 최적화합니다.

#### 🤖 사용 모델

```
LightGBM — Gradient Boosting 기반 테이블형 시계열 예측
  objective  : tweedie (수요량의 0포함 우편향 분포에 적합)
  metric     : RMSE
  학습 방식  : Early Stopping + 시간 가중치 (최근 45일 강조, exp decay)
  범주형 처리: warehouse / m_cat / season_name → LightGBM 내장 categorical
```

| 함수 | 역할 |
|---|---|
| `run_optuna()` | Optuna TPESampler로 50 trial 하이퍼파라미터 탐색, 탐색 범위: `num_leaves` 63~255, `lr` 0.02~0.1, `max_depth` 6~12 등 |
| `train_model()` | Optuna best params로 최종 학습, 100 round마다 WMAPE 출력, early stopping 500 |
| `evaluate()` | 연휴 직후 Valid셋 WMAPE 산출 + **카테고리별 편향 보정 계수(bias_corr)** 생성 |
| `print_feature_importance()` | Gain 기준 피처 중요도 Top 20 출력 |
| `save_model()` | `.pkl` + `.txt` 형식 이중 저장 |

#### 📊 Optuna 탐색 결과 (Best Trial #43)

```python
tweedie_variance_power : 1.283
num_leaves             : 245
max_depth              : 12
min_child_samples      : 31
learning_rate          : 0.0999
subsample              : 0.632
colsample_bytree       : 0.713
reg_alpha              : 0.00131
reg_lambda             : 0.999
```

#### 🔍 피처 중요도 Top 5

```
same_dow_4w_mean   ██████████████████████████████  (1위)
same_dow_8w_mean   ███████████████████████████
total_weight       ███████████████████
stockout_lag1      ███████████
lag_7              ██████
```

---

### `demand_forecast_part3_order.py` — 발주량 산출 & 결과 저장

> 학습된 모델로 2026-01-02 수요를 예측하고, 재고를 차감하여 발주 권고량을 산출합니다.

| 함수 | 역할 |
|---|---|
| `encode_forecast_rows()` | 예측 행을 학습 시와 동일한 카테고리 코드값으로 변환 |
| `predict_demand()` | 모델 예측 → 카테고리 편향 보정(bias_corr) → 고오류 5개 카테고리 룰 기반 대체 |
| `calculate_order_qty()` | `발주권고량 = max(0, 예측수요 × 안전재고계수 - 현재고)` |
| `print_wmape_report()` | 카테고리별 / 창고별 WMAPE 상세 리포트 |
| `save_results()` | Excel 다중 시트(전체계획·발주필요SKU·피처중요도·요약) + CSV 저장 |

#### 📦 발주 정책

```
발주 권고량 = max(0, 예측수요 × 안전재고계수 - 사용가능재고)

안전재고계수 (카테고리별 신선도 반영):
  신선 채소류 (엽채/나물/가금육)  → 1.20  (유통기한 짧음)
  육류/유제품                     → 1.10~1.15
  가공/냉동/과자                   → 1.05  (유통기한 김)
```
---
#### 📋 예측 보정 2단계

```
1단계 — 카테고리 편향 보정 (bias_corr)
         : Valid셋 기준 실측합/예측합 비율로 체계적 과소예측 수치 보정

2단계 — 룰 기반 예측 대체 (고오류 5개 카테고리)
         : 과자 / 냉동육 / 라면·면 / 가공식품 / 육가공
         → Valid 연휴직후 실측 평균값으로 대체 (WMAPE 추가 개선)
```

---
---
#### 📈 최종 성능

| 카테고리 | WMAPE |
|---|---|
| 나물류 | **5.21%** ✅ |
| 엽채류 | 5.43% ✅ |
| 과채류 | 6.60% ✅ |
| 우유 | 6.40% ✅ |
| 냉동육 | 14.73% ⚠️ |
| **전체** | **8.33%** ✅ |

| 창고 | WMAPE |
|---|---|
| A~E 센터 전체 | 8.26% ~ 8.36% ✅ |

---


---
####**출력 파일**
```
order_plan_20260102.xlsx   # 발주 계획 (전체 / 발주필요 / 피처중요도 / 요약 시트)
order_plan_20260102.csv    # 발주 계획 CSV

ame_dow_4w_mean   ██████████████████████████████  (1위)
same_dow_8w_mean   ███████████████████████████
total_weight       ███████████████████
stockout_lag1      ███████████
lag_7              ██████
```

---

### `demand_forecast_part3_order.py` — 발주량 산출 & 결과 저장

> 학습된 모델로 2026-01-02 수요를 예측하고, 재고를 차감하여 발주 권고량을 산출합니다.

| 함수 | 역할 |
|---|---|
| `encode_forecast_rows()` | 예측 행을 학습 시와 동일한 카테고리 코드값으로 변환 |
| `predict_demand()` | 모델 예측 → 카테고리 편향 보정(bias_corr) → 고오류 5개 카테고리 룰 기반 대체 |
| `calculate_order_qty()` | `발주권고량 = max(0, 예측수요 × 안전재고계수 - 현재고)` |
| `print_wmape_report()` | 카테고리별 / 창고별 WMAPE 상세 리포트 |
| `save_results()` | Excel 다중 시트(전체계획·발주필요SKU·피처중요도·요약) + CSV 저장 |
---
---
#### 📦 발주 정책

```
발주 권고량 = max(0, 예측수요 × 안전재고계수 - 사용가능재고)

안전재고계수 (카테고리별 신선도 반영):
  신선 채소류 (엽채/나물/가금육)  → 1.20  (유통기한 짧음)
  육류/유제품                     → 1.10~1.15
  가공/냉동/과자                   → 1.05  (유통기한 김)
```
---
---
#### 📋 예측 보정 2단계

```
1단계 — 카테고리 편향 보정 (bias_corr)
         : Valid셋 기준 실측합/예측합 비율로 체계적 과소예측 수치 보정

2단계 — 룰 기반 예측 대체 (고오류 5개 카테고리)
         : 과자 / 냉동육 / 라면·면 / 가공식품 / 육가공
         → Valid 연휴직후 실측 평균값으로 대체 (WMAPE 추가 개선)
```

---
---
#### 📈 최종 성능

| 카테고리 | WMAPE |
|---|---|
| 나물류 | **5.21%** ✅ |
| 엽채류 | 5.43% ✅ |
| 과채류 | 6.60% ✅ |
| 우유 | 6.40% ✅ |
| 냉동육 | 14.73% ⚠️ |
| **전체** | **8.33%** ✅ |

| 창고 | WMAPE |
|---|---|
| A~E 센터 전체 | 8.26% ~ 8.36% ✅ |

---

---
**출력 파일**


order_plan_20260102.xlsx   # 발주 계획 (전체 / 발주필요 / 피처중요도 / 요약 시트)

order_plan_20260102.csv    # 발주 계획 CSV

```
---

![footer](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=footer)
