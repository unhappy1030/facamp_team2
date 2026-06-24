# 2026 FIFA World Cup 예측 모델 — CatBoost 파이프라인 최종 보고서

> 작성일: 2026-06-24  
> 모델: CatBoost Classifier + Regressor (5-Fold CV)  
> 최종 제출 파일: `pred_cat/submission_catboost_w_0.20.csv`

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [폴더 구조](#2-폴더-구조)
3. [실행 방법](#3-실행-방법)
4. [데이터 파이프라인](#4-데이터-파이프라인)
5. [피처 설계](#5-피처-설계)
6. [모델 구조](#6-모델-구조)
7. [검증 결과 (리더보드)](#7-검증-결과-리더보드)
8. [토너먼트 시뮬레이션 결과](#8-토너먼트-시뮬레이션-결과)
9. [배당률 블렌딩 전략](#9-배당률-블렌딩-전략)
10. [파일별 역할 요약](#10-파일별-역할-요약)

---

## 1. 프로젝트 개요

2026 FIFA 월드컵(48개국 체제) 전 경기의 결과를 예측하는 머신러닝 파이프라인입니다.

- **예측 대상**: 각 경기의 홈 승(Win) / 무승부(Draw) / 원정 승(Loss) 확률 + 예상 스코어
- **핵심 모델**: CatBoost Classifier (분류) + CatBoost Regressor (득점 예측)
- **블렌딩**: 모델 예측 확률 × 시장 배당률 로그-가중 블렌딩
- **평가 지표**: RPS (Ranked Probability Score) — 낮을수록 좋음

### 최종 성적 요약

| 지표 | 값 (최적 w=0.20) |
|------|-----------------|
| **RPS** | **0.1756** |
| LogLoss | 0.9315 |
| Accuracy | 64.58% |
| 검증 경기 수 | 48경기 |

---

## 2. 폴더 구조

```
final/
├── train_catboost.py              # 메인 학습·시뮬레이션 스크립트
├── preprocess_base.py             # 전처리 파이프라인 (out/ 생성)
├── REPORT.md                      # 이 보고서
│
├── dataset/                       # 원천 데이터
│   ├── results.csv                # 국제 A매치 경기 결과 (전체)
│   ├── fifa_ranking-2026-04-01.csv  # FIFA 랭킹 (2026-04-01 기준)
│   ├── elo_ratings_wc2026.csv     # ELO 레이팅 + 본선 48개국 목록
│   ├── main_dataset.csv           # 고급 전력 지표 (ELO·WCI·PI·WhoScored)
│   ├── wc_2026_capital_distance.csv # 각 국가 수도 → 북미 개최지 거리(km)
│   └── odds_avg.parquet           # 경기별 시장 평균 배당률
│
├── out/                           # 전처리 산출물 (preprocess_base.py 생성)
│   ├── train_base.parquet         # 학습 데이터셋 (896경기, 88컬럼)
│   ├── predict_base.parquet       # 2026 본선 예측셋 (72경기)
│   └── final_base.parquet         # 전체 피처 데이터셋 (7,002경기)
│
├── dataset_scripts_out/           # 대회 구조 데이터
│   ├── wc_2026_teams.csv          # 48개국 조 편성표
│   └── wc_2026_fixtures.csv       # 토너먼트 대진표 (16강~결승)
│
├── validation/
│   └── validation_ground_truth.csv  # 외부 검증 정답 (48경기)
│
├── tournament/
│   └── third_place_assignments_2026.csv  # 3위팀 와일드카드 배정 규정표
│
├── submission_format/
│   └── template.csv               # 조별리그 경기 템플릿 (예측 대상 경기 목록)
│
└── pred_cat/                      # 출력 디렉토리 (실행 시 자동 생성)
    ├── submission_catboost_w_0.00.csv ~ w_1.00.csv  # 블렌딩 가중치별 제출 파일
    ├── leaderboard_catboost.csv   # 가중치별 검증 성적 비교표
    ├── best_hyperparameters_catboost.csv  # Fold별 최적 하이퍼파라미터 캐시
    └── selected_features.csv      # (--select-features 사용 시) 선택된 피처 목록
```

---

## 3. 실행 방법

### 환경

```bash
conda activate fac
# 또는 직접 실행:
/home/unhappy1030/miniconda3/envs/fac/bin/python
```

### Step 1 — 전처리 (out/ 재생성이 필요한 경우)

```bash
/home/unhappy1030/miniconda3/envs/fac/bin/python \
    /home/unhappy1030/repo/facamp/final/preprocess_base.py
```

> `out/train_base.parquet`, `out/predict_base.parquet`, `out/final_base.parquet` 3개 파일이 생성됩니다.  
> 이미 `out/` 파일이 존재하면 이 단계는 생략해도 됩니다.

### Step 2 — CatBoost 학습 + 시뮬레이션 (기본 실행)

```bash
/home/unhappy1030/miniconda3/envs/fac/bin/python \
    /home/unhappy1030/repo/facamp/final/train_catboost.py
```

### Step 3 — 옵션별 실행

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--tune` | Optuna 하이퍼파라미터 탐색 (15 trials/fold) | `--tune` |
| `--select-features` | Ablation 분석으로 중요 피처만 선택 | `--select-features` |
| `--external-val <path>` | 검증 CSV 경로 변경 | `--external-val /path/to/csv` |
| `--subsample-rate <float>` | 검증 서브샘플링 비율 (기본 0.8) | `--subsample-rate 0.7` |

### Step 4 — GPU 사용

`train_catboost.py`는 CatBoost `task_type="GPU"` 설정이 기본값입니다.  
GPU 없이 실행하려면 코드 상단 두 곳의 `task_type="GPU"` 를 `task_type="CPU"` 로 변경하세요.

```python
# train_catboost.py 29~36번째 줄
MODELS["CatBoost"] = lambda: CatBoostClassifier(
    iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=5,
    random_seed=42, verbose=0, task_type="CPU"   # ← 변경
)
REGRESSORS["CatBoost"] = lambda: CatBoostRegressor(
    iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=5,
    random_seed=42, verbose=0, task_type="CPU"   # ← 변경
)
```

---

## 4. 데이터 파이프라인

### 4-1. 원천 데이터 설명

| 파일 | 내용 | 비고 |
|------|------|------|
| `results.csv` | 국제 A매치 전체 경기 결과 | 날짜·홈팀·원정팀·스코어·대회명·중립경기여부 |
| `fifa_ranking-2026-04-01.csv` | 공식 FIFA 랭킹 스냅샷 | 2026-04-01 기준, 랭킹 포인트·순위 포함 |
| `elo_ratings_wc2026.csv` | ELO 레이팅 + 본선 48개국 명단 | 전력 기반 누적 레이팅, WC 참가국 필터용 |
| `main_dataset.csv` | 고급 전력 지표 통합본 | ELO·PI(Player Impact)·WCI(스쿼드 가치)·WhoScored 지표 |
| `wc_2026_capital_distance.csv` | 수도 → 개최지 거리(km) | 이동 피로도 피처 생성용 (현재는 보조 용도) |
| `odds_avg.parquet` | 시장 평균 배당률 | P(home win), P(draw), P(away win) |

### 4-2. 전처리 7단계 파이프라인 (`preprocess_base.py`)

```
results.csv + fifa_ranking.csv
        │
        ▼
[1] prep_rank()
    - 2018-08-01 이후 랭킹만 사용
    - 국가명 통일 (Korea Republic → South Korea 등)
    - 경기일까지 일별 forward-fill 연장
        │
        ▼
[2] merge_match_rank()
    - 경기 결과 ↔ 랭킹 inner join (경기일 기준 당일 랭킹 사용)
    - 데이터 누수 방지: 경기 당일 랭킹만 매핑
        │
        ▼
[3] feature_engineer()
    - 팀 단위 long-format 변환
    - shift(1) 적용 후 누적 평균(expanding) + 롤링 평균(window=5)
    - 생성 통계: goals_avg, goals_suf_avg, gd_avg, rank_avg, points_avg
        │
        ▼
[4] merge_home_away()
    - 팀 단위 통계를 다시 경기 단위로 피벗 (_home/_away 접미사)
        │
        ▼
[5] add_diffs()
    - 홈-원정 차이(diff) 피처 추가
    - 실시간 FIFA 랭킹·스코어 결합
        │
        ▼
[6] add_target()
    - 홈팀 기준 결과 인코딩: 패(0) / 무(1) / 승(2)
        │
        ▼
[6.5] main_dataset.csv 고급 지표 left join
    - ELO 레이팅, PI 리더십 점수, WCI 스쿼드 가치, WhoScored 지표
    - 예측셋(2026 본선)은 각 팀의 최신 스냅샷으로 결측 보정
        │
        ▼
[7] split_dataset()
    - WC 2026 본선 48개국 경기만 필터
    - 결과 있음 → train_base.parquet (학습)
    - 결과 없음 → predict_base.parquet (2026 본선 예측)
```

### 4-3. 데이터셋 통계

| 항목 | 값 |
|------|----|
| 학습 데이터(train_base) | **896경기** |
| 학습 기간 | 2018-09-09 ~ 2026-06-07 |
| 예측 데이터(predict_base) | **72경기** (2026 본선 조별리그) |
| 전체 피처 데이터(final_base) | **7,002경기**, 88컬럼 |
| 피처 수 (학습 사용) | **74개** |
| 타깃 분포 | Win(2): 402 / Draw(1): 244 / Loss(0): 250 |

---

## 5. 피처 설계

### 5-1. 피처 그룹 분류

총 74개 피처를 5개 그룹으로 분류합니다.

#### 그룹 A — FIFA 랭킹 기반 통계 (30개)

누적(expanding) 평균과 최근 5경기(rolling) 평균으로 계산. 홈·원정·차이(diff) 3방향.

| 피처 | 설명 |
|------|------|
| `goals_avg_{home/away/diff}` | 누적 평균 득점 |
| `goals_avg_lastn_{home/away/diff}` | 최근 5경기 평균 득점 |
| `goals_suf_avg_{home/away/diff}` | 누적 평균 실점 |
| `goals_suf_avg_lastn_{home/away/diff}` | 최근 5경기 평균 실점 |
| `gd_avg_{home/away/diff}` | 누적 평균 득실차 |
| `gd_avg_lastn_{home/away/diff}` | 최근 5경기 평균 득실차 |
| `rank_avg_{home/away/diff}` | 누적 평균 FIFA 랭킹 (상대 팀 랭킹) |
| `rank_avg_lastn_{home/away/diff}` | 최근 5경기 평균 FIFA 랭킹 |
| `points_avg_{home/away/diff}` | 누적 평균 FIFA 포인트 |
| `points_avg_lastn_{home/away/diff}` | 최근 5경기 평균 FIFA 포인트 |

#### 그룹 B — 실시간 FIFA 랭킹 (3개)

| 피처 | 설명 |
|------|------|
| `rank_home` | 경기 당일 홈팀 FIFA 순위 |
| `rank_away` | 경기 당일 원정팀 FIFA 순위 |
| `rank_diff` | 순위 차이 (홈 - 원정) |

#### 그룹 C — ELO 레이팅 (4개)

| 피처 | 설명 |
|------|------|
| `h_elo_rating`, `a_elo_rating` | 누적 ELO 점수 (홈·원정) |
| `h_elo_rank`, `a_elo_rank` | ELO 기반 순위 |
| `elo_rating_diff` | ELO 점수 차이 |
| `elo_win_rate_diff` | ELO 기준 승률 차이 × 1000 |

#### 그룹 D — WhoScored & PI 지표 (24개)

| 피처 | 설명 |
|------|------|
| `h/a_form_goals_avg` | 최근 폼 득점 평균 |
| `h/a_form_goals_avg_l10` | 최근 10경기 폼 득점 |
| `h/a_form_goals_suf_avg` | 최근 폼 실점 평균 |
| `h/a_form_win_rate` | 최근 폼 승률 |
| `h/a_ws_shots_pg` | 경기당 슈팅 수 |
| `h/a_ws_possession_pct` | 평균 점유율(%) |
| `h/a_ws_pass_pct` | 패스 성공률(%) |
| `h/a_ws_ws_rating` | WhoScored 종합 레이팅 |
| `h/a_ws_xg_pg` | 경기당 기대 득점(xG) |
| `h/a_ws_xg_diff_pg` | 경기당 xG 차이 |
| `h/a_pi_best_lead_score` | PI 최고 리더십 점수 |
| `h/a_pi_avg_lead_score` | PI 평균 리더십 점수 |

#### 그룹 E — 종합 차이(diff) 지표 (10개)

| 피처 | 설명 |
|------|------|
| `form_goals_diff` | 폼 득점 차 |
| `form_ga_diff` | 폼 실점 차 |
| `form_wr_diff` | 폼 승률 차 |
| `ws_poss_diff` | 점유율 차 |
| `ws_rating_diff` | WhoScored 레이팅 차 |
| `ws_xg_diff` | xG 차 |
| `pi_leader_diff` | PI 최고 리더십 점수 차 |
| `pi_n_leaders_diff` | 핵심 선수 수 차 |
| `squad_value_diff` | WCI 스쿼드 가치 차 |
| `h/a_wci_squad_value` | 스쿼드 시장 가치 |

---

## 6. 모델 구조

### 6-1. CatBoost 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `iterations` | 300 | 트리 개수 |
| `depth` | 4 | 트리 깊이 |
| `learning_rate` | 0.03 | 학습률 |
| `l2_leaf_reg` | 5 | L2 정규화 |
| `random_seed` | 42 | 재현성 |
| `task_type` | GPU | RTX 4090 가속 |

### 6-2. 학습 전략

```
전체 학습 데이터 (896경기)
        │
        ▼ StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  ┌─────┴─────┐
  │  Fold 1~5  │  각 Fold마다:
  │            │   - CatBoostClassifier 학습 (3클래스 분류)
  │            │   - CatBoostRegressor 학습 × 2 (홈 득점, 원정 득점)
  └─────┬─────┘
        │
        ▼
  5개 Classifier + 5개 HomeRegressor + 5개 AwayRegressor
  (앙상블: 예측 시 5 모델 평균)
```

### 6-3. 예측 로직 (`Predictor.predict`)

```python
# 1) 분류 확률 앙상블
prob = mean([model.predict_proba(x) for model in trained_models])
# → [P(Loss), P(Draw), P(Win)]

# 2) 배당률 블렌딩 (log-space)
log_prob = w * log(prob) + (1-w) * log(market_odds)
prob = softmax(log_prob)

# 3) 스코어 예측
h_score = mean([reg_h.predict(x) for reg_h in regressors_home])
a_score = mean([reg_a.predict(x) for reg_a in regressors_away])

# 4) 분류 결과와 스코어 정합성 보정
if pred_class == Win and h_score <= a_score: h_score = a_score + 1
if pred_class == Loss and a_score <= h_score: a_score = h_score + 1
if pred_class == Draw and h_score != a_score: h_score = a_score = round(mean)
```

---

## 7. 검증 결과 (리더보드)

검증 데이터: `validation/validation_ground_truth.csv` — **48경기** (2026 WC 그룹 스테이지 일부)

### 블렌딩 가중치(w)별 성적

> w = 모델 신뢰도 (1.0 = 모델 100%, 0.0 = 배당률 100%)

| 순위 | w | RPS ↓ | LogLoss | Accuracy |
|------|---|-------|---------|----------|
| **1** | **0.20** | **0.17556** | 0.9315 | 64.58% |
| 2 | 0.10 | 0.17556 | 0.9321 | 64.58% |
| 3 | 0.00 | 0.17559 | 0.9329 | 64.58% |
| 4 | 0.30 | 0.17560 | 0.9313 | 64.58% |
| 5 | 0.40 | 0.17571 | 0.9312 | 64.58% |
| 6 | 0.50 | 0.17590 | 0.9315 | 64.58% |
| 7 | 0.60 | 0.17618 | 0.9320 | 64.58% |
| 8 | 0.70 | 0.17656 | 0.9329 | 64.58% |
| 9 | 0.80 | 0.17706 | 0.9341 | 64.58% |
| 10 | 0.90 | 0.17769 | 0.9357 | 64.58% |
| 11 | 1.00 | 0.17844 | 0.9376 | 64.58% |

### 해석

- **최적 w = 0.20**: 모델 20% + 배당률 80% 블렌딩이 RPS 기준 최적
- w가 낮을수록(배당률 의존도 높을수록) RPS가 낮아지는 경향 → 시장 배당률이 여전히 강력한 신호
- Accuracy는 모든 w에서 동일(64.58%) → 분류 경계 자체는 바뀌지 않고 확률 분포만 조정됨
- w=0.20 vs w=0.00 간 RPS 차이가 미미(0.00004) → 순수 배당률 대비 모델이 소폭 개선

### RPS 기준

| 수준 | RPS |
|------|-----|
| 순수 배당률 (w=0.00) | 0.1756 |
| **최적 블렌딩 (w=0.20)** | **0.1756** |
| 균등 확률 (1/3씩) | ~0.222 |
| WC 2022 당시 dl/ 파이프라인 | 0.223 |

---

## 8. 토너먼트 시뮬레이션 결과

> 최적 가중치 **w=0.20** 기준 시뮬레이션 (`submission_catboost_w_0.20.csv`)

### 2026 WC 조 편성

| 조 | 팀 |
|----|----|
| A | Mexico, South Africa, South Korea, Czechia |
| B | Canada, Switzerland, Qatar, Bosnia and Herzegovina |
| C | Brazil, Morocco, Haiti, Scotland |
| D | USA, Paraguay, Australia, Türkiye |
| E | Germany, Curaçao, Ivory Coast, Ecuador |
| F | Netherlands, Japan, Sweden, Tunisia |
| G | Belgium, Egypt, Iran, New Zealand |
| H | Spain, Cape Verde, Saudi Arabia, Uruguay |
| I | France, Senegal, Norway, Iraq |
| J | Argentina, Algeria, Austria, Jordan |
| K | Portugal, DR Congo, Uzbekistan, Colombia |
| L | England, Croatia, Ghana, Panama |

### 준결승 이후 예측 결과

| 라운드 | 팀1 | 스코어 | 팀2 | 팀1 확률 | 팀2 확률 |
|--------|-----|--------|-----|---------|---------|
| 준결승 | Ecuador | 1 - 2 | **Japan** | 49% | 51% |
| 준결승 | **Argentina** | 2 - 1 | Morocco | 57% | 43% |
| 3위 결정전 | Ecuador | 1 - 2 | **Morocco** | 50% | 50% |
| **결승** | **Japan** | 2 - 1 | Argentina | 52% | 48% |

### 우승 예측: 🇯🇵 Japan

Japan이 52% 확률로 Argentina를 꺾고 우승 예측. Ecuador·Morocco가 4강 진출하는 이변도 예측.

---

## 9. 배당률 블렌딩 전략

### 블렌딩 공식

```
log_p_blend = w * log(p_model) + (1-w) * log(p_odds)
p_blend = softmax(log_p_blend)
```

- **p_model**: CatBoost 앙상블 예측 확률 [P(Loss), P(Draw), P(Win)]
- **p_odds**: 시장 평균 배당률에서 역산한 확률 (같은 순서)
- **w**: 모델 가중치 (0.0~1.0)
- log-space 블렌딩은 확률의 기하 평균으로, 극단적 확률이 부드럽게 처리됨

### 배당률 데이터 (`odds_avg.parquet`)

- 컬럼: `home_team`, `away_team`, `p_home`, `p_draw`, `p_away`
- 이미 확률로 변환된 값(implied probability) 저장

### 전략 근거

시장 배당률은 수천 명의 베터가 종합한 정보로 단기 이벤트 예측에서 강력한 기준선(baseline)을 제공합니다. 단순 모델 단독보다 배당률을 80% 반영하는 것이 검증 RPS를 개선한다는 점이 리더보드에서 확인됩니다.

---

## 10. 파일별 역할 요약

### 실행 스크립트

| 파일 | 역할 | 주요 입력 | 주요 출력 |
|------|------|----------|----------|
| `preprocess_base.py` | 전처리 파이프라인 | `dataset/*.csv` | `out/*.parquet` |
| `train_catboost.py` | 학습 + 검증 + 시뮬레이션 | `out/*.parquet`, `dataset/*.csv` | `pred_cat/*.csv` |

### 데이터 파일

| 파일 | 생성 주체 | 소비 주체 |
|------|----------|----------|
| `dataset/results.csv` | 외부 | `preprocess_base.py` |
| `dataset/fifa_ranking-2026-04-01.csv` | 외부 | `preprocess_base.py` |
| `dataset/elo_ratings_wc2026.csv` | 외부 | `preprocess_base.py` |
| `dataset/main_dataset.csv` | 외부 | `preprocess_base.py` |
| `dataset/odds_avg.parquet` | 외부 | `train_catboost.py` |
| `dataset/wc_2026_capital_distance.csv` | 외부 | `train_catboost.py` |
| `out/train_base.parquet` | `preprocess_base.py` | `train_catboost.py` |
| `out/predict_base.parquet` | `preprocess_base.py` | `train_catboost.py` |
| `out/final_base.parquet` | `preprocess_base.py` | `train_catboost.py` |
| `dataset_scripts_out/wc_2026_teams.csv` | 외부 | `train_catboost.py` |
| `dataset_scripts_out/wc_2026_fixtures.csv` | 외부 | `train_catboost.py` |
| `validation/validation_ground_truth.csv` | 외부 | `train_catboost.py` |
| `tournament/third_place_assignments_2026.csv` | 외부 | `train_catboost.py` |
| `submission_format/template.csv` | 외부 | `train_catboost.py` |

### 출력 파일 (`pred_cat/`)

| 파일 | 설명 |
|------|------|
| `submission_catboost_w_0.00.csv` ~ `w_1.00.csv` | 블렌딩 가중치별 전 경기 예측 결과 |
| `leaderboard_catboost.csv` | 가중치별 RPS/LogLoss/Acc 비교표 |
| `best_hyperparameters_catboost.csv` | Fold별 최적 파라미터 캐시 (--tune 사용 시) |
| `selected_features.csv` | 선택된 필수 피처 목록 캐시 (--select-features 사용 시) |

---

*이 보고서는 `final/` 디렉토리 기준으로 작성되었습니다. 모든 경로는 `/home/unhappy1030/repo/facamp/final/` 을 루트로 합니다.*
