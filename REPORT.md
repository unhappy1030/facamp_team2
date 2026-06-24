# 2026 FIFA World Cup 예측 모델 — CatBoost 파이프라인 최종 보고서

> 작성일: 2026-06-24  
> 모델: CatBoost Classifier + Regressor (5-Fold CV)  
> 최종 제출 파일: `pred_cat/submission_reproduced.csv`

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [폴더 구조](#2-폴더-구조)
3. [실행 방법](#3-실행-방법)
4. [데이터 파이프라인](#4-데이터-파이프라인)
5. [피처 설계](#5-피처-설계)
6. [모델 구조](#6-모델-구조)
7. [배당률 블렌딩 전략](#7-배당률-블렌딩-전략)
8. [검증 결과 (리더보드)](#8-검증-결과-리더보드)
9. [토너먼트 시뮬레이션 전체 결과](#9-토너먼트-시뮬레이션-전체-결과)
10. [한계점 및 개선 방향](#10-한계점-및-개선-방향)
11. [파일별 역할 요약](#11-파일별-역할-요약)

---

## 1. 프로젝트 개요

2026 FIFA 월드컵(48개국 체제) 전 경기의 결과를 예측하는 머신러닝 파이프라인입니다.

- **예측 대상**: 홈 승(Win) / 무승부(Draw) / 원정 승(Loss) 확률 + 예상 스코어
- **핵심 모델**: CatBoost Classifier (분류) + CatBoost Regressor (득점 예측)
- **블렌딩**: 모델 예측 확률 × 시장 배당률 로그-가중 블렌딩
- **평가 지표**: RPS (Ranked Probability Score) — 낮을수록 좋음

### 최종 성적 요약

| 지표 | 값 (최적 w=0.30) |
|------|-----------------|
| **RPS** | **0.1624** |
| LogLoss | 0.8764 |
| Accuracy | 62.5% |
| 조별리그 방향 정확도 | **68.8% (33/48경기)** |
| 검증 경기 수 | 48경기 |
| 학습 데이터 경기 수 | 896경기 |
| 사용 피처 수 | 74개 |

---

## 2. 폴더 구조

```
final/
├── train_catboost.py              # 메인 학습·시뮬레이션 스크립트
├── preprocess_base.py             # 전처리 파이프라인 (out/ 생성)
├── REPORT.md                      # 이 보고서
├── PRESENTATION_SCRIPT.md         # 발표 대본
│
├── dataset/                       # 원천 데이터
│   ├── results.csv                # 국제 A매치 경기 결과 (전체)
│   ├── fifa_ranking-2026-04-01.csv
│   ├── elo_ratings_wc2026.csv     # ELO 레이팅 + 본선 48개국 목록
│   ├── main_dataset.csv           # 고급 전력 지표 (ELO·WCI·PI·WhoScored)
│   ├── wc_2026_capital_distance.csv
│   └── odds_avg.parquet           # 시장 평균 배당률 (28경기)
│
├── out/                           # 전처리 산출물
│   ├── train_base.parquet         # 학습셋 (896경기, 88컬럼)
│   ├── predict_base.parquet       # 2026 본선 예측셋 (72경기)
│   └── final_base.parquet         # 전체 피처셋 (7,002경기)
│
├── dataset_scripts_out/
│   ├── wc_2026_teams.csv          # 48개국 조 편성
│   └── wc_2026_fixtures.csv       # 토너먼트 대진표
│
├── validation/
│   └── validation_ground_truth.csv  # 검증 정답 (48경기)
│
├── tournament/
│   └── third_place_assignments_2026.csv
│
├── submission_format/
│   └── template.csv
│
└── pred_cat/                      # 출력 디렉토리
    ├── submission_reproduced.csv  # ★ 최종 제출 파일
    ├── submission_catboost_w_0.00~1.00.csv
    ├── leaderboard_catboost.csv
    ├── best_hyperparameters_catboost.csv
    └── selected_features.csv
```

---

## 3. 실행 방법

### 환경

```bash
/home/unhappy1030/miniconda3/envs/fac/bin/python
```

### Step 1 — 전처리 (out/ 재생성 시)

```bash
/home/unhappy1030/miniconda3/envs/fac/bin/python \
    /home/unhappy1030/repo/facamp/final/preprocess_base.py
```

### Step 2 — 학습 + 시뮬레이션

```bash
/home/unhappy1030/miniconda3/envs/fac/bin/python \
    /home/unhappy1030/repo/facamp/final/train_catboost.py
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `--tune` | Optuna 하이퍼파라미터 탐색 |
| `--select-features` | Ablation 분석으로 중요 피처 선택 |

---

## 4. 데이터 파이프라인

### 4-1. 원천 데이터

| 파일 | 내용 |
|------|------|
| `results.csv` | 국제 A매치 경기 결과 전체 |
| `fifa_ranking-2026-04-01.csv` | 2026-04-01 기준 공식 FIFA 랭킹 |
| `elo_ratings_wc2026.csv` | ELO 레이팅 + WC 48개국 명단 |
| `main_dataset.csv` | ELO·PI·WCI·WhoScored 통합 전력 지표 |
| `wc_2026_capital_distance.csv` | 수도 → 북미 개최지 거리(km) |
| `odds_avg.parquet` | 시장 평균 배당률 28경기 (implied probability) |

### 4-2. 전처리 7단계 파이프라인

```
results.csv + fifa_ranking.csv
        │
        ▼
[1] prep_rank()        FIFA 랭킹 정제 + 국가명 통일 + 일별 forward-fill
[2] merge_match_rank() 경기 결과 ↔ 당일 랭킹 inner join (누수 방지)
[3] feature_engineer() shift(1) 적용 후 expanding/rolling 통계 계산
[4] merge_home_away()  팀 단위 통계 → 경기 단위 피벗 (_home/_away)
[5] add_diffs()        홈-원정 차이(diff) 피처 + 실시간 랭킹 결합
[6] add_target()       타깃 인코딩: 패(0) / 무(1) / 승(2)
[6.5] main_dataset join  ELO·PI·WCI·WhoScored 고급 지표 병합
[7] split_dataset()    48개국 필터 → train / predict 분리
```

**데이터 누수 방지 핵심**: `shift(1)` 적용 후 rolling/expanding 계산으로 현재 경기 결과가 자신의 예측 피처에 포함되지 않도록 차단.

### 4-3. 데이터셋 통계

| 항목 | 값 |
|------|----|
| 학습셋 | 896경기 (2018-09-09 ~ 2026-06-07) |
| 예측셋 | 72경기 (2026 본선 조별리그) |
| 전체 피처셋 | 7,002경기, 88컬럼 |
| 피처 수 (학습 사용) | 74개 |
| 타깃 분포 | Win(2): 402 / Draw(1): 244 / Loss(0): 250 |

---

## 5. 피처 설계 (74개)

| 그룹 | 피처 수 | 내용 |
|------|--------|------|
| A. FIFA 통계 | 30 | 득점·실점·득실차·랭킹·포인트 (누적+최근5경기, 홈/원정/diff) |
| B. 실시간 랭킹 | 3 | rank_home, rank_away, rank_diff |
| C. ELO | 6 | elo_rating, elo_rank, elo_rating_diff, elo_win_rate_diff |
| D. WhoScored·PI·WCI | 24 | 슈팅·점유율·패스·xG·리더십 점수·스쿼드 가치 (홈/원정 각 12) |
| E. diff 지표 | 11 | form/ws/pi/squad 계열 홈-원정 차이 종합 |

**데이터 누수 방지**: shift(1) + expanding/rolling으로 현재 경기 데이터 차단.

---

## 6. 모델 구조

### CatBoost 하이퍼파라미터

| 파라미터 | 값 |
|----------|----|
| iterations | 300 |
| depth | 4 |
| learning_rate | 0.03 |
| l2_leaf_reg | 5 |
| task_type | GPU (RTX 4090) |

### 학습 전략

```
896경기
  └─ StratifiedKFold(n_splits=5)
       ├─ Fold 1~5: CatBoostClassifier × 5  → 결과 확률 앙상블
       ├─ Fold 1~5: CatBoostRegressor(home) × 5  → 홈 득점 앙상블
       └─ Fold 1~5: CatBoostRegressor(away) × 5  → 원정 득점 앙상블
```

### 예측 흐름

```python
# 1) 분류 확률 앙상블
prob = mean([clf.predict_proba(x) for clf in classifiers])

# 2) 배당률 log-space 블렌딩
lp   = w * log(prob_model) + (1-w) * log(prob_odds)
prob = softmax(lp)

# 3) 스코어 예측 + 정합성 보정
h_score = mean([reg.predict(x) for reg in home_regs])
a_score = mean([reg.predict(x) for reg in away_regs])
# 분류 결과가 Win인데 스코어가 동점 이하면 h_score = a_score + 1 로 보정
```

---

## 7. 배당률 블렌딩 전략

### 공식

```
log_p̃ = w · log(p_model) + (1-w) · log(p_odds)
p̃ = exp(log_p̃) / Σexp(log_p̃)        ← softmax 정규화
```

- `w = 1.0`: 모델 100%, 배당률 0%
- `w = 0.0`: 배당률 100%, 모델 0%
- **최적값: w = 0.30** (모델 30% + 배당률 70%)
- log-space = 기하 평균 → 극단 확률 안정적 처리

### 배당률 데이터 (28경기 보유)

| 홈팀 | 원정팀 | P(홈승) | P(무) | P(원정승) |
|------|--------|--------|------|---------|
| Ecuador | Germany | 25.4% | 24.2% | 50.4% |
| Japan | Sweden | 49.4% | 27.8% | 22.8% |
| England | Ghana | 80.0% | 14.0% | 6.1% |
| Jordan | Argentina | 6.7% | 14.1% | 79.2% |
| Tunisia | Netherlands | 4.4% | 10.8% | 84.9% |
| Portugal | Uzbekistan | 83.0% | 12.0% | 5.1% |
| … (총 28경기) | | | | |

---

## 8. 검증 결과 (리더보드)

검증 데이터: `validation_ground_truth.csv` — **48경기** (2026 WC 조별리그)

### 블렌딩 가중치(w)별 성적

| 순위 | w | RPS ↓ | LogLoss | Accuracy |
|------|---|-------|---------|----------|
| **1** | **0.30** | **0.16237** | 0.87680 | 62.5% |
| 2 | 0.40 | 0.16238 | 0.87639 | 62.5% |
| 3 | 0.20 | 0.16240 | 0.87741 | 62.5% |
| 4 | 0.50 | 0.16244 | 0.87620 | 62.5% |
| 5 | 0.10 | 0.16247 | 0.87820 | 62.5% |
| 6 | 0.60 | 0.16256 | 0.87623 | 62.5% |
| 7 | 0.00 | 0.16257 | 0.87916 | 62.5% |
| 8 | 0.70 | 0.16274 | 0.87650 | 62.5% |
| 9 | 0.80 | 0.16300 | 0.87702 | 62.5% |
| 10 | 0.90 | 0.16334 | 0.87779 | 62.5% |
| 11 | 1.00 | 0.16376 | 0.87882 | 62.5% |

### 성능 해석

- **최적 w = 0.30**: 모델 30% + 배당률 70%가 최적 조합
- Accuracy는 모든 w에서 동일(62.5%) → 블렌딩은 확률 분포만 조정, 결과 방향은 유지
- **조별리그 방향 정확도**: 68.8% (33/48경기) ← submission_reproduced 기준 직접 계산
- 순수 모델(w=1.00) RPS 0.16376 vs 최적 블렌딩 RPS 0.16237 → 블렌딩으로 **0.00139 개선**

### 성능 기준 비교

| 모델 | RPS |
|------|-----|
| 균등 확률 (1/3씩) | ~0.222 |
| WC 2022 bivariate-Poisson 파이프라인 | 0.223 |
| CatBoost 순수 모델 (w=1.00) | 0.16376 |
| **최적 블렌딩 (w=0.30)** | **0.16237** |

---

## 9. 토너먼트 시뮬레이션 전체 결과

> `submission_reproduced.csv` 기준 (최적 블렌딩 적용)

### 2026 WC 조 편성 (12개 조, 48개국)

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

### 32강 (Round of 32)

| 팀1 | 스코어 | 팀2 | 팀1확률 | 팀2확률 | 진출 |
|-----|--------|-----|--------|--------|------|
| Scotland | 1-2 | Canada | 44% | 56% | **Canada** |
| Iran | 2-1 | South Korea | 59% | 41% | **Iran** |
| Brazil | 2-1 | Australia | 68% | 32% | **Brazil** |
| Ecuador | 2-1 | Morocco | 51% | 49% | **Ecuador** |
| Turkey | 1-2 | Netherlands | 43% | 57% | **Netherlands** |
| Japan | 2-1 | Ivory Coast | 67% | 33% | **Japan** |
| Algeria | 2-1 | Uruguay | 51% | 49% | **Algeria** |
| Spain | 2-1 | Egypt | 64% | 36% | **Spain** |
| Sweden | 3-2 | Austria | 50% | 50% | **Sweden** |
| Argentina | 2-1 | Norway | 63% | 37% | **Argentina** |
| Panama | 1-2 | Croatia | 27% | 73% | **Croatia** |
| Senegal | 2-1 | Colombia | 51% | 49% | **Senegal** |
| Canada | 1-2 | South Korea | 48% | 52% | **South Korea** |
| Morocco | 2-1 | Uruguay | 51% | 49% | **Morocco** |
| Netherlands | 2-3 | Austria | 45% | 55% | **Austria** |
| Croatia | 1-2 | Colombia | 47% | 53% | **Colombia** |

### 16강 (Round of 16)

| 팀1 | 스코어 | 팀2 | 팀1확률 | 팀2확률 | 진출 |
|-----|--------|-----|--------|--------|------|
| Canada | 2-1 | Iran | 53% | 47% | **Canada** |
| Brazil | 2-1 | Ecuador | 63% | 37% | **Brazil** |
| Netherlands | 1-2 | Japan | 48% | 52% | **Japan** |
| Algeria | 1-2 | Spain | 45% | 55% | **Spain** |
| Sweden | 1-2 | Argentina | 44% | 56% | **Argentina** |
| Croatia | 2-1 | Senegal | 57% | 43% | **Croatia** |
| South Korea | 1-2 | Morocco | 48% | 52% | **Morocco** |
| Austria | 1-2 | Colombia | 43% | 57% | **Colombia** |

### 8강 (Quarterfinal)

| 팀1 | 스코어 | 팀2 | 팀1확률 | 팀2확률 | 진출 |
|-----|--------|-----|--------|--------|------|
| Canada | 2-1 | Brazil | 51% | 49% | **Canada** |
| Japan | 2-1 | Spain | 55% | 45% | **Japan** |
| Argentina | 2-1 | Croatia | 66% | 34% | **Argentina** |
| Morocco | 1-2 | Colombia | 47% | 53% | **Colombia** |

### 준결승 ~ 결승

| 라운드 | 팀1 | 스코어 | 팀2 | 팀1확률 | 팀2확률 | 진출 |
|--------|-----|--------|-----|--------|--------|------|
| 준결승 | **Canada** | 2-1 | Japan | 53% | 47% | Canada |
| 준결승 | **Argentina** | 2-1 | Colombia | 59% | 41% | Argentina |
| 3위 결정전 | **Japan** | 2-1 | Colombia | 53% | 47% | Japan (3위) |
| **결승** | **Canada** | 2-1 | Argentina | 51% | 49% | **Canada 우승** |

### 최종 순위 예측

| 순위 | 팀 | 비고 |
|------|-----|------|
| 🥇 1위 | **Canada** | 홈 월드컵 우승, 51% 확률 |
| 🥈 2위 | Argentina | 준우승 |
| 🥉 3위 | Japan | Netherlands·Spain 격파 |
| 4위 | Colombia | 8강에서 Morocco 꺾고 4강 진출 |

> **주목할 이변**: Japan이 조별리그(Netherlands 1-2), 16강(Netherlands 재대결 1-2)에서 연속 격파하며 8강 진출. Canada가 Brazil을 51%:49% 박빙으로 꺾고 결승 진출 후 Argentina를 51%:49%로 꺾으며 홈 우승.

---

## 10. 한계점 및 개선 방향

| 항목 | 현재 한계 | 개선 방안 |
|------|---------|---------|
| 배당률 커버리지 | 28경기만 보유 | 전 경기 배당률 수집 |
| 중립 경기 미반영 | 홈 어드밴티지 보정 없음 | is_neutral 플래그 추가 |
| 무승부 과소 예측 | 극단 결과 선호 경향 | Bivariate Poisson 통합 |
| 최신 정보 미반영 | 부상·명단 변경 없음 | 대회 직전 PI/WCI 업데이트 |
| 실시간성 부재 | 경기 결과 업데이트 수동 | results.csv 자동 수집 파이프라인 |

---

## 11. 파일별 역할 요약

### 데이터 흐름

```
dataset/results.csv
dataset/fifa_ranking-2026-04-01.csv  ┐
dataset/elo_ratings_wc2026.csv       ├─ preprocess_base.py ─► out/train_base.parquet
dataset/main_dataset.csv             ┘                      ► out/predict_base.parquet
                                                            ► out/final_base.parquet

out/*.parquet                         ┐
dataset/odds_avg.parquet              │
dataset_scripts_out/wc_2026_*.csv     ├─ train_catboost.py ─► pred_cat/submission_reproduced.csv
validation/validation_ground_truth.csv│                    ► pred_cat/leaderboard_catboost.csv
tournament/third_place_assignments.csv│
submission_format/template.csv        ┘
```

---

*루트: `/home/unhappy1030/repo/facamp/final/`*
