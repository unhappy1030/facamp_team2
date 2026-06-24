"""
2026 FIFA World Cup — 가벼운 모델(XGBoost) 학습용 기본 전처리 파이프라인 (Base - FIFA Rank & Results 기반)

day1 노트북(02~06)의 전처리 로직을 계승하며, 평균 득실차(GD - Goal Difference) 피처를 추가하고,
결측 처리와 데이터 누수 방지를 고려하여 XGBoost 모델 학습이 가능한 형태로 정제합니다.

파이프라인:
  1. prep_rank()        : FIFA 랭킹 정제 + 국가별 일별 forward-fill   -> rank_base.parquet
  2. merge_match_rank() : 경기 결과에 양 팀 랭킹 매칭(inner)          -> match_rank_base.parquet
  3. feature_engineer() : 팀 단위 long-format + expanding/rolling 통계 (득실차 gd_avg 추가) -> stats_base.parquet
  4. merge_home_away()  : 다시 경기 단위로 피벗(_home/_away)          -> features_base.parquet
  5. add_diffs()        : 양 팀 stat/rank 차이(diff)                  -> final_base.parquet (전체)
  6. add_target()       : 홈팀 기준 정수형 타깃 레이블(loss: 0, draw: 1, win: 2) 추가
  7. split_dataset()    : 2026 본선 48개국 경기만 필터링하여 학습셋/예측셋 분리
                          -> train_base.parquet (라벨有) / predict_base.parquet (본선 fixture)

leakage 방지: 롤링/누적 통계는 shift(1), 랭킹 결합은 경기일 기준 당일 랭킹 사용.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
DATASET_DIR = Path("/home/unhappy1030/repo/facamp/final/dataset")
OUT_DIR = Path("/home/unhappy1030/repo/facamp/final/out")

RESULTS_CSV = DATASET_DIR / "results.csv"
RANKING_CSV = DATASET_DIR / "fifa_ranking-2026-04-01.csv"
ELO_CSV = DATASET_DIR / "elo_ratings_wc2026.csv"  # 본선 진출국 리스트 로딩용

RANK_START = "2018-08-01"   # 이 날짜 이후 랭킹만 사용 (day1과 동일)
ROLL_WINDOW = 5             # 최근 5경기 롤링 윈도우

# 국가명 통일 (results.csv 표기 기준)
RANK_NAME_MAP = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Cabo Verde": "Cape Verde",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Kyrgyz Republic": "Kyrgyzstan",
    "The Gambia": "Gambia",
    "Türkiye": "Turkey",
}

ELO_NAME_MAP = {
    "Czechia": "Czech Republic",
}

MAIN_NAME_MAP = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
}

def build_main_latest_snapshots(df_main: pd.DataFrame) -> dict:
    """
    각 국가의 main_dataset.csv 내 전력 지표들이 결측치(NaN)가 아닌 가장 최근 경기 기준으로 스냅샷을 빌드합니다.
    """
    feature_keys = {
        "elo_rating": ("h_elo_rating", "a_elo_rating"),
        "elo_rank": ("h_elo_rank", "a_elo_rank"),
        "pi_best_lead_score": ("h_pi_best_lead_score", "a_pi_best_lead_score"),
        "pi_avg_lead_score": ("h_pi_avg_lead_score", "a_pi_avg_lead_score"),
        "wci_squad_value": ("h_wci_squad_value", "a_wci_squad_value"),
        "form_goals_avg": ("h_form_goals_avg", "a_form_goals_avg"),
        "form_goals_avg_l10": ("h_form_goals_avg_l10", "a_form_goals_avg_l10"),
        "form_goals_suf_avg": ("h_form_goals_suf_avg", "a_form_goals_suf_avg"),
        "form_win_rate": ("h_form_win_rate", "a_form_win_rate"),
        "ws_shots_pg": ("h_ws_shots_pg", "a_ws_shots_pg"),
        "ws_possession_pct": ("h_ws_possession_pct", "a_ws_possession_pct"),
        "ws_pass_pct": ("h_ws_pass_pct", "a_ws_pass_pct"),
        "ws_ws_rating": ("h_ws_ws_rating", "a_ws_ws_rating"),
        "ws_xg_pg": ("h_ws_xg_pg", "a_ws_xg_pg"),
        "ws_xg_diff_pg": ("h_ws_xg_diff_pg", "a_ws_xg_diff_pg"),
        "elo_wins": ("h_elo_wins", "a_elo_wins"),
        "elo_matches_total": ("h_elo_matches_total", "a_elo_matches_total"),
        "pi_n_key_leaders": ("h_pi_n_key_leaders", "a_pi_n_key_leaders")
    }
    
    teams = set(df_main["home_team"].unique()) | set(df_main["away_team"].unique())
    snapshots = {}
    
    for team in teams:
        team_matches = df_main[(df_main["home_team"] == team) | (df_main["away_team"] == team)].sort_values("date", ascending=False)
        if len(team_matches) == 0:
            continue
            
        team_snap = {}
        for key, (h_col, a_col) in feature_keys.items():
            val = np.nan
            for _, match in team_matches.iterrows():
                is_home = match["home_team"] == team
                col = h_col if is_home else a_col
                if not pd.isna(match[col]):
                    val = match[col]
                    break
            
            # 기록이 모두 NaN일 때 디폴트
            if pd.isna(val):
                is_home = team_matches.iloc[0]["home_team"] == team
                col = h_col if is_home else a_col
                val = team_matches.iloc[0][col]
                
            team_snap[key] = val
            
        snapshots[team] = team_snap
        
    return snapshots

# ---------------------------------------------------------------------------
# 1) FIFA 랭킹 정제 + 일별 격자 연장 (Forward-fill)
# ---------------------------------------------------------------------------
def prep_rank(end_date: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(RANKING_CSV)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    df = df[df["rank_date"] >= RANK_START].reset_index(drop=True)
    df["country_full"] = df["country_full"].replace(RANK_NAME_MAP)


    # 2026 본선 경기일정까지 랭킹을 연장하기 위한 일별 date_range 생성
    full_idx = pd.date_range(RANK_START, end_date, freq="D")

    def fill_group(g):
        g = g.set_index("rank_date").sort_index()
        g = g[~g.index.duplicated(keep="last")]
        return g.reindex(full_idx).ffill()

    df = (
        df.groupby("country_full", group_keys=True)
        .apply(fill_group, include_groups=False)
        .rename_axis(["country_full", "rank_date"])
        .reset_index()
    )
    df = df.dropna(subset=["rank"]).reset_index(drop=True)
    df = df.astype({"rank": int, "rank_change": int})
    return df

# ---------------------------------------------------------------------------
# 2) 경기 결과 + 랭킹 병합
# ---------------------------------------------------------------------------
def merge_match_rank(df_rank: pd.DataFrame) -> pd.DataFrame:
    df_rank = df_rank.drop(columns=["country_abrv", "previous_points", "confederation"])

    df_match = pd.read_csv(RESULTS_CSV)
    df_match["date"] = pd.to_datetime(df_match["date"])

    df_home = pd.merge(
        df_match, df_rank,
        left_on=["date", "home_team"],
        right_on=["rank_date", "country_full"],
        how="inner",
    )
    df = pd.merge(
        df_home, df_rank,
        left_on=["date", "away_team"],
        right_on=["rank_date", "country_full"],
        suffixes=["_home", "_away"],
        how="inner",
    )
    df = df.drop(columns=["country_full_home", "country_full_away"])
    df = df.astype({"home_score": "float64", "away_score": "float64"})
    return df

# ---------------------------------------------------------------------------
# 3) 피처 엔지니어링: 롤링/누적 통계 계산 (득실차 gd_avg 추가)
# ---------------------------------------------------------------------------
STAT_COLS = [
    "goals_avg", "goals_avg_lastn",
    "goals_suf_avg", "goals_suf_avg_lastn",
    "gd_avg", "gd_avg_lastn",
    "rank_avg", "rank_avg_lastn",
    "points_avg", "points_avg_lastn",
]

def _rolling_team_stats(df, window):
    rows = []
    for team in df["team"].unique():
        team_df = df[df["team"] == team].sort_values("date").copy()
        
        # 경기 기록 매칭
        gs = team_df["goals_scored"]
        gc = team_df["goals_conceded"]
        gd = gs - gc  # 실시간 득실차
        rk = team_df["opp_rank"]
        pts = team_df["team_points"]

        # Expanding (누적 평균) - shift(1)로 현재 경기 데이터 차단
        team_df["goals_avg"] = gs.shift(1).expanding().mean()
        team_df["goals_suf_avg"] = gc.shift(1).expanding().mean()
        team_df["gd_avg"] = gd.shift(1).expanding().mean()
        team_df["rank_avg"] = rk.shift(1).expanding().mean()
        team_df["points_avg"] = pts.shift(1).expanding().mean()

        # Rolling (최근 N경기 평균) - shift(1)로 현재 경기 데이터 차단
        team_df["goals_avg_lastn"] = gs.shift(1).rolling(window, min_periods=1).mean()
        team_df["goals_suf_avg_lastn"] = gc.shift(1).rolling(window, min_periods=1).mean()
        team_df["gd_avg_lastn"] = gd.shift(1).rolling(window, min_periods=1).mean()
        team_df["rank_avg_lastn"] = rk.shift(1).rolling(window, min_periods=1).mean()
        team_df["points_avg_lastn"] = pts.shift(1).rolling(window, min_periods=1).mean()

        rows.append(team_df)

    return pd.concat(rows).sort_index()

def feature_engineer(df_match_rank: pd.DataFrame) -> pd.DataFrame:
    col_names = [
        "date", "team", "opponent", "goals_scored", "goals_conceded",
        "team_rank", "opp_rank", "team_points", "neutral", "tournament", "is_home",
    ]

    # 홈팀 관점 분리
    df_home = df_match_rank[[
        "date", "home_team", "away_team", "home_score", "away_score",
        "rank_home", "rank_away", "total_points_home", "neutral", "tournament",
    ]].copy()
    df_home["is_home"] = 1
    df_home.columns = col_names

    # 어웨이팀 관점 분리
    df_away = df_match_rank[[
        "date", "away_team", "home_team", "away_score", "home_score",
        "rank_away", "rank_home", "total_points_away", "neutral", "tournament",
    ]].copy()
    df_away["is_home"] = 0
    df_away.columns = col_names

    df_concat = pd.concat([df_home, df_away]).sort_values("date")

    # 롤링/누적 통계 계산
    df_stats = _rolling_team_stats(df_concat, window=ROLL_WINDOW)
    
    # 초창기 이력이 없어 롤링/누적 통계가 NaN인 데이터는 제거하되, 스코어(NaN)가 비어있는 예측 대상 경기는 보존
    df_stats = df_stats.dropna(subset=STAT_COLS)
    return df_stats

# ---------------------------------------------------------------------------
# 4) 경기 단위로 다시 피벗 (home / away 통합)
# ---------------------------------------------------------------------------
def merge_home_away(df_stats: pd.DataFrame) -> pd.DataFrame:
    basic_cols = ["date", "team", "opponent", "goals_scored"]

    df_home = df_stats[df_stats.is_home == 1].loc[:, basic_cols + STAT_COLS].copy()
    df_away = df_stats[df_stats.is_home == 0].loc[:, basic_cols + STAT_COLS].copy()

    df = pd.merge(
        df_home, df_away,
        left_on=["date", "team", "opponent"],
        right_on=["date", "opponent", "team"],
        suffixes=["_home", "_away"],
        how="inner",
    )
    df = df.rename(columns={"team_home": "home_team", "opponent_home": "away_team"})
    df = df.drop(columns=["team_away", "opponent_away"])
    return df

# ---------------------------------------------------------------------------
# 5) 양 팀 간 피처 격차(diff) 계산
# ---------------------------------------------------------------------------
def add_diffs(df_features: pd.DataFrame, df_match_rank: pd.DataFrame) -> pd.DataFrame:
    df = df_features.copy()
    for stat in STAT_COLS:
        df[f"{stat}_diff"] = df[f"{stat}_home"] - df[f"{stat}_away"]

    # 실시간 FIFA 랭킹 및 스코어 정보 결합
    mr = df_match_rank[[
        "date", "home_team", "away_team",
        "home_score", "away_score", "rank_home", "rank_away",
    ]].copy()
    mr["rank_diff"] = mr["rank_home"] - mr["rank_away"]

    df = pd.merge(df, mr, on=["date", "home_team", "away_team"], how="left")
    return df

# ---------------------------------------------------------------------------
# 6) 홈팀 기준 정수형 타깃 레이블 추가 (loss: 0, draw: 1, win: 2)
# ---------------------------------------------------------------------------
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    diff = df["home_score"] - df["away_score"]

    def encode_label(d):
        if pd.isna(d):
            return np.nan
        return 2 if d > 0 else (0 if d < 0 else 1)

    df["result"] = diff.map(encode_label)
    return df

# ---------------------------------------------------------------------------
# 7) 2026 본선 진출국 필터 및 학습셋/예측셋 분리 저장
# ---------------------------------------------------------------------------
def load_wc_teams() -> set:
    elo = pd.read_csv(ELO_CSV)
    return set(elo["country"].replace(ELO_NAME_MAP).unique())

def filter_wc(df: pd.DataFrame, wc_teams: set) -> pd.DataFrame:
    mask = df["home_team"].isin(wc_teams) & df["away_team"].isin(wc_teams)
    return df[mask].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 메인 전처리 실행 흐름
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # fixture 종료일 확인 (results.csv 전체에서 가장 늦은 날짜)
    results_df = pd.read_csv(RESULTS_CSV)
    end_date = pd.to_datetime(results_df["date"]).max()
    print(f"전처리 최종 일정 한계일 (results.csv 최종일자): {end_date.date()}")

    print("[1/7] FIFA 랭킹 정제 및 연장...")
    rank = prep_rank(end_date)
    print(f"      랭킹 정제 완료: {len(rank):,} 행, 국가 수: {rank.country_full.nunique()}")

    print("[2/7] 경기 결과와 FIFA 랭킹 병합...")
    match_rank = merge_match_rank(rank)
    print(f"      매칭 완료: {len(match_rank):,} 경기 ({match_rank.date.min().date()} ~ {match_rank.date.max().date()})")

    print("[3/7] 피처 엔지니어링 (누적/롤링 및 득실차 gd_avg 계산)...")
    stats = feature_engineer(match_rank)
    print(f"      팀 단위 통계 계산 완료: {len(stats):,} 행")

    print("[4/7] 경기 단위 피벗 및 홈/어웨이 통합...")
    features = merge_home_away(stats)
    print(f"      경기 단위 피벗 완료: {len(features):,} 경기")

    print("[5/7] 피처 격차(diff) 계산 및 실시간 랭킹 결합...")
    final = add_diffs(features, match_rank)
    print(f"      격차 피처 완료 (전체 컬럼 수: {final.shape[1]})")

    print("[6/7] 정수형 타깃 레이블 추가...")
    final = add_target(final)
    
    print("[6.5/7] main_dataset.csv 고급 전력 지표 결합...")
    MAIN_COLS = [
        "h_elo_rating", "h_elo_rank", "h_pi_best_lead_score", "h_pi_avg_lead_score",
        "h_wci_squad_value", "h_form_goals_avg", "h_form_goals_avg_l10",
        "h_form_goals_suf_avg", "h_form_win_rate", "h_ws_shots_pg",
        "h_ws_possession_pct", "h_ws_pass_pct", "h_ws_ws_rating", "h_ws_xg_pg", "h_ws_xg_diff_pg",
        "a_elo_rating", "a_elo_rank", "a_pi_best_lead_score", "a_pi_avg_lead_score",
        "a_wci_squad_value", "a_form_goals_avg", "a_form_goals_avg_l10",
        "a_form_goals_suf_avg", "a_form_win_rate", "a_ws_shots_pg",
        "a_ws_possession_pct", "a_ws_pass_pct", "a_ws_ws_rating", "a_ws_xg_pg", "a_ws_xg_diff_pg",
        "elo_rating_diff", "elo_win_rate_diff", "pi_leader_diff", "pi_n_leaders_diff",
        "squad_value_diff", "form_goals_diff", "form_ga_diff", "form_wr_diff",
        "ws_poss_diff", "ws_rating_diff", "ws_xg_diff",
        "h_elo_wins", "h_elo_matches_total", "h_pi_n_key_leaders",
        "a_elo_wins", "a_elo_matches_total", "a_pi_n_key_leaders"
    ]
    
    main_df = pd.read_csv(DATASET_DIR / "main_dataset.csv")
    main_df["date"] = pd.to_datetime(main_df["date"])
    main_df["home_team"] = main_df["home_team"].replace(MAIN_NAME_MAP)
    main_df["away_team"] = main_df["away_team"].replace(MAIN_NAME_MAP)
    
    # 병합용 서브셋 추출
    merge_cols = ["date", "home_team", "away_team"] + MAIN_COLS
    main_subset = main_df[merge_cols].drop_duplicates(subset=["date", "home_team", "away_team"])
    
    # final 데이터셋에 merge (과거 학습 경기용)
    final = pd.merge(final, main_subset, on=["date", "home_team", "away_team"], how="left")
    
    print("[7/7] 본선 48개국 필터링 및 데이터셋 분리...")
    wc_teams = load_wc_teams()
    wc_final = filter_wc(final, wc_teams)

    # 학습 데이터셋(결과 존재) vs 예측 데이터셋(결과 NaN) 분리
    train = wc_final[wc_final["result"].notna()].reset_index(drop=True)
    predict = wc_final[wc_final["result"].isna()].reset_index(drop=True)
    
    # 예측 데이터셋(predict) 내 결측 전력 피처들을 스냅샷에서 채워줌
    print("      예측 데이터셋 내 결측 피처 보정 중...")
    snapshots = build_main_latest_snapshots(main_df)
    
    # 매핑할 단일 피처 키들
    feature_keys = {
        "elo_rating": ("h_elo_rating", "a_elo_rating", "elo_rating_diff"),
        "elo_rank": ("h_elo_rank", "a_elo_rank", None),
        "pi_best_lead_score": ("h_pi_best_lead_score", "a_pi_best_lead_score", "pi_leader_diff"),
        "pi_avg_lead_score": ("h_pi_avg_lead_score", "a_pi_avg_lead_score", None),
        "wci_squad_value": ("h_wci_squad_value", "a_wci_squad_value", "squad_value_diff"),
        "form_goals_avg": ("h_form_goals_avg", "a_form_goals_avg", "form_goals_diff"),
        "form_goals_avg_l10": ("h_form_goals_avg_l10", "a_form_goals_avg_l10", None),
        "form_goals_suf_avg": ("h_form_goals_suf_avg", "a_form_goals_suf_avg", "form_ga_diff"),
        "form_win_rate": ("h_form_win_rate", "a_form_win_rate", "form_wr_diff"),
        "ws_shots_pg": ("h_ws_shots_pg", "a_ws_shots_pg", None),
        "ws_possession_pct": ("h_ws_possession_pct", "a_ws_possession_pct", "ws_poss_diff"),
        "ws_pass_pct": ("h_ws_pass_pct", "a_ws_pass_pct", None),
        "ws_ws_rating": ("h_ws_ws_rating", "a_ws_ws_rating", "ws_rating_diff"),
        "ws_xg_pg": ("h_ws_xg_pg", "a_ws_xg_pg", "ws_xg_diff"),
        "ws_xg_diff_pg": ("h_ws_xg_diff_pg", "a_ws_xg_diff_pg", None)
    }
    
    for idx, row in predict.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        
        h_snap = snapshots.get(h_team, {})
        a_snap = snapshots.get(a_team, {})
        
        # 1) 단일 팀 속성 및 diff 채우기
        for base_key, (h_col, a_col, diff_col) in feature_keys.items():
            h_val = h_snap.get(base_key, np.nan)
            a_val = a_snap.get(base_key, np.nan)
            
            predict.at[idx, h_col] = h_val
            predict.at[idx, a_col] = a_val
            
            if diff_col:
                if not pd.isna(h_val) and not pd.isna(a_val):
                    predict.at[idx, diff_col] = h_val - a_val
                else:
                    predict.at[idx, diff_col] = np.nan
                    
        # 2) 수치 매핑되지 않는 특수 diff 및 메타데이터 채우기 (elo_win_rate_diff, pi_n_leaders_diff)
        # pi_n_leaders_diff = h_pi_n_key_leaders - a_pi_n_key_leaders
        h_leaders = h_snap.get("pi_n_key_leaders", np.nan)
        a_leaders = a_snap.get("pi_n_key_leaders", np.nan)
        
        predict.at[idx, "h_pi_n_key_leaders"] = h_leaders
        predict.at[idx, "a_pi_n_key_leaders"] = a_leaders
        if not pd.isna(h_leaders) and not pd.isna(a_leaders):
            predict.at[idx, "pi_n_leaders_diff"] = h_leaders - a_leaders
            
        # elo_win_rate_diff = (h_elo_wins/h_elo_matches_total - a_elo_wins/a_elo_matches_total) * 1000
        h_wins = h_snap.get("elo_wins", np.nan)
        h_total = h_snap.get("elo_matches_total", np.nan)
        a_wins = a_snap.get("elo_wins", np.nan)
        a_total = a_snap.get("elo_matches_total", np.nan)
        
        predict.at[idx, "h_elo_wins"] = h_wins
        predict.at[idx, "h_elo_matches_total"] = h_total
        predict.at[idx, "a_elo_wins"] = a_wins
        predict.at[idx, "a_elo_matches_total"] = a_total
        if not pd.isna(h_wins) and not pd.isna(h_total) and h_total > 0 and not pd.isna(a_wins) and not pd.isna(a_total) and a_total > 0:
            h_wr = h_wins / h_total
            a_wr = a_wins / a_total
            predict.at[idx, "elo_win_rate_diff"] = (h_wr - a_wr) * 1000

    # float 형식 타깃을 정수형으로 형변환 (XGBoost 호환)
    train = train.astype({"result": int})

    # 저장
    final.to_parquet(OUT_DIR / "final_base.parquet")
    train.to_parquet(OUT_DIR / "train_base.parquet")
    predict.to_parquet(OUT_DIR / "predict_base.parquet")

    print("\n================ 전처리 파이프라인 산출물 요약 ================")
    print(f"1. 전체 피처 데이터셋: {OUT_DIR / 'final_base.parquet'} ({len(final):,} rows, {final.shape[1]} cols)")
    print(f"2. XGBoost 학습셋    : {OUT_DIR / 'train_base.parquet'} ({len(train):,} rows, {train.shape[1]} cols)")
    print(f"3. 2026 본선 예측셋  : {OUT_DIR / 'predict_base.parquet'} ({len(predict):,} rows, {predict.shape[1]} cols)")
    print(f"\n학습셋 타깃 분포 (2: Win, 1: Draw, 0: Loss):\n{train['result'].value_counts()}")
    print("==============================================================")

if __name__ == "__main__":
    main()
