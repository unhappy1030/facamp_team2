"""
2026 FIFA World Cup — CatBoost 단독 모델 학습 및 가중치별 블렌딩 검증 파이프라인 (train_catboost.py)

CatBoost 모델(Classifier & Regressor)만 단독으로 학습하며, 이전에 드롭했던 피처(elo_rating_diff, rank_avg_lastn_diff 등)를 모두 복구하여 학습에 사용합니다.
외부 검답 데이터셋(리더보드)을 기반으로 배당률과의 블렌딩 가중치 w (0.3 ~ 1.0)별 검증 성적을 비교하고, 각 w 값에 해당하는 월드컵 토너먼트 시뮬레이션 결과를 각각의 파일로 pred_cat 디렉토리에 저장합니다.
"""

import argparse
import csv
from pathlib import Path
import re
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, accuracy_score
from catboost import CatBoostClassifier, CatBoostRegressor

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

MODELS = {}
REGRESSORS = {}

# CatBoost 단독 모델 정의 (RTX 4090 GPU 가속 지원)
MODELS["CatBoost"] = lambda: CatBoostClassifier(
    iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=5,
    random_seed=42, verbose=0, task_type="GPU"
)
REGRESSORS["CatBoost"] = lambda: CatBoostRegressor(
    iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=5,
    random_seed=42, verbose=0, task_type="GPU"
)

# 국가명 매핑 사전
FIX2RES = {
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Cape Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "USA": "United States",
    "United States": "United States",
    "Türkiye": "Turkey",
    "Turkey": "Turkey"
}

RES2OUT = {
    "Czech Republic": "Czechia",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
}

def res(name):
    return FIX2RES.get(name, name)

def sub(name):
    return RES2OUT.get(name, name)

def ranked_probability_score(y_prob, y_true):
    """
    Ranked Probability Score (RPS) 계산 (3클래스 순서형).
    y_prob: shape (N, 3) - 예측 확률 [P(Loss), P(Draw), P(Win)]
    y_true: shape (N,) - 실제 클래스 인덱스 (0: Loss, 1: Draw, 2: Win)
    """
    N = len(y_true)
    y_true_oh = np.eye(3)[y_true]
    
    cum_prob = np.cumsum(y_prob, axis=1)
    cum_true = np.cumsum(y_true_oh, axis=1)
    
    rps = np.sum((cum_prob - cum_true) ** 2, axis=1) / 2.0
    return float(np.mean(rps))

def build_snapshots(df_predict):
    snap = {}
    stat_bases = [
        "goals_avg", "goals_avg_lastn", "goals_suf_avg", "goals_suf_avg_lastn",
        "gd_avg", "gd_avg_lastn", "rank_avg", "rank_avg_lastn", "points_avg", "points_avg_lastn",
        "elo_rating", "elo_rank", "pi_best_lead_score", "pi_avg_lead_score",
        "wci_squad_value", "form_goals_avg", "form_goals_avg_l10", "form_goals_suf_avg",
        "form_win_rate", "ws_shots_pg", "ws_possession_pct", "ws_pass_pct",
        "ws_ws_rating", "ws_xg_pg", "ws_xg_diff_pg",
        "elo_wins", "elo_matches_total", "pi_n_key_leaders"
    ]
    for _, r in df_predict.iterrows():
        for team, side in [(r["home_team"], "home"), (r["away_team"], "away")]:
            if team in snap:
                continue
            s = {}
            for b in stat_bases:
                if b in ["goals_avg", "goals_avg_lastn", "goals_suf_avg", "goals_suf_avg_lastn",
                         "gd_avg", "gd_avg_lastn", "rank_avg", "rank_avg_lastn", "points_avg", "points_avg_lastn"]:
                    col_name = f"{b}_{side}"
                else:
                    col_name = f"h_{b}" if side == "home" else f"a_{b}"
                
                if col_name in df_predict.columns:
                    s[b] = r[col_name]
                else:
                    s[b] = np.nan
            s["rank"] = r[f"rank_{side}"]
            snap[team] = s
    return snap

def load_capital_distances(csv_path):
    dist = {}
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            c_name = res(row["country"])
            dist[c_name] = float(row["distance_km"])
    return dist

def load_market_odds(parquet_path):
    df = pd.read_parquet(parquet_path)
    odds = {}
    for _, r in df.iterrows():
        h = res(r["home_team"])
        a = res(r["away_team"])
        odds[(h, a)] = np.array([float(r["p_away"]), float(r["p_draw"]), float(r["p_home"])])
    return odds

def _row(t1, t2, snap, feature_cols, capital_dists):
    s1, s2 = snap[t1], snap[t2]
    row = {}
    stat_bases = [
        "goals_avg", "goals_avg_lastn", "goals_suf_avg", "goals_suf_avg_lastn",
        "gd_avg", "gd_avg_lastn", "rank_avg", "rank_avg_lastn", "points_avg", "points_avg_lastn"
    ]
    for b in stat_bases:
        row[f"{b}_home"] = s1[b]
        row[f"{b}_away"] = s2[b]
        row[f"{b}_diff"] = s1[b] - s2[b]
    row["rank_home"] = s1["rank"]
    row["rank_away"] = s2["rank"]
    row["rank_diff"] = s1["rank"] - s2["rank"]
    
    # main_dataset.csv 유래 피처 및 격차(diff) 처리
    main_bases = [
        ("elo_rating", "h_elo_rating", "a_elo_rating", "elo_rating_diff"),
        ("elo_rank", "h_elo_rank", "a_elo_rank", None),
        ("pi_best_lead_score", "h_pi_best_lead_score", "a_pi_best_lead_score", "pi_leader_diff"),
        ("pi_avg_lead_score", "h_pi_avg_lead_score", "a_pi_avg_lead_score", None),
        ("wci_squad_value", "h_wci_squad_value", "a_wci_squad_value", "squad_value_diff"),
        ("form_goals_avg", "h_form_goals_avg", "a_form_goals_avg", "form_goals_diff"),
        ("form_goals_avg_l10", "h_form_goals_avg_l10", "a_form_goals_avg_l10", None),
        ("form_goals_suf_avg", "h_form_goals_suf_avg", "a_form_goals_suf_avg", "form_ga_diff"),
        ("form_win_rate", "h_form_win_rate", "a_form_win_rate", "form_wr_diff"),
        ("ws_shots_pg", "h_ws_shots_pg", "a_ws_shots_pg", None),
        ("ws_possession_pct", "h_ws_possession_pct", "a_ws_possession_pct", "ws_poss_diff"),
        ("ws_pass_pct", "h_ws_pass_pct", "a_ws_pass_pct", None),
        ("ws_ws_rating", "h_ws_ws_rating", "a_ws_ws_rating", "ws_rating_diff"),
        ("ws_xg_pg", "h_ws_xg_pg", "a_ws_xg_pg", "ws_xg_diff"),
        ("ws_xg_diff_pg", "h_ws_xg_diff_pg", "a_ws_xg_diff_pg", None)
    ]
    
    for base_key, h_col, a_col, diff_col in main_bases:
        row[h_col] = s1[base_key]
        row[a_col] = s2[base_key]
        if diff_col:
            if not pd.isna(s1[base_key]) and not pd.isna(s2[base_key]):
                row[diff_col] = s1[base_key] - s2[base_key]
            else:
                row[diff_col] = np.nan
                
    # pi_n_leaders_diff 수동 계산
    h_leaders = s1.get("pi_n_key_leaders", np.nan)
    a_leaders = s2.get("pi_n_key_leaders", np.nan)
    if not pd.isna(h_leaders) and not pd.isna(a_leaders):
        row["pi_n_leaders_diff"] = h_leaders - a_leaders
    else:
        row["pi_n_leaders_diff"] = np.nan
        
    # elo_win_rate_diff 수동 계산
    h_wins = s1.get("elo_wins", np.nan)
    h_total = s1.get("elo_matches_total", np.nan)
    a_wins = s2.get("elo_wins", np.nan)
    a_total = s2.get("elo_matches_total", np.nan)
    if not pd.isna(h_wins) and not pd.isna(h_total) and h_total > 0 and not pd.isna(a_wins) and not pd.isna(a_total) and a_total > 0:
        row["elo_win_rate_diff"] = (h_wins / h_total - a_wins / a_total) * 1000
    else:
        row["elo_win_rate_diff"] = np.nan
        
    # 메타데이터 컬럼들 row에 지정 (reindex 시 drop_cols 로 제외됨)
    for side, s in [("h", s1), ("a", s2)]:
        row[f"{side}_elo_wins"] = s.get("elo_wins", np.nan)
        row[f"{side}_elo_matches_total"] = s.get("elo_matches_total", np.nan)
        row[f"{side}_pi_n_key_leaders"] = s.get("pi_n_key_leaders", np.nan)
        
    return pd.DataFrame([row]).reindex(columns=feature_cols)

class Predictor:
    def __init__(self, trained_models, trained_regressors_home, trained_regressors_away, feature_cols, snap, capital_dists, market_odds=None, w_model=0.40):
        self.trained_models = trained_models
        self.trained_regressors_home = trained_regressors_home
        self.trained_regressors_away = trained_regressors_away
        self.feature_cols = feature_cols
        self.snap = snap
        self.capital_dists = capital_dists
        self.market_odds = market_odds
        self.w = w_model

    def predict(self, t1, t2, allow_blend=True):
        x = _row(t1, t2, self.snap, self.feature_cols, self.capital_dists)
        x = x.fillna(0.0)
        
        probs = []
        for model in self.trained_models:
            probs.append(model.predict_proba(x)[0])
        prob = np.mean(probs, axis=0)
        
        if allow_blend and self.market_odds is not None:
            mk = self.market_odds.get((t1, t2))
            if mk is None and (t2, t1) in self.market_odds:
                v = self.market_odds[(t2, t1)]
                mk = np.array([v[2], v[1], v[0]])
            if mk is not None:
                p_m = np.clip(prob, 1e-6, 1.0)
                p_o = np.clip(mk, 1e-6, 1.0)
                lp = self.w * np.log(p_m) + (1.0 - self.w) * np.log(p_o)
                prob = np.exp(lp)
                prob = prob / prob.sum()
                
        loss, draw, win = prob[0], prob[1], prob[2]
        p1 = win + 0.5 * draw
        p2 = loss + 0.5 * draw
        
        # Regressor 기반 골 기대치 예측
        h_score_preds = []
        a_score_preds = []
        for reg_h in self.trained_regressors_home:
            h_score_preds.append(reg_h.predict(x)[0])
        for reg_a in self.trained_regressors_away:
            a_score_preds.append(reg_a.predict(x)[0])
            
        h_score_pred = np.mean(h_score_preds) if len(h_score_preds) > 0 else 1.0
        a_score_pred = np.mean(a_score_preds) if len(a_score_preds) > 0 else 1.0
        
        s1 = int(round(max(0.0, h_score_pred)))
        s2 = int(round(max(0.0, a_score_pred)))
        
        pred_class = np.argmax(prob)
        if pred_class == 2:
            if s1 <= s2: s1 = s2 + 1
        elif pred_class == 0:
            if s2 <= s1: s2 = s1 + 1
        else:
            if s1 != s2: s1 = s2 = int(round(max(0.0, (h_score_pred + a_score_pred)/2.0)))
            
        return p1, draw, p2, s1, s2

    def play(self, t1, t2, knockout=False):
        p1, draw, p2, s1, s2 = self.predict(t1, t2, allow_blend=not knockout)
        if knockout and s1 == s2:
            if p1 >= p2:
                s1 += 1
            else:
                s2 += 1
        tot = p1 + p2
        return {
            "team1": t1, "team2": t2, "team1_score": s1, "team2_score": s2,
            "team1_prob": round(p1 / tot, 2), "team2_prob": round(p2 / tot, 2),
            "winner": t1 if (s1 > s2 or (s1 == s2 and p1 >= p2)) else t2,
        }

def group_table(predictor, template_df, teams_df):
    team_group = {res(r["team"]): r["group"] for _, r in teams_df.iterrows()}
    groups = {}
    for t, g in team_group.items():
        groups.setdefault(g, []).append(t)
    stats = {t: {"pts": 0, "gf": 0, "ga": 0, "rank": predictor.snap[t]["rank"]}
             for t in team_group}

    group_rows = []
    for _, r in template_df.iterrows():
        t1, t2 = res(r["team1"]), res(r["team2"])
        m = predictor.play(t1, t2, knockout=False)
        s1, s2 = m["team1_score"], m["team2_score"]
        stats[t1]["gf"] += s1; stats[t1]["ga"] += s2
        stats[t2]["gf"] += s2; stats[t2]["ga"] += s1
        if s1 > s2:
            stats[t1]["pts"] += 3
        elif s2 > s1:
            stats[t2]["pts"] += 3
        else:
            stats[t1]["pts"] += 1; stats[t2]["pts"] += 1
        group_rows.append({**m, "type": f"Group Stage({team_group[t1]})"})

    def key(t):
        s = stats[t]
        return (s["pts"], s["gf"] - s["ga"], s["gf"], -s["rank"])

    standings = {g: sorted(ts, key=key, reverse=True) for g, ts in groups.items()}
    return group_rows, standings, stats

def best_thirds(standings, stats, n=8):
    thirds = [ts[2] for ts in standings.values()]
    thirds.sort(key=lambda t: (stats[t]["pts"], stats[t]["gf"] - stats[t]["ga"],
                               stats[t]["gf"], -stats[t]["rank"]), reverse=True)
    return thirds[:n]

def load_third_place_rules(csv_path):
    SLOTS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
    rules = {}
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            assignment = {slot: row[slot] for slot in SLOTS}
            key = tuple(sorted(value[1:] for value in assignment.values()))
            rules[key] = assignment
    return rules

TYPE_MAP = {
    "Round of 32": "Round of 32", "Round of 16": "Round of 16",
    "Quarter-final": "Quarterfinal", "Semi-final": "Semifinal",
    "3rd Place Match": "Third Place", "Final": "Final",
}

def simulate(predictor, fixtures, standings, thirds, qualified_groups, rules):
    assignment = rules.get(qualified_groups)
    if assignment is None:
        raise ValueError(f"해당 조합을 규정표에서 찾지 못했습니다: {qualified_groups}")

    wildcard_1st_teams = {
        "Best 3rd #1": standings["B"][0],
        "Best 3rd #2": standings["A"][0],
        "Best 3rd #3": standings["D"][0],
        "Best 3rd #4": standings["E"][0],
        "Best 3rd #5": standings["G"][0],
        "Best 3rd #6": standings["I"][0],
        "Best 3rd #7": standings["L"][0],
        "Best 3rd #8": standings["K"][0],
    }

    def slot(token, ctx):
        token = str(token)
        if token in wildcard_1st_teams:
            return wildcard_1st_teams[token]
            
        m = re.fullmatch(r"([12])([A-L])", token)
        if m:
            slot_num = int(m.group(1))
            group_name = m.group(2)
            if slot_num == 1 and token in assignment:
                target_3rd_group = assignment[token][1:]
                return standings[target_3rd_group][2]
            else:
                return standings[group_name][slot_num - 1]
                
        for pat, lst in [(r"R32 W(\d+)", "r32"), (r"QF(\d+)", "r16"),
                         (r"SF(\d+)", "qf"), (r"Finalist (\d+)", "sf_w")]:
            m = re.fullmatch(pat, token)
            if m:
                return ctx[lst][int(m.group(1)) - 1]
        if token == "3rd Place":
            return ctx["sf_l"].pop(0)
        raise ValueError(f"미해결 슬롯: {token}")

    ctx = {"r32": [], "r16": [], "qf": [], "sf_w": [], "sf_l": []}
    out_rows = []
    for stage, label in TYPE_MAP.items():
        winners = []
        for _, r in fixtures[fixtures["stage"] == stage].iterrows():
            t1, t2 = slot(r["team1"], ctx), slot(r["team2"], ctx)
            m = predictor.play(t1, t2, knockout=True)
            out_rows.append({**m, "type": label})
            winners.append(m["winner"])
            if stage == "Semi-final":
                ctx["sf_l"].append(t2 if m["winner"] == t1 else t1)
        key = {"Round of 32": "r32", "Round of 16": "r16",
               "Quarter-final": "qf", "Semi-final": "sf_w"}.get(stage)
        if key:
            ctx[key] = winners
    return out_rows

def tune_model_optuna(model_name, X_tr, y_tr, n_trials=15):
    import logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        if model_name == "CatBoost":
            depth = trial.suggest_int("depth", 3, 8)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
            iterations = trial.suggest_int("iterations", 50, 300, step=50)
            model = MODELS["CatBoost"]()
            model.set_params(depth=depth, learning_rate=learning_rate, iterations=iterations)
        else:
            return 0.0
            
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params

def load_external_val(ext_path, df_final, feature_cols):
    df_ext = pd.read_csv(ext_path)
    matched_rows = []
    
    df_final_wc = df_final[pd.to_datetime(df_final["date"]).dt.year == 2026].copy()
    
    for idx, row in df_ext.iterrows():
        t1 = res(row["team1"])
        t2 = res(row["team2"])
        s1, s2 = int(row["team1_score"]), int(row["team2_score"])
        
        match_f = df_final_wc[
            ((df_final_wc["home_team"] == t1) & (df_final_wc["away_team"] == t2)) |
            ((df_final_wc["home_team"] == t2) & (df_final_wc["away_team"] == t1))
        ]
        
        if len(match_f) == 0:
            match_f = df_final[
                ((df_final["home_team"] == t1) & (df_final["away_team"] == t2)) |
                ((df_final["home_team"] == t2) & (df_final["away_team"] == t1))
            ]
            
        if len(match_f) == 0:
            print(f"[Warning] final_base에서 매치를 찾을 수 없습니다: {t1} vs {t2}")
            continue
            
        f_row = match_f.iloc[-1].copy()
        
        h_team = f_row["home_team"]
        if h_team == t1:
            h_score, a_score = s1, s2
        else:
            h_score, a_score = s2, s1
            
        diff = h_score - a_score
        result = 2 if diff > 0 else (0 if diff < 0 else 1)
        
        f_row["result"] = result
        matched_rows.append(f_row)
        
    if len(matched_rows) == 0:
        return None, None, None
        
    df_matched = pd.DataFrame(matched_rows).reset_index(drop=True)
    X_ext = df_matched[feature_cols].copy()
    y_ext = df_matched["result"].astype(int).values
    return X_ext, y_ext, df_matched

def select_features_ablation(X, y, feature_cols):
    """
    각 피처를 하나씩 제거하면서 3-Fold CV RPS를 비교하고,
    뺐을 때 성능이 하락하는(꼭 필요한) 필수 피처들만 식별합니다.
    """
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def evaluate_feature_subset(subset_cols):
        rps_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X.iloc[train_idx][subset_cols].fillna(0.0), y[train_idx]
            X_va, y_va = X.iloc[val_idx][subset_cols].fillna(0.0), y[val_idx]
            
            # 빠른 탐색을 위해 기본 하이퍼파라미터의 CatBoost 사용
            model = MODELS["CatBoost"]()
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)
            rps_scores.append(ranked_probability_score(preds, y_va))
        return float(np.mean(rps_scores))

    print("\n" + "="*30 + " 🔍 피처 선택: Baseline 평가 " + "="*30)
    print("[Feature Selection] Baseline 평가 중 (모든 피처 사용)...")
    baseline_rps = evaluate_feature_subset(feature_cols)
    print(f"      Baseline RPS: {baseline_rps:.5f}")
    
    essential_features = []
    redundant_features = []
    
    print(f"\n[Feature Selection] 총 {len(feature_cols)}개 피처 대상 Ablation Analysis 시작...")
    
    for idx, f in enumerate(feature_cols, 1):
        test_features = [col for col in feature_cols if col != f]
        test_rps = evaluate_feature_subset(test_features)
        
        # RPS는 낮을수록 좋음!
        # 따라서 피처 f를 뺐을 때 test_rps가 baseline_rps보다 커졌다면 -> f는 필수(Essential) 피처임
        if test_rps > baseline_rps:
            essential_features.append(f)
            status = "Essential (성능 저하)"
        else:
            redundant_features.append(f)
            status = "Redundant/Noise (성능 유지 또는 향상)"
            
        print(f"      [{idx:02d}/{len(feature_cols)}] 제외 피처: {f:<30} | 제외시 RPS: {test_rps:.5f} | {status}")
        
    print("\n" + "="*20 + " 📊 피처 선택 결과 요약 " + "="*20)
    print(f"   - 전체 피처 수: {len(feature_cols)}개")
    print(f"   - 필수 피처 수 (학습 사용): {len(essential_features)}개")
    print(f"   - 제외 피처 수 (노이즈/중복): {len(redundant_features)}개")
    print(f"   * 제외된 피처 목록: {redundant_features}")
    print("="*60 + "\n")
    
    return essential_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample-rate", type=float, default=0.8, help="검증 시 동적 서브샘플링 비율")
    parser.add_argument("--n-subsamples", type=int, default=5, help="검증 폴드당 랜덤 서브샘플링 횟수")
    parser.add_argument("--external-val", type=str, default="/home/unhappy1030/repo/facamp/final/validation/validation_ground_truth.csv", help="정답 validation CSV 경로")
    parser.add_argument("--tune", action="store_true", help="하이퍼파라미터 튜닝 수행 여부")
    parser.add_argument("--select-features", action="store_true", help="피처 절제 연구 기반 중요 피처 선택 수행 여부")
    args = parser.parse_args()

    train_path = Path("/home/unhappy1030/repo/facamp/final/out/train_base.parquet")
    predict_path = Path("/home/unhappy1030/repo/facamp/final/out/predict_base.parquet")
    final_path = Path("/home/unhappy1030/repo/facamp/final/out/final_base.parquet")

    # 결과 저장 경로를 pred_cat 으로 변경
    out_dir = Path("/home/unhappy1030/repo/facamp/final/pred_cat")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"학습용 데이터셋이 존재하지 않습니다: {train_path}. 전처리를 먼저 수행해 주세요.")

    # 데이터 로드
    df_train = pd.read_parquet(train_path)
    df_predict = pd.read_parquet(predict_path)
    df_final = pd.read_parquet(final_path) if final_path.exists() else None

    # elo_rating_diff, rank_avg_lastn_diff, distance_km_diff 등의 핵심 피처를 살리기 위해 drop_cols를 대폭 축소 (피처 복구 완료)
    drop_cols = [
        "date", "home_team", "away_team", "home_score", "away_score", 
        "tournament", "result", "goals_scored_home", "goals_scored_away",
        "h_elo_wins", "h_elo_matches_total", "h_pi_n_key_leaders",
        "a_elo_wins", "a_elo_matches_total", "a_pi_n_key_leaders"
    ]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    
    X = df_train[feature_cols].copy()
    y = df_train["result"].astype(int).values
    y_home = df_train["home_score"].astype(float).values
    y_away = df_train["away_score"].astype(float).values

    # 피처 선택 옵션 처리 (캐시 저장 및 로드 연동)
    if args.select_features:
        features_cache_csv = out_dir / "selected_features.csv"
        if features_cache_csv.exists():
            try:
                df_features_cache = pd.read_csv(features_cache_csv)
                essential_features = df_features_cache["Feature"].tolist()
                print(f"[Info] {len(essential_features)}개의 기존 캐시된 필수 피처 목록을 성공적으로 로드했습니다.")
            except Exception as e:
                print(f"[Warning] 캐시된 피처 파일 로드 실패: {e}. 피처 절제 연구를 재수행합니다.")
                essential_features = select_features_ablation(X, y, feature_cols)
                # 캐시 저장
                pd.DataFrame({"Feature": essential_features}).to_csv(features_cache_csv, index=False)
                print(f"[Info] 피처 선택 결과를 '{features_cache_csv}'에 저장했습니다.")
        else:
            essential_features = select_features_ablation(X, y, feature_cols)
            # 캐시 저장
            pd.DataFrame({"Feature": essential_features}).to_csv(features_cache_csv, index=False)
            print(f"[Info] 피처 선택 결과를 '{features_cache_csv}'에 저장했습니다.")
            
        feature_cols = essential_features
        X = X[feature_cols].copy()

    print(f"[Info] CatBoost 단독 학습 시작 - 데이터 크기: {X.shape[0]}행, 피처 수: {X.shape[1]}개")
    print(f"[Info] 복구 및 사용된 피처 수: {len(feature_cols)}개")
    print(f"피처 목록: {feature_cols}\n")

    # 외부 리소스 로드 (이동거리 & 배당률)
    dist_csv = Path("/home/unhappy1030/repo/facamp/final/dataset/wc_2026_capital_distance.csv")
    odds_parquet = Path("/home/unhappy1030/repo/facamp/final/dataset/odds_avg.parquet")
    
    capital_dists = load_capital_distances(dist_csv) if dist_csv.exists() else {}
    market_odds = load_market_odds(odds_parquet) if odds_parquet.exists() else None
    snap = build_snapshots(df_predict)

    # 외부 검증 데이터셋 로드
    X_ext, y_ext, df_ext_matched = None, None, None
    if args.external_val and df_final is not None:
        ext_path = Path(args.external_val)
        if ext_path.exists():
            X_ext, y_ext, df_ext_matched = load_external_val(ext_path, df_final, feature_cols)
            if X_ext is not None:
                print(f"[Info] 외부 검증 데이터셋(리더보드 정답) {len(X_ext)}경기 피처 복원 및 복구 완료!")
        else:
            print(f"[Warning] 외부 검증 데이터 파일 없음: {args.external_val}")

    # 기존 튜닝 결과 캐시 로드
    cached_params = {}
    params_csv = out_dir / "best_hyperparameters_catboost.csv"
    if params_csv.exists():
        try:
            df_cache = pd.read_csv(params_csv)
            import ast
            for _, row in df_cache.iterrows():
                m_name = row["Model"]
                f_num = int(row["Fold"])
                p_dict = ast.literal_eval(row["Best_Parameters"])
                cached_params[(m_name, f_num)] = p_dict
            print(f"[Info] {len(cached_params)}개의 기존 CatBoost 최적 파라미터 캐시를 성공적으로 로드했습니다.")
        except Exception as e:
            print(f"[Warning] 캐시된 파라미터 파일 로드 실패: {e}")

    # CatBoost 5-Fold 교차 검증 훈련
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained_models = []
    trained_regressors_home = []
    trained_regressors_away = []
    best_params_records = []

    print("[CatBoost] 5-Fold 교차 검증 학습 중...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx].fillna(0.0), y[train_idx]
        X_va, y_va = X.iloc[val_idx].fillna(0.0), y[val_idx]

        model = MODELS["CatBoost"]()
        p_key = ("CatBoost", fold)
        
        if p_key in cached_params:
            print(f"      [Fold {fold}] 캐시된 최적 파라미터 적용: {cached_params[p_key]}")
            model.set_params(**cached_params[p_key])
            best_params_records.append({
                "Model": "CatBoost",
                "Fold": fold,
                "Best_Parameters": str(cached_params[p_key])
            })
        elif args.tune:
            if not HAS_OPTUNA:
                raise ImportError("[Error] --tune 옵션에는 optuna가 필요합니다.")
            print(f"      [Fold {fold}] Optuna 최적 파라미터 탐색 중...")
            best_params = tune_model_optuna("CatBoost", X_tr, y_tr, n_trials=15)
            model.set_params(**best_params)
            print(f"      [Fold {fold}] Best Params: {best_params}")
            best_params_records.append({
                "Model": "CatBoost",
                "Fold": fold,
                "Best_Parameters": str(best_params)
            })
        
        model.fit(X_tr, y_tr)
        trained_models.append(model)

        # Regressor 학습
        reg_h = REGRESSORS["CatBoost"]()
        reg_a = REGRESSORS["CatBoost"]()
        reg_h.fit(X_tr, y_home[train_idx])
        reg_a.fit(X_tr, y_away[train_idx])
        trained_regressors_home.append(reg_h)
        trained_regressors_away.append(reg_a)

    print("[Info] CatBoost 5-Fold 교차 검증 학습 완료!")

    # 외부 검증 셋 예측값 계산
    if X_ext is not None and df_ext_matched is not None:
        ext_preds_list = []
        for model in trained_models:
            ext_preds_list.append(model.predict_proba(X_ext.fillna(0.0)))
        ext_avg_probs = np.mean(ext_preds_list, axis=0)

        # 블렌딩 비율별 결과 탐색 및 시뮬레이션 저장
        # blend_weights: w=0.00 ~ w=1.00 (0.1 단위)
        blend_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        leaderboard = []

        # 토너먼트 시뮬레이션용 데이터 파일 검증
        teams_csv = Path("/home/unhappy1030/repo/facamp/final/dataset_scripts_out/wc_2026_teams.csv")
        fixtures_csv = Path("/home/unhappy1030/repo/facamp/final/dataset_scripts_out/wc_2026_fixtures.csv")
        third_place_csv = Path("/home/unhappy1030/repo/facamp/final/tournament/third_place_assignments_2026.csv")
        template_csv = Path("/home/unhappy1030/repo/facamp/final/submission_format/template.csv")

        resources_ok = teams_csv.exists() and fixtures_csv.exists() and third_place_csv.exists() and template_csv.exists()
        if resources_ok:
            df_teams = pd.read_csv(teams_csv)
            df_fixtures = pd.read_csv(fixtures_csv)
            df_template = pd.read_csv(template_csv)
            third_place_rules = load_third_place_rules(third_place_csv)
            team_group = {res(r["team"]): r["group"] for _, r in df_teams.iterrows()}
        else:
            print("[Warning] 시뮬레이션 리소스 파일 일부 누락. submission_*.csv는 생성되지 않습니다.")

        for w_val in blend_weights:
            blended_probs = []
            for i, row in df_ext_matched.iterrows():
                h = row["home_team"]
                a = row["away_team"]
                prob = ext_avg_probs[i]

                if market_odds is not None and w_val < 1.00:
                    mk = market_odds.get((h, a))
                    if mk is None and (a, h) in market_odds:
                        v = market_odds[(a, h)]
                        mk = np.array([v[2], v[1], v[0]])
                    if mk is not None:
                        p_m = np.clip(prob, 1e-6, 1.0)
                        p_o = np.clip(mk, 1e-6, 1.0)
                        lp = w_val * np.log(p_m) + (1.0 - w_val) * np.log(p_o)
                        prob = np.exp(lp)
                        prob = prob / prob.sum()
                blended_probs.append(prob)

            blended_probs = np.array(blended_probs)
            w_rps = ranked_probability_score(blended_probs, y_ext)
            w_logloss = log_loss(y_ext, blended_probs, labels=[0, 1, 2])
            w_acc = accuracy_score(y_ext, np.argmax(blended_probs, axis=1))

            leaderboard.append({
                "Blending_Weight(w)": f"w={w_val:.2f}",
                "Ext_RPS": w_rps,
                "Ext_LogLoss": w_logloss,
                "Ext_Acc": w_acc
            })
            print(f"      [Weight w={w_val:.2f}] RPS: {w_rps:.4f} | LogLoss: {w_logloss:.4f} | Accuracy: {w_acc:.3%}")

            # 월드컵 본선 예측 시뮬레이션 및 각 w별 결과 저장
            if resources_ok:
                predictor = Predictor(
                    trained_models, trained_regressors_home, trained_regressors_away,
                    feature_cols, snap, capital_dists, market_odds, w_model=w_val
                )
                # 1. 조별리그 시뮬레이션
                group_rows, standings, stats = group_table(predictor, df_template, df_teams)
                # 2. 각 조 3위 추출 및 와일드카드 조합 결정
                thirds = best_thirds(standings, stats)
                qualified_groups = tuple(sorted(team_group[t] for t in thirds))
                # 3. 토너먼트 시뮬레이션
                ko_rows = simulate(predictor, df_fixtures, standings, thirds, qualified_groups, third_place_rules)
                # 4. 결과 병합 및 저장
                all_rows = group_rows + ko_rows
                df_sub = pd.DataFrame([{
                    "team1": sub(r["team1"]),
                    "team2": sub(r["team2"]),
                    "team1_score": r["team1_score"],
                    "team2_score": r["team2_score"],
                    "team1_prob": r["team1_prob"],
                    "team2_prob": r["team2_prob"],
                    "type": r["type"],
                } for r in all_rows])

                sub_csv = out_dir / f"submission_catboost_w_{w_val:.2f}.csv"
                df_sub.to_csv(sub_csv, index=False)

        # 리더보드 저장 및 출력
        df_leaderboard = pd.DataFrame(leaderboard).sort_values("Ext_RPS")
        leaderboard_csv = out_dir / "leaderboard_catboost.csv"
        df_leaderboard.to_csv(leaderboard_csv, index=False)

        # 최적 하이퍼파라미터 CSV 저장
        if best_params_records:
            df_params = pd.DataFrame(best_params_records)
            df_params.to_csv(params_csv, index=False)
            print(f"\n[Info] CatBoost Fold별 최적 하이퍼파라미터가 '{params_csv}'에 저장되었습니다.")

        # Predictor와 모델을 재사용할 수 있도록 pickle 파일로 저장
        import pickle
        assets = {
            "trained_models": trained_models,
            "trained_regressors_home": trained_regressors_home,
            "trained_regressors_away": trained_regressors_away,
            "feature_cols": feature_cols,
            "snap": snap,
            "capital_dists": capital_dists,
            "market_odds": market_odds
        }
        pickle_path = out_dir / "predictor_assets.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(assets, f)
        print(f"[Info] 추론을 위한 Predictor 에셋이 '{pickle_path}'에 저장되었습니다.")

        print("\n" + "="*20 + " 🏆 CatBoost Blending Weight (w) 리더보드 " + "="*20)
        print(df_leaderboard.to_string(index=False))
        print(f"\n[Info] 각 w별 시뮬레이션 결과 및 리더보드가 '{out_dir}' 폴더에 저장되었습니다.")
        print("="*75)

if __name__ == "__main__":
    main()
