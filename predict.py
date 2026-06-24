"""
2026 FIFA World Cup — Saved CatBoost Predictor Inference Runner (predict.py)

This script loads the trained model assets (including CatBoost models and rolling features)
from 'pred_cat/predictor_assets.pkl' and generates the final submission file.
No training is performed.
"""

import pickle
from pathlib import Path
import re
import numpy as np
import pandas as pd
import warnings

# Avoid warnings
warnings.filterwarnings("ignore")

# Define Predictor class exactly as in train_catboost.py for unpickling compatibility
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
        from train_catboost import _row
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

# Import helper functions from train_catboost
from train_catboost import (
    res, sub, group_table, best_thirds, load_third_place_rules, simulate
)

def main():
    base_dir = Path(__file__).resolve().parent
    pickle_path = base_dir / "pred_cat" / "predictor_assets.pkl"

    if not pickle_path.exists():
        print(f"[Error] Model assets pickle file not found at: {pickle_path}")
        print("Please train the model first to generate the pickle file.")
        return

    print(f"[Info] Loading trained model assets from: {pickle_path} ...")
    with open(pickle_path, "rb") as f:
        assets = pickle.load(f)

    # 1. Initialize Predictor with default blending weight w=0.40
    # You can change the blending weight w here if needed.
    default_w = 0.40
    print(f"[Info] Initializing Predictor with blending weight w = {default_w:.2f}")
    predictor = Predictor(
        trained_models=assets["trained_models"],
        trained_regressors_home=assets["trained_regressors_home"],
        trained_regressors_away=assets["trained_regressors_away"],
        feature_cols=assets["feature_cols"],
        snap=assets["snap"],
        capital_dists=assets["capital_dists"],
        market_odds=assets["market_odds"],
        w_model=default_w
    )

    # 2. Run simulation
    teams_csv = base_dir / "dataset_scripts_out" / "wc_2026_teams.csv"
    fixtures_csv = base_dir / "dataset_scripts_out" / "wc_2026_fixtures.csv"
    third_place_csv = base_dir / "tournament" / "third_place_assignments_2026.csv"
    template_csv = base_dir / "submission_format" / "template.csv"

    if not (teams_csv.exists() and fixtures_csv.exists() and third_place_csv.exists() and template_csv.exists()):
        print("[Error] Tournament resources (CSV files) are missing. Cannot run simulation.")
        return

    df_teams = pd.read_csv(teams_csv)
    df_fixtures = pd.read_csv(fixtures_csv)
    df_template = pd.read_csv(template_csv)
    third_place_rules = load_third_place_rules(third_place_csv)
    team_group = {res(r["team"]): r["group"] for _, r in df_teams.iterrows()}

    print("[Info] Simulating 2026 World Cup Tournament...")
    # Group stage
    group_rows, standings, stats = group_table(predictor, df_template, df_teams)
    # Best 3rd place teams
    thirds = best_thirds(standings, stats)
    qualified_groups = tuple(sorted(team_group[t] for t in thirds))
    # Knockout stage
    ko_rows = simulate(predictor, df_fixtures, standings, thirds, qualified_groups, third_place_rules)

    # Combine and save submission
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

    output_csv = base_dir / "pred_cat" / "submission_reproduced.csv"
    df_sub.to_csv(output_csv, index=False)
    print(f"[Success] Simulation finished! Results saved to '{output_csv}'")

if __name__ == "__main__":
    main()
