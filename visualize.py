"""
2026 FIFA WC CatBoost 모델 — 피처 분석 시각화
출력: final/figures/ 디렉토리에 PNG 저장
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import poisson
from sklearn.feature_selection import mutual_info_classif

# ── 경로 설정
BASE = Path(__file__).parent
FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

TRAIN_PATH   = BASE / "out" / "train_base.parquet"
FEAT_PATH    = BASE / "pred_cat" / "selected_features.csv"
LEADER_PATH  = BASE / "pred_cat" / "leaderboard_catboost.csv"

# ── 데이터 로드
df = pd.read_parquet(TRAIN_PATH)
features = pd.read_csv(FEAT_PATH)["Feature"].tolist()
leaderboard = pd.read_csv(LEADER_PATH)

X = df[features].copy()
y = df["result"].copy()          # 0=원정승, 1=무, 2=홈승
goals_home = df["goals_scored_home"].dropna().astype(int)
goals_away = df["goals_scored_away"].dropna().astype(int)

RESULT_LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}
RESULT_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2980b9"}
PALETTE = [RESULT_COLORS[k] for k in [0, 1, 2]]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ────────────────────────────────────────────────
# 1. 클래스 분포
# ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = y.value_counts().sort_index()
bars = ax.bar(
    [RESULT_LABELS[i] for i in counts.index],
    counts.values,
    color=PALETTE,
    edgecolor="white",
    linewidth=0.8,
    width=0.55,
)
for bar, cnt in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        f"{cnt}\n({cnt/len(y)*100:.1f}%)",
        ha="center", va="bottom", fontsize=10,
    )
ax.set_title("Match Outcome Distribution (Training Set, n=896)", fontsize=13, pad=12)
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.18)
ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
plt.tight_layout()
plt.savefig(FIGURES / "01_outcome_distribution.png")
plt.close()
print("saved 01_outcome_distribution.png")


# ────────────────────────────────────────────────
# 2. 골 분포 + 포아송 적합
# ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, goals, label, color, mu_color in zip(
    axes,
    [goals_home, goals_away],
    ["Home Goals", "Away Goals"],
    ["#2980b9", "#e74c3c"],
    ["#1a5276", "#922b21"],
):
    mu = goals.mean()
    max_g = int(goals.max())
    k_vals = np.arange(0, max_g + 1)

    # 실제 빈도
    actual_freq = goals.value_counts().sort_index().reindex(k_vals, fill_value=0) / len(goals)
    ax.bar(k_vals - 0.2, actual_freq, width=0.35, color=color, alpha=0.75,
           label="Observed", edgecolor="white")

    # 포아송 적합
    poisson_prob = poisson.pmf(k_vals, mu)
    ax.bar(k_vals + 0.2, poisson_prob, width=0.35, color=mu_color, alpha=0.55,
           label=f"Poisson(λ={mu:.3f})", edgecolor="white")

    ax.set_title(f"{label} Distribution", fontsize=12)
    ax.set_xlabel("Goals")
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

fig.suptitle("Goal Scoring Distribution vs Poisson Fit", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES / "02_poisson_goals.png", bbox_inches="tight")
plt.close()
print("saved 02_poisson_goals.png")


# ────────────────────────────────────────────────
# 3. 블렌딩 가중치 w별 RPS / LogLoss
# ────────────────────────────────────────────────
lb = leaderboard.copy()
lb["w"] = lb["Blending_Weight(w)"].str.replace("w=", "").astype(float)
lb = lb.sort_values("w")

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax2 = ax1.twinx()

line1, = ax1.plot(lb["w"], lb["Ext_RPS"], "o-", color="#2980b9", linewidth=2,
                  markersize=6, label="RPS (lower=better)")
line2, = ax2.plot(lb["w"], lb["Ext_LogLoss"], "s--", color="#e74c3c", linewidth=1.8,
                  markersize=5, label="Log Loss")

best_idx = lb["Ext_RPS"].idxmin()
best_w = lb.loc[best_idx, "w"]
best_rps = lb.loc[best_idx, "Ext_RPS"]
ax1.axvline(best_w, color="#27ae60", linestyle=":", linewidth=1.6)
ax1.annotate(f"w={best_w:.2f}\nRPS={best_rps:.5f}",
             xy=(best_w, best_rps), xytext=(best_w + 0.06, best_rps + 0.0003),
             fontsize=9, color="#27ae60",
             arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))

ax1.set_xlabel("Blending Weight (w)  — 0=odds only, 1=model only", fontsize=10)
ax1.set_ylabel("RPS", color="#2980b9", fontsize=10)
ax2.set_ylabel("Log Loss", color="#e74c3c", fontsize=10)
ax1.tick_params(axis="y", labelcolor="#2980b9")
ax2.tick_params(axis="y", labelcolor="#e74c3c")

# Accuracy 주석 (모든 w에서 동일)
ax1.text(0.02, 0.06, "Accuracy: 62.5% for ALL w values",
         transform=ax1.transAxes, fontsize=9, color="#555",
         bbox=dict(facecolor="lightyellow", edgecolor="#ccc", boxstyle="round,pad=0.3"))

lines = [line1, line2]
ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
ax1.set_title("Blending Weight Sweep: RPS & Log Loss", fontsize=13)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
plt.tight_layout()
plt.savefig(FIGURES / "03_blending_weight_sweep.png")
plt.close()
print("saved 03_blending_weight_sweep.png")


# ────────────────────────────────────────────────
# 4. 주요 피처 상관관계 히트맵
# ────────────────────────────────────────────────
corr_df = X.dropna()
corr = corr_df.corr()

# 읽기 편하도록 짧은 이름으로 매핑
short_names = {
    "goals_avg_lastn_home":   "gls_lastn_H",
    "goals_suf_avg_lastn_home": "suf_lastn_H",
    "rank_avg_home":          "rank_avg_H",
    "rank_avg_lastn_home":    "rank_lastn_H",
    "rank_avg_away":          "rank_avg_A",
    "points_avg_lastn_away":  "pts_lastn_A",
    "goals_avg_lastn_diff":   "gls_lastn_D",
    "rank_avg_lastn_diff":    "rank_lastn_D",
    "points_avg_diff":        "pts_avg_D",
    "h_elo_rating":           "elo_rating_H",
    "h_elo_rank":             "elo_rank_H",
    "h_pi_best_lead_score":   "pi_best_H",
    "h_pi_avg_lead_score":    "pi_avg_H",
    "h_form_goals_suf_avg":   "form_suf_H",
    "h_form_win_rate":        "form_win_H",
    "h_ws_shots_pg":          "ws_shots_H",
    "h_ws_pass_pct":          "ws_pass_H",
    "h_ws_ws_rating":         "ws_rating_H",
    "a_pi_best_lead_score":   "pi_best_A",
    "a_wci_squad_value":      "wci_squad_A",
    "a_form_goals_avg_l10":   "form_gls_A",
    "a_form_goals_suf_avg":   "form_suf_A",
    "a_form_win_rate":        "form_win_A",
    "a_ws_shots_pg":          "ws_shots_A",
    "a_ws_possession_pct":    "ws_poss_A",
    "pi_n_leaders_diff":      "pi_lead_D",
    "form_goals_diff":        "form_gls_D",
    "form_ga_diff":           "form_suf_D",
    "ws_poss_diff":           "ws_poss_D",
    "ws_rating_diff":         "ws_rate_D",
    "ws_xg_diff":             "ws_xg_D",
}
corr.index = [short_names.get(c, c) for c in corr.index]
corr.columns = [short_names.get(c, c) for c in corr.columns]

cmap = LinearSegmentedColormap.from_list(
    "rw_b", ["#c0392b", "#ffffff", "#2980b9"], N=256
)

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Pearson r", fontsize=10)

n = len(corr)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7.5)
ax.set_yticklabels(corr.index, fontsize=7.5)

for i in range(n):
    for j in range(n):
        val = corr.values[i, j]
        if abs(val) > 0.45:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color="white" if abs(val) > 0.7 else "black")

ax.set_title("Feature Correlation Heatmap (31 Selected Features)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(FIGURES / "04_feature_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("saved 04_feature_correlation_heatmap.png")


# ────────────────────────────────────────────────
# 5. 결과별 핵심 피처 바이올린 플롯
# ────────────────────────────────────────────────
key_feats = [
    ("ws_xg_diff",          "xG Diff (H−A)"),
    ("ws_rating_diff",      "WhoScored Rating Diff"),
    ("form_goals_diff",     "Form Goals Diff"),
    ("rank_avg_lastn_diff", "FIFA Rank Diff (recent)"),
    ("h_elo_rating",        "Home ELO Rating"),
    ("h_form_win_rate",     "Home Form Win Rate"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, (feat, title) in zip(axes, key_feats):
    data_by_result = [
        df.loc[y == r, feat].dropna().values
        for r in [0, 1, 2]
    ]
    parts = ax.violinplot(data_by_result, positions=[0, 1, 2],
                          showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], PALETTE):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    parts["cbars"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Away Win", "Draw", "Home Win"], fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(feat, fontsize=7.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

fig.suptitle("Key Feature Distributions by Match Outcome", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES / "05_feature_violin_by_outcome.png", bbox_inches="tight")
plt.close()
print("saved 05_feature_violin_by_outcome.png")


# ────────────────────────────────────────────────
# 6. Mutual Information (피처 중요도 대리 지표)
# ────────────────────────────────────────────────
X_filled = X.fillna(X.median())
mi_scores = mutual_info_classif(X_filled, y, random_state=42)
mi_df = pd.DataFrame({"feature": features, "mi": mi_scores})
mi_df = mi_df.sort_values("mi", ascending=True)

fig, ax = plt.subplots(figsize=(8, 9))
colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(mi_df)))
bars = ax.barh(mi_df["feature"], mi_df["mi"], color=colors, edgecolor="white", height=0.7)

for bar, val in zip(bars, mi_df["mi"]):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=7.5)

ax.set_xlabel("Mutual Information Score")
ax.set_title("Feature Importance Proxy\n(Mutual Information w/ Outcome)", fontsize=12, pad=10)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.005))
plt.tight_layout()
plt.savefig(FIGURES / "06_mutual_information.png", bbox_inches="tight")
plt.close()
print("saved 06_mutual_information.png")


# ────────────────────────────────────────────────
# 7. ELO Rating Diff vs 결과 (산점도 + 빈도 히스토그램)
# ────────────────────────────────────────────────
elo_diff_col = "h_elo_rating"
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

result_names = ["Away Win (0)", "Draw (1)", "Home Win (2)"]
bins = np.linspace(
    df[elo_diff_col].quantile(0.01),
    df[elo_diff_col].quantile(0.99),
    30,
)

for ax, r, name, color in zip(axes, [0, 1, 2], result_names, PALETTE):
    vals = df.loc[y == r, elo_diff_col].dropna()
    ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor="white")
    ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={vals.mean():.0f}")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Home ELO Rating")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

fig.suptitle("Home Team ELO Rating Distribution by Outcome", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES / "07_elo_by_outcome.png", bbox_inches="tight")
plt.close()
print("saved 07_elo_by_outcome.png")


# ────────────────────────────────────────────────
# 8. 피처 그룹별 결측치 비율
# ────────────────────────────────────────────────
null_pct = X.isnull().mean().sort_values(ascending=True) * 100
null_pct = null_pct[null_pct > 0]

if len(null_pct) > 0:
    fig, ax = plt.subplots(figsize=(8, max(3, len(null_pct) * 0.38)))
    colors_null = ["#e74c3c" if v > 20 else "#f39c12" if v > 10 else "#2980b9"
                   for v in null_pct.values]
    bars = ax.barh(null_pct.index, null_pct.values, color=colors_null,
                   edgecolor="white", height=0.65)
    for bar, val in zip(bars, null_pct.values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8.5)
    ax.set_xlabel("Missing Rate (%)")
    ax.set_title("Missing Values by Feature (Training Set)", fontsize=12, pad=10)
    ax.axvline(10, color="#f39c12", linestyle=":", linewidth=1.2, label="10%")
    ax.axvline(20, color="#e74c3c", linestyle=":", linewidth=1.2, label="20%")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "08_missing_values.png", bbox_inches="tight")
    plt.close()
    print("saved 08_missing_values.png")
else:
    print("08_missing_values.png skipped (no missing values)")


# ────────────────────────────────────────────────
# 9. Annexe C 공식 32강 브래킷 PNG
# ────────────────────────────────────────────────
import base64, csv, io, json, re
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont as PILFont

NOTEBOOK_PATH = BASE.parent / "world_cup_2026_third_place_bracket.ipynb"
ANNEX_PATH    = BASE / "tournament" / "third_place_assignments_2026.csv"
SUB_PATH      = BASE / "pred_cat" / "submission_reproduced.csv"

BRACKET_LABELS = {
    "1E": {"xy": (115, 106),  "box": (18,   88,  190, 124)},
    "1I": {"xy": (115, 190),  "box": (18,  172,  190, 208)},
    "1D": {"xy": (115, 607),  "box": (18,  589,  190, 625)},
    "1G": {"xy": (115, 690),  "box": (18,  672,  190, 708)},
    "1A": {"xy": (1246, 274), "box": (1168, 256, 1340, 292)},
    "1L": {"xy": (1246, 354), "box": (1168, 336, 1340, 372)},
    "1B": {"xy": (1246, 607), "box": (1168, 589, 1340, 625)},
    "1K": {"xy": (1246, 690), "box": (1168, 672, 1340, 708)},
}
SLOTS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")

if NOTEBOOK_PATH.exists() and ANNEX_PATH.exists() and SUB_PATH.exists():
    # base64 PNG 추출
    nb = json.loads(NOTEBOOK_PATH.read_text())
    src = "".join(nb["cells"][1]["source"])
    m = re.search(r'BRACKET_TEMPLATE_BASE64\s*=\s*"""(.*?)"""', src, re.DOTALL)
    bracket_img = Image.open(io.BytesIO(base64.b64decode(m.group(1).strip()))).convert("RGBA")

    # submission에서 조별 순위 계산
    sub_rows = list(csv.DictReader(SUB_PATH.open()))
    b_stats  = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0})
    b_groups = defaultdict(list)
    for r in sub_rows:
        if not r["type"].startswith("Group Stage("): continue
        g = r["type"][len("Group Stage("):-1]
        for t in [r["team1"], r["team2"]]:
            if t not in b_groups[g]: b_groups[g].append(t)
        s1, s2 = int(r["team1_score"]), int(r["team2_score"])
        b_stats[r["team1"]]["gf"] += s1; b_stats[r["team1"]]["ga"] += s2
        b_stats[r["team2"]]["gf"] += s2; b_stats[r["team2"]]["ga"] += s1
        if s1 > s2:   b_stats[r["team1"]]["pts"] += 3
        elif s2 > s1: b_stats[r["team2"]]["pts"] += 3
        else: b_stats[r["team1"]]["pts"] += 1; b_stats[r["team2"]]["pts"] += 1

    b_standings = {g: sorted(ts, key=lambda t: (b_stats[t]["pts"],
                   b_stats[t]["gf"] - b_stats[t]["ga"], b_stats[t]["gf"]), reverse=True)
                   for g, ts in b_groups.items()}

    thirds_ranked = sorted(
        [(b_standings[g][2], b_stats[b_standings[g][2]], g) for g in b_standings],
        key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]), reverse=True,
    )
    top8_groups = tuple(sorted(g for _, _, g in thirds_ranked[:8]))

    # Annexe C 룩업
    annex_rules = {}
    for row in csv.DictReader(ANNEX_PATH.open()):
        a = {s: row[s] for s in SLOTS}
        key = tuple(sorted(v[1:] for v in a.values()))
        annex_rules[key] = {"option": row["option"], **a}
    annex = annex_rules[top8_groups]

    # 폰트
    b_font = None
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if Path(fp).exists():
            b_font = PILFont.truetype(fp, size=26)
            break
    if b_font is None:
        b_font = PILFont.load_default()

    draw = ImageDraw.Draw(bracket_img)

    def _centered(draw, xy, text, font, fill=(0, 0, 0, 255)):
        box = draw.textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)

    for slot, layout in BRACKET_LABELS.items():
        group = annex[slot][1:]
        team  = b_standings[group][2]
        label = f"{group}3  {team}"
        draw.rectangle(layout["box"], fill=(255, 255, 255, 255))
        _centered(draw, layout["xy"], label, b_font)

    out_path = FIGURES / "09_annexe_c_bracket.png"
    bracket_img.save(out_path)
    print(f"saved 09_annexe_c_bracket.png  (option {annex['option']}, groups: {''.join(top8_groups)})")
else:
    print("09_annexe_c_bracket.png skipped (notebook or annex CSV not found)")


# ────────────────────────────────────────────────
# 완료 요약
# ────────────────────────────────────────────────
saved = sorted(FIGURES.glob("*.png"))
print(f"\n총 {len(saved)}개 파일 저장: {FIGURES}/")
for f in saved:
    print(f"  {f.name}")
