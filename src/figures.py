"""Custom figures for the governance report. Palette values are validated
against the CVD/contrast checks (blue #2a78d6 vs red #e34948: CVD dE 21.6)."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(BASE, "figures")
BLUE, RED, GREEN, GREY = "#2a78d6", "#e34948", "#1baf7a", "#8a8880"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#d8d7d2",
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "axes.grid": True, "grid.color": "#eceae5", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})


def fig_impact_ratios():
    fa = json.load(open(os.path.join(BASE, "reports", "fairness_audit.json")))
    race_h = pd.DataFrame(fa["naive_gbm"]["tables"]["race"])
    race_g = pd.DataFrame(fa["governed_logistic"]["tables"]["race"])
    order = ["White", "Asian", "Hispanic/Latino", "Black", "Two or more/Other"]
    h = race_h.set_index("group").reindex(order)["impact_ratio"]
    g = race_g.set_index("group").reindex(order)["impact_ratio"]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    y = np.arange(len(order)); bh = 0.36
    ax.barh(y + bh/2 + 0.02, h.values, height=bh, color=RED, label="Naive model (proxies included)")
    ax.barh(y - bh/2 - 0.02, g.values, height=bh, color=BLUE, label="Governed model (proxies removed)")
    ax.axvline(0.80, color=INK, lw=1.4, ls="--")
    ax.text(0.795, len(order)-0.30, "four-fifths threshold  ", fontsize=8.4, color=INK,
            va="center", ha="right")
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel("Impact ratio (selection rate relative to highest-selected group)")
    ax.set_xlim(0, 1.12); ax.grid(axis="y", visible=False)
    for yy, v in zip(y + bh/2 + 0.02, h.values):
        ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    for yy, v in zip(y - bh/2 - 0.02, g.values):
        ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8.4, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.30))
    ax.set_title("Removing proxy features lifts every group above the four-fifths threshold",
                 fontsize=11, loc="left", pad=14)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "impact_ratios.png"), dpi=190, bbox_inches="tight"); plt.close()


def fig_performance_tradeoff():
    m = json.load(open(os.path.join(BASE, "reports", "model_metrics.json")))
    labels = ["Naive\nlogistic", "Naive\nGBM", "Governed\nlogistic", "Governed\nGBM"]
    keys = ["naive_logistic", "naive_gbm", "governed_logistic", "governed_gbm"]
    hist = [m[k]["auc_observed_label"] for k in keys]
    true = [m[k]["auc_vs_latent_qualification"] for k in keys]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w/2 - 0.01, hist, width=w, color=GREY, label="AUC vs historical recruiter decisions")
    ax.bar(x + w/2 + 0.01, true, width=w, color=BLUE, label="AUC vs true latent qualification")
    for xi, v in zip(x - w/2 - 0.01, hist):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
    for xi, v in zip(x + w/2 + 0.01, true):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("AUC"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")
    ax.set_title("The model that best imitates recruiters is not the model that best finds talent",
                 fontsize=11, loc="left", pad=12)
    plt.figtext(0.012, -0.02, "Selecting on the grey bars — the only metric available in production — "
                "would ship the naive model.", fontsize=7.8, color=INK2)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "performance_tradeoff.png"), dpi=190, bbox_inches="tight"); plt.close()


def fig_monitoring():
    h = pd.read_csv(os.path.join(BASE, "reports", "monitoring_history.csv"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.8, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1], "hspace": 0.26})
    a1.plot(h["month"], h["auc"], color=GREY, lw=2, marker="o", ms=6, label="AUC")
    a1.plot(h["month"], h["max_psi"], color=RED, lw=2, marker="s", ms=6, label="Max feature PSI")
    a1.axhline(0.25, color=RED, ls=":", lw=1.2)
    a1.text(4.6, 0.285, "PSI alert 0.25", fontsize=7.6, color=RED)
    a1.set_ylim(-0.05, 1.18); a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_ylabel("value")
    a1.legend(frameon=False, fontsize=8.2, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    a1.text(6.05, h["auc"].iloc[-1], " AUC flat", fontsize=7.8, color=GREY, va="center")
    a1.set_title("Performance holds steady while drift and fairness diverge",
                 fontsize=11, loc="left", pad=10)

    a2.plot(h["month"], h["ir_race"], color=BLUE, lw=2, marker="o", ms=6, label="Impact ratio (race)")
    a2.plot(h["month"], h["ir_sex"], color=GREEN, lw=2, marker="^", ms=6, label="Impact ratio (sex)")
    a2.axhline(0.80, color=INK, ls="--", lw=1.3)
    a2.text(4.35, 0.812, "four-fifths SLO", fontsize=7.6, color=INK)
    brk = h[h["ir_race"] < 0.80]
    if len(brk):
        a2.scatter(brk["month"], brk["ir_race"], s=170, facecolors="none", edgecolors=RED, lw=2, zorder=5)
        a2.annotate("SLO breach: AUC still 0.867,\nPSI only 0.065 — a performance\ndashboard shows all green",
                    xy=(brk.iloc[0]["month"], brk.iloc[0]["ir_race"] - 0.008), xytext=(0.15, 0.685),
                    fontsize=8, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2,
                                    connectionstyle="arc3,rad=-0.15"))
    a2.set_ylim(0.63, 1.06); a2.set_xlim(-0.25, 6.6)
    a2.set_xlabel("months since deployment"); a2.set_ylabel("impact ratio")
    a2.legend(frameon=False, fontsize=8.2, ncol=1, loc="upper right")
    plt.savefig(os.path.join(FIG, "monitoring_timeline.png"), dpi=190, bbox_inches="tight"); plt.close()


def fig_energy():
    s = json.load(open(os.path.join(BASE, "reports", "sustainability.json")))["annual_projection"]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    cats = ["This system\n(classical model)", "LLM per-resume\n(low estimate)", "LLM per-resume\n(high estimate)"]
    vals = [s["annual_total_kwh"], s["llm_alternative_annual_kwh_low"], s["llm_alternative_annual_kwh_high"]]
    ax.barh(cats[::-1], vals[::-1], color=[RED, RED, BLUE], height=0.55)
    ax.set_xscale("log"); ax.set_xlabel("Annual energy, kWh (log scale) — 250,000 resumes/year")
    for i, v in enumerate(vals[::-1]):
        ax.text(v * 1.15, i, f"{v:,.2f} kWh", va="center", fontsize=8.4, color=INK2)
    ax.set_xlim(0.05, 3000); ax.grid(axis="y", visible=False)
    ax.set_title(f"Whole-pipeline energy: {s['ratio_low']:,.0f}x to {s['ratio_high']:,.0f}x advantage",
                 fontsize=11, loc="left", pad=12)
    plt.figtext(0.012, -0.06, "Both bars include identical parsing and serving overhead. The model's own "
                "arithmetic is 0.0004% of this system's total.", fontsize=7.8, color=INK2)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "energy_comparison.png"), dpi=190, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    fig_impact_ratios(); fig_performance_tradeoff(); fig_monitoring(); fig_energy()
    print("figures written:", sorted(os.listdir(FIG)))
