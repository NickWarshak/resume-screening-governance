"""
SHAP explainability analysis.

Two distinct governance jobs are served here, and they are often conflated:

  GLOBAL explanation  - what does the model rely on in aggregate? This is a
                        model-risk and bias-detection tool. It is how we caught
                        the naive model routing most of its decision through
                        referral status and school prestige.

  LOCAL explanation   - why was THIS applicant scored this way? This is a legal
                        artifact. Illinois AIVIA and the EU AI Act both create
                        candidate-facing explanation duties, and NYC LL144
                        requires disclosure of the qualifications the tool
                        assesses. A SHAP waterfall is the raw material for an
                        adverse-action-style notice.

IMPORTANT CAVEAT, stated here because it belongs in the code and not only the
report: SHAP explains the MODEL, not the WORLD. A SHAP value of +0.30 for
"referral" means the model raised its score because the applicant was referred.
It does not mean referrals cause job success. Treating SHAP attributions as
causal is the most common misuse of explainability tooling, and it is how an
organization talks itself into believing a biased feature is a legitimate one.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generate_data import GOVERNED_FEATURES, NAIVE_FEATURES
from train import load, split, SEED

BASE = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(BASE, "figures")

C_BLUE, C_ORANGE, C_RED, C_GREEN = "#2a78d6", "#eb6834", "#e34948", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "axes.grid": True, "grid.color": "#eceae5",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False,
})

PRETTY = {
    "years_experience": "Years of experience", "education_level": "Education level",
    "num_relevant_skills": "Relevant skills matched", "keyword_match_score": "Keyword match score",
    "gpa": "GPA", "num_certifications": "Certifications", "portfolio_projects": "Portfolio projects",
    "num_prior_roles": "Prior roles", "avg_tenure_months": "Avg tenure (months)",
    "leadership_indicators": "Leadership indicators", "num_publications": "Publications",
    "resume_length_words": "Resume length (words)", "employment_gap_months": "Employment gap (months)",
    "top_school": "Attended 'top' school", "referral": "Employee referral",
    "distance_from_office_km": "Distance from office (km)",
}
PROXY_SET = {"employment_gap_months", "top_school", "referral", "distance_from_office_km"}


def fit_raw(feats, train, algo="gbm"):
    X, y = train[feats], train["advanced_to_onsite"]
    if algo == "gbm":
        m = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_depth=5, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=SEED)
    else:
        m = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
    m.fit(X, y)
    return m


def shap_values_for(model, X, algo):
    if algo == "gbm":
        ex = shap.TreeExplainer(model)
        return ex(X), ex
    bg = shap.utils.sample(X, 100, random_state=SEED)
    ex = shap.Explainer(lambda d: model.predict_proba(d)[:, 1], bg)
    return ex(X), ex


def mean_abs_table(sv, feats):
    vals = np.abs(sv.values).mean(axis=0)
    t = pd.DataFrame({"feature": feats, "mean_abs_shap": vals})
    t["pretty"] = t["feature"].map(PRETTY)
    t["is_proxy"] = t["feature"].isin(PROXY_SET)
    t = t.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    t["share_of_total"] = (t["mean_abs_shap"] / t["mean_abs_shap"].sum()).round(4)
    return t


def run():
    df = load()
    train, val, test = split(df)
    sample = test.sample(600, random_state=SEED)

    out = {}
    for name, feats, algo in [("naive", NAIVE_FEATURES, "gbm"),
                              ("governed", GOVERNED_FEATURES, "gbm")]:
        model = fit_raw(feats, train, algo)
        X = sample[feats]
        sv, ex = shap_values_for(model, X, algo)
        tbl = mean_abs_table(sv, feats)
        out[name] = {"sv": sv, "X": X, "table": tbl, "model": model, "feats": feats}

        # Beeswarm
        plt.figure(figsize=(7.2, 4.6))
        sv_named = sv
        sv_named.feature_names = [PRETTY.get(f, f) for f in feats]
        shap.plots.beeswarm(sv_named, max_display=12, show=False, color_bar=True)
        plt.title(f"SHAP feature attributions — {name} model", fontsize=10.5, loc="left", pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"shap_beeswarm_{name}.png"), dpi=190, bbox_inches="tight")
        plt.close()

        print(f"\n{'='*72}\nGLOBAL SHAP IMPORTANCE — {name.upper()} MODEL\n{'='*72}")
        print(f"{'feature':<32}{'mean |SHAP|':>13}{'share':>9}  {'proxy?':>7}")
        print("-" * 72)
        for _, r in tbl.head(10).iterrows():
            print(f"{r['pretty']:<32}{r['mean_abs_shap']:>13.4f}{r['share_of_total']:>9.1%}"
                  f"  {'YES' if r['is_proxy'] else '':>7}")
        prox_share = tbl.loc[tbl.is_proxy, "share_of_total"].sum()
        print("-" * 72)
        print(f"Share of total attribution carried by demographic proxy features: {prox_share:.1%}")
        out[name]["proxy_share"] = float(prox_share)

    # ---- Figure: proxy reliance comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    # Color encodes FEATURE TYPE, consistently across both panels -- not panel
    # identity. Blue/red validated at CVD dE 21.6 (protan); the orange/red pair
    # this replaced failed at dE 5.6.
    for ax, name in [(axes[0], "naive"), (axes[1], "governed")]:
        t = out[name]["table"].head(9).iloc[::-1]
        colors = [C_RED if p else C_BLUE for p in t["is_proxy"]]
        ax.barh(t["pretty"], t["share_of_total"], color=colors, height=0.62)
        ax.set_xlabel("Share of total |SHAP| attribution")
        ax.set_xlim(0, max(0.42, t["share_of_total"].max() * 1.18))
        ax.xaxis.set_major_formatter(lambda x, p: f"{x:.0%}")
        ps = out[name]["proxy_share"]
        ax.set_title(f"{name.capitalize()} model\nproxy features carry {ps:.0%} of the decision",
                     fontsize=10, loc="left", pad=10)
        for y, v in enumerate(t["share_of_total"]):
            ax.text(v + 0.008, y, f"{v:.0%}", va="center", fontsize=8, color=INK2)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Removing proxy features redistributes the decision onto merit signals",
                 fontsize=11.5, x=0.012, ha="left", y=0.995)
    plt.figtext(0.012, 0.005, "Red bars are demographic proxy features (referral, school prestige, "
                "employment gap, geographic distance).", fontsize=7.8, color=INK2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(os.path.join(FIG, "proxy_reliance.png"), dpi=190, bbox_inches="tight")
    plt.close()

    # ---- Local explanation: an adverse-action-style case ----
    gov = out["governed"]
    scores = gov["model"].predict_proba(gov["X"])[:, 1]
    # pick a borderline rejected candidate -- the population that most needs an explanation
    order = np.argsort(np.abs(scores - 0.30))
    idx = int(order[0])
    sv_one = gov["sv"][idx]
    plt.figure(figsize=(7.4, 4.4))
    shap.plots.waterfall(sv_one, max_display=10, show=False)
    plt.title("Local explanation for a borderline applicant (adverse-action basis)",
              fontsize=10.5, loc="left", pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "shap_waterfall_local.png"), dpi=190, bbox_inches="tight")
    plt.close()

    # Human-readable adverse action text
    contrib = pd.DataFrame({
        "feature": gov["feats"],
        "value": gov["X"].iloc[idx].values,
        "shap": sv_one.values,
    }).sort_values("shap")
    top_neg = contrib.head(4)
    print(f"\n{'='*72}\nGENERATED CANDIDATE-FACING EXPLANATION (score={scores[idx]:.3f})\n{'='*72}")
    print("The four factors that most reduced this application's score:\n")
    for i, (_, r) in enumerate(top_neg.iterrows(), 1):
        print(f"  {i}. {PRETTY.get(r['feature'], r['feature'])}: your value was "
              f"{r['value']:.4g}  (impact {r['shap']:+.3f})")
    print("\nThis text is machine-generated from SHAP attributions and is reviewed by a")
    print("human recruiter before release. It describes the model's reasoning, not a")
    print("judgment about the applicant's ability.")

    contrib.to_csv(os.path.join(BASE, "reports", "local_explanation_example.csv"), index=False)
    for name in out:
        out[name]["table"].drop(columns=["pretty"]).to_csv(
            os.path.join(BASE, "reports", f"shap_importance_{name}.csv"), index=False)

    return {k: {"proxy_share": v["proxy_share"], "top": v["table"].head(5)[["pretty","share_of_total"]].to_dict("records")}
            for k, v in out.items()}


if __name__ == "__main__":
    run()
