"""
Fairness audit for the resume screening system.

Implements the disparate-impact analysis that NYC Local Law 144 requires of
automated employment decision tools (AEDTs), plus the error-rate and
calibration diagnostics that a bare impact ratio does not capture.

METRICS
  Selection rate      P(shortlisted | group)
  Impact ratio        selection_rate(group) / selection_rate(best group)
                      < 0.80 is prima facie adverse impact under the EEOC
                      Uniform Guidelines (29 CFR 1607.4D, the "four-fifths rule")
  TPR / FPR gap       equalized-odds violation; measured against the
                      counterfactual fair label, since measuring error rates
                      against a biased label launders the bias
  Calibration         mean predicted score vs actual rate, by group

A model can satisfy any one of these and fail the others. Kleinberg et al.
(2016) proved calibration and equalized odds are mathematically incompatible
when base rates differ, so the audit reports all three rather than optimizing
a single number.
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
FOUR_FIFTHS = 0.80


def impact_table(df, group_col, pred_col, truth_col=None, min_n=50):
    """Selection rates and impact ratios for one protected attribute."""
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        if len(sub) < min_n:
            continue
        row = {
            "group": str(g),
            "n": int(len(sub)),
            "selection_rate": float(sub[pred_col].mean()),
        }
        if truth_col is not None:
            pos, neg = sub[sub[truth_col] == 1], sub[sub[truth_col] == 0]
            row["tpr"] = float(pos[pred_col].mean()) if len(pos) else np.nan
            row["fpr"] = float(neg[pred_col].mean()) if len(neg) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    best = out["selection_rate"].max()
    out["impact_ratio"] = (out["selection_rate"] / best).round(4)
    out["passes_4_5ths"] = out["impact_ratio"] >= FOUR_FIFTHS
    out["selection_rate"] = out["selection_rate"].round(4)
    if truth_col is not None:
        out["tpr"] = out["tpr"].round(4)
        out["fpr"] = out["fpr"].round(4)
    return out.sort_values("selection_rate", ascending=False).reset_index(drop=True)


def qualified_miss_rate(df, group_col, pred_col, truth_col):
    """Share of genuinely qualified applicants in each group who are screened out.

    This is the metric that matters to the individual applicant: given that you
    would have succeeded, what was the chance the system rejected you.
    """
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        q = sub[sub[truth_col] == 1]
        if len(q) < 25:
            continue
        rows.append({
            "group": str(g),
            "n_qualified": int(len(q)),
            "missed_rate": round(float(1 - q[pred_col].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("missed_rate").reset_index(drop=True)


def calibration_by_group(df, group_col, score_col, truth_col, min_n=50):
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        if len(sub) < min_n:
            continue
        rows.append({
            "group": str(g),
            "mean_score": round(float(sub[score_col].mean()), 4),
            "actual_rate": round(float(sub[truth_col].mean()), 4),
            "calibration_gap": round(float(sub[score_col].mean() - sub[truth_col].mean()), 4),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def intersectional(df, pred_col, min_n=60):
    """LL144 requires intersectional sex x race/ethnicity reporting."""
    d = df.copy()
    d["intersection"] = d["sex"].astype(str) + " / " + d["race_ethnicity"].astype(str)
    return impact_table(d, "intersection", pred_col, min_n=min_n)


def audit_model(df, key, truth_col="advanced_counterfactual"):
    pred_col, score_col = f"pred_{key}", f"score_{key}"
    return {
        "sex": impact_table(df, "sex", pred_col, truth_col),
        "race": impact_table(df, "race_ethnicity", pred_col, truth_col),
        "age": impact_table(df, "age_group", pred_col, truth_col),
        "intersectional": intersectional(df, pred_col),
        "missed_qualified": qualified_miss_rate(df, "race_ethnicity", pred_col, truth_col),
        "calibration": calibration_by_group(df, "race_ethnicity", score_col, truth_col),
    }


def worst_ratio(tbl):
    return float(tbl["impact_ratio"].min())


def run():
    df = pd.read_csv(os.path.join(BASE, "data", "test_scored.csv"))
    summary = {}

    print("=" * 92)
    print("DISPARATE IMPACT AUDIT  (four-fifths threshold = 0.80)".center(92))
    print("=" * 92)

    # Baseline: the human process the model was trained to imitate
    human = impact_table(df, "sex", "advanced_to_onsite", "advanced_counterfactual")
    human_race = impact_table(df, "race_ethnicity", "advanced_to_onsite", "advanced_counterfactual")
    print("\n### BASELINE: historical human recruiter decisions")
    print(human[["group", "n", "selection_rate", "impact_ratio", "passes_4_5ths"]].to_string(index=False))
    print(human_race[["group", "n", "selection_rate", "impact_ratio", "passes_4_5ths"]].to_string(index=False))
    summary["human_baseline"] = {
        "sex_worst_impact_ratio": worst_ratio(human),
        "race_worst_impact_ratio": worst_ratio(human_race),
    }

    for key in ["naive_gbm", "governed_logistic"]:
        res = audit_model(df, key)
        label = "NAIVE MODEL (all features incl. proxies)" if key == "naive_gbm" \
            else "GOVERNED MODEL (proxy features removed)"
        print("\n" + "=" * 92)
        print(f"### {label}   [{key}]")
        print("\n-- Sex --")
        print(res["sex"][["group", "n", "selection_rate", "impact_ratio", "tpr", "fpr", "passes_4_5ths"]].to_string(index=False))
        print("\n-- Race / ethnicity --")
        print(res["race"][["group", "n", "selection_rate", "impact_ratio", "tpr", "fpr", "passes_4_5ths"]].to_string(index=False))
        print("\n-- Age --")
        print(res["age"][["group", "n", "selection_rate", "impact_ratio", "passes_4_5ths"]].to_string(index=False))
        print("\n-- Intersectional (sex x race) --")
        print(res["intersectional"][["group", "n", "selection_rate", "impact_ratio", "passes_4_5ths"]].to_string(index=False))
        print("\n-- Qualified applicants wrongly screened out --")
        print(res["missed_qualified"].to_string(index=False))

        summary[key] = {
            "sex_worst_impact_ratio": worst_ratio(res["sex"]),
            "race_worst_impact_ratio": worst_ratio(res["race"]),
            "age_worst_impact_ratio": worst_ratio(res["age"]),
            "intersectional_worst_impact_ratio": worst_ratio(res["intersectional"]),
            "intersectional_worst_group": res["intersectional"].iloc[-1]["group"],
            "max_tpr_gap_race": round(float(res["race"]["tpr"].max() - res["race"]["tpr"].min()), 4),
            "max_fpr_gap_race": round(float(res["race"]["fpr"].max() - res["race"]["fpr"].min()), 4),
            "missed_qualified_spread": round(
                float(res["missed_qualified"]["missed_rate"].max()
                      - res["missed_qualified"]["missed_rate"].min()), 4),
            "tables": {k: v.to_dict(orient="records") for k, v in res.items()},
        }

    print("\n" + "=" * 92)
    print("SUMMARY: worst-case impact ratio by system".center(92))
    print("=" * 92)
    print(f"{'system':<42}{'sex':>10}{'race':>10}{'intersectional':>18}")
    print("-" * 92)
    print(f"{'Human recruiters (status quo)':<42}"
          f"{summary['human_baseline']['sex_worst_impact_ratio']:>10.3f}"
          f"{summary['human_baseline']['race_worst_impact_ratio']:>10.3f}{'--':>18}")
    for key, name in [("naive_gbm", "Naive model (proxies included)"),
                      ("governed_logistic", "Governed model (proxies removed)")]:
        print(f"{name:<42}{summary[key]['sex_worst_impact_ratio']:>10.3f}"
              f"{summary[key]['race_worst_impact_ratio']:>10.3f}"
              f"{summary[key]['intersectional_worst_impact_ratio']:>18.3f}")
    print("-" * 92)
    print("Values below 0.80 constitute prima facie adverse impact.")

    with open(os.path.join(BASE, "reports", "fairness_audit.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


if __name__ == "__main__":
    run()
