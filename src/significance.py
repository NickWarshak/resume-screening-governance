"""
Statistical reliability of the impact-ratio estimates.

An impact ratio computed on 68 applicants is not the same evidence as one
computed on 1,113. Reporting a bare point estimate of 0.65 as a compliance
finding, without an interval, overstates what the audit knows. This module
bootstraps confidence intervals so the report can distinguish a real disparity
from sampling noise -- and can state honestly when the audit is underpowered.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
rng = np.random.default_rng(7)


def bootstrap_impact_ratio(df, group_col, pred_col, target_group, n_boot=5000, min_n=30):
    rates_ref, rates_tgt = [], []
    idx = np.arange(len(df))
    grp = df[group_col].values
    pred = df[pred_col].values
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        g, p = grp[s], pred[s]
        best = 0.0
        for u in np.unique(g):
            m = g == u
            if m.sum() >= min_n:
                best = max(best, p[m].mean())
        m = g == target_group
        if m.sum() < 5 or best == 0:
            continue
        rates_tgt.append(p[m].mean() / best)
    a = np.array(rates_tgt)
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5), (a < 0.80).mean()


def run():
    df = pd.read_csv(os.path.join(BASE, "data", "test_scored.csv"))
    df["intersection"] = df["sex"].astype(str) + " / " + df["race_ethnicity"].astype(str)

    print("=" * 96)
    print("BOOTSTRAP CONFIDENCE INTERVALS ON IMPACT RATIOS  (5,000 resamples)".center(96))
    print("=" * 96)

    checks = [
        ("governed_logistic", "sex", "Female"),
        ("governed_logistic", "race_ethnicity", "Black"),
        ("governed_logistic", "race_ethnicity", "Hispanic/Latino"),
        ("governed_logistic", "intersection", "Female / Hispanic/Latino"),
        ("governed_logistic", "intersection", "Female / Black"),
        ("naive_gbm", "sex", "Female"),
        ("naive_gbm", "race_ethnicity", "Black"),
        ("naive_gbm", "intersection", "Female / Black"),
    ]
    print(f"{'model':<20}{'attribute':<18}{'group':<28}{'n':>6}{'ratio':>8}{'95% CI':>18}{'P(<0.80)':>10}")
    print("-" * 96)
    rows = []
    for model, attr, grp in checks:
        n = int((df[attr] == grp).sum())
        # Match the reference-group threshold used by the fairness audit
        # (min_n=60 for intersections, 50 for marginals) so the bootstrap
        # centre is comparable to the reported point estimate.
        min_n = 60 if attr == "intersection" else 50
        mean, lo, hi, p_fail = bootstrap_impact_ratio(df, attr, f"pred_{model}", grp, min_n=min_n)
        print(f"{model:<20}{attr:<18}{grp:<28}{n:>6}{mean:>8.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}{p_fail:>10.3f}")
        rows.append(dict(model=model, attribute=attr, group=grp, n=n,
                         ratio=round(mean,4), ci_low=round(lo,4), ci_high=round(hi,4),
                         prob_below_threshold=round(p_fail,4)))
    print("-" * 96)
    print("P(<0.80) = bootstrap probability the true impact ratio breaches the four-fifths rule.")
    print("Wide intervals indicate the audit is underpowered for that subgroup, not that it is safe.")
    pd.DataFrame(rows).to_csv(os.path.join(BASE, "reports", "impact_ratio_ci.csv"), index=False)

    # Minimum detectable sample size note
    print("\nSubgroup sizes in the 2,400-applicant test set:")
    print(df["intersection"].value_counts().to_string())


if __name__ == "__main__":
    run()
