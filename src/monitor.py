"""
Drift detection and production monitoring.

Resume screening drifts in ways generic ML monitoring misses. Three distinct
failure modes, each needing its own detector:

  FEATURE DRIFT   - the applicant pool changes (a layoff wave floods the market
                    with senior candidates; a new sourcing channel shifts the
                    education mix). Detected via PSI and KS.

  LABEL DRIFT     - what recruiters reward changes. Because our label is a
                    recruiter decision, a change in recruiter behavior silently
                    invalidates the model without any feature moving.

  FAIRNESS DRIFT  - THE ONE THAT MATTERS MOST AND IS MONITORED LEAST. Impact
                    ratios can degrade while AUC, PSI and every accuracy metric
                    hold perfectly steady. A dashboard that tracks only
                    performance will show all-green while the system develops
                    an actionable disparate impact. This module treats the
                    impact ratio as a first-class SLO.

  FEEDBACK LOOP   - the model's own shortlists become tomorrow's training
                    labels. Left alone this is self-confirming: the model
                    proposes, the recruiter ratifies, the ratification trains
                    the next model. Mitigated by a randomized holdout of
                    human-only screening, described below.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), "..")

PSI_WARN, PSI_ALERT = 0.10, 0.25   # industry-standard population stability bands
IMPACT_RATIO_SLO = 0.80


def psi(expected, actual, bins=10):
    """Population Stability Index between a reference and a current sample."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference, current, features):
    rows = []
    for f in features:
        p = psi(reference[f].values, current[f].values)
        ks, pval = stats.ks_2samp(reference[f].values, current[f].values)
        status = "ALERT" if p >= PSI_ALERT else ("WARN" if p >= PSI_WARN else "OK")
        rows.append({"feature": f, "psi": round(p, 4), "ks_stat": round(float(ks), 4),
                     "ks_pvalue": round(float(pval), 5), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def fairness_slo(df, pred_col, protected_col, min_n=30):
    rates = df.groupby(protected_col, observed=True)[pred_col].agg(["mean", "size"])
    rates = rates[rates["size"] >= min_n]
    if len(rates) < 2:
        return None
    ratio = float(rates["mean"].min() / rates["mean"].max())
    return {"impact_ratio": round(ratio, 4),
            "breach": ratio < IMPACT_RATIO_SLO,
            "worst_group": str(rates["mean"].idxmin())}


def simulate_month(df, month, rng):
    """Simulate a production month with a realistic, compounding drift scenario.

    Narrative: a large sector layoff progressively shifts the applicant pool
    toward senior candidates with recent employment gaps, and a new university
    sourcing partnership changes the education mix. Neither event is a model
    change; both silently alter who gets shortlisted.
    """
    d = df.sample(len(df), replace=True, random_state=1000 + month).reset_index(drop=True)
    intensity = month / 6.0
    d["years_experience"] = np.clip(d["years_experience"] + rng.normal(2.4 * intensity, 0.6, len(d)), 0, 50)
    d["employment_gap_months"] = np.clip(
        d["employment_gap_months"] + rng.gamma(1.4, 2.4 * intensity + 0.01, len(d)), 0, 72)
    keep = rng.random(len(d)) > (0.12 * intensity)
    d.loc[~keep, "education_level"] = np.clip(d.loc[~keep, "education_level"] - 1, 0, 4)
    return d


def run():
    import joblib
    art = joblib.load(os.path.join(BASE, "models", "governed_logistic.joblib"))
    model, feats, thr = art["model"], art["features"], art["threshold"]

    full = pd.read_csv(os.path.join(BASE, "data", "applicants.csv"))
    reference = pd.read_csv(os.path.join(BASE, "data", "test_scored.csv"))
    rng = np.random.default_rng(11)

    print("=" * 96)
    print("SIMULATED 6-MONTH PRODUCTION MONITORING".center(96))
    print("=" * 96)
    print(f"{'month':<8}{'max PSI':>10}{'drifted feat':>22}{'AUC':>9}{'sel.rate':>10}"
          f"{'IR(sex)':>10}{'IR(race)':>11}{'status':>12}")
    print("-" * 96)

    history = []
    for m in range(0, 7):
        cur = reference.copy() if m == 0 else simulate_month(full, m, rng)
        score = model.predict_proba(cur[feats])[:, 1]
        cur["_score"] = score
        cur["_pred"] = (score >= thr).astype(int)

        dr = drift_report(reference[feats], cur[feats], feats)
        max_psi, worst_feat = dr.iloc[0]["psi"], dr.iloc[0]["feature"]

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(cur["advanced_to_onsite"], score)
        ir_sex = fairness_slo(cur, "_pred", "sex")
        ir_race = fairness_slo(cur, "_pred", "race_ethnicity")

        breaches = []
        if max_psi >= PSI_ALERT: breaches.append("DRIFT")
        if ir_sex and ir_sex["breach"]: breaches.append("FAIRNESS")
        if ir_race and ir_race["breach"]: breaches.append("FAIRNESS")
        if auc < 0.80: breaches.append("PERF")
        status = "OK" if not breaches else "|".join(sorted(set(breaches)))

        print(f"{m:<8}{max_psi:>10.3f}{worst_feat[:20]:>22}{auc:>9.3f}"
              f"{cur['_pred'].mean():>10.3f}{ir_sex['impact_ratio']:>10.3f}"
              f"{ir_race['impact_ratio']:>11.3f}{status:>12}")
        history.append({"month": m, "max_psi": round(float(max_psi), 4), "auc": round(float(auc), 4),
                        "selection_rate": round(float(cur["_pred"].mean()), 4),
                        "ir_sex": ir_sex["impact_ratio"], "ir_race": ir_race["impact_ratio"],
                        "status": status})

    print("-" * 96)
    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(BASE, "reports", "monitoring_history.csv"), index=False)

    auc_drop = hist.iloc[0]["auc"] - hist.iloc[-1]["auc"]
    ir_drop = hist.iloc[0]["ir_race"] - hist.iloc[-1]["ir_race"]
    print(f"Over 6 simulated months:  AUC moved {auc_drop:+.3f}   "
          f"race impact ratio moved {-ir_drop:+.3f}")
    print("PSI alert threshold 0.25; fairness SLO 0.80.")

    print("\n" + "=" * 96)
    print("ALERT ROUTING".center(96))
    print("=" * 96)
    routes = [
        ("PSI >= 0.10 on any feature", "P3", "Data science reviews within 5 business days"),
        ("PSI >= 0.25 on any feature", "P2", "Retraining assessment opened; recruiters notified"),
        ("AUC drops > 0.05 from baseline", "P2", "Model owner review; consider rollback"),
        ("Impact ratio < 0.80, any attribute", "P1", "Auto-widen shortlist; notify Legal + HR within 24h"),
        ("Impact ratio < 0.70, any attribute", "P0", "KILL SWITCH: model disabled, 100% human screening"),
        ("Adversarial flag rate > 3x baseline", "P2", "Security review of intake pipeline"),
        ("Abstention band > 40% of volume", "P3", "Recalibration; capacity planning with recruiting"),
    ]
    print(f"{'trigger':<44}{'sev':<6}{'response'}")
    print("-" * 96)
    for t, s, r in routes:
        print(f"{t:<44}{s:<6}{r}")

    print("\n" + "=" * 96)
    print("FEEDBACK LOOP CONTROL".center(96))
    print("=" * 96)
    print("""  The model's shortlists become the next cycle's training labels, which makes the
  system self-confirming unless it is broken deliberately. Three controls:

  1. RANDOMIZED HOLDOUT. 5% of applicants are screened by humans with the model
     output hidden. This preserves an unbiased label stream and is the only
     source of counterfactual data about candidates the model would have ranked
     low. It is a permanent cost of running the system, not a pilot phase.

  2. OUTCOME LABELS, NOT SCREENING LABELS. Retraining targets 12-month
     performance and retention where available, not "did a recruiter advance
     them." Screening labels are used only where outcome labels do not exist,
     and that substitution is recorded in the model card.

  3. REJECTED-CANDIDATE SAMPLING. A stratified sample of low-scored applicants
     is interviewed anyway each quarter, to estimate the false-negative rate the
     system otherwise never observes.""")

    return hist


if __name__ == "__main__":
    run()
