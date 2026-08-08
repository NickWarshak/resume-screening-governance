"""
Export the trained models to JSON so the browser demo can score applicants
client-side, with no server.

WHY THIS IS FAITHFUL AND NOT AN APPROXIMATION:
The production candidate is a logistic regression - a standardizer, a weight
vector, and an isotonic calibration curve. All three are exactly representable
as numbers, and evaluating them is arithmetic a browser can do. This script
retrains with train.py's exact procedure and seed, extracts those numbers, and
then *verifies* that a dependency-free reimplementation of the scoring math
reproduces scikit-learn's predict_proba on the full test set to within 1e-12.
If that check fails the export aborts, so the demo can never silently drift
from the audited model.

The same closed form gives exact local attributions. For a linear model the
Shapley value of feature j is coef_j * (z_j - E[z_j]) under feature
independence - no sampling, no approximation, no shap dependency.

Run:  python export_web_model.py
Out:  docs/model.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_data import GOVERNED_FEATURES, MERIT_FEATURES, NAIVE_FEATURES, PROXY_FEATURES
from train import load, pick_threshold_for_rate, split

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = 42

# Presentation metadata for the demo UI. step/decimals control the slider only;
# they never affect scoring.
FEATURE_UI = {
    "years_experience":       ("Years of experience",        "yr",    0.5, 1),
    "education_level":        ("Education level",            "",      1,   0),
    "num_relevant_skills":    ("Relevant skills matched",    "",      1,   0),
    "keyword_match_score":    ("Keyword match vs. req",      "",      0.01, 2),
    "gpa":                    ("GPA",                        "",      0.01, 2),
    "num_certifications":     ("Certifications",             "",      1,   0),
    "portfolio_projects":     ("Portfolio projects",         "",      1,   0),
    "num_prior_roles":        ("Prior roles",                "",      1,   0),
    "avg_tenure_months":      ("Average tenure",             "mo",    1,   0),
    "leadership_indicators":  ("Leadership indicators",      "",      1,   0),
    "num_publications":       ("Publications",               "",      1,   0),
    "resume_length_words":    ("Resume length",              "words", 10,  0),
    "employment_gap_months":  ("Employment gap",             "mo",    0.5, 1),
    "top_school":             ("Attended a 'top' school",    "",      1,   0),
    "referral":               ("Has an internal referral",   "",      1,   0),
    "distance_from_office_km": ("Distance from office",      "km",    1,   0),
}

EDUCATION_LABELS = ["High school", "Associate", "Bachelor's", "Master's", "PhD"]


def fit_logistic(Xtr, ytr, Xva, yva):
    """Reproduce train.py's logistic pipeline + isotonic calibration exactly."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
    ])
    pipe.fit(Xtr, ytr)
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    cal.fit(Xva, yva)
    return pipe, cal


def extract(pipe, cal, feats, Xtr):
    """Pull the scaler, weights and isotonic knots out of the fitted objects."""
    scaler = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]

    calibrated = cal.calibrated_classifiers_
    assert len(calibrated) == 1, f"expected 1 calibrator (FrozenEstimator), got {len(calibrated)}"
    iso = calibrated[0].calibrators[0]

    # Mean of the standardized training features. StandardScaler centres on the
    # training mean so this is ~0, but carrying the measured value keeps the
    # attribution decomposition exact rather than nearly exact.
    z_mean = ((Xtr.to_numpy(dtype=float) - scaler.mean_) / scaler.scale_).mean(axis=0)

    return {
        "features": list(feats),
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "z_mean": [float(v) for v in z_mean],
        "iso_x": [float(v) for v in iso.X_thresholds_],
        "iso_y": [float(v) for v in iso.y_thresholds_],
    }


# --------------------------------------------------------------------------
# Dependency-free reimplementation of the scoring path.
# This mirrors, line for line, what demo.js does in the browser. Keeping the
# reference here is what makes the parity assertion below meaningful.
# --------------------------------------------------------------------------
def score_pure(m, X):
    z = (np.asarray(X, dtype=float) - np.array(m["mean"])) / np.array(m["scale"])
    margin = z @ np.array(m["coef"]) + m["intercept"]          # decision_function
    return np.interp(margin, m["iso_x"], m["iso_y"])            # isotonic, clipped at ends


def run():
    df = load()
    train, val, test = split(df)

    out = {"seed": SEED, "models": {}, "features": {}, "presets": []}

    configs = {"naive": NAIVE_FEATURES, "governed": GOVERNED_FEATURES}
    max_err = 0.0

    for name, feats in configs.items():
        Xtr, ytr = train[feats], train["advanced_to_onsite"]
        Xva, yva = val[feats], val["advanced_to_onsite"]
        Xte = test[feats]

        pipe, cal = fit_logistic(Xtr, ytr, Xva, yva)
        m = extract(pipe, cal, feats, Xtr)

        # ---- parity gate: pure-arithmetic path vs scikit-learn ----
        ref = cal.predict_proba(Xte)[:, 1]
        mine = score_pure(m, Xte.to_numpy(dtype=float))
        err = float(np.max(np.abs(ref - mine)))
        max_err = max(max_err, err)
        if err > 1e-12:
            raise SystemExit(f"ABORT: {name} scoring parity failed, max |diff| = {err:.3e}")

        # ---- metrics, to confirm we retrained the audited model ----
        yte = test["advanced_to_onsite"]
        latent = (test["latent_qualification"] > test["latent_qualification"].median()).astype(int)
        m["auc_observed"] = round(float(roc_auc_score(yte, ref)), 4)
        m["auc_latent"] = round(float(roc_auc_score(latent, ref)), 4)
        m["threshold_25pct"] = round(float(pick_threshold_for_rate(ref, 0.25)), 6)

        # Score distribution over the 2,400-applicant test set, as 101 quantiles.
        # Lets the demo place a candidate in the population without shipping the data.
        m["score_quantiles"] = [round(float(q), 6) for q in np.quantile(ref, np.linspace(0, 1, 101))]

        out["models"][name] = m
        print(f"{name:<10} AUC(hist)={m['auc_observed']:.4f}  AUC(true qual)={m['auc_latent']:.4f}  "
              f"parity max|diff|={err:.2e}")

    # ---- feature UI metadata, derived from the training distribution ----
    for f in NAIVE_FEATURES:
        col = train[f].astype(float)
        label, unit, step, decimals = FEATURE_UI[f]
        binary = bool(set(np.unique(col.to_numpy())) <= {0.0, 1.0})
        out["features"][f] = {
            "label": label,
            "unit": unit,
            "step": step,
            "decimals": decimals,
            "binary": binary,
            "is_proxy": f in PROXY_FEATURES,
            "min": float(np.floor(col.quantile(0.01) / step) * step) if not binary else 0.0,
            "max": float(np.ceil(col.quantile(0.99) / step) * step) if not binary else 1.0,
            "median": float(col.median()),
            "mean": float(col.mean()),
        }
    out["features"]["education_level"]["choices"] = EDUCATION_LABELS

    out["merit_features"] = MERIT_FEATURES
    out["proxy_features"] = PROXY_FEATURES

    # ---- preset applicants drawn from the real test split ----
    # Chosen by governed-model percentile so the demo opens on recognisable cases.
    gov = out["models"]["governed"]
    gov_scores = score_pure(gov, test[GOVERNED_FEATURES].to_numpy(dtype=float))
    order = np.argsort(gov_scores)
    picks = {
        "Strong candidate": order[int(0.96 * len(order))],
        "Borderline candidate": order[int(0.62 * len(order))],
        "Weak match": order[int(0.18 * len(order))],
    }
    for label, idx in picks.items():
        row = test.iloc[int(idx)]
        out["presets"].append({
            "label": label,
            "values": {f: float(row[f]) for f in NAIVE_FEATURES},
        })

    # ---- reconcile UI grid with the values we actually ship ----
    # A range input silently snaps its thumb to the step grid and clamps to
    # [min,max]. If a median or preset value sits off-grid or out of range, the
    # slider shows one number while the model scores another. Align them here so
    # what the user sees is exactly what gets scored.
    def snap(v, step):
        return round(round(v / step) * step, 6)

    for f in NAIVE_FEATURES:
        meta = out["features"][f]
        if meta["binary"] or meta.get("choices"):
            continue
        step = meta["step"]
        vals = [meta["median"]] + [p["values"][f] for p in out["presets"]]
        meta["median"] = snap(meta["median"], step)
        for p in out["presets"]:
            p["values"][f] = snap(p["values"][f], step)
        meta["min"] = snap(min([meta["min"]] + vals), step)
        meta["max"] = snap(max([meta["max"]] + vals), step)
        assert meta["min"] < meta["max"], f"{f}: degenerate slider range"
        for v in [meta["median"]] + [p["values"][f] for p in out["presets"]]:
            assert meta["min"] <= v <= meta["max"], f"{f}: {v} outside slider range"
            assert abs(v / step - round(v / step)) < 1e-9, f"{f}: {v} off the step grid"

    dest = os.path.join(BASE, "docs", "model.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))

    size = os.path.getsize(dest)
    print(f"\nparity verified on {len(test):,} test applicants, worst error {max_err:.2e}")
    print(f"wrote {dest} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    run()
