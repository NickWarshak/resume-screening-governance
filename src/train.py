"""
Model training and evaluation for the resume screening system.

Trains two models to make the central governance argument empirically:

  NAIVE model    - every available feature, including known demographic proxies
                   (referral, top_school, employment_gap, distance). This is the
                   "maximize AUC and ship it" model.

  GOVERNED model - proxy features removed, calibrated, with an abstention band.
                   Lower headline accuracy against the biased label.

The critical evaluation is measuring BOTH models against two targets:
  (a) the observed historical label  -> rewards imitating biased recruiters
  (b) latent true qualification      -> rewards actually identifying talent
A model can win on (a) while losing on (b). That divergence is the finding.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generate_data import GOVERNED_FEATURES, NAIVE_FEATURES

BASE = os.path.join(os.path.dirname(__file__), "..")
SEED = 42


def load():
    return pd.read_csv(os.path.join(BASE, "data", "applicants.csv"))


def split(df):
    """Stratified 60/20/20 train/validation/test split."""
    train, temp = train_test_split(
        df, test_size=0.40, random_state=SEED, stratify=df["advanced_to_onsite"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=SEED, stratify=temp["advanced_to_onsite"]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_models():
    return {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        ]),
        "gbm": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_depth=5,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=SEED,
        ),
    }


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "selection_rate": round(float(pred.mean()), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def pick_threshold_for_rate(proba, target_rate):
    """Choose the score cutoff that advances a fixed fraction of applicants.

    Operationally this is how screening actually works: recruiter capacity is
    fixed, so the business picks a shortlist size, not a probability cutoff.
    """
    return float(np.quantile(proba, 1.0 - target_rate))


def run():
    df = load()
    train, val, test = split(df)
    results = {}
    artifacts = {}

    configs = {
        "naive": NAIVE_FEATURES,
        "governed": GOVERNED_FEATURES,
    }

    for cfg_name, feats in configs.items():
        Xtr, ytr = train[feats], train["advanced_to_onsite"]
        Xva, yva = val[feats], val["advanced_to_onsite"]
        Xte, yte = test[feats], test["advanced_to_onsite"]

        for model_name, model in build_models().items():
            key = f"{cfg_name}_{model_name}"

            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            fit_seconds = time.perf_counter() - t0

            # Probability calibration on the held-out validation split
            cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
            cal.fit(Xva, yva)

            t1 = time.perf_counter()
            proba_te = cal.predict_proba(Xte)[:, 1]
            infer_seconds = time.perf_counter() - t1

            # Evaluation target (a): the observed, biased historical label
            auc_observed = roc_auc_score(yte, proba_te)
            ap_observed = average_precision_score(yte, proba_te)
            brier = brier_score_loss(yte, proba_te)

            # Evaluation target (b): the ground truth we actually care about.
            # Unobservable in production -- available here only by construction.
            auc_latent = roc_auc_score(
                (test["latent_qualification"] > test["latent_qualification"].median()).astype(int),
                proba_te,
            )
            auc_counterfactual = roc_auc_score(test["advanced_counterfactual"], proba_te)

            thr = pick_threshold_for_rate(proba_te, target_rate=0.25)

            results[key] = {
                "feature_set": cfg_name,
                "algorithm": model_name,
                "n_features": len(feats),
                "auc_observed_label": round(float(auc_observed), 4),
                "avg_precision_observed": round(float(ap_observed), 4),
                "brier_score": round(float(brier), 4),
                "auc_vs_latent_qualification": round(float(auc_latent), 4),
                "auc_vs_counterfactual_fair_label": round(float(auc_counterfactual), 4),
                "fit_seconds": round(fit_seconds, 3),
                "inference_seconds_per_1k": round(infer_seconds / len(Xte) * 1000, 5),
                "operating_point": metrics_at(yte, proba_te, thr),
            }
            artifacts[key] = {"model": cal, "features": feats, "threshold": thr}

    # Persist scored test set for the fairness audit
    best = artifacts["governed_gbm"]
    scored = test.copy()
    for key, art in artifacts.items():
        scored[f"score_{key}"] = art["model"].predict_proba(test[art["features"]])[:, 1]
        scored[f"pred_{key}"] = (scored[f"score_{key}"] >= art["threshold"]).astype(int)
    scored.to_csv(os.path.join(BASE, "data", "test_scored.csv"), index=False)

    import joblib
    for key, art in artifacts.items():
        joblib.dump(
            {"model": art["model"], "features": art["features"], "threshold": art["threshold"]},
            os.path.join(BASE, "models", f"{key}.joblib"),
        )

    with open(os.path.join(BASE, "reports", "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary ----
    print("=" * 88)
    print("MODEL PERFORMANCE".center(88))
    print("=" * 88)
    hdr = f"{'model':<20}{'AUC(hist)':>11}{'AP':>8}{'Brier':>8}{'AUC(true qual)':>16}{'AUC(fair)':>11}"
    print(hdr)
    print("-" * 88)
    for k, v in results.items():
        print(f"{k:<20}{v['auc_observed_label']:>11.4f}{v['avg_precision_observed']:>8.4f}"
              f"{v['brier_score']:>8.4f}{v['auc_vs_latent_qualification']:>16.4f}"
              f"{v['auc_vs_counterfactual_fair_label']:>11.4f}")
    print("-" * 88)
    print("AUC(hist)      = ranks applicants the way past recruiters did")
    print("AUC(true qual) = ranks applicants by actual latent ability")
    print()
    print("Operating point @ 25% shortlist rate:")
    for k, v in results.items():
        op = v["operating_point"]
        print(f"  {k:<20} precision={op['precision']:.3f}  recall={op['recall']:.3f}  f1={op['f1']:.3f}")
    return results


if __name__ == "__main__":
    run()
