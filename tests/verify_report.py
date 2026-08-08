"""Cross-check: every number in the report has to match what the code computed.

A governance report whose numbers have drifted from its own pipeline is worse than no report,
because it looks auditable while being wrong. This runs as a gate.

Each check asserts two things: that the claim literally appears in the document, and that it
agrees numerically with the computed output. A number that is right but missing from the report,
or present but stale, fails either way.
"""
import json, os, re, sys
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
html = open(os.path.join(BASE, "reports", "report.html"), encoding="utf-8").read()
mc   = open(os.path.join(BASE, "reports", "MODEL_CARD.md"), encoding="utf-8").read()
met  = json.load(open(os.path.join(BASE, "reports", "model_metrics.json")))
fair = json.load(open(os.path.join(BASE, "reports", "fairness_audit.json")))
sus  = json.load(open(os.path.join(BASE, "reports", "sustainability.json")))
ci   = pd.read_csv(os.path.join(BASE, "reports", "impact_ratio_ci.csv"))
hist = pd.read_csv(os.path.join(BASE, "reports", "monitoring_history.csv"))
shap_n = pd.read_csv(os.path.join(BASE, "reports", "shap_importance_naive.csv"))
shap_g = pd.read_csv(os.path.join(BASE, "reports", "shap_importance_governed.csv"))

checks, failures = [], []


def flat(s):
    """Collapse whitespace so an assertion is not defeated by HTML line wrapping."""
    return re.sub(r"\s+", " ", s)


def fmt(x, places=3):
    """Round for display the way a person writes it (half away from zero).

    Python's format() inherits binary rounding, so 0.6795 prints as 0.679 while
    anyone writing the number by hand puts 0.680. Deriving the expected string
    this way keeps the gate honest without arguing about the last digit.
    """
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def check(name, claim, computed, tol=0.0015, in_doc=None):
    """Claim must appear verbatim in the document AND match the computed value."""
    doc = flat(in_doc if in_doc is not None else html)
    present = str(claim) in doc
    numeric_ok = abs(float(str(claim).replace(",", "")) - float(computed)) <= tol
    ok = present and numeric_ok
    checks.append((name, claim, round(float(computed), 5), "PASS" if ok else "FAIL"))
    if not ok:
        failures.append(f"{name}: report says {claim}, computed {computed}, in_doc={present}")


def check_num(name, computed, tol=0.0015, in_doc=None, places=3):
    """Like check(), but derives the expected string from the computed value.

    A value stored at 4 dp is genuinely ambiguous at 3 dp when it ends in a 5
    (0.9215 is equally 0.921 or 0.922), so either neighbour is accepted. The
    numeric tolerance below still catches a report that has actually drifted.
    """
    doc = flat(in_doc if in_doc is not None else html)
    candidates = {fmt(computed, places), f"{float(computed):.{places}f}"}
    found = next((c for c in sorted(candidates) if c in doc), None)
    ok = found is not None and abs(float(found) - float(computed)) <= tol
    shown = found if found else "/".join(sorted(candidates))
    checks.append((name, shown, round(float(computed), 5), "PASS" if ok else "FAIL"))
    if not ok:
        failures.append(f"{name}: expected one of {sorted(candidates)}, computed {computed}, found={found}")


def check_text(name, needle, doc=None):
    doc = flat(doc if doc is not None else html)
    ok = flat(needle) in doc
    checks.append((name, needle[:44], "-", "PASS" if ok else "FAIL"))
    if not ok:
        failures.append(f"{name}: missing text {needle!r}")


# --- performance table -----------------------------------------------------
PERF = [("naive_logistic", "naïve log"), ("naive_gbm", "naïve gbm"),
        ("governed_logistic", "gov log"), ("governed_gbm", "gov gbm")]
for key, label in PERF:
    m = met[key]
    check_num(f"AUC recruiter {label}", m["auc_observed_label"])
    check_num(f"AUC merit {label}", m["auc_vs_latent_qualification"])
    check_num(f"precision {label}", m["avg_precision_observed"])
    check_num(f"brier {label}", m["brier_score"])
    check_num(f"P@25 {label}", m["operating_point"]["precision"])
    check_num(f"R@25 {label}", m["operating_point"]["recall"])

# --- fairness --------------------------------------------------------------
check("human sex IR", "0.607", fair["human_baseline"]["sex_worst_impact_ratio"])
check("human ethnicity IR", "0.583", fair["human_baseline"]["race_worst_impact_ratio"])
check("naive sex IR", "0.612", fair["naive_gbm"]["sex_worst_impact_ratio"])
check("naive ethnicity IR", "0.633", fair["naive_gbm"]["race_worst_impact_ratio"])
check("naive intersection IR", "0.488", fair["naive_gbm"]["intersectional_worst_impact_ratio"])
check("gov sex IR", "0.967", fair["governed_logistic"]["sex_worst_impact_ratio"])
check("gov ethnicity IR", "0.833", fair["governed_logistic"]["race_worst_impact_ratio"])
check("gov intersection IR", "0.652", fair["governed_logistic"]["intersectional_worst_impact_ratio"])
check_text("worst group naive", "Female / Black")
check_text("worst group governed", "Female / Hispanic")
check_text("four-fifths threshold stated", "0.80 is prima facie evidence")

# --- SHAP ------------------------------------------------------------------
proxy_share = shap_n[shap_n.is_proxy].share_of_total.sum()
check("naive attribution via biased features", "37.7", proxy_share * 100, tol=0.15)
check("referral share", "15.5", shap_n[shap_n.feature == "referral"].share_of_total.iloc[0] * 100, tol=0.15)
check("keyword match naive", "18.8", shap_n[shap_n.feature == "keyword_match_score"].share_of_total.iloc[0] * 100, tol=0.15)
check("keyword match governed", "33.3", shap_g[shap_g.feature == "keyword_match_score"].share_of_total.iloc[0] * 100, tol=0.15)
gov_proxy = shap_g[shap_g.is_proxy].share_of_total.sum()
assert gov_proxy == 0, "the governed model must contain zero biased features"
checks.append(("governed biased-feature share == 0", "0", 0, "PASS"))

# --- guardrails ------------------------------------------------------------
check_text("red team 7 of 7", "7 of 7 red-team cases pass")
check_text("proxy pairs flagged", "11 of 48 feature-attribute pairs flagged")
check_text("abstention share", "22.8% of applicants route to human review")
check_text("abstention band bounds", "0.25–0.55")
check_text("no reject path", "not to have any code allowing it to reject")

# --- monitoring ------------------------------------------------------------
m2 = hist[hist.month == 2].iloc[0]
m6 = hist[hist.month == 6].iloc[0]
check("month 2 ethnicity IR", "0.779", m2.ir_race)
check("month 2 AUC", "0.867", m2.auc)
check("month 2 PSI", "0.065", m2.max_psi)
check("month 6 PSI", "0.683", m6.max_psi)
check("month 6 AUC", "0.868", m6.auc)

# --- sustainability --------------------------------------------------------
pm = sus["per_model"]
ap = sus["annual_projection"]
# Model sizes are deterministic.
check("size governed logistic", "2.1", pm["governed_logistic"]["model_size_kb"], tol=0.05)
check("size governed gbm", "201.9", pm["governed_gbm"]["model_size_kb"], tol=0.05)
check("size naive gbm", "240.9", pm["naive_gbm"]["model_size_kb"], tol=0.05)
# Wall-clock timings jitter between runs, so these carry a loose tolerance. They
# still fail if the report drifts from the JSON by more than rounding noise.
JITTER = 0.06
check("train cpu governed logistic", "0.02", pm["governed_logistic"]["train_cpu_seconds"], tol=JITTER)
check("train cpu governed gbm", "0.37", pm["governed_gbm"]["train_cpu_seconds"], tol=JITTER)
check("train cpu naive gbm", "0.39", pm["naive_gbm"]["train_cpu_seconds"], tol=JITTER)
check("inference governed logistic", "0.6", pm["governed_logistic"]["inference_ms_per_1k_resumes"], tol=JITTER)
check("inference governed gbm", "2.71", pm["governed_gbm"]["inference_ms_per_1k_resumes"], tol=JITTER)
check("inference naive gbm", "3.40", pm["naive_gbm"]["inference_ms_per_1k_resumes"], tol=JITTER)
check("train energy governed logistic", "0.0001", pm["governed_logistic"]["train_energy_wh"], tol=0.0005)
check("train energy governed gbm", "0.0017", pm["governed_gbm"]["train_energy_wh"], tol=0.0005)
check("train energy naive gbm", "0.0018", pm["naive_gbm"]["train_energy_wh"], tol=0.0005)
check("annual kWh", "0.19", ap["annual_total_kwh"], tol=0.005)
check("energy ratio low", "403", ap["ratio_low"], tol=0.5)
check("energy ratio high", "4,019", ap["ratio_high"], tol=0.5)
check_text("model share of pipeline energy", "0.0004%")

# --- regulation ------------------------------------------------------------
for cite in ["NYC Local Law 144", "EEOC Uniform Guidelines", "Title VII",
             "Illinois AIVIA", "Colorado SB 24-205"]:
    check_text(f"cite {cite}", cite)

# --- report structure ------------------------------------------------------
for sec in ["Model Overview", "Explainability", "Guardrails", "Monitoring Plan",
            "Compliance &amp; Documentation", "Sustainability", "Conclusion"]:
    check_text(f"section {sec}", sec)

# --- honesty: the unresolved disparity must stay disclosed -----------------
check_text("intersection disclosed as unfixed", "remains an unfixed problem")
check_text("scope limited to ranking", "not in place of one")

# --- model card consistency ------------------------------------------------
check("MC gov ethnicity IR", "0.833", fair["governed_logistic"]["race_worst_impact_ratio"], in_doc=mc)
check("MC gov intersection IR", "0.652", fair["governed_logistic"]["intersectional_worst_impact_ratio"], in_doc=mc)
check("MC AUC merit", "0.961", met["governed_logistic"]["auc_vs_latent_qualification"], in_doc=mc)
check("MC AUC recruiter", "0.865", met["governed_logistic"]["auc_observed_label"], in_doc=mc)
check("MC size", "2.1", pm["governed_logistic"]["model_size_kb"], tol=0.05, in_doc=mc)
check_text("MC inference", "about 0.6 ms per 1,000 applicants", doc=mc)
check_text("MC no-reject", "never a rejection", doc=mc)
check_text("MC discloses the intersection", "What is still broken", doc=mc)

# --- confidence intervals (reported on the site and in the model card) -----
r = ci[(ci.model == "governed_logistic") & (ci.group == "Female / Hispanic/Latino")].iloc[0]
idx = os.path.join(BASE, "docs", "index.html")
if os.path.exists(idx):
    site = open(idx, encoding="utf-8").read()
    check("site CI low", "0.370", r.ci_low, tol=0.002, in_doc=site)
    check("site CI high", "0.949", r.ci_high, tol=0.002, in_doc=site)
    check("site P(breach)", "87", r.prob_below_threshold * 100, tol=1.0, in_doc=site)

df = pd.DataFrame(checks, columns=["check", "report_claim", "computed", "result"])
print("=" * 92)
print("REPORT CLAIM VERIFICATION".center(92))
print("=" * 92)
print(df.to_string(index=False))
print("-" * 92)
n_fail = (df.result == "FAIL").sum()
print(f"{len(df) - n_fail}/{len(df)} checks passed")
if failures:
    print("\nFAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("\nEvery number in the report matches what the pipeline computed.")
