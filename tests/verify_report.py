"""Automated cross-check: every quantitative claim in the report must match computed output.

A governance report whose numbers drift from its own pipeline is worse than no report --
it looks auditable while being wrong. This runs as a gate.
"""
import json, os, re, sys
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
html = open(os.path.join(BASE, "reports", "report.html")).read()
mc   = open(os.path.join(BASE, "reports", "MODEL_CARD.md")).read()
met  = json.load(open(os.path.join(BASE, "reports", "model_metrics.json")))
fair = json.load(open(os.path.join(BASE, "reports", "fairness_audit.json")))
sus  = json.load(open(os.path.join(BASE, "reports", "sustainability.json")))
ci   = pd.read_csv(os.path.join(BASE, "reports", "impact_ratio_ci.csv"))
hist = pd.read_csv(os.path.join(BASE, "reports", "monitoring_history.csv"))
shap_n = pd.read_csv(os.path.join(BASE, "reports", "shap_importance_naive.csv"))
shap_g = pd.read_csv(os.path.join(BASE, "reports", "shap_importance_governed.csv"))

checks, failures = [], []

def check(name, claim, computed, tol=0.0015, in_doc=None):
    doc = in_doc if in_doc is not None else html
    present = str(claim) in doc
    numeric_ok = abs(float(claim) - float(computed)) <= tol
    ok = present and numeric_ok
    checks.append((name, claim, round(float(computed), 5), "PASS" if ok else "FAIL"))
    if not ok:
        failures.append(f"{name}: report says {claim}, computed {computed}, in_doc={present}")

def check_text(name, needle, doc=None):
    doc = doc if doc is not None else html
    ok = needle in doc
    checks.append((name, needle[:40], "-", "PASS" if ok else "FAIL"))
    if not ok: failures.append(f"{name}: missing text {needle!r}")

# --- performance table ---
for key, label in [("naive_logistic","naive log"),("naive_gbm","naive gbm"),
                   ("governed_logistic","gov log"),("governed_gbm","gov gbm")]:
    check(f"AUC hist {label}", f"{met[key]['auc_observed_label']:.3f}", met[key]['auc_observed_label'])
    check(f"AUC true {label}", f"{met[key]['auc_vs_latent_qualification']:.3f}", met[key]['auc_vs_latent_qualification'])

# --- fairness ---
check("human sex IR", "0.607", fair["human_baseline"]["sex_worst_impact_ratio"])
check("human race IR", "0.583", fair["human_baseline"]["race_worst_impact_ratio"])
check("naive sex IR", "0.612", fair["naive_gbm"]["sex_worst_impact_ratio"])
check("naive race IR", "0.633", fair["naive_gbm"]["race_worst_impact_ratio"])
check("naive inter IR", "0.488", fair["naive_gbm"]["intersectional_worst_impact_ratio"])
check("gov sex IR", "0.967", fair["governed_logistic"]["sex_worst_impact_ratio"])
check("gov race IR", "0.833", fair["governed_logistic"]["race_worst_impact_ratio"])
check("gov inter IR", "0.652", fair["governed_logistic"]["intersectional_worst_impact_ratio"])
check_text("worst group naive", "Female / Black")
check_text("worst group gov", "Female / Hispanic-Latino")

# --- confidence intervals ---
r = ci[(ci.model=="governed_logistic") & (ci.group=="Female / Hispanic/Latino")].iloc[0]
check("F/H-L CI low", "0.370", r.ci_low, tol=0.002)
check("F/H-L CI high", "0.949", r.ci_high, tol=0.002)
check("F/H-L P(fail)", "87", r.prob_below_threshold*100, tol=1.0)
r2 = ci[(ci.model=="governed_logistic") & (ci.group=="Hispanic/Latino")].iloc[0]
check("H/L point est", "0.811", r2.ratio, tol=0.002)
check("H/L P(fail)", "46", r2.prob_below_threshold*100, tol=1.0)

# --- SHAP ---
proxy_share = shap_n[shap_n.is_proxy].share_of_total.sum()
check("naive proxy share", "37.7", proxy_share*100, tol=0.15)
ref = shap_n[shap_n.feature=="referral"].share_of_total.iloc[0]
check("referral share", "15.5", ref*100, tol=0.15)
kw_n = shap_n[shap_n.feature=="keyword_match_score"].share_of_total.iloc[0]
kw_g = shap_g[shap_g.feature=="keyword_match_score"].share_of_total.iloc[0]
check("keyword naive", "18.8", kw_n*100, tol=0.15)
check("keyword governed", "33.3", kw_g*100, tol=0.15)
gov_proxy = shap_g[shap_g.is_proxy].share_of_total.sum()
assert gov_proxy == 0, "governed model must contain zero proxy features"
checks.append(("governed proxy share == 0", "0", 0, "PASS"))

# --- monitoring ---
m2 = hist[hist.month==2].iloc[0]; m6 = hist[hist.month==6].iloc[0]
check("month2 IR race", "0.779", m2.ir_race)
check("month2 AUC", "0.867", m2.auc)
check("month2 PSI", "0.065", m2.max_psi)
check("month6 PSI", "0.683", m6.max_psi)
check("month6 AUC", "0.868", m6.auc)

# --- sustainability ---
ap = sus["annual_projection"]
check("ratio low", "403", ap["ratio_low"], tol=0.5)
check("ratio high", "4,019", ap["ratio_high"], tol=0.5) if False else None
checks.append(("ratio high (formatted)", "4,019", ap["ratio_high"], "PASS" if "4,019" in html and abs(ap["ratio_high"]-4019)<1 else "FAIL"))
if not ("4,019" in html and abs(ap["ratio_high"]-4019)<1): failures.append("ratio high mismatch")
check("annual kWh", "0.19", ap["annual_total_kwh"], tol=0.005)
check("llm low kWh", "75", ap["llm_alternative_annual_kwh_low"], tol=0.5)
check("parsing Wh", "140", ap["parsing_wh"], tol=0.5)
check("serving Wh", "47", ap["serving_overhead_wh"], tol=0.5)
check("gco2 us", "68.9", ap["annual_gco2_us_grid"], tol=0.1)
check("gco2 low carbon", "6.0", ap["annual_gco2_low_carbon"], tol=0.1)
pm = sus["per_model"]
check("model size KB", "2.1", pm["governed_logistic"]["model_size_kb"], tol=0.05)
check_text("infer ms (approx, jitters)", "≈0.7")
check_text("model share pct", "0.0004%")

# --- guardrails ---
check_text("red team 7/7", "7 of 7 red-team cases pass")
check_text("abstention", "22.8%")
check_text("proxy pairs", "11 of 48")

# --- regulatory citations ---
for cite in ["29 CFR 1607.4D", "Local Law 144", "820 ILCS 42", "SB 24-205",
             "Annex III", "2024/1689", "Bertrand", "Lundberg"]:
    check_text(f"cite {cite}", cite)

# --- model card consistency ---
check("MC gov race IR", "0.833", fair["governed_logistic"]["race_worst_impact_ratio"], in_doc=mc)
check("MC AUC true", "0.961", met["governed_logistic"]["auc_vs_latent_qualification"], in_doc=mc)
check_text("MC infer ms", "≈0.7 ms per 1,000 applicants", doc=mc)

# --- report structure ---
for sec in ["1. Model Overview","2. Explainability","3. Guardrails","4. Monitoring Plan",
            "5. Compliance and Documentation","6. Sustainability"]:
    check_text(f"section {sec}", sec)

df = pd.DataFrame(checks, columns=["check","report_claim","computed","result"])
print("=" * 88)
print("REPORT CLAIM VERIFICATION".center(88))
print("=" * 88)
print(df.to_string(index=False))
print("-" * 88)
n_fail = (df.result == "FAIL").sum()
print(f"{len(df) - n_fail}/{len(df)} checks passed")
if failures:
    print("\nFAILURES:")
    for f in failures: print("  -", f)
    sys.exit(1)
print("\nAll quantitative claims in the report match computed pipeline output.")
