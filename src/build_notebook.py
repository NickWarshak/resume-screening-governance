import json, os
BASE = os.path.join(os.path.dirname(__file__), "..")

def _src(t): return t.splitlines(keepends=True)
def md(t): return {"cell_type":"markdown","metadata":{},"source":_src(t)}
def code(t): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":_src(t)}

cells = [
md("""# Responsible AI Resume Screening — End-to-End Notebook

**Author:** Nick Warshak  
**System:** Applicant shortlist ranking for software engineering requisitions  
**Classification:** High-risk AI system (EU AI Act Annex III §4; NYC Local Law 144 AEDT)

---

## The argument this notebook makes

Resume screening is the domain where AI hiring tools have most publicly failed. Amazon
scrapped an internal recruiting model in 2018 after finding it penalized resumes containing
the word "women's." The failure was not a bug. The model worked exactly as designed: it
learned to reproduce ten years of human hiring decisions, and those decisions were biased.

This notebook rebuilds that failure deliberately, measures it, and then fixes it — so that
each governance control can be evaluated against evidence rather than asserted.

**The central finding, stated up front:** the model that best predicts historical recruiter
behavior is *not* the model that best identifies qualified candidates. Optimizing the metric
available in production selects the discriminatory model."""),

md("""## 1. Setup and data generation

Real resume corpora contain PII and cannot lawfully be repurposed for model training. We
generate a synthetic population in which *true qualification is known by construction* —
which is what makes the fairness audit verifiable. In real data, true qualification is never
observed, so you can never prove the model is wrong; here we can."""),

code("""import sys, os
sys.path.insert(0, os.path.abspath("../src"))
import numpy as np, pandas as pd
pd.set_option("display.width", 160)

from generate_data import generate, MERIT_FEATURES, PROXY_FEATURES, GOVERNED_FEATURES, NAIVE_FEATURES

df = generate(n=12000, seed=42)
print(f"{len(df):,} applicants, {df.advanced_to_onsite.mean():.1%} advanced to onsite")
df.head()"""),

md("""### The four bias channels

The label is **not** "was this person good at the job." It is "did a recruiter advance them."
That gap — a proxy label standing in for the construct of interest — is the single most
important governance flaw in commercial resume screeners. Four documented mechanisms are
injected:

| Channel | Mechanism | Real-world basis |
|---|---|---|
| Referral homophily | Referrals flow through existing employee networks | Rubineau & Fernandez (2013) |
| Employment-gap penalty | Gaps penalized; caregiving unequally distributed | BLS time-use data |
| Prestige proxy | "Top school" tracks parental income, which is racialized | Chetty et al. (2020) |
| Residual direct bias | Name-based callback discrimination | Bertrand & Mullainathan (2004) |

Critically, **latent qualification is generated independent of protected class.** In this
world there is no real ability difference between groups, so every disparity the audit finds
is pure measurement bias."""),

code("""print("Advance rate by sex:")
print(df.groupby("sex").advanced_to_onsite.mean().round(4))
print("\\nAdvance rate by race/ethnicity:")
print(df.groupby("race_ethnicity").advanced_to_onsite.mean().round(4))
print("\\nMean TRUE qualification by race (all ~0 -> no real ability gap):")
print(df.groupby("race_ethnicity").latent_qualification.mean().round(4))"""),

md("""## 2. Training: two models, two philosophies

**Naive** — every available feature, including known demographic proxies. The
"maximize AUC and ship it" model.

**Governed** — proxy features removed, calibrated, abstention band applied."""),

code("""from train import run as train_run
metrics = train_run()"""),

md("""### Reading the result

Compare the two AUC columns. The naive model wins on `AUC(hist)` — predicting what
recruiters did. The governed model wins on `AUC(true qual)` — identifying who was actually
qualified.

In production **you only ever observe the first column.** Standard model selection therefore
picks the naive model, and picks the worse one. This is the mechanism by which a
well-intentioned team ships a discriminatory system while following every ML best practice."""),

md("""## 3. Fairness audit

Implements what NYC Local Law 144 requires of automated employment decision tools:
selection rates and impact ratios by sex, race/ethnicity, and their intersection.

An impact ratio below 0.80 is prima facie evidence of adverse impact under the EEOC Uniform
Guidelines (29 CFR 1607.4D)."""),

code("""from fairness import run as fairness_run
audit = fairness_run()"""),

md("""### Statistical honesty

An impact ratio computed on 68 applicants is not the same evidence as one computed on 1,113.
Reporting a bare point estimate as a compliance finding overstates what the audit knows."""),

code("""from significance import run as sig_run
sig_run()"""),

md("""**Two findings that only appear with intervals:**

1. Female / Hispanic-Latino shows a 92% bootstrap probability of a genuine four-fifths breach
   — a real finding, not noise, despite n=68.
2. Hispanic/Latino overall has a *passing* point estimate of 0.811 but a 46% probability the
   true ratio breaches. Reporting the point estimate alone would be misleading.

The governed model passes marginal audits for sex and race while still failing
intersectionally. This is precisely why LL144 mandates intersectional reporting."""),

md("""## 4. Explainability

Two distinct jobs, often conflated:

- **Global** — what does the model rely on? A bias-detection tool.
- **Local** — why was *this* applicant scored this way? A legal artifact, feeding
  adverse-action-style notices required under Illinois AIVIA and the EU AI Act.

**Caveat that belongs in the code, not just the report:** SHAP explains the *model*, not the
*world*. A +0.30 attribution for "referral" means the model raised its score because the
applicant was referred. It does not mean referrals cause job success. Treating SHAP
attributions as causal is how an organization talks itself into believing a biased feature is
a legitimate one."""),

code("""from explain import run as explain_run
shap_summary = explain_run()"""),

md("""## 5. Guardrails and red teaming

Six layers. The load-bearing design decision: **the model cannot reject anyone.** It
produces a ranked shortlist and a routing recommendation; rejection remains a human act.
There is deliberately no code path returning `REJECT`.

Prompt injection is a live threat, not hypothetical: any pipeline passing resume text to an
LLM is one where *the applicant controls part of the prompt*."""),

code("""from guardrails import run as guard_run
guard = guard_run()"""),

md("""### A red-team finding that changed the design

Case **RT-07** originally failed. An early revision matched the bare phrase
`prompt injection`, which flagged legitimate ML-security engineers who simply described their
own work — a guardrail that discriminated against applicants in the AI safety field.

The fix: detection keys on *instructional framing directed at the system*, never on subject
matter. RT-07 is retained as a standing regression test. This is what red teaming is for —
the defect was in the defense, not the model."""),

md("""## 6. Monitoring

Three failure modes, each needing its own detector. The one that matters most and is
monitored least is **fairness drift**: impact ratios degrade while AUC, PSI, and every
accuracy metric hold steady."""),

code("""from monitor import run as monitor_run
history = monitor_run()"""),

md("""At month 2 the fairness SLO breaches (race impact ratio 0.779) while AUC sits at 0.867 and
max PSI is 0.065 — *below even the warning band*. A performance-only dashboard shows all
green during an active four-fifths breach.

By month 6 the inverse holds: PSI 0.683 (well past alert) with AUC unmoved at 0.868. Drift
alarms fire with no performance degradation. Both directions of the decoupling are real, and
neither is visible from a single dashboard."""),

md("""## 7. Sustainability"""),

code("""from sustainability import run as sustain_run
sustain = sustain_run()"""),

md("""**A methodological correction worth stating.** An earlier revision counted only the
model's own arithmetic and produced a defensible-looking claim of an ~8-order-of-magnitude
advantage over an LLM. That was dishonest accounting. Priced across the whole serving
envelope — parsing, serialization, logging — the advantage is roughly 3 orders of magnitude.
Still decisive, and it has the advantage of being true.

The more useful finding: the model is **0.0004%** of pipeline energy. Optimizing it further
would be optimizing the wrong thing."""),

md("""## 8. Conclusion

| Question | Answer |
|---|---|
| Does removing proxies fix disparate impact? | Largely — worst race impact ratio 0.63 → 0.83 |
| Does it fix it completely? | **No.** Female/Hispanic-Latino remains at 0.59 |
| Does fairness cost accuracy? | Against the biased label yes (−0.046 AUC); against true qualification **no**, it gains +0.046 |
| Is the system safe to deploy fully automated? | **No** — and the design forecloses it |

The honest conclusion is that this system is suitable for **ranking and routing under human
review**, not for automated rejection. A governance report that concluded otherwise would be
recommending something the evidence in this notebook does not support."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
      "language_info": {"name":"python","version":"3.11"}}, "nbformat":4, "nbformat_minor":5}
path = os.path.join(BASE, "notebooks", "resume_screening_governance.ipynb")
json.dump(nb, open(path,"w"), indent=1)
print("wrote", path, "-", len(cells), "cells")
