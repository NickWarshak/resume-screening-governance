"""Builds notebooks/resume_screening_governance.ipynb.

The notebook's prose lives here rather than in the .ipynb so it can be edited
and diffed as text. If the notebook already exists, any captured code outputs
are carried over, so rebuilding the narration does not throw away a run.
"""
import json, os

BASE = os.path.join(os.path.dirname(__file__), "..")

def _src(t): return t.splitlines(keepends=True)
def md(t): return {"cell_type":"markdown","metadata":{},"source":_src(t)}
def code(t): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":_src(t)}

cells = [
md("""# Responsible AI Resume Screening

**Author:** Nick Warshak
**System:** Ranking applicants for software engineering roles
**Status:** Reference implementation, not approved for production

---

## What this notebook is arguing

Resume screening is a domain in which AI has very publicly come up short. Amazon cut an internal
recruitment model in 2018 when it found that the model deducted from an application with the word
"women's." That was not a bug. The model was trained on human decisions and had baked in human
bias, which is exactly what it was asked to do.

This notebook rebuilds that failure on purpose, measures it, and then constrains it, so that the
governance can be checked against real evidence instead of just asserted.

**What I found, up front:** the model that best predicts what recruiters did is not the model that
best finds qualified people. Picking the model with the best number you can actually see in
production gets you the discriminatory one."""),

md("""## 1. Setup and building the data

Real resumes contain information that can't, and shouldn't, be used for model training without
consent. So I generate a synthetic pool instead, where **qualification is known** and generated
separately from protected class.

That last part is what makes the fairness audit mean anything. On real data you never observe
whether someone would actually have been good at the job, only whether they got hired, so you can
never prove the model is wrong. Here I can."""),

code("""import sys, os
sys.path.insert(0, os.path.abspath("../src"))
import numpy as np, pandas as pd
pd.set_option("display.width", 160)

from generate_data import generate, MERIT_FEATURES, PROXY_FEATURES, GOVERNED_FEATURES, NAIVE_FEATURES

df = generate(n=12000, seed=42)
print(f"{len(df):,} applicants, {df.advanced_to_onsite.mean():.1%} advanced to onsite")
df.head()"""),

md("""### The four kinds of bias I put in

The training label has never been "was this person good at the job." It is "did this person get
advanced by the recruiter." That gap is the single biggest problem with resume screeners, and it
brings in biases that are neither legal nor productive for recruiting. I built in four of them:

| Channel | What it is |
|---|---|
| Referrals | Internal references, which flow through the people already working there |
| Gaps | Time not employed, which penalizes anyone who had to step away for personal reasons |
| Prestige | Which school someone attended, which carries obvious class and income factors |
| Distance | How far the candidate lives from the office |

The important part is that **qualification is generated separately from protected class.** There
is no real ability difference between groups in this world, so anything the audit finds is bias I
put there on purpose and can measure."""),

code("""print("Advance rate by sex:")
print(df.groupby("sex").advanced_to_onsite.mean().round(4))
print("\\nAdvance rate by race/ethnicity:")
print(df.groupby("race_ethnicity").advanced_to_onsite.mean().round(4))
print("\\nMean TRUE qualification by race (all ~0 -> no real ability gap):")
print(df.groupby("race_ethnicity").latent_qualification.mean().round(4))"""),

md("""## 2. Training two models

I trained two models so they could be compared directly.

**Naïve** uses all sixteen features, including the four biased ones. This is the model you get if
you just maximize the score and ship it.

**Governed** uses only the twelve merit-based features, calibrated, with an abstention band."""),

code("""from train import run as train_run
metrics = train_run()"""),

md("""### Reading that table

Compare the two AUC columns. The naïve model wins on `AUC(hist)`, which is predicting what
recruiters did. The governed model wins on `AUC(true qual)`, which is finding who was actually
qualified.

In production **you only ever see the first column.** So standard practice picks the naïve model,
and picks the worse one. This is how a team following every best practice still ends up shipping
a discriminatory system."""),

md("""## 3. Fairness audit

This measures fairness the way the law does, specifically NYC: selection rates and impact ratios
by sex, race/ethnicity, and their intersection.

An impact ratio below 0.80 is prima facie evidence of adverse impact under the EEOC Uniform
Guidelines."""),

code("""from fairness import run as fairness_run
audit = fairness_run()"""),

md("""### Being honest about small groups

An impact ratio worked out on 68 applicants is not the same evidence as one worked out on 1,113.
Reporting a bare number as a compliance finding claims more than the audit actually knows, so I
bootstrap intervals around them."""),

code("""from significance import run as sig_run
sig_run()"""),

md("""**Two things only show up once you have intervals:**

1. Female / Hispanic has an 87% chance of a genuine breach. That is a real finding and not noise,
   even at n=68.
2. Hispanic/Latino overall has a *passing* point estimate of 0.811, but a 46% chance the true
   ratio actually breaches. Reporting just the point estimate would have been misleading.

The governed model passes on sex and on ethnicity separately while still failing at the
intersection. That is exactly why the law asks for intersectional reporting."""),

md("""## 4. Explainability

Two different jobs that get confused with each other:

- **Global** is how the model makes decisions as a whole. This is what catches bias.
- **Local** is why one specific decision came out the way it did. This is the artifact you would
  have to show a candidate who asked.

One caveat that belongs in the code and not just the report: SHAP explains the **model**, not the
**world**. A +0.30 for "referral" means the model raised its score because the applicant was
referred. It does not mean referrals cause job success. Treating these as causal is how a team
talks itself into believing a biased feature is a legitimate one."""),

code("""from explain import run as explain_run
shap_summary = explain_run()"""),

md("""## 5. Guardrails and red teaming

Six layers. The most important design choice was **not letting the model reject anybody.** It
outputs a shortlist and a recommendation, and there is deliberately no code path that returns a
rejection. That keeps it a ranking tool rather than an automated decision maker.

Prompt injection is a real threat here, not a hypothetical one. Applicants will try tricks to get
scored highly, which defeats the point of having the tool at all."""),

code("""from guardrails import run as guard_run
guard = guard_run()"""),

md("""### A red-team finding that changed the design

Case **RT-07** originally failed. An early version matched the bare phrase `prompt injection`,
which flagged legitimate ML and AI-safety engineers who were simply describing their own work. The
guardrail was discriminating against applicants in that field.

The fix was to detect *framing directed at the system* rather than subject matter. I kept RT-07 in
the suite so it can't come back. This is what red teaming is for. The defect was in the defense,
not the model."""),

md("""## 6. Monitoring

Resume screening drifts in three ways, and each one needs its own detector. The one that matters
most and gets watched least is **fairness drift**, where impact ratios fall apart while accuracy
and every drift metric sit still."""),

code("""from monitor import run as monitor_run
history = monitor_run()"""),

md("""In month 2 the race impact ratio breaches at 0.779 while AUC is 0.867 and max PSI is 0.065,
which is *below even the warning band*. A dashboard that only tracks performance shows all green
during an active breach.

By month 6 it runs the other way: PSI is 0.683, well past the alert line, while AUC has not moved
at 0.868. Drift alarms fire with no loss of accuracy at all. Both directions are real and neither
one is visible from a single dashboard."""),

md("""## 7. Sustainability"""),

code("""from sustainability import run as sustain_run
sustain = sustain_run()"""),

md("""**Something I corrected along the way.** An earlier version of this counted only the model's
own arithmetic and produced a very impressive looking claim of about eight orders of magnitude
against an LLM. That was not honest accounting. Once you price the whole thing, including parsing
and serving, the advantage is roughly three orders of magnitude. Still decisive, and it has the
advantage of being true.

The more useful finding is that the model is **0.0004%** of the pipeline's energy. Optimizing it
further would be optimizing the wrong thing."""),

md("""## 8. Conclusion

| Question | Answer |
|---|---|
| Does dropping the biased features fix disparate impact? | Mostly. Worst ethnicity ratio goes 0.63 → 0.83 |
| Does it fix it completely? | **No.** Female / Hispanic is still at 0.652 |
| Does fairness cost accuracy? | Against the biased label yes, −0.046 AUC. Against merit it *gains* +0.046 |
| Is this safe to run fully automated? | **No**, and the design does not allow it |

The honest answer is that this system belongs in front of a recruiter, not in place of one. It
ranks and it routes, and a person still decides. A report that concluded anything else would be
recommending something the evidence here does not support."""),
]

path = os.path.join(BASE, "notebooks", "resume_screening_governance.ipynb")

# Carry over captured outputs so rebuilding the prose does not discard a run.
carried = 0
if os.path.exists(path):
    old = json.load(open(path, encoding="utf-8"))
    old_code = [c for c in old["cells"] if c["cell_type"] == "code"]
    new_code = [c for c in cells if c["cell_type"] == "code"]
    if len(old_code) == len(new_code):
        for o, n in zip(old_code, new_code):
            if o.get("outputs"):
                n["outputs"] = o["outputs"]
                n["execution_count"] = o.get("execution_count")
                carried += 1
    else:
        print(f"WARNING: code cell count changed ({len(old_code)} -> {len(new_code)}); "
              "outputs not carried over")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
      "language_info": {"name":"python","version":"3.11"}}, "nbformat":4, "nbformat_minor":5}
json.dump(nb, open(path,"w",encoding="utf-8"), indent=1)
print("wrote", path, "-", len(cells), "cells,", carried, "code outputs carried over")
