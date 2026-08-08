# Responsible AI Resume Screening — Governance Project

**Live site → <https://nickwarshak.github.io/resume-screening-governance/>**
&nbsp;·&nbsp; **[Interactive demo](https://nickwarshak.github.io/resume-screening-governance/demo.html)**
&nbsp;·&nbsp; [Report](https://nickwarshak.github.io/resume-screening-governance/report.html)
&nbsp;·&nbsp; [Notebook](https://nickwarshak.github.io/resume-screening-governance/notebook.html)
&nbsp;·&nbsp; [Model card](https://nickwarshak.github.io/resume-screening-governance/model-card.html)

A ranking system for software engineering applicants, built to show how you would actually
develop, evaluate, and govern an AI tool in hiring.

**What I found:** the model that best predicts what recruiters did is not the model that best
finds qualified people. The naïve model wins on the only number you can see in production (AUC
0.911 against recruiter decisions). The governed model loses there (0.865) and wins on the thing
that matters (0.961 vs 0.915 against merit). Picking the model with the best visible number gets
you the discriminatory one, while doing everything else right.

## What is in here

| File | What it is |
|---|---|
| `reports/IS7085_Governance_Report.pdf` | The governance report (main deliverable) |
| `reports/report.html` | Same report, built for reading in a browser |
| `docs/demo.html` | Interactive scorer — both models, live, runs in your browser |
| `notebooks/resume_screening_governance.ipynb` | The whole pipeline end to end |
| `reports/MODEL_CARD.md` | Model card: intended use, limits, what is still broken |
| `src/` | All the source modules |
| `figures/` | Generated charts |
| `reports/*.json`, `*.csv` | Raw computed metrics |
| `tests/verify_report.py` | Checks that the report's numbers match what the code computed |
| `tests/verify_web_demo.py` | Checks the demo's browser math against scikit-learn |
| `tests/verify_demo_render.py` | Checks the demo page actually renders |

## Source modules

| Module | What it does |
|---|---|
| `generate_data.py` | Builds the synthetic applicant pool with four kinds of bias put in on purpose |
| `train.py` | Trains the naïve and governed models, scores both against recruiter decisions and against merit |
| `fairness.py` | Impact ratios by sex, ethnicity, and their intersection |
| `significance.py` | Bootstrap confidence intervals, so a small group is not read as a certain finding |
| `explain.py` | SHAP, global and local |
| `guardrails.py` | The six guardrail layers and the red-team suite |
| `monitor.py` | Drift detection and the fairness SLO |
| `sustainability.py` | Energy and carbon accounting for the pipeline |
| `figures.py` | The report charts |
| `export_web_model.py` | Exports the model to `docs/model.json` for the browser demo |

## Running it

```bash
pip install scikit-learn pandas numpy matplotlib shap scipy joblib
cd src
python3 generate_data.py && python3 train.py && python3 fairness.py \
  && python3 significance.py && python3 explain.py && python3 monitor.py \
  && python3 sustainability.py && python3 figures.py
python3 export_web_model.py                 # refresh the demo's parameters
cd ../tests && python3 verify_report.py     # report claims vs computed output
python3 verify_web_demo.py                  # browser JS vs scikit-learn
python3 verify_demo_render.py               # demo page renders correctly
```

Seeded at 42. A clean rebuild gives the same numbers.

## Results

| Metric | Naïve model | Governed model |
|---|---|---|
| AUC vs recruiter decisions | 0.911 | 0.865 |
| AUC vs merit | 0.915 | **0.961** |
| Worst impact ratio (sex) | 0.612 | **0.967** |
| Worst impact ratio (ethnicity) | 0.633 | **0.833** |
| Worst intersection | 0.488 | 0.652 (still failing) |
| Attribution through biased features | 37.7% | 0.0% |
| Red-team cases passed | — | 7 of 7 |
| Annual pipeline energy | — | 0.19 kWh |

## The interactive demo

[`docs/demo.html`](docs/demo.html) scores applicants against both models right in the browser,
with no server behind it. The governed model is a logistic regression, which is a standardizer, a
list of weights, and a calibration curve. That is arithmetic a browser can do, so this is the
real model and not a re-creation of it.

`export_web_model.py` writes those numbers to `docs/model.json`, and it refuses to write the file
at all unless plain arithmetic reproduces scikit-learn's output on all 2,400 test applicants. It
currently matches exactly. `tests/verify_web_demo.py` checks it from the other end: it pulls the
scoring functions straight out of the shipped `demo.html` and runs those exact characters under
Node against scikit-learn. They agree to 1e-13, which is just floating point rounding.

The part worth playing with is the proxy panel. Take the borderline applicant the demo opens on
and turn the referral on. The naïve model jumps 44.8 points. The governed model does not move at
all, because that feature is not in it.

## What I did not fix

1. The governed model **still fails** at the intersection. Female / Hispanic applicants come out
   at 0.652. I tried a few things and none of them were honest fixes, so it is reported rather
   than hidden.
2. `years_experience` still moves with age. I kept it because the job needs it, but that is a
   real exposure and not a solved problem.
3. The data is synthetic. The numbers show the method works. They do not tell you what would
   happen on a real applicant pool.
4. This belongs in front of a recruiter, not in place of one. It ranks and it routes. There is no
   code path in it that rejects anybody.
