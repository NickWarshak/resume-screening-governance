# Responsible AI Resume Screening — Governance Project

**Live site → <https://nickwarshak.github.io/resume-screening-governance/>**
&nbsp;·&nbsp; [Report](https://nickwarshak.github.io/resume-screening-governance/report.html)
&nbsp;·&nbsp; [Notebook](https://nickwarshak.github.io/resume-screening-governance/notebook.html)
&nbsp;·&nbsp; [Model card](https://nickwarshak.github.io/resume-screening-governance/model-card.html)

A predictive classification system for ranking software-engineering applicants, built as a
demonstration of responsible AI development, deployment, and governance.

**The central finding:** the model that best predicts historical recruiter decisions is *not*
the model that best identifies qualified candidates. Optimizing the only metric available in
production (AUC 0.911) selects the discriminatory model; the governed model scores lower there
(0.865) and higher against true qualification (0.961 vs 0.915).

## Deliverables

| File | What it is |
|---|---|
| `reports/governance_report.pdf` | 7-page governance report (primary deliverable) |
| `notebooks/resume_screening_governance.ipynb` | Executed end-to-end notebook with outputs |
| `reports/MODEL_CARD.md` | Model card with intended use, limitations, known disparities |
| `src/` | All source modules |
| `figures/` | Generated figures (SHAP, fairness, monitoring, energy) |
| `reports/*.json`, `*.csv` | Raw computed metrics |
| `tests/verify_report.py` | Automated gate: 64 checks that report claims match pipeline output |

## Source modules

| Module | Purpose |
|---|---|
| `generate_data.py` | Synthetic applicant population with four injected bias channels |
| `train.py` | Trains naive vs governed models; dual evaluation |
| `fairness.py` | LL144-style disparate impact audit, equalized odds, calibration |
| `significance.py` | Bootstrap confidence intervals on impact ratios |
| `explain.py` | SHAP global and local explainability |
| `guardrails.py` | Six guardrail layers + red-team suite |
| `monitor.py` | PSI/KS drift, fairness SLO, incident routing, feedback controls |
| `sustainability.py` | Whole-pipeline energy and carbon accounting |
| `figures.py` | Report figures |

## Reproducing

```bash
pip install scikit-learn pandas numpy matplotlib shap scipy joblib
cd src
python3 generate_data.py && python3 train.py && python3 fairness.py \
  && python3 significance.py && python3 explain.py && python3 monitor.py \
  && python3 sustainability.py && python3 figures.py
cd ../tests && python3 verify_report.py     # 64/64 checks
```

Seeded at 42; verified to reproduce identical metrics across a clean rebuild.

## Key results

| Metric | Naive model | Governed model |
|---|---|---|
| AUC vs recruiter decisions | 0.911 | 0.865 |
| AUC vs true qualification | 0.915 | **0.961** |
| Worst impact ratio (sex) | 0.612 | **0.967** |
| Worst impact ratio (race) | 0.633 | **0.833** |
| Worst intersectional | 0.488 | 0.652 (still failing) |
| SHAP attribution via proxies | 37.7% | 0.0% |
| Red-team cases passed | — | 7 / 7 |
| Annual pipeline energy | — | 0.19 kWh (403–4,019× below LLM) |

## Honest limitations

1. The governed model **still fails** the four-fifths rule intersectionally for
   Female/Hispanic-Latino applicants (0.652, 87% probability of genuine breach). Disclosed,
   not resolved.
2. `years_experience` remains age-correlated (1.47 SD gap). Retained on business-necessity
   grounds; a live ADEA exposure, monitored quarterly.
3. Training data is synthetic. Metrics demonstrate the governance method; they do not transfer
   to a real applicant pool.
4. The system is suitable for **ranking and routing under human review**, not automated
   rejection. The architecture contains no `REJECT` code path.
