# Model Card — Resume Screening Shortlist Ranker

**Version:** 1.0 · **Date:** 2026-08-08 · **Owner:** Nick Warshak
**Status:** Reference implementation. **Not approved for production use.**

## Model details

| Field | Value |
|---|---|
| Model type | Logistic regression (L2), standardized inputs, isotonic calibration |
| Production candidate | `governed_logistic` |
| Input features | 12 merit-derived resume attributes |
| Excluded features | `referral`, `top_school`, `employment_gap_months`, `distance_from_office_km` |
| Output | Calibrated probability + routing action (never a rejection) |
| Model size | 2.1 KB |
| Training time | 0.02 s CPU |
| Inference | ≈0.7 ms per 1,000 applicants (mean of 20 runs) |

## Intended use

**In scope.** Producing a *ranked* shortlist of software engineering applicants for human
recruiter review; routing uncertain applicants to unaided human screening.

**Out of scope, explicitly.** Automated rejection of any applicant. Ranking for job families
other than software engineering. Any use where the model output is the final decision.
Use as evidence of an applicant's ability.

## Performance

| Metric | Value |
|---|---|
| AUC vs historical recruiter decisions | 0.865 |
| AUC vs true latent qualification | 0.961 |
| Average precision | 0.680 |
| Brier score | 0.122 |
| Precision @ 25% shortlist | 0.556 |
| Recall @ 25% shortlist | 0.742 |

The naive comparison model scores **higher** on the first metric (0.911) and **lower** on the
second (0.915). Selection on production-observable metrics alone favors the worse model.

## Fairness

Worst-case impact ratios (four-fifths threshold = 0.80):

| System | Sex | Race | Intersectional |
|---|---|---|---|
| Human recruiters (status quo) | 0.607 | 0.583 | — |
| Naive model | 0.612 | 0.633 | 0.488 |
| **Governed model** | **0.967** | **0.833** | **0.652** |

**Known unresolved disparity.** Female / Hispanic-Latino applicants show an impact ratio of
0.652 (95% CI 0.370–0.949; 87% bootstrap probability of genuine breach). The model passes
marginal audits while failing this intersection. This is disclosed, not resolved.

## Limitations

1. **The label is a proxy.** Trained on recruiter decisions, not job performance. The model
   can only be as fair as what it imitates, minus what governance removes.
2. **Age-correlated features retained.** `years_experience` shows a 1.47 SD gap across age
   groups. Retained on business-necessity grounds; monitored quarterly. This is a live ADEA
   exposure, not a solved problem.
3. **Synthetic training data.** Distributions are constructed, not observed. Metrics
   demonstrate the governance method; they do not transfer to a real applicant pool.
4. **Intersectional audit underpowered.** Several cells fall below n=100. Wide intervals
   indicate the audit cannot see clearly, not that the system is safe.

## Ethical considerations

The most likely harm is not a wrong score. It is a wrong score acted on automatically, at
scale, with no human in the loop and no applicant recourse. Every architectural choice —
abstention band, no-reject constraint, batch circuit breaker, randomized human holdout — is
aimed at that failure mode rather than at maximizing accuracy.

## Regulatory posture

Subject to NYC Local Law 144 (annual independent bias audit, published results, 10-business-day
candidate notice), EEOC Uniform Guidelines (29 CFR 1607), Title VII, ADEA, ADA, Illinois AIVIA,
Colorado SB 24-205, and EU AI Act Annex III §4 obligations where applicable.
