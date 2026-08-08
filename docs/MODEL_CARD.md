# Model Card — Resume Screening Shortlist Ranker

**Version:** 1.0 · **Date:** August 2026 · **Owner:** Nick Warshak
**Status:** Reference implementation. **Not approved for production use.**

## Model details

| Field | Value |
|---|---|
| Model type | Logistic regression, standardized inputs, isotonic calibration |
| Production candidate | `governed_logistic` |
| Input features | 12 merit-based resume attributes |
| Excluded features | `referral`, `top_school`, `employment_gap_months`, `distance_from_office_km` |
| Output | Calibrated score plus a routing action, never a rejection |
| Model size | 2.1 KB |
| Training time | 0.02 s CPU |
| Inference | about 0.6 ms per 1,000 applicants |

## Intended use

**What it is for.** Ranking software engineering applicants so a recruiter sees the best fits
first, and routing the ones the model is unsure about to a person who reads them without a score.

**What it is not for.** Rejecting anybody. Ranking for jobs other than software engineering. Any
situation where the model output is the final decision. Being treated as evidence of how good an
applicant actually is.

## Performance

| Metric | Value |
|---|---|
| AUC vs recruiter decisions | 0.865 |
| AUC vs merit | 0.961 |
| Precision | 0.680 |
| Brier score | 0.122 |
| Precision @ 25% shortlist | 0.556 |
| Recall @ 25% shortlist | 0.742 |

The naïve model scores higher on the first line (0.911) and lower on the second (0.915). Only
the first line is observable in production, so picking the model with the best visible number
gets you the worse one.

## Fairness

Worst impact ratios, where 0.80 is the threshold:

| System | Sex | Ethnicity | Intersection |
|---|---|---|---|
| Human recruiter | 0.607 | 0.583 | — |
| Naïve model | 0.612 | 0.633 | 0.488 |
| **Governed model** | **0.967** | **0.833** | **0.652** |

**What is still broken.** Female / Hispanic applicants come out at 0.652. The model passes on sex
and on ethnicity when you look at them one at a time, then fails when you look at both together.
I tried a few things to fix it and none of them were honest fixes, so I am reporting it rather
than hiding it.

## Limitations

1. **The label is the wrong thing.** The model is trained on whether a recruiter advanced
   someone, not on whether they were good at the job. It can only be as fair as the thing it is
   copying, minus whatever governance takes back out.
2. **Experience still tracks age.** `years_experience` is kept because the job genuinely needs
   it, but it moves with age group. That is a real exposure, not a solved problem.
3. **The data is synthetic.** The numbers show that the method works. They do not tell you what
   would happen on a real applicant pool.
4. **The intersection audit is small.** Several of those groups are under 100 people. Wide
   intervals mean the audit cannot see clearly, not that the system is fine.

## Ethical considerations

The thing most likely to go wrong here is not a bad score. It is a bad score that gets acted on
automatically, at volume, with nobody reading it and no way for the applicant to respond. The
abstention band, the no-reject rule, the batch circuit breaker and the human holdout are all
aimed at that, rather than at squeezing out more accuracy.

## Regulatory posture

Built against NYC Local Law 144, the EEOC Uniform Guidelines, Title VII, Illinois AIVIA, and
Colorado SB 24-205.
