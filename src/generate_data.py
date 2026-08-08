"""
Synthetic resume-screening dataset generator.

WHY SYNTHETIC: Real resume corpora contain personally identifiable information
(names, employers, contact details) and cannot be lawfully repurposed for model
training without consent. This generator produces a structurally realistic
substitute in which the *causal* relationship between qualification, protected
class, and historical hiring decisions is known by construction. That known
ground truth is what makes the fairness audit in this project verifiable:
we can measure not only whether the model is biased, but whether it is biased
*relative to true qualification* -- something impossible with real-world data
where true qualification is never observed.

DESIGN OF THE BIAS:
The label is NOT "was this person good at the job." It is "did a human recruiter
advance this person to an onsite interview." This is a PROXY LABEL, and the gap
between the proxy and the construct of interest is the single most important
governance flaw in commercial resume screeners. We inject four bias channels
that mirror documented failure modes in real hiring:

  1. REFERRAL HOMOPHILY  - referrals flow through existing employee networks,
     which mirror the demographics of the existing workforce (Rubineau &
     Fernandez, 2013). Recruiters weight referrals heavily.
  2. EMPLOYMENT-GAP PENALTY - career gaps are penalized; gaps are unequally
     distributed because caregiving labor is unequally distributed.
  3. PRESTIGE PROXY - "top school" attendance correlates with parental income
     and thus with race in the US; recruiters over-weight it.
  4. RESIDUAL DIRECT BIAS - a small unexplained penalty applied at screening,
     representing name-based discrimination documented in audit studies
     (Bertrand & Mullainathan, 2004: identical resumes with White-sounding
     names received 50% more callbacks).

Author: Nick Warshak
"""

import numpy as np
import pandas as pd

RNG_SEED = 42

# --- Population composition (approximates US CS-degree pipeline, not general pop) ---
SEX_LEVELS = ["Male", "Female"]
SEX_PROBS = [0.72, 0.28]

RACE_LEVELS = ["White", "Asian", "Hispanic/Latino", "Black", "Two or more/Other"]
RACE_PROBS = [0.47, 0.31, 0.10, 0.08, 0.04]

AGE_LEVELS = ["Under 40", "40 and over"]
AGE_PROBS = [0.78, 0.22]


def generate(n=12000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. PROTECTED ATTRIBUTES
    #    Collected for audit purposes only. NEVER passed to the model.
    #    In production these come from voluntary EEO self-identification,
    #    stored in a separate system from the screening features.
    # ------------------------------------------------------------------
    sex = rng.choice(SEX_LEVELS, size=n, p=SEX_PROBS)
    race = rng.choice(RACE_LEVELS, size=n, p=RACE_PROBS)
    age_group = rng.choice(AGE_LEVELS, size=n, p=AGE_PROBS)
    disability = rng.choice([0, 1], size=n, p=[0.93, 0.07])

    is_female = (sex == "Female").astype(int)
    # Groups underrepresented in the referring workforce (drives homophily channel)
    underrep = np.isin(race, ["Black", "Hispanic/Latino", "Two or more/Other"]).astype(int)
    is_older = (age_group == "40 and over").astype(int)

    # ------------------------------------------------------------------
    # 2. LATENT TRUE QUALIFICATION  (Q)
    #    The construct we actually care about: would this person succeed in
    #    the role. Generated INDEPENDENT of protected class -- by construction
    #    there is no real ability difference between groups in this world.
    #    Any disparity the audit finds is therefore pure measurement bias.
    # ------------------------------------------------------------------
    latent_q = rng.normal(0, 1, size=n)

    # ------------------------------------------------------------------
    # 3. OBSERVED RESUME FEATURES
    #    Signal features load on latent_q. Some also pick up demographic
    #    structure through real-world social mechanisms.
    # ------------------------------------------------------------------

    # Experience: older applicants have more of it (mechanical, not bias)
    years_experience = np.clip(
        2.0 + 1.6 * latent_q + 7.5 * is_older + rng.gamma(2.0, 1.3, n), 0, 35
    ).round(1)

    # Education
    edu_raw = 1.4 + 0.85 * latent_q + rng.normal(0, 0.9, n)
    education_level = np.clip(np.round(edu_raw), 0, 4).astype(int)  # 0=HS .. 4=PhD

    # Skills matched against the job requisition
    num_relevant_skills = np.clip(
        np.round(6 + 3.1 * latent_q + rng.normal(0, 1.9, n)), 0, 20
    ).astype(int)

    # Keyword match score vs job description (TF-IDF cosine, 0-1)
    keyword_match_score = np.clip(
        0.42 + 0.17 * latent_q + rng.normal(0, 0.10, n), 0.0, 1.0
    ).round(3)

    # PROXY CHANNEL 3: prestige. Access to selective universities tracks
    # parental income; US wealth distribution is racialized.
    top_school_logit = -1.7 + 0.75 * latent_q - 0.62 * underrep + rng.normal(0, 0.8, n)
    top_school = (1 / (1 + np.exp(-top_school_logit)) > rng.uniform(size=n)).astype(int)

    gpa = np.clip(3.05 + 0.30 * latent_q + 0.10 * top_school + rng.normal(0, 0.32, n), 1.8, 4.0).round(2)

    num_certifications = rng.poisson(np.clip(1.1 + 0.65 * latent_q, 0.1, None)).astype(int)
    portfolio_projects = rng.poisson(np.clip(2.0 + 1.25 * latent_q, 0.1, None)).astype(int)

    num_prior_roles = np.clip(
        np.round(1 + 0.28 * years_experience + rng.normal(0, 1.1, n)), 0, 15
    ).astype(int)

    avg_tenure_months = np.clip(
        np.where(num_prior_roles > 0, (years_experience * 12) / np.maximum(num_prior_roles, 1), 0)
        + rng.normal(0, 5, n), 0, 200
    ).round(1)

    # PROXY CHANNEL 2: employment gaps. Caregiving burden is unequally
    # distributed by sex; this is a documented driver of resume-gap disparity.
    gap_base = rng.gamma(1.1, 2.6, n)
    employment_gap_months = np.clip(
        gap_base + is_female * rng.gamma(2.3, 3.4, n) - 0.5 * latent_q, 0, 72
    ).round(1)

    # PROXY CHANNEL 1: referral homophily. Referrals come from the existing
    # workforce, so underrepresented applicants have less access to them.
    ref_logit = -1.35 + 0.42 * latent_q - 0.88 * underrep - 0.30 * is_female + rng.normal(0, 0.7, n)
    referral = (1 / (1 + np.exp(-ref_logit)) > rng.uniform(size=n)).astype(int)

    leadership_indicators = np.clip(
        np.round(0.42 * np.sqrt(np.maximum(years_experience, 0)) + 0.55 * latent_q + rng.normal(0, 0.7, n)),
        0, 8
    ).astype(int)

    resume_length_words = np.clip(
        np.round(420 + 22 * years_experience + 55 * latent_q + rng.normal(0, 110, n)), 120, 2200
    ).astype(int)

    num_publications = rng.poisson(
        np.clip(0.08 + 0.55 * (education_level >= 3) * (1 + latent_q), 0.01, None)
    ).astype(int)

    # Geographic distance -- historically redlined areas sit further from
    # tech-corridor offices. Included deliberately as a trap feature.
    distance_from_office_km = np.clip(
        rng.gamma(2.0, 8.0, n) + underrep * rng.gamma(1.6, 7.0, n), 0.5, 160
    ).round(1)

    # ------------------------------------------------------------------
    # 4. HISTORICAL RECRUITER DECISION  (the training label)
    #    A biased human process. This is what the model learns to imitate.
    # ------------------------------------------------------------------
    merit_component = (
        0.62 * latent_q
        + 0.055 * num_relevant_skills
        + 0.150 * education_level
        + 1.05 * keyword_match_score
        + 0.045 * years_experience
        + 0.085 * portfolio_projects
        + 0.070 * leadership_indicators
    )

    bias_component = (
        1.15 * referral                       # channel 1: network access
        - 0.052 * employment_gap_months       # channel 2: gap penalty
        + 0.68 * top_school                   # channel 3: prestige
        - 0.34 * underrep                     # channel 4: residual direct bias
        - 0.22 * is_female
        - 0.30 * is_older                     # age screening
        - 0.0055 * distance_from_office_km    # geographic screening
    )

    decision_logit = -2.55 + merit_component + bias_component + rng.logistic(0, 0.55, n)
    advanced = (decision_logit > 0).astype(int)

    # Counterfactual: what an unbiased recruiter would have decided.
    # Used ONLY for evaluation -- never available in real deployment.
    fair_logit = -2.55 + merit_component + 0.55 + rng.logistic(0, 0.55, n)
    advanced_fair = (fair_logit > 0).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"A{100000+i}" for i in range(n)],
        # --- model features ---
        "years_experience": years_experience,
        "education_level": education_level,
        "num_relevant_skills": num_relevant_skills,
        "keyword_match_score": keyword_match_score,
        "gpa": gpa,
        "num_certifications": num_certifications,
        "portfolio_projects": portfolio_projects,
        "num_prior_roles": num_prior_roles,
        "avg_tenure_months": avg_tenure_months,
        "leadership_indicators": leadership_indicators,
        "num_publications": num_publications,
        "resume_length_words": resume_length_words,
        # --- contested features (proxies; excluded in the governed model) ---
        "employment_gap_months": employment_gap_months,
        "top_school": top_school,
        "referral": referral,
        "distance_from_office_km": distance_from_office_km,
        # --- protected attributes: AUDIT ONLY, never features ---
        "sex": sex,
        "race_ethnicity": race,
        "age_group": age_group,
        "disability_disclosed": disability,
        # --- labels ---
        "advanced_to_onsite": advanced,          # observed, biased label
        "advanced_counterfactual": advanced_fair, # unobservable fair label
        "latent_qualification": latent_q.round(4),
    })
    return df


# Feature sets -------------------------------------------------------------
MERIT_FEATURES = [
    "years_experience", "education_level", "num_relevant_skills",
    "keyword_match_score", "gpa", "num_certifications", "portfolio_projects",
    "num_prior_roles", "avg_tenure_months", "leadership_indicators",
    "num_publications", "resume_length_words",
]

PROXY_FEATURES = [
    "employment_gap_months", "top_school", "referral", "distance_from_office_km",
]

NAIVE_FEATURES = MERIT_FEATURES + PROXY_FEATURES   # "ship it" model
GOVERNED_FEATURES = MERIT_FEATURES                 # proxy-restricted model

PROTECTED = ["sex", "race_ethnicity", "age_group", "disability_disclosed"]


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    df = generate()
    path = os.path.join(out_dir, "applicants.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows -> {os.path.abspath(path)}")
    print(f"\nOverall advance rate (observed/biased): {df.advanced_to_onsite.mean():.3f}")
    print(f"Overall advance rate (counterfactual):  {df.advanced_counterfactual.mean():.3f}")
    print("\nAdvance rate by sex:")
    print(df.groupby("sex", observed=True).advanced_to_onsite.mean().round(4).to_string())
    print("\nAdvance rate by race/ethnicity:")
    print(df.groupby("race_ethnicity", observed=True).advanced_to_onsite.mean().round(4).to_string())
    print("\nAdvance rate by age group:")
    print(df.groupby("age_group", observed=True).advanced_to_onsite.mean().round(4).to_string())
    print("\nMean latent qualification by race (should be ~0 for all -- no real ability gap):")
    print(df.groupby("race_ethnicity", observed=True).latent_qualification.mean().round(4).to_string())
