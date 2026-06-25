"""
Statistical Engine
==================
Implements Z-test, t-test, and confidence interval computations
for blood parameter hypothesis testing.

Curriculum Connection:
- Z-test for known population σ
- T-test for sample data with unknown σ
- Confidence intervals at 95% and 99% levels
- P-value interpretation and hypothesis decision
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats

from blood_parameters import BloodParameter


class Status(Enum):
    """Result classification based on p-value thresholds."""
    NORMAL = "Normal"
    BORDERLINE = "Borderline"
    FLAGGED = "Flagged"


@dataclass
class TestResult:
    """Complete result from a single hypothesis test."""
    parameter_name: str
    unit: str
    patient_value: float
    population_mean: float
    population_std: float
    normal_range_low: float
    normal_range_high: float
    test_type: str            # "Z-test" or "t-test"
    test_statistic: float     # z-score or t-statistic
    p_value: float
    status: Status
    ci_95: tuple[float, float]  # 95% confidence interval
    ci_99: tuple[float, float]  # 99% confidence interval
    degrees_of_freedom: int | None = None  # For t-test only
    sample_size: int | None = None         # For t-test only
    description: str = ""


def compute_z_test(value: float, param: BloodParameter) -> TestResult:
    """
    Perform a two-tailed Z-test for a single observation against
    the known population distribution.

    Hypotheses:
        H₀: x comes from N(μ, σ²)  — the value is normal
        H₁: x does NOT come from N(μ, σ²) — the value is abnormal

    Z = (x - μ) / σ
    p-value = 2 × P(Z > |z|)  [two-tailed]
    """
    mu = param.population_mean
    sigma = param.population_std

    # ── Z-score ───────────────────────────────────────────────────────────
    z_score = (value - mu) / sigma

    # ── P-value (two-tailed) ──────────────────────────────────────────────
    p_value = 2 * stats.norm.sf(abs(z_score))

    # ── Confidence Intervals ──────────────────────────────────────────────
    # CI for the population mean based on the observation
    # For a single observation, we use σ directly
    ci_95 = (
        value - 1.96 * sigma,
        value + 1.96 * sigma,
    )
    ci_99 = (
        value - 2.576 * sigma,
        value + 2.576 * sigma,
    )

    # ── Status classification ─────────────────────────────────────────────
    status = _classify_p_value(p_value)

    return TestResult(
        parameter_name=param.name,
        unit=param.unit,
        patient_value=value,
        population_mean=mu,
        population_std=sigma,
        normal_range_low=param.normal_range_low,
        normal_range_high=param.normal_range_high,
        test_type="Z-test",
        test_statistic=z_score,
        p_value=p_value,
        status=status,
        ci_95=ci_95,
        ci_99=ci_99,
        description=param.description,
    )


def compute_t_test(values: list[float], param: BloodParameter) -> TestResult:
    """
    Perform a two-tailed one-sample t-test when the user provides
    multiple readings for the same parameter.

    Hypotheses:
        H₀: x̄ = μ  — the sample mean equals the population mean
        H₁: x̄ ≠ μ  — the sample mean differs from the population mean

    t = (x̄ - μ) / (s / √n)
    p-value from t-distribution with df = n - 1
    """
    n = len(values)
    sample_mean = np.mean(values)
    sample_std = np.std(values, ddof=1)  # Bessel's correction
    mu = param.population_mean

    # ── T-statistic ───────────────────────────────────────────────────────
    se = sample_std / np.sqrt(n)  # Standard error
    if se == 0:
        # All values identical — treat as z-test with population σ
        return compute_z_test(sample_mean, param)

    t_stat = (sample_mean - mu) / se
    df = n - 1

    # ── P-value (two-tailed) ──────────────────────────────────────────────
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    # ── Confidence Intervals ──────────────────────────────────────────────
    t_crit_95 = stats.t.ppf(0.975, df)
    t_crit_99 = stats.t.ppf(0.995, df)

    ci_95 = (
        sample_mean - t_crit_95 * se,
        sample_mean + t_crit_95 * se,
    )
    ci_99 = (
        sample_mean - t_crit_99 * se,
        sample_mean + t_crit_99 * se,
    )

    # ── Status classification ─────────────────────────────────────────────
    status = _classify_p_value(p_value)

    return TestResult(
        parameter_name=param.name,
        unit=param.unit,
        patient_value=sample_mean,
        population_mean=mu,
        population_std=param.population_std,
        normal_range_low=param.normal_range_low,
        normal_range_high=param.normal_range_high,
        test_type="t-test",
        test_statistic=t_stat,
        p_value=p_value,
        status=status,
        ci_95=ci_95,
        ci_99=ci_99,
        degrees_of_freedom=df,
        sample_size=n,
        description=param.description,
    )


def _classify_p_value(p_value: float) -> Status:
    """
    Classify a result based on its p-value:
        p > 0.05        → Normal   (fail to reject H₀ at 95%)
        0.01 < p ≤ 0.05 → Borderline (reject at 95%, not at 99%)
        p ≤ 0.01        → Flagged  (reject H₀ at 99%)
    """
    if p_value > 0.05:
        return Status.NORMAL
    elif p_value > 0.01:
        return Status.BORDERLINE
    else:
        return Status.FLAGGED


def run_analysis(
    values: dict[str, float | list[float]],
    parameters: dict[str, "BloodParameter"],
) -> list[TestResult]:
    """
    Run the full statistical analysis on all provided blood values.

    Args:
        values: Dict mapping parameter key → single value (float) or
                list of readings (list[float]) for t-test.
        parameters: Dict of BloodParameter objects (filtered by gender).

    Returns:
        List of TestResult objects, one per parameter tested.
    """
    results: list[TestResult] = []

    for key, value in values.items():
        if key not in parameters:
            continue

        param = parameters[key]

        if isinstance(value, list):
            if len(value) >= 2:
                result = compute_t_test(value, param)
            elif len(value) == 1:
                result = compute_z_test(value[0], param)
            else:
                continue
        else:
            result = compute_z_test(float(value), param)

        results.append(result)

    return results
