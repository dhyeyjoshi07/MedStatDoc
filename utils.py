"""
Utility Functions
=================
Plain-English explanation generator, p-value formatting, and
status display helpers for MedStatDoc.
"""

from statistical_engine import TestResult, Status


def generate_explanation(result: TestResult) -> str:
    """
    Generate a plain-English explanation of a hypothesis test result.

    Example output:
        "Your Hemoglobin (18.2 g/dL) is 2.7 standard deviations above
        the population mean of 15.5 g/dL. The p-value of 0.007 means
        there's only a 0.7% chance this deviation occurred by random
        chance. This result is statistically significant at the 99%
        confidence level, suggesting your value is genuinely abnormal."
    """
    value = result.patient_value
    name = result.parameter_name
    unit = result.unit
    z = result.test_statistic
    p = result.p_value
    mu = result.population_mean
    test_type = result.test_type

    # ── Direction ─────────────────────────────────────────────────────────
    if z > 0:
        direction = "above"
    elif z < 0:
        direction = "below"
    else:
        direction = "exactly at"

    # ── Opening sentence ──────────────────────────────────────────────────
    explanation = (
        f"Your **{name}** ({value:.2f} {unit}) is "
        f"**{abs(z):.2f} standard deviations {direction}** the "
        f"population mean of {mu:.2f} {unit}."
    )

    # ── Test type detail ──────────────────────────────────────────────────
    if test_type == "t-test" and result.sample_size:
        explanation += (
            f" This was evaluated using a **{test_type}** with "
            f"{result.sample_size} readings (df = {result.degrees_of_freedom})."
        )
    else:
        explanation += f" This was evaluated using a **{test_type}**."

    # ── P-value interpretation ────────────────────────────────────────────
    p_pct = p * 100
    if p < 0.001:
        p_str = f"< 0.1%"
    else:
        p_str = f"{p_pct:.1f}%"

    explanation += (
        f" The p-value of **{format_p_value(p)}** means there is a "
        f"**{p_str} probability** that this deviation occurred purely "
        f"by random chance."
    )

    # ── Conclusion ────────────────────────────────────────────────────────
    if result.status == Status.FLAGGED:
        explanation += (
            " 🔴 This result is **statistically significant at the 99% "
            "confidence level**, strongly suggesting your value deviates "
            "from the healthy population norm."
        )
    elif result.status == Status.BORDERLINE:
        explanation += (
            " 🟡 This result is **statistically significant at the 95% "
            "confidence level** but not at 99%. It may warrant monitoring "
            "or a follow-up test."
        )
    else:
        explanation += (
            " 🟢 This result is **not statistically significant** — your "
            "value falls within normal population variation."
        )

    # ── Confidence interval note ──────────────────────────────────────────
    ci_low, ci_high = result.ci_95
    explanation += (
        f"\n\n**95% Confidence Interval:** [{ci_low:.2f}, {ci_high:.2f}] {unit}"
    )

    return explanation


def format_p_value(p: float) -> str:
    """
    Format a p-value for display, using scientific notation for
    very small values.
    """
    if p < 0.0001:
        return f"{p:.2e}"
    elif p < 0.001:
        return f"{p:.4f}"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.4f}"


def get_status_emoji(status: Status) -> str:
    """Map status to color emoji for display."""
    mapping = {
        Status.NORMAL: "🟢",
        Status.BORDERLINE: "🟡",
        Status.FLAGGED: "🔴",
    }
    return mapping.get(status, "⚪")


def get_status_color(status: Status) -> str:
    """Get hex color for a status level."""
    mapping = {
        Status.NORMAL: "#10b981",      # Emerald green
        Status.BORDERLINE: "#f59e0b",  # Amber
        Status.FLAGGED: "#ef4444",     # Red
    }
    return mapping.get(status, "#6b7280")


def get_risk_color(risk_category: str) -> str:
    """Get hex color for a risk category."""
    mapping = {
        "Low Risk": "#10b981",
        "Moderate Risk": "#f59e0b",
        "High Risk": "#ef4444",
        "Insufficient Data": "#6b7280",
    }
    return mapping.get(risk_category, "#6b7280")


def get_risk_emoji(risk_category: str) -> str:
    """Get emoji for a risk category."""
    mapping = {
        "Low Risk": "✅",
        "Moderate Risk": "⚠️",
        "High Risk": "🚨",
        "Insufficient Data": "❓",
    }
    return mapping.get(risk_category, "❓")
