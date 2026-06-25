"""
Blood Parameters Database
=========================
Defines all supported blood test parameters with population statistics,
normal ranges, and gender-specific variants for hypothesis testing.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BloodParameter:
    """Represents a single blood test parameter with its statistical properties."""
    name: str
    unit: str
    population_mean: float       # μ — population mean
    population_std: float        # σ — population standard deviation
    normal_range_low: float      # Lower bound of clinical normal range
    normal_range_high: float     # Upper bound of clinical normal range
    description: str             # Brief clinical description
    gender: Optional[str] = None # None = applies to all, "male" or "female"
    category: str = "General"    # Grouping category for UI


# ─── Complete Parameter Registry ──────────────────────────────────────────────

BLOOD_PARAMETERS: dict[str, BloodParameter] = {
    # ── Hematology ────────────────────────────────────────────────────────────
    "hemoglobin_male": BloodParameter(
        name="Hemoglobin",
        unit="g/dL",
        population_mean=15.5,
        population_std=1.0,
        normal_range_low=13.5,
        normal_range_high=17.5,
        description="Oxygen-carrying protein in red blood cells",
        gender="male",
        category="Hematology",
    ),
    "hemoglobin_female": BloodParameter(
        name="Hemoglobin",
        unit="g/dL",
        population_mean=14.0,
        population_std=1.0,
        normal_range_low=12.0,
        normal_range_high=16.0,
        description="Oxygen-carrying protein in red blood cells",
        gender="female",
        category="Hematology",
    ),
    "rbc_male": BloodParameter(
        name="RBC Count",
        unit="million/μL",
        population_mean=5.4,
        population_std=0.35,
        normal_range_low=4.7,
        normal_range_high=6.1,
        description="Red blood cell count — carries oxygen throughout the body",
        gender="male",
        category="Hematology",
    ),
    "rbc_female": BloodParameter(
        name="RBC Count",
        unit="million/μL",
        population_mean=4.8,
        population_std=0.30,
        normal_range_low=4.2,
        normal_range_high=5.4,
        description="Red blood cell count — carries oxygen throughout the body",
        gender="female",
        category="Hematology",
    ),
    "wbc": BloodParameter(
        name="WBC Count",
        unit="cells/μL",
        population_mean=7500,
        population_std=1750,
        normal_range_low=4500,
        normal_range_high=11000,
        description="White blood cell count — key indicator of immune function",
        gender=None,
        category="Hematology",
    ),
    "platelet": BloodParameter(
        name="Platelet Count",
        unit="thousand/μL",
        population_mean=275,
        population_std=60,
        normal_range_low=150,
        normal_range_high=400,
        description="Platelets help blood clot and prevent bleeding",
        gender=None,
        category="Hematology",
    ),

    # ── Metabolic Panel ───────────────────────────────────────────────────────
    "fasting_blood_sugar": BloodParameter(
        name="Fasting Blood Sugar",
        unit="mg/dL",
        population_mean=85,
        population_std=10,
        normal_range_low=70,
        normal_range_high=100,
        description="Blood glucose level after fasting — screens for diabetes",
        gender=None,
        category="Metabolic",
    ),
    "creatinine": BloodParameter(
        name="Creatinine",
        unit="mg/dL",
        population_mean=1.0,
        population_std=0.15,
        normal_range_low=0.7,
        normal_range_high=1.3,
        description="Waste product filtered by kidneys — indicates kidney function",
        gender=None,
        category="Metabolic",
    ),

    # ── Lipid Panel ───────────────────────────────────────────────────────────
    "total_cholesterol": BloodParameter(
        name="Total Cholesterol",
        unit="mg/dL",
        population_mean=170,
        population_std=30,
        normal_range_low=0,
        normal_range_high=200,
        description="Total blood cholesterol level — cardiovascular risk indicator",
        gender=None,
        category="Lipid Panel",
    ),
    "ldl": BloodParameter(
        name="LDL Cholesterol",
        unit="mg/dL",
        population_mean=100,
        population_std=25,
        normal_range_low=0,
        normal_range_high=100,
        description="'Bad' cholesterol — high levels increase heart disease risk",
        gender=None,
        category="Lipid Panel",
    ),
    "hdl": BloodParameter(
        name="HDL Cholesterol",
        unit="mg/dL",
        population_mean=55,
        population_std=12,
        normal_range_low=40,
        normal_range_high=200,
        description="'Good' cholesterol — higher levels are protective",
        gender=None,
        category="Lipid Panel",
    ),
    "triglycerides": BloodParameter(
        name="Triglycerides",
        unit="mg/dL",
        population_mean=120,
        population_std=40,
        normal_range_low=0,
        normal_range_high=150,
        description="Fat in blood — elevated levels increase cardiovascular risk",
        gender=None,
        category="Lipid Panel",
    ),

    # ── Endocrine ─────────────────────────────────────────────────────────────
    "tsh": BloodParameter(
        name="TSH",
        unit="mIU/L",
        population_mean=2.0,
        population_std=0.8,
        normal_range_low=0.4,
        normal_range_high=4.0,
        description="Thyroid-stimulating hormone — regulates metabolism",
        gender=None,
        category="Endocrine",
    ),

    # ── Vitamins ──────────────────────────────────────────────────────────────
    "vitamin_d": BloodParameter(
        name="Vitamin D",
        unit="ng/mL",
        population_mean=40,
        population_std=12,
        normal_range_low=30,
        normal_range_high=100,
        description="Essential for bone health and immune function",
        gender=None,
        category="Vitamins",
    ),
}


def get_parameters_for_gender(gender: str) -> dict[str, BloodParameter]:
    """
    Filter parameters based on selected gender.
    Returns gender-specific variants + gender-neutral parameters.
    """
    result = {}
    for key, param in BLOOD_PARAMETERS.items():
        if param.gender is None:
            result[key] = param
        elif param.gender == gender.lower():
            result[key] = param
    return result


def get_parameter_categories(params: dict[str, BloodParameter]) -> dict[str, list[str]]:
    """Group parameter keys by their category for organized UI display."""
    categories: dict[str, list[str]] = {}
    for key, param in params.items():
        if param.category not in categories:
            categories[param.category] = []
        categories[param.category].append(key)
    return categories
