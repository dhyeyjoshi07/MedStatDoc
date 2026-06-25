"""
Naive Bayes Health Risk Classifier
===================================
Uses Gaussian Naive Bayes from scikit-learn to estimate overall health
risk category based on the z-scores of all blood parameters.

Training data is synthetically generated from realistic medical
distributions so the model trains on app startup (< 1 second).

Risk Categories:
    - Low Risk:      Most values near population mean
    - Moderate Risk: Some values deviating significantly
    - High Risk:     Multiple values far from normal
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from statistical_engine import TestResult, Status


# ─── Synthetic Training Data Generation ───────────────────────────────────────

RISK_LABELS = ["Low Risk", "Moderate Risk", "High Risk"]
N_FEATURES = 14  # Max number of blood parameters


def _generate_synthetic_data(
    n_samples: int = 600,
    n_features: int = N_FEATURES,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data where features are z-scores.

    Distribution logic:
        Low Risk:      z-scores ~ N(0, 0.6)   — values clustered near mean
        Moderate Risk: z-scores ~ N(0, 1.2)   — wider spread, some outliers
        High Risk:     z-scores ~ N(±1.5, 1.0) — shifted away from mean

    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    samples_per_class = n_samples // 3

    # ── Low Risk: tight cluster around 0 ──────────────────────────────────
    X_low = rng.normal(loc=0.0, scale=0.6, size=(samples_per_class, n_features))
    y_low = np.full(samples_per_class, 0)

    # ── Moderate Risk: wider spread ───────────────────────────────────────
    X_mod = rng.normal(loc=0.0, scale=1.2, size=(samples_per_class, n_features))
    # Ensure at least 2-4 features are moderately abnormal
    for i in range(samples_per_class):
        n_abnormal = rng.randint(2, 5)
        abnormal_idx = rng.choice(n_features, n_abnormal, replace=False)
        X_mod[i, abnormal_idx] = rng.normal(loc=0, scale=1.8, size=n_abnormal)
    y_mod = np.full(samples_per_class, 1)

    # ── High Risk: multiple features significantly off ────────────────────
    X_high = rng.normal(loc=0.0, scale=1.0, size=(samples_per_class, n_features))
    for i in range(samples_per_class):
        n_abnormal = rng.randint(4, n_features)
        abnormal_idx = rng.choice(n_features, n_abnormal, replace=False)
        # Shift mean to ±1.5 or higher
        signs = rng.choice([-1, 1], size=n_abnormal)
        X_high[i, abnormal_idx] = signs * rng.uniform(1.5, 3.5, size=n_abnormal)
    y_high = np.full(samples_per_class, 2)

    X = np.vstack([X_low, X_mod, X_high])
    y = np.concatenate([y_low, y_mod, y_high])

    # Shuffle
    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


# ─── Classifier ───────────────────────────────────────────────────────────────

class HealthRiskClassifier:
    """
    Gaussian Naive Bayes classifier for predicting overall health risk.

    The classifier is trained on synthetic z-score data and predicts
    one of three risk categories based on the patient's z-score profile.
    """

    def __init__(self):
        self.model = GaussianNB()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(RISK_LABELS)
        self._is_trained = False
        self._train()

    def _train(self):
        """Train on synthetic data. Called once at initialization."""
        X, y = _generate_synthetic_data()
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, test_results: list[TestResult]) -> dict:
        """
        Predict health risk category from test results.

        Args:
            test_results: List of TestResult objects from the statistical engine.

        Returns:
            Dictionary with:
                - risk_category: str
                - probabilities: dict mapping category → probability
                - feature_contributions: list of (param_name, z_score, impact)
        """
        if not test_results:
            return {
                "risk_category": "Insufficient Data",
                "probabilities": {label: 0.0 for label in RISK_LABELS},
                "feature_contributions": [],
            }

        # ── Build feature vector ──────────────────────────────────────────
        z_scores = [r.test_statistic for r in test_results]

        # Pad or truncate to N_FEATURES
        feature_vector = np.zeros(N_FEATURES)
        for i, z in enumerate(z_scores[:N_FEATURES]):
            feature_vector[i] = z

        X_input = feature_vector.reshape(1, -1)

        # ── Predict ──────────────────────────────────────────────────────
        prediction = self.model.predict(X_input)[0]
        probabilities = self.model.predict_proba(X_input)[0]

        risk_category = RISK_LABELS[int(prediction)]
        prob_dict = {
            RISK_LABELS[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        }

        # ── Feature contributions (by |z-score| magnitude) ───────────────
        contributions = []
        for result in test_results:
            abs_z = abs(result.test_statistic)
            if abs_z >= 2.576:
                impact = "High Impact"
            elif abs_z >= 1.96:
                impact = "Moderate Impact"
            elif abs_z >= 1.0:
                impact = "Low Impact"
            else:
                impact = "Minimal"
            contributions.append((
                result.parameter_name,
                round(result.test_statistic, 3),
                impact,
            ))

        # Sort by absolute z-score descending
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "risk_category": risk_category,
            "probabilities": prob_dict,
            "feature_contributions": contributions,
        }
