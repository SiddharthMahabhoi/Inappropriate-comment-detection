"""Stage 7: Severity Grading and Provisional Risk Estimation.

Ground-Truth Audit:
- Jigsaw contains genuine binary annotation for 'severe_toxic' (1,595 positive samples).
- Wikipedia talk pages contain continuous 'meta_toxicity' scores.
- Neither dataset natively contains four-class ordinal labels ('Low', 'Medium', 'High', 'Critical').

Implementation Strategy:
1. Supervised predictor for genuine 'severe_toxic' ground-truth labels.
2. Explicitly documented, rule-grounded Provisional Severity Estimator combining multi-label
   harm probabilities, severe toxicity, and conversation derailment forecasting into transparent risk bands:
   - Low, Medium, High, Critical
3. Clear architectural and UI separation between genuine trained predictions and provisional estimates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SeverityAssessment:
    """Represents a structured severity grading outcome."""

    severity_band: str  # "Low", "Medium", "High", "Critical"
    numerical_risk_score: float  # [0.0, 1.0] continuous composite risk index
    is_provisional: bool  # True: Derived via rule-grounded composite estimator
    primary_drivers: List[str]  # Which factors elevated the severity
    explanation: str  # Human-readable rationale for moderators
    trained_severe_toxic_prob: Optional[float] = None  # Supervised severe_toxic probability if available


class SeverityGradingEngine:
    """Evaluates comment severity using supervised signals and explicit composite risk mapping."""

    SEVERITY_BANDS = ["Low", "Medium", "High", "Critical"]

    def __init__(
        self,
        weight_threat: float = 0.35,
        weight_severe_toxic: float = 0.25,
        weight_identity_hate: float = 0.20,
        weight_insult: float = 0.10,
        weight_toxic: float = 0.10,
        weight_derailment: float = 0.15,
    ):
        """Configure composite risk weights based on legal and platform harm hierarchies."""
        self.weights = {
            "threat": weight_threat,
            "severe_toxic": weight_severe_toxic,
            "identity_hate": weight_identity_hate,
            "insult": weight_insult,
            "toxic": weight_toxic,
            "derailment": weight_derailment,
        }

    def assess_severity(
        self,
        label_probabilities: Dict[str, float],
        derailment_probability: Optional[float] = None,
    ) -> SeverityAssessment:
        """Computes severity assessment distinguishing genuine trained probabilities from provisional bands.

        Args:
            label_probabilities: Dictionary of multi-label prediction probabilities.
            derailment_probability: Optional forecasted probability of thread derailment.

        Returns:
            Structured SeverityAssessment object with clear provisional disclosure.
        """
        p_toxic = float(label_probabilities.get("toxic", 0.0))
        p_severe = float(label_probabilities.get("severe_toxic", 0.0))
        p_threat = float(label_probabilities.get("threat", 0.0))
        p_hate = float(label_probabilities.get("identity_hate", 0.0))
        p_insult = float(label_probabilities.get("insult", 0.0))
        p_obscene = float(label_probabilities.get("obscene", 0.0))
        p_derail = float(derailment_probability) if derailment_probability is not None else 0.0

        # Continuous composite harm score [0.0, 1.0]
        composite_score = (
            self.weights["toxic"] * p_toxic
            + self.weights["insult"] * p_insult
            + self.weights["severe_toxic"] * p_severe
            + self.weights["identity_hate"] * p_hate
            + self.weights["threat"] * p_threat
            + self.weights["derailment"] * p_derail
        )
        composite_score = round(min(max(composite_score, 0.0), 1.0), 4)

        drivers = []
        # Critical Tier Triggers: Direct threats of violence or extreme identity-targeted severe harm
        if p_threat >= 0.25:
            severity = "Critical"
            drivers.append("High-confidence threat of physical violence or harm")
        elif p_severe >= 0.60 and p_hate >= 0.50:
            severity = "Critical"
            drivers.append("Extreme severe toxicity combined with identity-directed hate")
        # High Tier Triggers: Severe toxicity, identity hate, or extreme co-occurring abuse
        elif p_severe >= 0.30 or p_hate >= 0.35:
            severity = "High"
            if p_severe >= 0.30:
                drivers.append("Severe toxicity detected")
            if p_hate >= 0.35:
                drivers.append("Identity-directed hate detected")
        elif (p_toxic >= 0.70 and p_insult >= 0.60) or sum(1 for p in [p_toxic, p_insult, p_obscene] if p >= 0.65) >= 2:
            severity = "High"
            drivers.append("Multiple severe co-occurring abuse categories")
        # Medium Tier Triggers: Moderate toxicity, isolated profanity or personal insult
        elif p_toxic >= 0.35 or p_insult >= 0.35 or p_obscene >= 0.40 or p_derail >= 0.55:
            severity = "Medium"
            if p_toxic >= 0.35:
                drivers.append("Moderate toxicity")
            if p_insult >= 0.35:
                drivers.append("Interpersonal insult")
            if p_obscene >= 0.45:
                drivers.append("Profanity / obscenity")
            if p_derail >= 0.60:
                drivers.append("Elevated conversation derailment risk")
        # Low Tier: Civil comment or mild disagreement
        else:
            severity = "Low"
            drivers.append("No critical violations detected; within standard conversational norms")

        explanation = (
            f"Provisional Severity: [{severity}] (Composite Score: {composite_score:.2f}). "
            f"Key factors: {'; '.join(drivers)}."
        )

        return SeverityAssessment(
            severity_band=severity,
            numerical_risk_score=composite_score,
            is_provisional=True,
            primary_drivers=drivers,
            explanation=explanation,
            trained_severe_toxic_prob=p_severe if "severe_toxic" in label_probabilities else None,
        )


def document_conceptual_distinctions() -> Dict[str, str]:
    """Returns authoritative technical distinctions between toxicity, derailment, and severity."""
    return {
        "toxicity_probability": (
            "Estimates whether the current comment's text, in isolation, violates community guidelines "
            "regarding incivility, vulgarity, or personal insults."
        ),
        "conversation_derailment_risk": (
            "A sequential forecasting trajectory estimating whether the multi-turn conversational thread "
            "is degenerating toward future hostile breakdown or moderator intervention."
        ),
        "severity_grade": (
            "An ordinal impact assessment (Low, Medium, High, Critical) reflecting the potential harm, "
            "legal liability, or safety hazard of the comment, prioritizing threats and hate speech."
        ),
    }
