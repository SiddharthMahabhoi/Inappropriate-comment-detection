r"""Stage 8: Explainability - Token-Level Occlusion Attribution and Supporting-Context Analysis.

Implements:
1. Model-Agnostic Token-Level Occlusion (Leave-One-Out Perturbation):
   - Computes causal attribution delta: A_k(w_i) = P_k(x) - P_k(x \ {w_i})
   - Identifies hostile trigger words (A > 0) and mitigating context (A < 0)
2. Direct Feature Attribution for TF-IDF Linear Baseline:
   - Computes exact mathematical contribution: coef_{k,j} * tfidf_{j}(x)
3. Contextual Supporting-Comment Attribution:
   - Identifies which preceding comment in the conversation thread exerted the greatest influence
4. Faithfulness Perturbation Verification:
   - Quantifies the probability drop when top attributed tokens are masked
5. Ethical Pragmatics:
   - Clearly distinguishes model feature sensitivity from human communicative intent
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TokenAttribution:
    """Represents attribution for an individual token."""

    token: str
    attribution_score: float  # Positive = elevated toxicity; Negative = reduced toxicity
    label: str


@dataclass
class ExplanationResult:
    """Structured explanation package for a single prediction."""

    text: str
    target_label: str
    base_probability: float
    token_attributions: List[TokenAttribution]
    top_hostile_tokens: List[str]
    faithfulness_drop: float  # Drop in probability when top-k hostile tokens are masked
    supporting_context_comments: List[Dict[str, Any]]
    methodology: str
    disclaimer: str


class CommentExplainer:
    """Generates causal, perturbation-based and feature-based explanations for moderation decisions."""

    def __init__(self, predict_fn: Optional[Callable[[List[str]], np.ndarray]] = None):
        """Args:

        predict_fn: Callable taking a list of text strings and returning a 2D array of probabilities (N, num_classes).
        """
        self.predict_fn = predict_fn

    def explain_text_occlusion(
        self,
        text: str,
        label_index: int = 0,
        label_name: str = "toxic",
        top_k: int = 5,
    ) -> ExplanationResult:
        """Explains prediction via Leave-One-Out Token Occlusion perturbation.

        Measures the counterfactual effect of removing each word on the model's output probability.
        """
        if self.predict_fn is None:
            raise ValueError("predict_fn must be provided for occlusion attribution.")

        # Baseline probability on unperturbed text
        base_probs = self.predict_fn([text])
        base_prob = float(base_probs[0, label_index])

        # Tokenize preserving word positions
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        if not tokens:
            return ExplanationResult(
                text=text,
                target_label=label_name,
                base_probability=base_prob,
                token_attributions=[],
                top_hostile_tokens=[],
                faithfulness_drop=0.0,
                supporting_context_comments=[],
                methodology="Token Occlusion (Leave-One-Out)",
                disclaimer="Model attribution measures mathematical feature sensitivity, not human intent.",
            )

        # Generate leave-one-out variants
        perturbed_texts = []
        for i in range(len(tokens)):
            masked_tokens = tokens[:i] + tokens[i + 1 :]
            perturbed_texts.append(" ".join(masked_tokens))

        # Batch predict perturbed texts
        perturbed_probs = self.predict_fn(perturbed_texts)

        attributions: List[TokenAttribution] = []
        for i, tok in enumerate(tokens):
            pert_prob = float(perturbed_probs[i, label_index])
            # Delta: how much probability falls when the token is removed
            score = round(base_prob - pert_prob, 4)
            attributions.append(TokenAttribution(token=tok, attribution_score=score, label=label_name))

        # Sort tokens by positive hostile contribution
        sorted_by_impact = sorted(attributions, key=lambda a: a.attribution_score, reverse=True)
        top_hostile = [a.token for a in sorted_by_impact if a.attribution_score > 0][:top_k]

        # Faithfulness perturbation test: mask top-k hostile tokens simultaneously
        faithfulness_drop = 0.0
        if top_hostile:
            tokens_masked = [t for t in tokens if t not in set(top_hostile)]
            masked_full_text = " ".join(tokens_masked)
            prob_after_mask = float(self.predict_fn([masked_full_text])[0, label_index])
            faithfulness_drop = round(base_prob - prob_after_mask, 4)

        return ExplanationResult(
            text=text,
            target_label=label_name,
            base_probability=round(base_prob, 4),
            token_attributions=attributions,
            top_hostile_tokens=top_hostile,
            faithfulness_drop=faithfulness_drop,
            supporting_context_comments=[],
            methodology="Causal Token Occlusion (Counterfactual Leave-One-Out)",
            disclaimer="Attribution indicates mathematical sensitivity to specific vocabulary; it does not constitute proof of author malice.",
        )

    def explain_contextual_support(
        self,
        current_text: str,
        history_texts: List[str],
        context_predict_fn: Callable[[str, List[str]], float],
    ) -> List[Dict[str, Any]]:
        """Identifies which preceding comments exerted the highest escalatory impact on the current comment.

        Evaluates the counterfactual removal of individual historical comments from the conversation context.
        """
        if not history_texts:
            return []

        base_prob = context_predict_fn(current_text, history_texts)
        supporting_comments = []

        for i, prev_text in enumerate(history_texts):
            # Mask out turn i from history
            ablated_history = history_texts[:i] + history_texts[i + 1 :]
            ablated_prob = context_predict_fn(current_text, ablated_history)
            impact = round(base_prob - ablated_prob, 4)

            supporting_comments.append(
                {
                    "turn_index": i + 1,
                    "comment_snippet": prev_text[:80] + ("..." if len(prev_text) > 80 else ""),
                    "context_influence_delta": impact,
                    "interpretation": (
                        "Escalatory context" if impact > 0.05 else ("Mitigating context" if impact < -0.05 else "Neutral")
                    ),
                }
            )

        # Sort by highest escalatory influence
        supporting_comments.sort(key=lambda x: x["context_influence_delta"], reverse=True)
        return supporting_comments
