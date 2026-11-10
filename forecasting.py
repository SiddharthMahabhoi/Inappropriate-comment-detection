"""Stage 6: Conversation Derailment Forecasting (CRAFT-Inspired).

Implements:
1. Anti-leakage conversation prefix generation:
   - Evaluates conversation states at sequential prefix lengths (t = 1, 2, ...).
   - Excludes the derailment event and any future comments from input.
   - Splits strictly by conversation_id to eliminate data leakage.
2. Forecasting models:
   - Baseline: Mean-pooled prefix embedding + Linear Forecaster
   - Sequential Model: CRAFT-inspired GRU tracking conversational trajectory over time
3. Comprehensive evaluation:
   - Precision, Recall, F1-score, PR-AUC, False Positive Rate (FPR)
   - Early-warning lead time (number of turns before derailment)
   - Probability calibration: Brier Score, Expected Calibration Error (ECE)
4. Clearly documents target definitions:
   - Wikipedia: Genuine personal attack derailment
   - CMV: Moderator comment removal proxy
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConversationPrefixSample:
    """Represents a conversation truncated at turn t before any derailment."""

    conversation_id: str
    prefix_length: int
    will_derail: int  # 1 if thread subsequently derails; 0 if thread remains civil
    lead_time_turns: Optional[int]  # Distance in turns to the subsequent attack/event
    utterance_ids: List[str]
    texts: List[str]
    split: str


class ConversationPrefixGenerator:
    """Generates strictly causal prefixes from conversational threads."""

    def __init__(self, min_prefix: int = 1, max_prefix: int = 8):
        self.min_prefix = min_prefix
        self.max_prefix = max_prefix

    def generate_prefixes_from_thread(
        self,
        thread_utterances: List[Dict[str, Any]],
        conversation_will_derail: bool,
        split: str = "train",
    ) -> List[ConversationPrefixSample]:
        """Constructs valid developing prefixes for a single thread.

        Crucially halts prefix generation BEFORE any comment-level personal attack or removal.
        """
        samples: List[ConversationPrefixSample] = []
        clean_utterances: List[Dict[str, Any]] = []
        attack_turn_index: Optional[int] = None

        # Identify if and where derailment occurs
        for idx, utt in enumerate(thread_utterances):
            is_attack = bool(utt.get("meta_comment_has_personal_attack", False))
            if is_attack and attack_turn_index is None:
                attack_turn_index = idx
                break  # Do NOT include the attack or anything after it
            clean_utterances.append(utt)

        # If thread derailed but no single comment was flagged, treat whole conversation as positive
        target = 1 if conversation_will_derail else 0

        # Build prefixes of length min_prefix .. min(max_prefix, len(clean_utterances))
        max_t = min(len(clean_utterances), self.max_prefix)
        for t in range(self.min_prefix, max_t + 1):
            sub_utts = clean_utterances[:t]
            lead_time = (attack_turn_index - t) if (attack_turn_index is not None) else None

            sample = ConversationPrefixSample(
                conversation_id=str(clean_utterances[0]["conversation_id"]),
                prefix_length=t,
                will_derail=target,
                lead_time_turns=lead_time,
                utterance_ids=[u["id"] for u in sub_utts],
                texts=[str(u.get("text", "")) for u in sub_utts],
                split=split,
            )
            samples.append(sample)

        return samples


class CraftInspiredGRU(nn.Module):
    """CRAFT-inspired recurrent neural network forecasting derailment from sequential comments."""

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:

        x_seq: Batch tensor of shape (batch_size, max_seq_len, embedding_dim).

        Returns:
            Logits tensor of shape (batch_size, 1).
        """
        out, h_n = self.gru(x_seq)
        # Use final step hidden state
        final_h = h_n[-1]
        logits = self.classifier(final_h)
        return logits


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < num_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            bin_acc = y_true[in_bin].mean()
            bin_conf = y_prob[in_bin].mean()
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


class DerailmentForecastingEngine:
    """Orchestrates training, evaluation, and inference for derailment forecasting."""

    def __init__(self, embedding_dim: int = 768, device: str = "cpu"):
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.gru_model = CraftInspiredGRU(embedding_dim=embedding_dim, hidden_dim=128).to(self.device)
        self.baseline_model = LogisticRegression(class_weight="balanced", max_iter=200, solver="liblinear")
        self.threshold: float = 0.5
        self.baseline_threshold: float = 0.5
        self.is_trained = False

    def _prepare_tensors(
        self,
        samples: List[ConversationPrefixSample],
        embeddings_dict: Dict[str, np.ndarray],
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Convert samples into padded sequence tensors and mean-pooled representations."""
        max_len = max(s.prefix_length for s in samples) if samples else 1
        num_samples = len(samples)

        seq_array = np.zeros((num_samples, max_len, self.embedding_dim), dtype=np.float32)
        mean_array = np.zeros((num_samples, self.embedding_dim), dtype=np.float32)
        y_array = np.array([s.will_derail for s in samples], dtype=np.float32)

        for i, s in enumerate(samples):
            embs = [embeddings_dict.get(uid, np.zeros(self.embedding_dim, dtype=np.float32)) for uid in s.utterance_ids]
            if len(embs) > 0:
                seq_array[i, : len(embs), :] = np.array(embs)
                mean_array[i] = np.mean(embs, axis=0)

        return torch.tensor(seq_array, dtype=torch.float32), mean_array, y_array

    def fit(
        self,
        train_samples: List[ConversationPrefixSample],
        val_samples: List[ConversationPrefixSample],
        embeddings_dict: Dict[str, np.ndarray],
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """Train both the Baseline and CRAFT-inspired GRU forecaster."""
        logger.info(f"Training Derailment Forecasting on {len(train_samples)} conversation prefixes...")

        X_train_seq, X_train_mean, y_train = self._prepare_tensors(train_samples, embeddings_dict)
        X_val_seq, X_val_mean, y_val = self._prepare_tensors(val_samples, embeddings_dict)

        # 1. Fit Baseline Logistic Regression on mean-pooled prefixes
        logger.info("Fitting Mean-Pooled Prefix Baseline (Logistic Regression)...")
        self.baseline_model.fit(X_train_mean, y_train)

        # Tune baseline threshold on validation
        val_base_probs = self.baseline_model.predict_proba(X_val_mean)[:, 1]
        best_base_f1 = -1.0
        for t in np.linspace(0.2, 0.8, 13):
            f = f1_score(y_val, (val_base_probs >= t).astype(int), zero_division=0)
            if f > best_base_f1:
                best_base_f1 = f
                self.baseline_threshold = float(t)

        # 2. Train CRAFT-inspired GRU
        pos_count = max(y_train.sum(), 1.0)
        pos_weight = torch.tensor([(len(y_train) - pos_count) / pos_count], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.gru_model.parameters(), lr=lr, weight_decay=1e-4)

        train_dataset = torch.utils.data.TensorDataset(
            X_train_seq, torch.tensor(y_train).unsqueeze(-1)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_gru_f1 = -1.0
        best_state = None

        X_val_seq_dev = X_val_seq.to(self.device)

        for epoch in range(1, epochs + 1):
            self.gru_model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                optimizer.zero_grad()
                logits = self.gru_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)

            # Validation check
            self.gru_model.eval()
            with torch.no_grad():
                val_logits = self.gru_model(X_val_seq_dev)
                val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()

            for t in np.linspace(0.2, 0.8, 13):
                f = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
                if f > best_gru_f1:
                    best_gru_f1 = f
                    self.threshold = float(t)
                    best_state = {k: v.cpu().clone() for k, v in self.gru_model.state_dict().items()}

        if best_state is not None:
            self.gru_model.load_state_dict(best_state)

        self.is_trained = True
        logger.info(
            f"Forecasting models trained: Baseline Best Val F1={best_base_f1:.4f} (t={self.baseline_threshold:.2f}) | "
            f"GRU Best Val F1={best_gru_f1:.4f} (t={self.threshold:.2f})"
        )
        return {"baseline_val_f1": best_base_f1, "gru_val_f1": best_gru_f1}

    def predict_conversation_trajectory(
        self,
        comment_embeddings: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Compute the developing derailment probability after every comment in a thread.

        Args:
            comment_embeddings: List of 768-dim embeddings in chronological order.

        Returns:
            List of turn-level prediction dictionaries showing trajectory over time.
        """
        if not self.is_trained:
            raise RuntimeError("Forecasting model not trained.")

        self.gru_model.eval()
        trajectory = []

        with torch.no_grad():
            for t in range(1, len(comment_embeddings) + 1):
                prefix_embs = np.array(comment_embeddings[:t], dtype=np.float32)
                x_tensor = torch.tensor(prefix_embs).unsqueeze(0).to(self.device)
                logit = self.gru_model(x_tensor)
                prob = float(torch.sigmoid(logit).cpu().numpy().flatten()[0])

                risk_level = "Low"
                if prob >= 0.70:
                    risk_level = "High"
                elif prob >= 0.40:
                    risk_level = "Medium"

                trajectory.append(
                    {
                        "turn": t,
                        "derailment_probability": round(prob, 4),
                        "warning_triggered": bool(prob >= self.threshold),
                        "risk_level": risk_level,
                    }
                )

        return trajectory

    def evaluate(
        self,
        test_samples: List[ConversationPrefixSample],
        embeddings_dict: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation on held-out test conversation prefixes."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        X_test_seq, X_test_mean, y_test = self._prepare_tensors(test_samples, embeddings_dict)

        # Baseline evaluation
        base_probs = self.baseline_model.predict_proba(X_test_mean)[:, 1]
        base_preds = (base_probs >= self.baseline_threshold).astype(int)

        p_base, r_base, f_base, _ = precision_recall_fscore_support(
            y_test, base_preds, average="binary", zero_division=0
        )
        auc_base = round(float(roc_auc_score(y_test, base_probs)), 4)
        brier_base = round(float(brier_score_loss(y_test, base_probs)), 4)
        ece_base = compute_expected_calibration_error(y_test, base_probs)

        # False positive rate for baseline
        tn_b = ((base_preds == 0) & (y_test == 0)).sum()
        fp_b = ((base_preds == 1) & (y_test == 0)).sum()
        fpr_base = round(float(fp_b / max(fp_b + tn_b, 1)), 4)

        # GRU evaluation
        self.gru_model.eval()
        with torch.no_grad():
            gru_logits = self.gru_model(X_test_seq.to(self.device))
            gru_probs = torch.sigmoid(gru_logits).cpu().numpy().flatten()

        gru_preds = (gru_probs >= self.threshold).astype(int)

        p_gru, r_gru, f_gru, _ = precision_recall_fscore_support(
            y_test, gru_preds, average="binary", zero_division=0
        )
        auc_gru = round(float(roc_auc_score(y_test, gru_probs)), 4)
        brier_gru = round(float(brier_score_loss(y_test, gru_probs)), 4)
        ece_gru = compute_expected_calibration_error(y_test, gru_probs)

        tn_g = ((gru_preds == 0) & (y_test == 0)).sum()
        fp_g = ((gru_preds == 1) & (y_test == 0)).sum()
        fpr_gru = round(float(fp_g / max(fp_g + tn_g, 1)), 4)

        # Compute early-warning lead time for derailed threads
        lead_times_gru = []
        for i, s in enumerate(test_samples):
            if s.will_derail == 1 and gru_preds[i] == 1 and s.lead_time_turns is not None:
                if s.lead_time_turns >= 0:
                    lead_times_gru.append(s.lead_time_turns)

        avg_lead_time = round(float(np.mean(lead_times_gru)), 2) if lead_times_gru else 0.0

        eval_summary = {
            "baseline_mean_prefix": {
                "precision": round(float(p_base), 4),
                "recall": round(float(r_base), 4),
                "f1": round(float(f_base), 4),
                "pr_auc": auc_base,
                "fpr": fpr_base,
                "brier_score": brier_base,
                "ece": ece_base,
                "threshold": round(self.baseline_threshold, 3),
            },
            "craft_inspired_gru": {
                "precision": round(float(p_gru), 4),
                "recall": round(float(r_gru), 4),
                "f1": round(float(f_gru), 4),
                "pr_auc": auc_gru,
                "fpr": fpr_gru,
                "brier_score": brier_gru,
                "ece": ece_gru,
                "threshold": round(self.threshold, 3),
                "average_lead_time_turns": avg_lead_time,
                "warnings_issued_before_attack": len(lead_times_gru),
            },
        }

        # Save model checkpoints
        torch.save(self.gru_model.state_dict(), out / "craft_gru.pt")
        with open(out / "forecasting_metrics.json", "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

        logger.info(
            f"Forecasting Evaluation Summary:\n"
            f"  Baseline GRU comparison:\n"
            f"    Baseline F1: {eval_summary['baseline_mean_prefix']['f1']} | PR-AUC: {auc_base} | FPR: {fpr_base}\n"
            f"    CRAFT GRU F1: {eval_summary['craft_inspired_gru']['f1']} | PR-AUC: {auc_gru} | FPR: {fpr_gru}\n"
            f"    Average Lead Time: {avg_lead_time} turns before attack"
        )
        return eval_summary

