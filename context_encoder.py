"""Stage 5: Context Integration - Comment-Only vs Context-Aware Comparison.

Implements:
1. Pipeline A (Comment-Only): Classifies comments solely based on their own text representation.
2. Pipeline B (Comment + Conversation History): Integrates preceding conversation history
   using causal, chronological context aggregation without future leakage.
3. Strict anti-leakage splitting: Partitions strictly at the conversation level.
4. Comparative ablation: Evaluates both pipelines on identical held-out test splits.
5. Qualitative and quantitative error analysis: Identifies context-dependent gains and false alarms.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ContextualCommentSample:
    """Represents an utterance paired with its preceding conversational context."""

    utterance_id: str
    conversation_id: str
    speaker: str
    timestamp: float
    depth: int
    text: str
    has_attack: int  # Ground truth binary flag for personal attack/harm
    history_ids: List[str]
    history_texts: List[str]


class ConversationHistoryAggregator:
    """Extracts causal preceding context for each comment in a conversation thread."""

    def __init__(self, max_history_turns: int = 5, decay_factor: float = 0.85):
        """Args:

        max_history_turns: Maximum preceding turns to incorporate.
        decay_factor: Temporal discount weight for older comments.
        """
        self.max_history_turns = max_history_turns
        self.decay_factor = decay_factor

    def build_context_features(
        self,
        thread_comments: List[Dict[str, Any]],
        embeddings_dict: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Build causal context representation for every comment in a thread.

        Args:
            thread_comments: List of comments in chronological order.
            embeddings_dict: Mapping from utterance_id to 768-dim RoBERTa embedding.

        Returns:
            List of feature dictionaries containing current embedding and aggregated context embedding.
        """
        features = []
        embedding_dim = 768
        history_embs: List[np.ndarray] = []
        history_ids: List[str] = []
        history_texts: List[str] = []

        for idx, comment in enumerate(thread_comments):
            uid = comment["id"]
            curr_emb = embeddings_dict.get(uid, np.zeros(embedding_dim, dtype=np.float32))

            # Aggregate preceding history
            if len(history_embs) == 0:
                # Root post: context is zero vector
                context_emb = np.zeros(embedding_dim, dtype=np.float32)
            else:
                # Use up to max_history_turns most recent comments
                selected_embs = history_embs[-self.max_history_turns :]
                # Compute recency-weighted mean
                weights = [self.decay_factor ** (len(selected_embs) - 1 - i) for i in range(len(selected_embs))]
                norm_weights = np.array(weights) / sum(weights)
                context_emb = np.sum([w * e for w, e in zip(norm_weights, selected_embs)], axis=0).astype(np.float32)

            features.append(
                {
                    "utterance_id": uid,
                    "conversation_id": comment["conversation_id"],
                    "speaker": comment.get("speaker", ""),
                    "timestamp": comment.get("timestamp", 0.0),
                    "text": comment.get("text", ""),
                    "label": int(comment.get("meta_comment_has_personal_attack", 0)),
                    "curr_emb": curr_emb,
                    "context_emb": context_emb,
                    "history_ids": list(history_ids[-self.max_history_turns :]),
                    "history_texts": list(history_texts[-self.max_history_turns :]),
                    "history_len": len(history_embs),
                }
            )

            # Append current comment to history for subsequent comments (strict causality)
            history_embs.append(curr_emb)
            history_ids.append(uid)
            history_texts.append(comment.get("text", ""))

        return features


class CommentOnlyClassifier(nn.Module):
    """Pipeline A: Classifies comment based strictly on current comment embedding."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_curr: torch.Tensor, x_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x_curr)


class ContextAwareClassifier(nn.Module):
    """Pipeline B: Classifies comment using current comment concatenated with aggregated conversation context."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_curr: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_curr, x_ctx], dim=-1)
        return self.net(combined)


def train_context_model(
    model: nn.Module,
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> Tuple[nn.Module, float]:
    """Train either Pipeline A or Pipeline B using binary cross-entropy with pos_weight."""
    dev = torch.device(device)
    model.to(dev)

    y_train = np.array([s["label"] for s in train_samples], dtype=np.float32)
    pos_count = max(y_train.sum(), 1.0)
    pos_weight = torch.tensor([(len(y_train) - pos_count) / pos_count], dtype=torch.float32).to(dev)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    curr_train = torch.tensor(np.array([s["curr_emb"] for s in train_samples]), dtype=torch.float32)
    ctx_train = torch.tensor(np.array([s["context_emb"] for s in train_samples]), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    dataset = torch.utils.data.TensorDataset(curr_train, ctx_train, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    curr_val = torch.tensor(np.array([s["curr_emb"] for s in val_samples]), dtype=torch.float32).to(dev)
    ctx_val = torch.tensor(np.array([s["context_emb"] for s in val_samples]), dtype=torch.float32).to(dev)
    y_val = np.array([s["label"] for s in val_samples], dtype=np.float32)

    best_f1 = -1.0
    best_threshold = 0.5
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_curr, batch_ctx, batch_y in loader:
            batch_curr = batch_curr.to(dev)
            batch_ctx = batch_ctx.to(dev)
            batch_y = batch_y.to(dev)

            optimizer.zero_grad()
            logits = model(batch_curr, batch_ctx)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation evaluation & threshold tuning
        model.eval()
        with torch.no_grad():
            val_logits = model(curr_val, ctx_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()

        for t in np.linspace(0.2, 0.8, 13):
            preds = (val_probs >= t).astype(int)
            f = f1_score(y_val, preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_threshold = float(t)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(f"Model trained. Best Val F1: {best_f1:.4f} at threshold: {best_threshold:.3f}")
    return model, best_threshold


def evaluate_pipeline(
    model: nn.Module,
    threshold: float,
    test_samples: List[Dict[str, Any]],
    pipeline_name: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate pipeline on held-out test data."""
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    curr_test = torch.tensor(np.array([s["curr_emb"] for s in test_samples]), dtype=torch.float32).to(dev)
    ctx_test = torch.tensor(np.array([s["context_emb"] for s in test_samples]), dtype=torch.float32).to(dev)
    y_true = np.array([s["label"] for s in test_samples], dtype=int)

    with torch.no_grad():
        logits = model(curr_test, ctx_test)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    preds = (probs >= threshold).astype(int)

    p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

    auc_score = None
    try:
        auc_score = round(float(roc_auc_score(y_true, probs)), 4)
    except Exception:
        pass

    results = {
        "pipeline": pipeline_name,
        "threshold": round(threshold, 3),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": auc_score,
        "probabilities": probs,
        "predictions": preds,
        "y_true": y_true,
    }

    logger.info(
        f"[{pipeline_name}] Test F1: {results['f1']:.4f} | Prec: {results['precision']:.4f} | Recall: {results['recall']:.4f} | AUC: {results['roc_auc']}"
    )
    return results


def run_context_ablation(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Train and evaluate Pipeline A (Comment-Only) vs Pipeline B (Context-Aware) on identical splits."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- Training Pipeline A: Comment-Only ---")
    model_a = CommentOnlyClassifier(input_dim=768, hidden_dim=128)
    model_a, thresh_a = train_context_model(model_a, train_samples, val_samples, epochs=10)
    res_a = evaluate_pipeline(model_a, thresh_a, test_samples, "Pipeline A (Comment-Only)")

    logger.info("--- Training Pipeline B: Comment + Conversation History ---")
    model_b = ContextAwareClassifier(input_dim=768, hidden_dim=128)
    model_b, thresh_b = train_context_model(model_b, train_samples, val_samples, epochs=10)
    res_b = evaluate_pipeline(model_b, thresh_b, test_samples, "Pipeline B (Context-Aware)")

    # Qualitative and quantitative error analysis
    y_true = res_a["y_true"]
    preds_a = res_a["predictions"]
    preds_b = res_b["predictions"]

    # 1. Context helped: Pipeline A missed (FN) but Pipeline B detected (TP)
    gained_indices = np.where((preds_a == 0) & (preds_b == 1) & (y_true == 1))[0]
    # 2. Context hurt: Pipeline A was correct (TN) but Pipeline B false-alarmed (FP)
    lost_indices = np.where((preds_a == 0) & (preds_b == 1) & (y_true == 0))[0]

    gained_examples = []
    for idx in gained_indices[:5]:
        s = test_samples[idx]
        gained_examples.append(
            {
                "utterance_id": s["utterance_id"],
                "text": s["text"],
                "history": s["history_texts"],
                "prob_comment_only": round(float(res_a["probabilities"][idx]), 4),
                "prob_context_aware": round(float(res_b["probabilities"][idx]), 4),
            }
        )

    lost_examples = []
    for idx in lost_indices[:5]:
        s = test_samples[idx]
        lost_examples.append(
            {
                "utterance_id": s["utterance_id"],
                "text": s["text"],
                "history": s["history_texts"],
                "prob_comment_only": round(float(res_a["probabilities"][idx]), 4),
                "prob_context_aware": round(float(res_b["probabilities"][idx]), 4),
            }
        )

    summary = {
        "pipeline_a_comment_only": {
            "precision": res_a["precision"],
            "recall": res_a["recall"],
            "f1": res_a["f1"],
            "macro_f1": res_a["macro_f1"],
            "roc_auc": res_a["roc_auc"],
            "threshold": res_a["threshold"],
        },
        "pipeline_b_context_aware": {
            "precision": res_b["precision"],
            "recall": res_b["recall"],
            "f1": res_b["f1"],
            "macro_f1": res_b["macro_f1"],
            "roc_auc": res_b["roc_auc"],
            "threshold": res_b["threshold"],
        },
        "context_impact": {
            "f1_delta": round(res_b["f1"] - res_a["f1"], 4),
            "recall_delta": round(res_b["recall"] - res_a["recall"], 4),
            "precision_delta": round(res_b["precision"] - res_a["precision"], 4),
            "attacks_recovered_by_context": int(len(gained_indices)),
            "false_alarms_introduced_by_context": int(len(lost_indices)),
        },
        "qualitative_analysis": {
            "context_recovered_attacks": gained_examples,
            "context_introduced_false_positives": lost_examples,
        },
    }

    # Save models and ablation results
    torch.save(model_a.state_dict(), out_dir / "comment_only_model.pt")
    torch.save(model_b.state_dict(), out_dir / "context_aware_model.pt")
    with open(out_dir / "context_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Context ablation results saved to {out_dir / 'context_ablation_results.json'}")
    return summary

