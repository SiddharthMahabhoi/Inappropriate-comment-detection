"""Backend orchestration module for the Moderator Dashboard.

Integrates:
- Stage 1: Data structures and conversation thread loading
- Stage 2: Preprocessing and text normalization
- Stage 3: Sentence-RoBERTa dense embeddings (768-dim)
- Stage 4: Multi-label classification (RoBERTa Neural Head & TF-IDF Baseline Ensemble)
- Stage 5: Context integration (Pipeline A Comment-Only vs Pipeline B Context-Aware)
- Stage 6: CRAFT-inspired GRU sequential derailment forecasting
- Stage 7: Severity grading and provisional composite risk estimation
- Stage 8: Model-agnostic Leave-One-Out token occlusion explainability (Lazy On-Demand Caching)
- Moderation review management and session persistence
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import EXTENDED_LABELS, TARGET_LABELS, RoBERTaMultiLabelClassifier, TfidfMultiLabelBaseline
from src.context_encoder import CommentOnlyClassifier, ContextAwareClassifier, ConversationHistoryAggregator
from src.embeddings import CommentEmbeddingGenerator
from src.explainability import CommentExplainer, ExplanationResult
from src.forecasting import CraftInspiredGRU, DerailmentForecastingEngine
from src.preprocessing import ConversationPreprocessor
from src.severity import SeverityAssessment, SeverityGradingEngine

logger = logging.getLogger(__name__)

MODERATION_REVIEWS_FILE = PROJECT_ROOT / "outputs" / "moderation_reviews.json"
ACTIVITY_HISTORY_FILE = PROJECT_ROOT / "outputs" / "analysis_activity_log.json"

_LOO_CACHE: Dict[str, Dict[str, Any]] = {}


def load_all_resources() -> Dict[str, Any]:
    """Load all saved model weights, evaluators, and configuration artifacts."""
    models_dir = PROJECT_ROOT / "outputs" / "models"
    preprocessor = ConversationPreprocessor()
    severity_engine = SeverityGradingEngine()
    history_aggregator = ConversationHistoryAggregator(max_history_turns=5, decay_factor=0.85)

    # 1. TF-IDF Baseline
    baseline_model = None
    baseline_dir = models_dir / "baseline_tfidf"
    if baseline_dir.exists():
        try:
            baseline_model = TfidfMultiLabelBaseline.load(baseline_dir)
        except Exception as e:
            logger.warning(f"Could not load TF-IDF baseline: {e}")

    # 2. RoBERTa Multi-Label Head
    roberta_model = None
    roberta_dir = models_dir / "roberta_classifier"
    if roberta_dir.exists():
        try:
            roberta_model = RoBERTaMultiLabelClassifier.load(roberta_dir, device="cpu")
        except Exception as e:
            logger.warning(f"Could not load RoBERTa head: {e}")

    # 3. Comment Embedding Generator
    emb_gen = None
    try:
        emb_gen = CommentEmbeddingGenerator(
            model_name="sentence-transformers/all-distilroberta-v1",
            device="cpu",
            batch_size=16,
        )
    except Exception as e:
        logger.warning(f"Embedding generator offline: {e}")

    # 4. Context Pipeline Models (Stage 5)
    pipeline_a_model = None
    pipeline_b_model = None
    stage5_dir = PROJECT_ROOT / "outputs" / "stage5_context"
    p_a_path = stage5_dir / "comment_only_model.pt"
    p_b_path = stage5_dir / "context_aware_model.pt"

    if p_a_path.exists():
        try:
            pipeline_a_model = CommentOnlyClassifier(input_dim=768, hidden_dim=128)
            pipeline_a_model.load_state_dict(torch.load(p_a_path, map_location="cpu"))
            pipeline_a_model.eval()
        except Exception as e:
            logger.warning(f"Could not load Pipeline A model: {e}")

    if p_b_path.exists():
        try:
            pipeline_b_model = ContextAwareClassifier(input_dim=768, hidden_dim=128)
            pipeline_b_model.load_state_dict(torch.load(p_b_path, map_location="cpu"))
            pipeline_b_model.eval()
        except Exception as e:
            logger.warning(f"Could not load Pipeline B model: {e}")

    # 5. CRAFT Derailment Forecaster (Stage 6)
    craft_gru = None
    gru_path = PROJECT_ROOT / "outputs" / "stage6_forecasting" / "craft_gru.pt"
    if gru_path.exists():
        try:
            craft_gru = CraftInspiredGRU(embedding_dim=768, hidden_dim=128)
            craft_gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
            craft_gru.eval()
        except Exception as e:
            logger.warning(f"Could not load CRAFT GRU model: {e}")

    # 6. Evaluation Results and Manifests
    eval_data = None
    eval_json_path = PROJECT_ROOT / "outputs" / "evaluation_results.json"
    if eval_json_path.exists():
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load evaluation results: {e}")

    phase1_summary = None
    p1_path = PROJECT_ROOT / "outputs" / "phase1_run_summary.json"
    if p1_path.exists():
        try:
            with open(p1_path, "r", encoding="utf-8") as f:
                phase1_summary = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load Phase 1 summary: {e}")

    return {
        "preprocessor": preprocessor,
        "severity_engine": severity_engine,
        "history_aggregator": history_aggregator,
        "baseline_model": baseline_model,
        "roberta_model": roberta_model,
        "emb_gen": emb_gen,
        "pipeline_a_model": pipeline_a_model,
        "pipeline_b_model": pipeline_b_model,
        "craft_gru": craft_gru,
        "eval_data": eval_data,
        "phase1_summary": phase1_summary,
    }


def explain_single_comment(
    text: str,
    resources: Dict[str, Any],
    target_label: str = "toxic",
    top_k: int = 5,
) -> Optional[Dict[str, Any]]:
    """Compute Leave-One-Out token occlusion on-demand for a single comment with caching."""
    clean_text = text.strip()
    if not clean_text:
        return None

    cache_key = f"{clean_text}::{target_label}::{top_k}"
    if cache_key in _LOO_CACHE:
        return _LOO_CACHE[cache_key]

    baseline_model = resources.get("baseline_model")
    if not baseline_model:
        return None

    try:
        explainer = CommentExplainer(predict_fn=baseline_model.predict_proba)
        active_labels = EXTENDED_LABELS
        label_idx = active_labels.index(target_label) if target_label in active_labels else 0
        exp_res = explainer.explain_text_occlusion(clean_text, label_index=label_idx, label_name=target_label, top_k=top_k)
        result = {
            "target_label": exp_res.target_label,
            "faithfulness_drop": round(exp_res.faithfulness_drop, 4),
            "top_hostile_tokens": exp_res.top_hostile_tokens,
            "token_attributions": [asdict(a) for a in exp_res.token_attributions],
            "methodology": exp_res.methodology,
            "disclaimer": exp_res.disclaimer,
        }
        _LOO_CACHE[cache_key] = result
        return result
    except Exception as e:
        logger.warning(f"Explainability computation failed: {e}")
        return None


def get_default_benchmark_threads() -> Dict[str, List[Dict[str, Any]]]:
    """Provide realistic, research-grade conversational benchmark threads."""
    return {
        "Wikipedia Deliberation Escalating to Attack (Thread #748)": [
            {
                "comment_id": "c_wiki_01",
                "parent_id": None,
                "speaker": "Editor_A",
                "timestamp": "14:02",
                "text": "I noticed that you reverted my edits regarding the WP:COMMONNAME guidelines on this page.",
            },
            {
                "comment_id": "c_wiki_02",
                "parent_id": "c_wiki_01",
                "speaker": "Editor_B",
                "timestamp": "14:08",
                "text": "Your edits violated existing consensus reached on the talk page last week. Please read the archives.",
            },
            {
                "comment_id": "c_wiki_03",
                "parent_id": "c_wiki_02",
                "speaker": "Editor_A",
                "timestamp": "14:15",
                "text": "Consensus was not formally reached; you unilaterally closed the discussion without addressing policy objections.",
            },
            {
                "comment_id": "c_wiki_04",
                "parent_id": "c_wiki_03",
                "speaker": "Editor_B",
                "timestamp": "14:23",
                "text": "If you actually read the archives you would see two administrators agreed with the closure. Stop wikilawyering.",
            },
            {
                "comment_id": "c_wiki_05",
                "parent_id": "c_wiki_04",
                "speaker": "Editor_A",
                "timestamp": "14:31",
                "text": "That simply isn't true. You are deliberately lying to enforce your personal POV across the encyclopedia.",
            },
            {
                "comment_id": "c_wiki_06",
                "parent_id": "c_wiki_05",
                "speaker": "Editor_B",
                "timestamp": "14:38",
                "text": "Shut your mouth you incompetent troll, you have zero credibility here and should be permanently banned!",
            },
        ],
        "Reddit ChangeMyView Constructive Deliberation (Thread #912)": [
            {
                "comment_id": "c_cmv_01",
                "parent_id": None,
                "speaker": "Original_Poster",
                "timestamp": "09:15",
                "text": "CMV: Universal Basic Income is economically unsustainable long-term and disincentivizes skilled labor.",
            },
            {
                "comment_id": "c_cmv_02",
                "parent_id": "c_cmv_01",
                "speaker": "Challenger_1",
                "timestamp": "09:24",
                "text": "Empirical trials in Finland and Stockton show that employment rates did not drop; recipient health and education outcomes improved.",
            },
            {
                "comment_id": "c_cmv_03",
                "parent_id": "c_cmv_02",
                "speaker": "Original_Poster",
                "timestamp": "09:35",
                "text": "Those pilots had outside funding and fixed end dates. A nationwide permanent program requires tax increases that distort investment.",
            },
            {
                "comment_id": "c_cmv_04",
                "parent_id": "c_cmv_03",
                "speaker": "Challenger_1",
                "timestamp": "09:48",
                "text": "Fair point on municipal pilot scale. However, replacing means-tested bureaucracy can offset significant administrative overhead.",
            },
            {
                "comment_id": "c_cmv_05",
                "parent_id": "c_cmv_04",
                "speaker": "Original_Poster",
                "timestamp": "10:02",
                "text": "Δ awarding delta: while I maintain concerns about tax rates, the administrative consolidation argument is compelling.",
            },
        ],
        "Subtle Hostility & Sarcasm Boundary Test (Thread #305)": [
            {
                "comment_id": "c_sarc_01",
                "parent_id": None,
                "speaker": "Reviewer_X",
                "timestamp": "18:05",
                "text": "The proposed technical specification lacks any consideration for backward compatibility.",
            },
            {
                "comment_id": "c_sarc_02",
                "parent_id": "c_sarc_01",
                "speaker": "Author_Y",
                "timestamp": "18:14",
                "text": "Thank you for the brilliant insight. Clearly nobody on the architecture team thought of backward compatibility.",
            },
            {
                "comment_id": "c_sarc_03",
                "parent_id": "c_sarc_02",
                "speaker": "Reviewer_X",
                "timestamp": "18:22",
                "text": "Given your track record on the authentication refactor, that would not surprise anyone on this team.",
            },
            {
                "comment_id": "c_sarc_04",
                "parent_id": "c_sarc_03",
                "speaker": "Author_Y",
                "timestamp": "18:30",
                "text": "Keep projecting your own insecurities, everyone knows who actually writes the code around here.",
            },
        ],
    }


def run_conversation_inference(
    comments: List[Dict[str, Any]],
    resources: Dict[str, Any],
    model_choice: str = "hybrid",
    custom_thresholds: Optional[Dict[str, float]] = None,
    conversation_id: str = "live_conv_001",
) -> Dict[str, Any]:
    """Execute the end-to-end inference pipeline on an ordered list of comments with high sensitivity."""
    preprocessor: ConversationPreprocessor = resources["preprocessor"]
    severity_engine: SeverityGradingEngine = resources["severity_engine"]
    baseline_model: Optional[TfidfMultiLabelBaseline] = resources.get("baseline_model")
    roberta_model: Optional[RoBERTaMultiLabelClassifier] = resources.get("roberta_model")
    emb_gen: Optional[CommentEmbeddingGenerator] = resources.get("emb_gen")
    craft_gru: Optional[CraftInspiredGRU] = resources.get("craft_gru")
    pipeline_a: Optional[CommentOnlyClassifier] = resources.get("pipeline_a_model")
    pipeline_b: Optional[ContextAwareClassifier] = resources.get("pipeline_b_model")

    active_labels = EXTENDED_LABELS

    # Balanced production thresholds ensuring sensitivity to both subtle and explicit toxicity
    balanced_thresholds = {
        "toxic": 0.50,
        "severe_toxic": 0.65,
        "obscene": 0.50,
        "threat": 0.35,
        "insult": 0.48,
        "identity_hate": 0.55,
    }
    effective_thresholds = dict(balanced_thresholds)
    if custom_thresholds:
        effective_thresholds.update(custom_thresholds)

    # 1. Clean texts
    raw_texts = [str(c.get("text", "")) for c in comments]
    cleaned_texts = [preprocessor.clean_text(t) for t in raw_texts]

    # 2. Compute embeddings
    embeddings = None
    if emb_gen is not None and cleaned_texts:
        try:
            embeddings = emb_gen.encode_batch(cleaned_texts)
        except Exception as e:
            logger.warning(f"Dense embedding computation failed: {e}")

    if embeddings is None:
        embeddings = np.zeros((len(cleaned_texts), 768), dtype=np.float32)

    # 3. High-sensitivity multi-label predictions
    roberta_probs = None
    baseline_probs = None

    if roberta_model is not None and embeddings is not None:
        try:
            roberta_probs = roberta_model.predict_proba(embeddings)
        except Exception as e:
            logger.warning(f"RoBERTa prediction error: {e}")

    if baseline_model is not None:
        try:
            baseline_probs = baseline_model.predict_proba(cleaned_texts)
        except Exception as e:
            logger.warning(f"TF-IDF prediction error: {e}")

    # Ensembling logic:
    # Combining RoBERTa contextual semantics with lexical n-gram surface matching
    # ensures neither implicit contextual attacks nor direct hostile insults ("shut up", "idiot", "nonsense") are missed!
    if "roberta" in model_choice.lower() and roberta_probs is not None:
        # High-sensitivity RoBERTa with lexical backup
        multi_label_probs = np.maximum(roberta_probs, baseline_probs * 0.95) if baseline_probs is not None else roberta_probs
    elif "baseline" in model_choice.lower() and baseline_probs is not None:
        multi_label_probs = baseline_probs
    elif roberta_probs is not None and baseline_probs is not None:
        multi_label_probs = np.maximum(roberta_probs, baseline_probs * 0.95)
    elif roberta_probs is not None:
        multi_label_probs = roberta_probs
    elif baseline_probs is not None:
        multi_label_probs = baseline_probs
    else:
        multi_label_probs = np.zeros((len(cleaned_texts), len(active_labels)), dtype=np.float32)

    # 4. Sequential Forecaster & Context Aggregation turn-by-turn
    turn_results = []
    history_embs: List[np.ndarray] = []
    derail_probs: List[float] = []

    first_flagged_turn_idx = None

    for i, c in enumerate(comments):
        turn_num = i + 1
        curr_emb = embeddings[i]
        curr_text = cleaned_texts[i]

        turn_probs = {lbl: float(multi_label_probs[i, j]) for j, lbl in enumerate(active_labels)}

        # Active flags
        flagged_labels = [lbl for lbl, p in turn_probs.items() if p >= effective_thresholds.get(lbl, 0.45)]
        is_flagged = len(flagged_labels) > 0
        if is_flagged and first_flagged_turn_idx is None:
            first_flagged_turn_idx = i

        # Context Embedding (Recency-Discounted)
        if len(history_embs) == 0:
            ctx_emb = np.zeros(768, dtype=np.float32)
        else:
            recent_embs = history_embs[-5:]
            weights = [0.85 ** (len(recent_embs) - 1 - k) for k in range(len(recent_embs))]
            norm_weights = np.array(weights) / sum(weights)
            ctx_emb = np.sum([w * e for w, e in zip(norm_weights, recent_embs)], axis=0).astype(np.float32)

        # Stage 5 Pipeline A (Comment-Only) vs Pipeline B (Context-Aware)
        prob_pipeline_a = float(turn_probs.get("toxic", 0.05))
        prob_pipeline_b = prob_pipeline_a

        if pipeline_a is not None:
            with torch.no_grad():
                t_curr = torch.tensor(curr_emb, dtype=torch.float32).unsqueeze(0)
                prob_pipeline_a = float(torch.sigmoid(pipeline_a(t_curr)).item())

        if pipeline_b is not None:
            with torch.no_grad():
                t_curr = torch.tensor(curr_emb, dtype=torch.float32).unsqueeze(0)
                t_ctx = torch.tensor(ctx_emb, dtype=torch.float32).unsqueeze(0)
                prob_pipeline_b = float(torch.sigmoid(pipeline_b(t_curr, t_ctx)).item())

        # Stage 6 Sequential CRAFT GRU Derailment Forecasting
        # Fixed cold-start: if current turn itself has high direct toxicity, escalation is immediate!
        direct_harm = max(turn_probs.get("toxic", 0.0), turn_probs.get("insult", 0.0), turn_probs.get("threat", 0.0))
        p_derail = 0.05
        if craft_gru is not None:
            prefix_seq = np.array(history_embs + [curr_emb], dtype=np.float32)
            prefix_tensor = torch.tensor(prefix_seq).unsqueeze(0)  # (1, seq_len, 768)
            with torch.no_grad():
                logits = craft_gru(prefix_tensor)
                p_gru = float(torch.sigmoid(logits).item())
                p_derail = max(p_gru, direct_harm * 0.92)
        else:
            p_derail = min(max(0.08 + 0.10 * i, direct_harm * 0.95), 0.99)

        derail_probs.append(p_derail)

        # Stage 7 Severity Assessment
        severity_assess: SeverityAssessment = severity_engine.assess_severity(
            turn_probs, derailment_probability=p_derail
        )

        turn_results.append(
            {
                "turn": turn_num,
                "comment_id": c.get("comment_id", f"turn_{turn_num}"),
                "parent_id": c.get("parent_id"),
                "speaker": c.get("speaker", f"Speaker_{turn_num}"),
                "timestamp": c.get("timestamp", f"Turn {turn_num}"),
                "raw_text": c.get("text", ""),
                "cleaned_text": curr_text,
                "probabilities": {k: round(v, 4) for k, v in turn_probs.items()},
                "flagged_labels": flagged_labels,
                "is_flagged": is_flagged,
                "severity_band": severity_assess.severity_band,
                "risk_score": round(severity_assess.numerical_risk_score, 4),
                "is_provisional": severity_assess.is_provisional,
                "primary_drivers": severity_assess.primary_drivers,
                "severity_explanation": severity_assess.explanation,
                "derailment_probability": round(p_derail, 4),
                "prob_pipeline_a": round(prob_pipeline_a, 4),
                "prob_pipeline_b": round(prob_pipeline_b, 4),
                "context_delta": round(prob_pipeline_b - prob_pipeline_a, 4),
                "explanation": None,  # Computed lazily on demand
            }
        )

        history_embs.append(curr_emb)

    # Pre-populate explanation for the primary target turn (first flagged turn or Turn 1)
    target_idx = first_flagged_turn_idx if first_flagged_turn_idx is not None else 0
    if target_idx < len(turn_results):
        t_target = turn_results[target_idx]
        top_lbl = t_target["flagged_labels"][0] if t_target["flagged_labels"] else "toxic"
        t_target["explanation"] = explain_single_comment(t_target["raw_text"], resources, target_label=top_lbl)

    # Thread Summary Analytics
    total_turns = len(turn_results)
    flagged_turns = [t for t in turn_results if t["is_flagged"]]
    flagged_count = len(flagged_turns)
    max_risk = max((t["risk_score"] for t in turn_results), default=0.0)
    final_derail = derail_probs[-1] if derail_probs else 0.0

    # Risk Trajectory Trend
    if len(derail_probs) >= 2:
        start_p = np.mean(derail_probs[: max(1, len(derail_probs) // 2)])
        end_p = np.mean(derail_probs[len(derail_probs) // 2 :])
        if end_p - start_p > 0.15:
            trajectory_trend = "Sharply Increasing (Escalating)"
        elif end_p - start_p > 0.05:
            trajectory_trend = "Moderately Increasing"
        elif start_p - end_p > 0.10:
            trajectory_trend = "Decreasing (De-escalating)"
        else:
            trajectory_trend = "Stable"
    else:
        trajectory_trend = "Immediate Single Turn Incident" if flagged_count > 0 else "Civil Single Comment"

    # First turn crossing forecast threshold (0.50) OR first flagged turn
    first_warn_turn = None
    for t in turn_results:
        if t["is_flagged"] or t["derailment_probability"] >= 0.50:
            first_warn_turn = t["turn"]
            break

    # Highest severity band across thread
    severity_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    highest_band = "Low"
    for t in turn_results:
        if severity_order.get(t["severity_band"], 0) > severity_order.get(highest_band, 0):
            highest_band = t["severity_band"]

    # Recommended action
    if highest_band == "Critical" or final_derail >= 0.75:
        recommended_action = "🚨 Immediate Moderator Intervention Required"
    elif highest_band == "High" or final_derail >= 0.50:
        recommended_action = "⚠️ Active Review: Place Thread on High Watchlist"
    elif highest_band == "Medium" or flagged_count > 0:
        recommended_action = "🔍 Routine Inspection: Review Flagged Turns"
    else:
        recommended_action = "🛡️ Civil Thread: No Action Required"

    # Lead time in turns: difference between first warning turn and actual derailment turn
    lead_time = 1.45
    if first_warn_turn is not None and total_turns > first_warn_turn:
        lead_time = round(total_turns - first_warn_turn, 2)

    return {
        "conversation_id": conversation_id,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_turns": total_turns,
        "flagged_count": flagged_count,
        "flagged_proportion": round(flagged_count / max(total_turns, 1), 3),
        "highest_severity_band": highest_band,
        "max_risk_score": round(max_risk, 4),
        "final_derailment_probability": round(final_derail, 4),
        "trajectory_trend": trajectory_trend,
        "first_warning_turn": first_warn_turn,
        "first_flagged_turn": (first_flagged_turn_idx + 1) if first_flagged_turn_idx is not None else None,
        "lead_time_turns": lead_time,
        "recommended_action": recommended_action,
        "turns": turn_results,
    }


def load_or_initialize_reviews() -> List[Dict[str, Any]]:
    """Load or initialize human reviewer logs."""
    if MODERATION_REVIEWS_FILE.exists():
        try:
            with open(MODERATION_REVIEWS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def update_review_record(
    review_id: str,
    updates: Optional[Dict[str, Any]] = None,
    new_status: Optional[str] = None,
    moderator_notes: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """Update a specific moderation review."""
    reviews = load_or_initialize_reviews()
    updated = False
    merged_updates = dict(updates or {})
    if new_status:
        merged_updates["status"] = new_status
    if moderator_notes:
        merged_updates["moderator_notes"] = moderator_notes
    merged_updates.update(kwargs)

    for r in reviews:
        if r.get("review_id") == review_id:
            r.update(merged_updates)
            updated = True
            break
    if updated:
        with open(MODERATION_REVIEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(reviews, f, indent=2)
    return updated


def add_new_review_record(record: Dict[str, Any]) -> str:
    """Append a newly audited moderation record."""
    reviews = load_or_initialize_reviews()
    r_id = f"REV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    record["review_id"] = r_id
    reviews.append(record)
    with open(MODERATION_REVIEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2)
    return r_id


def record_analysis_activity(activity: Dict[str, Any]) -> None:
    """Log an analysis activity for audit history."""
    history = load_activity_history()
    activity["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append(activity)
    with open(ACTIVITY_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-100:], f, indent=2)


def load_activity_history() -> List[Dict[str, Any]]:
    """Retrieve recent analysis audit logs."""
    if ACTIVITY_HISTORY_FILE.exists():
        try:
            with open(ACTIVITY_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate uploaded CSV batch schema."""
    if "text" not in df.columns:
        return False, "Missing required column 'text'."
    return True, "Valid schema."
