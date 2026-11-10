"""Master Execution Script: Phase 2 and Phase 3 Implementation.

Orchestrates all subsequent stages (Stages 4 through 10):
- Stage 4: Multi-label classification (TF-IDF Baseline vs RoBERTa Neural Head)
- Stage 5: Context integration (Pipeline A Comment-Only vs Pipeline B Context-Aware)
- Stage 6: Conversation derailment forecasting (Prefix Baseline vs CRAFT GRU)
- Stage 7: Severity grading (Supervised severe_toxic + Provisional Risk Estimator)
- Stage 8: Explainability (Token occlusion attribution & contextual support)
- Stage 9 & 10: Unified evaluation, cross-dataset validation, and pragmatic error analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import (
    EXTENDED_LABELS,
    TARGET_LABELS,
    RoBERTaMultiLabelClassifier,
    TfidfMultiLabelBaseline,
    compute_multilabel_metrics,
    load_jigsaw_data,
)
from src.context_encoder import ConversationHistoryAggregator, run_context_ablation
from src.data_loader import ConversationDataLoader
from src.embeddings import CommentEmbeddingGenerator
from src.evaluator import UnifiedEvaluationRunner
from src.explainability import CommentExplainer
from src.forecasting import ConversationPrefixGenerator, DerailmentForecastingEngine
from src.preprocessing import ConversationPreprocessor
from src.severity import SeverityGradingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase2Phase3Pipeline")


def run_full_pipeline(
    jigsaw_path: str = r"C:\Users\ASUS\Downloads\NLP_Project_Datasets\Jigsaw\train.csv.gz",
    corpora_zip_path: str = r"C:\Users\ASUS\Downloads\NLP Dataset\corpora_csv.zip",
    output_dir: str = "outputs",
    sample_size_cls: int = 5000,
    num_threads_context: int = 150,
    epochs_cls: int = 8,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Executes the complete Phase 2 and Phase 3 research implementation."""
    total_start = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING PHASE 2 & 3: CONTEXT-AWARE INAPPROPRIATE COMMENTS SCANNER")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # STAGE 4: Multi-Label Inappropriate Comment Classification
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 4: MULTI-LABEL CLASSIFICATION (BASELINE VS ROBERTA)")
    logger.info("=" * 50)

    # Use the 5 target labels + severe_toxic for Stage 7 severity integration
    active_labels = EXTENDED_LABELS
    train_df, val_df, test_df = load_jigsaw_data(
        data_path=jigsaw_path,
        sample_size=sample_size_cls,
        random_state=42,
        labels=active_labels,
    )

    y_train = train_df[active_labels].values
    y_val = val_df[active_labels].values
    y_test = test_df[active_labels].values

    # Model 1: TF-IDF + Logistic Regression Baseline
    logger.info("\n--- Training TF-IDF Baseline ---")
    baseline = TfidfMultiLabelBaseline(labels=active_labels, max_features=8000, C=1.0)
    baseline.fit(train_df["cleaned_text"].tolist(), y_train, val_df["cleaned_text"].tolist(), y_val)
    baseline.save(models_dir / "baseline_tfidf")

    base_probs = baseline.predict_proba(test_df["cleaned_text"].tolist())
    base_preds = baseline.predict(test_df["cleaned_text"].tolist())
    base_metrics = compute_multilabel_metrics(y_test, base_preds, base_probs, active_labels)

    logger.info(
        f"Baseline Results: Macro-F1={base_metrics['macro_f1']} | Micro-F1={base_metrics['micro_f1']} | "
        f"Hamming Loss={base_metrics['hamming_loss']}"
    )

    # Model 2: RoBERTa / Sentence-BERT Representation + Neural Head
    logger.info("\n--- Training RoBERTa-Backed Multi-Label Neural Classifier ---")
    emb_gen = CommentEmbeddingGenerator(
        model_name="sentence-transformers/all-distilroberta-v1",
        device=device,
        batch_size=batch_size,
    )

    logger.info("Encoding text samples into 768-dim RoBERTa embeddings...")
    X_train_emb = emb_gen.encode_batch(train_df["cleaned_text"].tolist())
    X_val_emb = emb_gen.encode_batch(val_df["cleaned_text"].tolist())
    X_test_emb = emb_gen.encode_batch(test_df["cleaned_text"].tolist())

    roberta_cls = RoBERTaMultiLabelClassifier(
        labels=active_labels,
        embedding_dim=768,
        hidden_dim=256,
        learning_rate=1e-3,
        epochs=epochs_cls,
        batch_size=batch_size,
        device=device,
    )
    roberta_cls.fit(X_train_emb, y_train, X_val_emb, y_val)
    roberta_cls.save(models_dir / "roberta_classifier")

    roberta_probs = roberta_cls.predict_proba(X_test_emb)
    roberta_preds = roberta_cls.predict(X_test_emb)
    roberta_metrics = compute_multilabel_metrics(y_test, roberta_preds, roberta_probs, active_labels)

    logger.info(
        f"RoBERTa Results: Macro-F1={roberta_metrics['macro_f1']} | Micro-F1={roberta_metrics['micro_f1']} | "
        f"Hamming Loss={roberta_metrics['hamming_loss']}"
    )

    stage4_results = {
        "labels": active_labels,
        "sample_size": sample_size_cls,
        "baseline_tfidf": base_metrics,
        "roberta_neural_head": roberta_metrics,
    }

    # -------------------------------------------------------------------------
    # STAGE 5: Context Integration (Pipeline A vs Pipeline B)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 5: CONTEXT INTEGRATION (COMMENT-ONLY VS CONTEXT-AWARE)")
    logger.info("=" * 50)

    # Load Wikipedia conversational threads
    logger.info(f"Loading conversational data from {corpora_zip_path}...")
    z = zipfile.ZipFile(corpora_zip_path, "r")
    with z.open("csv/wiki_conversations.csv") as f:
        wiki_convs = pd.read_csv(f)
    with z.open("csv/wiki_utterances.csv") as f:
        wiki_utts = pd.read_csv(f)

    # Clean text
    preproc = ConversationPreprocessor()
    wiki_utts["cleaned_text"] = wiki_utts["text"].fillna("").apply(preproc.clean_text)

    # Filter section headers and select representative conversation subset
    valid_conv_ids = wiki_convs["conversation_id"].dropna().unique()[:num_threads_context]
    sub_utts = wiki_utts[wiki_utts["conversation_id"].isin(valid_conv_ids)].copy()

    # Pre-encode unique utterances to embeddings
    logger.info(f"Pre-encoding {len(sub_utts)} conversational comments for context modeling...")
    unique_texts = sub_utts["cleaned_text"].tolist()
    sub_embs = emb_gen.encode_batch(unique_texts)
    embeddings_dict = {uid: emb for uid, emb in zip(sub_utts["id"].tolist(), sub_embs)}

    # Build causal contextual features per thread
    aggregator = ConversationHistoryAggregator(max_history_turns=5, decay_factor=0.85)
    all_context_samples = []

    for cid in valid_conv_ids:
        c_utts = sub_utts[sub_utts["conversation_id"] == cid].sort_values("timestamp").to_dict("records")
        if c_utts:
            c_samples = aggregator.build_context_features(c_utts, embeddings_dict)
            all_context_samples.extend(c_samples)

    logger.info(f"Constructed {len(all_context_samples)} causal contextual samples.")

    # Split strictly by conversation ID (70% train, 15% val, 15% test)
    conv_list = list(valid_conv_ids)
    np.random.seed(42)
    np.random.shuffle(conv_list)
    n_train = int(len(conv_list) * 0.70)
    n_val = int(len(conv_list) * 0.15)

    train_cids = set(conv_list[:n_train])
    val_cids = set(conv_list[n_train : n_train + n_val])
    test_cids = set(conv_list[n_train + n_val :])

    ctx_train = [s for s in all_context_samples if s["conversation_id"] in train_cids]
    ctx_val = [s for s in all_context_samples if s["conversation_id"] in val_cids]
    ctx_test = [s for s in all_context_samples if s["conversation_id"] in test_cids]

    stage5_results = run_context_ablation(ctx_train, ctx_val, ctx_test, out_dir / "stage5_context")

    # -------------------------------------------------------------------------
    # STAGE 6: Conversation Derailment Forecasting (CRAFT-Inspired)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 6: CONVERSATION DERAILMENT FORECASTING (CRAFT-INSPIRED)")
    logger.info("=" * 50)

    prefix_gen = ConversationPrefixGenerator(min_prefix=1, max_prefix=6)
    train_prefixes = []
    val_prefixes = []
    test_prefixes = []

    conv_target_map = dict(zip(wiki_convs["conversation_id"], wiki_convs["conversation_has_personal_attack"]))

    for cid in valid_conv_ids:
        c_utts = sub_utts[sub_utts["conversation_id"] == cid].sort_values("timestamp").to_dict("records")
        if not c_utts:
            continue
        will_derail = bool(conv_target_map.get(cid, False))
        split = "train" if cid in train_cids else ("val" if cid in val_cids else "test")
        prefixes = prefix_gen.generate_prefixes_from_thread(c_utts, will_derail, split=split)

        if split == "train":
            train_prefixes.extend(prefixes)
        elif split == "val":
            val_prefixes.extend(prefixes)
        else:
            test_prefixes.extend(prefixes)

    logger.info(
        f"Forecasting prefixes generated: Train={len(train_prefixes)}, Val={len(val_prefixes)}, Test={len(test_prefixes)}"
    )

    forecaster = DerailmentForecastingEngine(embedding_dim=768, device=device)
    forecaster.fit(train_prefixes, val_prefixes, embeddings_dict, epochs=8, batch_size=32)
    stage6_results = forecaster.evaluate(test_prefixes, embeddings_dict, out_dir / "stage6_forecasting")

    # -------------------------------------------------------------------------
    # STAGE 7: Severity Grading & Provisional Risk Estimation
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 7: SEVERITY GRADING AND PROVISIONAL RISK ASSESSMENT")
    logger.info("=" * 50)

    severity_engine = SeverityGradingEngine()

    # Demonstration of severity assessment on high, medium, and low samples
    test_samples_severity = [
        ({"toxic": 0.05, "insult": 0.02, "threat": 0.01, "severe_toxic": 0.01, "identity_hate": 0.01}, 0.12),
        ({"toxic": 0.72, "insult": 0.65, "threat": 0.02, "severe_toxic": 0.05, "identity_hate": 0.04}, 0.45),
        ({"toxic": 0.94, "insult": 0.88, "threat": 0.68, "severe_toxic": 0.55, "identity_hate": 0.45}, 0.82),
    ]

    severity_demo = []
    for probs_dict, d_prob in test_samples_severity:
        assessment = severity_engine.assess_severity(probs_dict, derailment_probability=d_prob)
        severity_demo.append(
            {
                "input_probabilities": probs_dict,
                "derailment_probability": d_prob,
                "severity_band": assessment.severity_band,
                "numerical_risk_score": assessment.numerical_risk_score,
                "primary_drivers": assessment.primary_drivers,
                "is_provisional": assessment.is_provisional,
            }
        )

    logger.info("Demonstrated Severity Assessments:\n" + json.dumps(severity_demo, indent=2))

    # -------------------------------------------------------------------------
    # STAGE 8: Explainability (Token Occlusion & Context Support)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 8: EXPLAINABILITY & CAUSAL ATTRIBUTION")
    logger.info("=" * 50)

    explainer = CommentExplainer(predict_fn=baseline.predict_proba)
    sample_text = "You are a complete idiot and should leave this forum immediately."
    explanation = explainer.explain_text_occlusion(sample_text, label_index=0, label_name="toxic", top_k=4)

    logger.info(f"Sample Explanation for: '{sample_text}'")
    logger.info(f"  Base Toxic Probability: {explanation.base_probability}")
    logger.info(f"  Top Hostile Tokens: {explanation.top_hostile_tokens}")
    logger.info(f"  Faithfulness Probability Drop: {explanation.faithfulness_drop}")

    # -------------------------------------------------------------------------
    # STAGE 9 & 10: Unified Evaluation, Cross-Dataset & Pragmatic Analysis
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 9 & 10: CROSS-DATASET VALIDATION & PRAGMATIC ERROR ANALYSIS")
    logger.info("=" * 50)

    eval_runner = UnifiedEvaluationRunner(output_dir=out_dir)

    # Cross-dataset evaluation on Wikipedia
    cross_dataset_metrics = eval_runner.run_cross_dataset_evaluation(
        classifier=baseline,
        wiki_utterances_df=sub_utts,
        is_embedding_model=False,
    )

    # Pragmatic phenomena evaluation
    pragmatic_eval = eval_runner.evaluate_pragmatic_phenomena(
        classifier=baseline,
        is_embedding_model=False,
    )

    full_evaluation_package = {
        "stage4_multilabel": stage4_results,
        "stage5_context": stage5_results,
        "stage6_forecasting": stage6_results,
        "stage7_severity_demo": severity_demo,
        "stage8_explainability_sample": {
            "text": explanation.text,
            "target_label": explanation.target_label,
            "base_probability": explanation.base_probability,
            "top_hostile_tokens": explanation.top_hostile_tokens,
            "faithfulness_drop": explanation.faithfulness_drop,
            "methodology": explanation.methodology,
        },
        "stage10_cross_dataset": cross_dataset_metrics,
        "pragmatic_phenomena_evaluation": pragmatic_eval,
        "execution_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_execution_seconds": round(time.time() - total_start, 2),
            "device": device,
        },
    }

    # Save full machine-readable results
    eval_json_path = out_dir / "evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(full_evaluation_package, f, indent=2)

    # Export consolidated summary CSV
    summary_df = eval_runner.export_summary_table(full_evaluation_package)

    total_time = round(time.time() - total_start, 2)
    logger.info("=" * 80)
    logger.info(f"PHASE 2 & 3 PIPELINE COMPLETED SUCCESSFULLY IN {total_time}s")
    logger.info(f"Results saved to: {eval_json_path}")
    logger.info(f"Summary table saved to: {out_dir / 'evaluation_metrics_summary.csv'}")
    logger.info("=" * 80)

    return full_evaluation_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Execution Pipeline: NLP Project Phase 2 & 3")
    parser.add_argument(
        "--jigsaw-path",
        type=str,
        default=r"C:\Users\ASUS\Downloads\NLP_Project_Datasets\Jigsaw\train.csv.gz",
        help="Path to Jigsaw multi-label train.csv.gz",
    )
    parser.add_argument(
        "--corpora-path",
        type=str,
        default=r"C:\Users\ASUS\Downloads\NLP Dataset\corpora_csv.zip",
        help="Path to corpora_csv.zip containing Wikipedia and CMV threads",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Target outputs directory",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Sample size for multi-label classification training (default: 5000 for fast CPU execution)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=120,
        help="Number of conversation threads for context and forecasting modeling",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training epochs for neural heads",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for training and inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )

    args = parser.parse_args()

    run_full_pipeline(
        jigsaw_path=args.jigsaw_path,
        corpora_zip_path=args.corpora_path,
        output_dir=args.output_dir,
        sample_size_cls=args.sample_size,
        num_threads_context=args.num_threads,
        epochs_cls=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

