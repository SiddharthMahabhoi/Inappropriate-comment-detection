"""Main Pipeline Execution Script - Context-Aware Inappropriate Comments Scanner (Phase 1).

Executes the three authoritative Phase 1 stages:
  Stage 1: Input Conversation Thread (Load & Reconstruct CMV reply trees)
  Stage 2: Preprocessing (URLs, mentions, whitespace, preserve punctuation/emojis/casing)
  Stage 3: Text Representation (Pretrained RoBERTa / SBERT embeddings, mean pooling)

Usage (PowerShell):
  python src/pipeline.py --sample-size 50
  python src/pipeline.py --sample-size 500 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure utf-8 output on Windows PowerShell consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to python path for standalone runs
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ConversationDataLoader
from src.embeddings import CommentEmbeddingGenerator, check_system_resources
from src.preprocessing import ConversationPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase1Pipeline")


def run_phase1_pipeline(
    data_path: str,
    output_dir: str,
    sample_size: int = 50,
    model_name: str = "sentence-transformers/all-distilroberta-v1",
    device: str = "auto",
    batch_size: int = 16,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Execute complete Phase 1 pipeline across all three stages.

    Args:
        data_path: Path to corpora_csv.zip or extracted CMV data directory.
        output_dir: Target directory for artifacts.
        sample_size: Number of comment records to process.
        model_name: Sentence-BERT or RoBERTa pretrained model.
        device: 'auto', 'cpu', or 'cuda'.
        batch_size: Batch size for embedding inference.
        force_recompute: Whether to bypass cached embeddings.

    Returns:
        Dictionary containing summary statistics, file paths, and verification results.
    """
    logger.info("=" * 70)
    logger.info("STARTING PHASE 1 PIPELINE: CONTEXT-AWARE INAPPROPRIATE COMMENTS SCANNER")
    logger.info("=" * 70)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # STAGE 1: Input Conversation Thread Loading & Reconstruction
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 1] Loading Conversation Threads and Reply Structure ---")
    stage1_start = time.time()

    loader = ConversationDataLoader(data_path=data_path)
    convs_df, utts_df = loader.load_raw_data(sample_limit=None)

    total_convs = len(convs_df)
    total_utts = len(utts_df)
    logger.info(
        f"Successfully loaded CMV dataset: {total_convs} conversations, {total_utts} utterances."
    )
    logger.info(f"Conversations schema: {list(convs_df.columns)}")
    logger.info(f"Utterances schema: {list(utts_df.columns)}")

    # Reconstruct processed comment records with reply depths
    records = loader.load_processed_records(sample_limit=sample_size)
    logger.info(f"Reconstructed and selected {len(records)} comments across conversation threads.")

    # Select an example thread to visualize conversation reconstruction
    example_conv_id = records[0].conversation_id if records else convs_df.iloc[0]["conversation_id"]
    example_thread = loader.reconstruct_thread(example_conv_id)

    reconstruction_text = ""
    reconstruction_json_path = out_path / "conversation_reconstruction_sample.json"
    reconstruction_txt_path = out_path / "conversation_reconstruction_sample.txt"

    if example_thread:
        reconstruction_text = example_thread.format_tree()
        with open(reconstruction_txt_path, "w", encoding="utf-8") as f:
            f.write(reconstruction_text)

        thread_dict = {
            "conversation_id": example_thread.conversation_id,
            "split": example_thread.split,
            "has_removed_comment": example_thread.has_removed_comment,
            "comment_count": example_thread.comment_count,
            "comments": [c.to_dict() for c in example_thread.comments],
        }
        with open(reconstruction_json_path, "w", encoding="utf-8") as f:
            json.dump(thread_dict, f, indent=2)

        logger.info(f"Reconstructed thread example saved to {reconstruction_txt_path}")
        print("\n" + reconstruction_text + "\n")

    stage1_time = round(time.time() - stage1_start, 2)

    # -------------------------------------------------------------
    # STAGE 2: Preprocessing
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 2] Preprocessing Comments (Preserving Conversational Signals) ---")
    stage2_start = time.time()

    preprocessor = ConversationPreprocessor(
        url_placeholder="[URL]",
        user_placeholder="[USER]",
        subreddit_placeholder="[SUBREDDIT]",
        normalize_unicode=True,
        preserve_casing=True,
    )

    cleaned_records: List[Dict[str, Any]] = []
    for rec in records:
        cleaned_dict = preprocessor.preprocess_record(rec.to_dict())
        cleaned_records.append(cleaned_dict)

    cleaned_df = pd.DataFrame(cleaned_records)
    cleaned_csv_path = out_path / "cleaned_comments_sample.csv"
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    logger.info(f"Cleaned comments saved to {cleaned_csv_path} (Shape: {cleaned_df.shape})")

    # Display comparison for 2 sample comments
    sample_preview = cleaned_df[["comment_id", "original_text", "cleaned_text"]].head(2)
    logger.info("Sample preprocessing comparison:\n" + str(sample_preview.to_dict(orient="records")))

    stage2_time = round(time.time() - stage2_start, 2)

    # -------------------------------------------------------------
    # STAGE 3: Text Representation (RoBERTa / SBERT Embeddings)
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 3] Generating Pretrained Text Embeddings ---")
    stage3_start = time.time()

    # Resource check
    sys_res = check_system_resources()
    logger.info(
        f"System resource audit: Device={sys_res['device_name']}, CUDA={sys_res['cuda_available']}, "
        f"FreeDisk={sys_res['free_disk_gb']}GB, PyTorch={sys_res['torch_version']}"
    )

    emb_gen = CommentEmbeddingGenerator(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    embeddings, meta_df, manifest = emb_gen.generate_embeddings(
        records=cleaned_records,
        output_dir=out_path,
        force_recompute=force_recompute,
    )

    stage3_time = round(time.time() - stage3_start, 2)

    # -------------------------------------------------------------
    # VERIFICATION & INTEGRITY CHECKS
    # -------------------------------------------------------------
    logger.info("\n--- [VERIFICATION] Verifying Phase 1 Outputs & Numerical Integrity ---")

    npy_path = out_path / "comment_embeddings.npy"
    meta_path = out_path / "embedding_metadata.csv"
    manifest_path = out_path / "embedding_manifest.json"

    # 1. File existence
    assert npy_path.exists(), "Embedding .npy file missing!"
    assert meta_path.exists(), "Metadata CSV missing!"
    assert manifest_path.exists(), "Manifest JSON missing!"
    assert cleaned_csv_path.exists(), "Cleaned CSV missing!"
    assert reconstruction_txt_path.exists(), "Reconstruction text missing!"

    # 2. Reload and verify embeddings
    loaded_embs = np.load(npy_path)
    loaded_meta = pd.read_csv(meta_path)

    assert loaded_embs.shape[0] == len(records), (
        f"Embedding rows ({loaded_embs.shape[0]}) != records count ({len(records)})"
    )
    assert loaded_embs.shape[1] == manifest["embedding_dimensions"], (
        f"Embedding dimension ({loaded_embs.shape[1]}) != manifest dimension ({manifest['embedding_dimensions']})"
    )
    assert np.isfinite(loaded_embs).all(), "Non-finite numbers found in embeddings array!"
    assert len(loaded_meta) == loaded_embs.shape[0], "1-to-1 mapping row count mismatch!"

    logger.info("Verification PASSED:")
    logger.info(f"  - Embeddings matrix shape: {loaded_embs.shape} (dtype: {loaded_embs.dtype})")
    logger.info(f"  - Finite values verified: 100% finite (no NaN, no Inf)")
    logger.info(f"  - 1-to-1 comment mapping verified: {len(loaded_meta)} rows in {meta_path}")

    summary = {
        "dataset_selected": "Reddit ChangeMyView (corpora_csv.zip)",
        "total_corpus_conversations": total_convs,
        "total_corpus_utterances": total_utts,
        "sample_records_processed": len(records),
        "embedding_model": manifest["model_name"],
        "embedding_dimensions": manifest["embedding_dimensions"],
        "device_used": manifest["device_used"],
        "stage1_time_seconds": stage1_time,
        "stage2_time_seconds": stage2_time,
        "stage3_time_seconds": stage3_time,
        "total_pipeline_time_seconds": round(stage1_time + stage2_time + stage3_time, 2),
        "output_files": {
            "cleaned_csv": str(cleaned_csv_path),
            "embeddings_npy": str(npy_path),
            "embedding_metadata_csv": str(meta_path),
            "manifest_json": str(manifest_path),
            "reconstruction_txt": str(reconstruction_txt_path),
            "reconstruction_json": str(reconstruction_json_path),
        },
    }

    summary_file = out_path / "phase1_run_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nExecution summary saved to {summary_file}")
    logger.info("=" * 70)
    logger.info("PHASE 1 IMPLEMENTATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    return summary


def main() -> None:
    """CLI entry point for Phase 1 pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Pipeline: Context-Aware Inappropriate Comments Scanner"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=r"C:\Users\ASUS\Downloads\NLP Dataset\corpora_csv.zip",
        help="Path to corpora_csv.zip or directory containing CMV CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Directory to save generated outputs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of comments to process (default: 50). Use 0 or negative for all.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-distilroberta-v1",
        help="Pretrained Sentence-BERT / RoBERTa model identifier.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for embedding inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of embeddings even if cached outputs exist.",
    )

    args = parser.parse_args()

    sample_size = None if args.sample_size <= 0 else args.sample_size

    run_phase1_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=sample_size or 50,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
