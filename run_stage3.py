"""Run Stage 3 only on an existing cleaned-comments CSV.

Example (PowerShell):
  python src/run_stage3.py --input outputs/cleaned_comments_sample.csv --output-dir outputs --device cpu
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
from src.embeddings import CommentEmbeddingGenerator, check_system_resources

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
REQUIRED = {"comment_id", "conversation_id", "cleaned_text"}

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pretrained embeddings from cleaned comments CSV")
    parser.add_argument("--input", default="outputs/cleaned_comments_sample.csv", help="CSV created by Stage 2")
    parser.add_argument("--output-dir", default="outputs", help="Directory for Stage 3 artifacts")
    parser.add_argument("--model-name", default="sentence-transformers/all-distilroberta-v1", help="Pretrained Hugging Face/Sentence-Transformers model")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Stage 2 CSV not found: {input_path}. Run Stage 1+2 first or pass --input.")
    if args.batch_size < 1 or args.max_seq_length < 1:
        raise ValueError("--batch-size and --max-seq-length must be positive integers")
    df = pd.read_csv(input_path).fillna({"cleaned_text": "", "original_text": ""})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input CSV contains no comments; cannot generate embeddings")
    records = df.to_dict(orient="records")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.info("Resource check: %s", json.dumps(check_system_resources()))
    logging.info("Encoding %d comments with %s", len(records), args.model_name)
    generator = CommentEmbeddingGenerator(model_name=args.model_name, device=args.device,
        batch_size=args.batch_size, max_seq_length=args.max_seq_length)
    embeddings, metadata, manifest = generator.generate_embeddings(records, out, args.force_recompute)
    expected = (len(df), int(manifest["embedding_dimensions"]))
    if embeddings.shape != expected:
        raise RuntimeError(f"Embedding shape {embeddings.shape} did not match expected {expected}")
    if len(metadata) != len(df) or not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding integrity check failed: row mapping or finite-value check failed")
    if metadata["comment_id"].astype(str).tolist() != df["comment_id"].astype(str).tolist():
        raise RuntimeError("Metadata comment_id order does not match the input CSV")
    manifest["input_csv"] = str(input_path.resolve())
    manifest["input_rows"] = int(len(df))
    manifest["row_mapping_verified"] = True
    manifest["finite_values_verified"] = True
    with (out / "embedding_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    summary = {"status": "success", "input_rows": int(len(df)),
        "embedding_shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
        "all_values_finite": bool(np.isfinite(embeddings).all()), "row_mapping_verified": True,
        "model_name": manifest["model_name"], "device_used": manifest["device_used"],
        "outputs": [str((out / n).resolve()) for n in ["comment_embeddings.npy", "embedding_metadata.csv", "embedding_manifest.json"]]}
    with (out / "stage3_verification.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
