"""Stage 3: Text Representation - Context-Aware Inappropriate Comments Scanner.

This module generates comment-level text embeddings using a pretrained
RoBERTa-based or Sentence-BERT-compatible encoder (e.g., all-distilroberta-v1
or all-MiniLM-L6-v2), applying proper tokenization, attention masks, and
mean pooling in batched inference mode.

Outputs:
- .npy file: numerical matrix of comment embeddings (shape: [N, embedding_dim])
- metadata.csv: 1-to-1 mapping linking each row to comment_id, conversation_id, etc.
- manifest.json: records model architecture, dimensions, device, and run configuration.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_system_resources() -> Dict[str, Any]:
    """Inspect hardware resources, CUDA availability, and disk space."""
    import torch

    free_gb = 0.0
    try:
        total, used, free = shutil.disk_usage(os.getcwd())
        free_gb = round(free / (1024**3), 2)
    except Exception:
        pass

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "free_disk_gb": free_gb,
        "torch_version": torch.__version__,
    }


class CommentEmbeddingGenerator:
    """Generates and manages comment-level text embeddings using RoBERTa / SBERT encoders."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-distilroberta-v1",
        device: str = "auto",
        batch_size: int = 32,
        max_seq_length: int = 128,
    ):
        """Initialize embedding generator.

        Args:
            model_name: HuggingFace or Sentence-BERT model identifier.
                        Default is 'sentence-transformers/all-distilroberta-v1' (RoBERTa architecture).
            device: 'auto', 'cuda', or 'cpu'.
            batch_size: Mini-batch size for inference.
            max_seq_length: Maximum token length for truncation.
        """
        import torch

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Initializing CommentEmbeddingGenerator with model='{self.model_name}' on device='{self.device}'"
        )

        self._model = None
        self._tokenizer = None
        self._use_sbert = False
        self.embedding_dim: Optional[int] = None

    def _load_model(self) -> None:
        """Lazy-load the pretrained encoder and tokenizer."""
        if self._model is not None:
            return

        import torch

        # Attempt to load using sentence-transformers first for optimal pooling/caching
        try:
            from sentence_transformers import SentenceTransformer

            try:
                # Try loading from local cache first to avoid slow online metadata handshakes
                self._model = SentenceTransformer(
                    self.model_name, device=self.device, local_files_only=True
                )
            except Exception:
                logger.info(f"Loading model via SentenceTransformer: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)

            self._model.max_seq_length = self.max_seq_length
            self._model.eval()
            self._use_sbert = True

            # Determine vector dimension
            test_emb = self._model.encode(["test"], convert_to_numpy=True)
            self.embedding_dim = int(test_emb.shape[1])
            logger.info(f"SentenceTransformer loaded successfully. Embedding dim: {self.embedding_dim}")
            return
        except Exception as e:
            logger.warning(
                f"SentenceTransformer load failed ({e}). Falling back to transformers AutoModel + mean pooling."
            )

        # Fallback to direct transformers AutoModel + AutoTokenizer with mean pooling
        from transformers import AutoModel, AutoTokenizer

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
        except Exception:
            logger.info(f"Loading model via HuggingFace transformers: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)

        self._model.eval()
        self._use_sbert = False

        if hasattr(self._model.config, "hidden_size"):
            self.embedding_dim = self._model.config.hidden_size
        else:
            self.embedding_dim = 768
        logger.info(f"HuggingFace model loaded successfully. Embedding dim: {self.embedding_dim}")

    @staticmethod
    def _mean_pooling(model_output: Any, attention_mask: Any) -> Any:
        """Apply attention-mask-weighted mean pooling over token embeddings.

        This directly implements Sentence-BERT (Reimers & Gurevych, 2019) mean pooling.
        """
        import torch

        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into numerical embeddings.

        Args:
            texts: List of comment text strings.

        Returns:
            2D numpy float32 array of shape (len(texts), embedding_dim).
        """
        self._load_model()
        import torch

        # Sanitize empty or null strings to avoid tokenizer assertion errors
        cleaned_inputs = [t if (t and t.strip()) else "[EMPTY]" for t in texts]

        if self._use_sbert:
            with torch.no_grad():
                embeddings = self._model.encode(
                    cleaned_inputs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            return embeddings.astype(np.float32)

        # Direct HuggingFace transformers implementation with chunked batching
        assert self._tokenizer is not None
        assert self._model is not None

        all_embs = []
        for i in range(0, len(cleaned_inputs), self.batch_size):
            batch_chunk = cleaned_inputs[i : i + self.batch_size]
            encoded_input = self._tokenizer(
                batch_chunk,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                model_output = self._model(**encoded_input)
                sentence_embeddings = self._mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            all_embs.append(sentence_embeddings.cpu().numpy().astype(np.float32))

        return np.vstack(all_embs)

    def generate_embeddings(
        self,
        records: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
        """Generate embeddings for a list of comment records and persist them.

        Args:
            records: List of dictionaries containing 'cleaned_text', 'comment_id', 'conversation_id', etc.
            output_dir: Directory where .npy, metadata.csv, and manifest.json will be saved.
            force_recompute: If True, recompute even if existing files are present.

        Returns:
            Tuple of (embeddings_array, metadata_df, manifest_dict).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        npy_path = out_path / "comment_embeddings.npy"
        meta_path = out_path / "embedding_metadata.csv"
        manifest_path = out_path / "embedding_manifest.json"

        # Check for existing valid outputs for resumability
        if not force_recompute and npy_path.exists() and meta_path.exists() and manifest_path.exists():
            try:
                existing_meta = pd.read_csv(meta_path)
                if len(existing_meta) == len(records):
                    logger.info(
                        f"Found existing embeddings matching {len(records)} records in {out_path}. Loading cached output."
                    )
                    existing_embs = np.load(npy_path)
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    return existing_embs, existing_meta, manifest
            except Exception as e:
                logger.warning(f"Could not load cached embeddings ({e}); recomputing.")

        self._load_model()
        logger.info(f"Generating embeddings for {len(records)} comments in batches of {self.batch_size}...")

        all_embeddings: List[np.ndarray] = []
        meta_rows: List[Dict[str, Any]] = []

        start_time = time.time()
        for i in range(0, len(records), self.batch_size):
            batch_records = records[i : i + self.batch_size]
            batch_texts = [r.get("cleaned_text", r.get("original_text", "")) for r in batch_records]

            batch_embs = self.encode_batch(batch_texts)

            # Validate numerical properties: finite numbers, correct dimensions
            if not np.isfinite(batch_embs).all():
                raise ValueError(f"Batch {i // self.batch_size} produced non-finite embedding values!")

            all_embeddings.append(batch_embs)

            for idx, r in enumerate(batch_records):
                text_snippet = r.get("cleaned_text", "")
                if len(text_snippet) > 100:
                    text_snippet = text_snippet[:97] + "..."

                meta_rows.append(
                    {
                        "row_index": i + idx,
                        "comment_id": r.get("comment_id", ""),
                        "conversation_id": r.get("conversation_id", ""),
                        "speaker": r.get("speaker", ""),
                        "reply_to": r.get("reply_to", ""),
                        "timestamp": r.get("timestamp", 0.0),
                        "depth": r.get("depth", 0),
                        "split": r.get("split", "train"),
                        "text_length": len(r.get("cleaned_text", "")),
                        "cleaned_text_snippet": text_snippet,
                    }
                )

        final_embeddings = np.vstack(all_embeddings)
        duration = round(time.time() - start_time, 2)
        metadata_df = pd.DataFrame(meta_rows)

        # Save files
        np.save(npy_path, final_embeddings)
        metadata_df.to_csv(meta_path, index=False)

        manifest = {
            "model_name": self.model_name,
            "architecture": "RoBERTa / Sentence-BERT",
            "embedding_dimensions": int(final_embeddings.shape[1]),
            "num_comments": int(final_embeddings.shape[0]),
            "device_used": self.device,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "pooling_method": "mean_pooling_with_attention_mask",
            "generation_time_seconds": duration,
            "finite_values_verified": True,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Successfully saved embeddings: shape={final_embeddings.shape} to {npy_path} in {duration}s"
        )

        return final_embeddings, metadata_df, manifest
