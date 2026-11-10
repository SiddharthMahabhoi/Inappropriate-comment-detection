"""Stage 4: Multi-label Inappropriate-Comment Classification.

Implements:
1. Baseline: TF-IDF + One-vs-Rest Logistic Regression with class-weight balancing
2. Neural Model: RoBERTa / Sentence-BERT representation + PyTorch Multi-Label Head with BCEWithLogitsLoss and positive weighting
3. Per-class validation threshold tuning
4. Detailed evaluation metrics: per-label Precision/Recall/F1, Macro-F1, Micro-F1, Hamming Loss, confusion statistics
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from src.preprocessing import ConversationPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default 5 authoritative target labels from presentation (+ optional severe_toxic)
TARGET_LABELS = ["toxic", "insult", "threat", "obscene", "identity_hate"]
EXTENDED_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_jigsaw_data(
    data_path: Union[str, Path] = r"C:\Users\ASUS\Downloads\NLP_Project_Datasets\Jigsaw\train.csv.gz",
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
    labels: List[str] = TARGET_LABELS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Jigsaw multi-label toxic comment dataset and create reproducible splits.

    Stratifies or samples while preserving minority class occurrences.

    Args:
        data_path: Path to Jigsaw train.csv.gz or train.csv.
        sample_size: Number of records to sample for lightweight training. None for all.
        random_state: Random seed for reproducibility.
        labels: List of target label columns.

    Returns:
        Tuple of (train_df, val_df, test_df) with cleaned text.
    """
    logger.info(f"Loading multi-label dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Ensure all required labels are present
    for lbl in labels:
        if lbl not in df.columns:
            raise KeyError(f"Expected label '{lbl}' not found in dataset columns: {df.columns.tolist()}")

    # Preprocess text using Stage 2 preprocessor
    preprocessor = ConversationPreprocessor()
    df["original_text"] = df["comment_text"].fillna("").astype(str)
    
    # Stratified/balanced subsample if sample_size is requested
    if sample_size and sample_size < len(df):
        # Oversample harmful rows to preserve rare classes (threat, identity_hate)
        any_harm = df[labels].sum(axis=1) > 0
        harm_df = df[any_harm]
        clean_df = df[~any_harm]
        
        # Take all or significant portion of harmful comments
        harm_take = min(len(harm_df), int(sample_size * 0.4))
        clean_take = sample_size - harm_take
        
        harm_sample = harm_df.sample(n=harm_take, random_state=random_state)
        clean_sample = clean_df.sample(n=clean_take, random_state=random_state)
        df_sampled = pd.concat([harm_sample, clean_sample]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        df_sampled = df.copy()

    df_sampled["cleaned_text"] = df_sampled["original_text"].apply(preprocessor.clean_text)

    # 70% Train, 15% Validation, 15% Test
    train_df, test_val_df = train_test_split(df_sampled, test_size=0.30, random_state=random_state)
    val_df, test_df = train_test_split(test_val_df, test_size=0.50, random_state=random_state)

    logger.info(
        f"Data partitioned: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} | Total={len(df_sampled)}"
    )
    for lbl in labels:
        pos_pct = train_df[lbl].mean() * 100
        logger.info(f"  Train label '{lbl}': {train_df[lbl].sum()} positive ({pos_pct:.2f}%)")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class TfidfMultiLabelBaseline:
    """Reproducible Baseline: TF-IDF feature extraction + OneVsRest Logistic Regression."""

    def __init__(
        self,
        labels: List[str] = TARGET_LABELS,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.labels = labels
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents=None,  # Preprocessing already preserves unicode cleanly
        )
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=250,
                solver="liblinear",
                random_state=random_state,
            )
        )
        self.thresholds: Dict[str, float] = {l: 0.5 for l in self.labels}
        self.is_trained = False

    def fit(self, train_texts: List[str], y_train: np.ndarray, val_texts: Optional[List[str]] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit TF-IDF vectorizer and One-vs-Rest classifier on training data."""
        logger.info(f"Fitting TF-IDF vectorizer (max_features={self.max_features})...")
        X_train = self.vectorizer.fit_transform(train_texts)

        logger.info(f"Fitting OneVsRest Logistic Regression on {X_train.shape[0]} samples...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Tune decision thresholds on validation set if provided
        if val_texts is not None and y_val is not None:
            self.tune_thresholds(val_texts, y_val)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probability scores for each class."""
        if not self.is_trained:
            raise RuntimeError("Model is not fitted yet.")
        X = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(X)
        return probs

    def tune_thresholds(self, val_texts: List[str], y_val: np.ndarray) -> Dict[str, float]:
        """Tune per-class decision thresholds on validation data to maximize F1-score."""
        logger.info("Tuning per-class decision thresholds on validation data...")
        probs = self.predict_proba(val_texts)
        best_thresholds: Dict[str, float] = {}

        for idx, lbl in enumerate(self.labels):
            best_t = 0.5
            best_f1 = -1.0
            y_true_col = y_val[:, idx]

            # If no positive instances, leave threshold at 0.5
            if y_true_col.sum() == 0:
                best_thresholds[lbl] = 0.5
                continue

            for t in np.linspace(0.15, 0.85, 29):
                y_pred_col = (probs[:, idx] >= t).astype(int)
                score = f1_score(y_true_col, y_pred_col, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = float(t)

            best_thresholds[lbl] = round(best_t, 3)
            logger.info(f"  Tuned threshold for '{lbl}': {best_thresholds[lbl]:.3f} (Val F1: {best_f1:.4f})")

        self.thresholds = best_thresholds
        return best_thresholds

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict binary multi-label indicator using tuned thresholds."""
        probs = self.predict_proba(texts)
        preds = np.zeros_like(probs, dtype=int)
        for idx, lbl in enumerate(self.labels):
            t = self.thresholds.get(lbl, 0.5)
            preds[:, idx] = (probs[:, idx] >= t).astype(int)
        return preds

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save model artifacts, vectorizer, and thresholds to directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, out / "tfidf_vectorizer.joblib")
        joblib.dump(self.classifier, out / "tfidf_ovr_classifier.joblib")
        with open(out / "tfidf_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, indent=2)
        with open(out / "labels.json", "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)
        logger.info(f"TF-IDF baseline saved to {out}")

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "TfidfMultiLabelBaseline":
        """Load fitted TF-IDF baseline model from directory."""
        mdir = Path(model_dir)
        with open(mdir / "labels.json", "r", encoding="utf-8") as f:
            labels = json.load(f)
        with open(mdir / "tfidf_thresholds.json", "r", encoding="utf-8") as f:
            thresholds = json.load(f)

        instance = cls(labels=labels)
        instance.vectorizer = joblib.load(mdir / "tfidf_vectorizer.joblib")
        instance.classifier = joblib.load(mdir / "tfidf_ovr_classifier.joblib")
        instance.thresholds = thresholds
        instance.is_trained = True
        return instance


class MultiLabelClassificationHead(nn.Module):
    """PyTorch multi-label classification head operating on RoBERTa embeddings."""

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits for each class."""
        return self.net(x)


class RoBERTaMultiLabelClassifier:
    """Pretrained RoBERTa-backed Multi-label Classifier with class-weighted BCEWithLogitsLoss."""

    def __init__(
        self,
        labels: List[str] = TARGET_LABELS,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 15,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.labels = labels
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.model = MultiLabelClassificationHead(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=len(labels),
        ).to(self.device)

        self.thresholds: Dict[str, float] = {l: 0.5 for l in self.labels}
        self.is_trained = False

    def fit(
        self,
        X_train_emb: np.ndarray,
        y_train: np.ndarray,
        X_val_emb: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Train neural head using class-weighted BCE loss on RoBERTa embeddings."""
        logger.info(f"Training RoBERTa Multi-Label Head for {self.epochs} epochs on {self.device}...")

        # Calculate positive class weights for class imbalance
        num_samples = len(y_train)
        pos_counts = y_train.sum(axis=0)
        # Avoid division by zero
        pos_weights = (num_samples - pos_counts) / np.maximum(pos_counts, 1.0)
        pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        X_train_t = torch.tensor(X_train_emb, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        best_val_loss = float("inf")
        best_state = None
        history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)

            train_loss /= len(train_dataset)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_x_t = torch.tensor(X_val_emb, dtype=torch.float32).to(self.device)
                val_y_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                val_logits = self.model(val_x_t)
                val_loss = criterion(val_logits, val_y_t).item()

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if epoch % 5 == 0 or epoch == self.epochs:
                logger.info(f"Epoch {epoch:2d}/{self.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_trained = True

        # Tune thresholds on validation embeddings
        self.tune_thresholds(X_val_emb, y_val)
        return {"best_val_loss": best_val_loss, "history": history}

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict sigmoid probabilities from input embeddings."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            logits = self.model(x_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def tune_thresholds(self, X_val_emb: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Tune decision thresholds for optimal per-class F1."""
        probs = self.predict_proba(X_val_emb)
        best_thresholds: Dict[str, float] = {}

        for idx, lbl in enumerate(self.labels):
            best_t = 0.5
            best_f1 = -1.0
            y_true_col = y_val[:, idx]

            if y_true_col.sum() == 0:
                best_thresholds[lbl] = 0.5
                continue

            for t in np.linspace(0.15, 0.85, 29):
                y_pred_col = (probs[:, idx] >= t).astype(int)
                score = f1_score(y_true_col, y_pred_col, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = float(t)

            best_thresholds[lbl] = round(best_t, 3)
            logger.info(f"  RoBERTa Tuned threshold '{lbl}': {best_thresholds[lbl]:.3f} (Val F1: {best_f1:.4f})")

        self.thresholds = best_thresholds
        return best_thresholds

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict binary multi-label outputs using tuned thresholds."""
        probs = self.predict_proba(embeddings)
        preds = np.zeros_like(probs, dtype=int)
        for idx, lbl in enumerate(self.labels):
            t = self.thresholds.get(lbl, 0.5)
            preds[:, idx] = (probs[:, idx] >= t).astype(int)
        return preds

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save PyTorch weights and threshold configurations."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "roberta_head.pt")
        with open(out / "roberta_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, indent=2)
        with open(out / "labels.json", "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)
        config = {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "labels": self.labels,
        }
        with open(out / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"RoBERTa classifier saved to {out}")

    @classmethod
    def load(cls, model_dir: Union[str, Path], device: str = "cpu") -> "RoBERTaMultiLabelClassifier":
        """Load trained neural multi-label classifier."""
        mdir = Path(model_dir)
        with open(mdir / "model_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(mdir / "roberta_thresholds.json", "r", encoding="utf-8") as f:
            thresholds = json.load(f)

        instance = cls(
            labels=cfg["labels"],
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            device=device,
        )
        state_dict = torch.load(mdir / "roberta_head.pt", map_location=device, weights_only=True)
        instance.model.load_state_dict(state_dict)
        instance.thresholds = thresholds
        instance.is_trained = True
        return instance


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
) -> Dict[str, Any]:
    """Calculate comprehensive multi-label evaluation metrics."""
    metrics: Dict[str, Any] = {}

    # Aggregate metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    h_loss = hamming_loss(y_true, y_pred)

    metrics["macro_f1"] = round(float(macro_f1), 4)
    metrics["micro_f1"] = round(float(micro_f1), 4)
    metrics["hamming_loss"] = round(float(h_loss), 4)

    # Per-label metrics
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_label = {}

    for idx, lbl in enumerate(labels):
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # PR-AUC / ROC-AUC if positive examples exist
        auc_score = None
        try:
            if y_true[:, idx].sum() > 0:
                auc_score = round(float(roc_auc_score(y_true[:, idx], y_prob[:, idx])), 4)
        except Exception:
            pass

        per_label[lbl] = {
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1": round(float(f[idx]), 4),
            "support": int(s[idx]),
            "roc_auc": auc_score,
            "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        }

    metrics["per_label"] = per_label
    return metrics

