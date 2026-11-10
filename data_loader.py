"""Stage 1: Input Conversation Thread - Data Loading and Reconstruction.

This module loads conversation datasets (specifically Reddit ChangeMyView from
corpora_csv.zip), validates schemas, handles missing values/duplicates,
reconstructs conversation reply trees, and preserves comment ordering.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


@dataclass
class CommentRecord:
    """Represents a single comment/utterance within a conversation thread."""

    comment_id: str
    conversation_id: str
    speaker: str
    reply_to: Optional[str]
    timestamp: float
    original_text: str
    is_root: bool
    depth: int = 0
    split: str = "train"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to standard dictionary."""
        return {
            "comment_id": self.comment_id,
            "conversation_id": self.conversation_id,
            "speaker": self.speaker,
            "reply_to": self.reply_to,
            "timestamp": self.timestamp,
            "original_text": self.original_text,
            "is_root": self.is_root,
            "depth": self.depth,
            "split": self.split,
            **self.metadata,
        }


@dataclass
class ConversationThread:
    """Represents a complete reconstructed conversation thread."""

    conversation_id: str
    split: str
    has_removed_comment: Optional[bool]
    comments: List[CommentRecord] = field(default_factory=list)
    root_id: Optional[str] = None

    @property
    def comment_count(self) -> int:
        return len(self.comments)

    def format_tree(self) -> str:
        """Produce an ASCII visualization of the conversation reply tree."""
        lines = [
            f"=== Conversation: {self.conversation_id} (Split: {self.split}, Comments: {self.comment_count}) ==="
        ]
        for c in self.comments:
            indent = "    " * c.depth
            prefix = "+-- [OP]" if c.is_root else "\\-- [Reply]"
            snippet = c.original_text.replace("\n", " ").strip()
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            lines.append(
                f"{indent}{prefix} ({c.comment_id}) [{c.speaker} | t={int(c.timestamp)}]: \"{snippet}\""
            )
        return "\n".join(lines)


class ConversationDataLoader:
    """Loads and reconstructs conversation threads from CMV dataset archives or folders."""

    def __init__(self, data_path: Union[str, Path]):
        """Initialize data loader with either a path to corpora_csv.zip or an extracted directory.

        Args:
            data_path: Path to corpora_csv.zip, a folder containing it, or an extracted 'csv/' directory.
        """
        self.data_path = Path(data_path)
        self.conversations_df: Optional[pd.DataFrame] = None
        self.utterances_df: Optional[pd.DataFrame] = None
        self.schema_info: Dict[str, Any] = {}

    def _resolve_data_source(self) -> Tuple[Optional[zipfile.ZipFile], Optional[Path]]:
        """Resolve whether the target is a zip file, directory containing zip, or extracted folder."""
        p = self.data_path

        # If pointing directly to a zip file
        if p.is_file() and p.name.endswith(".zip"):
            return zipfile.ZipFile(p, "r"), None

        # If pointing to a directory that contains corpora_csv.zip
        if p.is_dir():
            zip_in_dir = p / "corpora_csv.zip"
            if zip_in_dir.is_file():
                return zipfile.ZipFile(zip_in_dir, "r"), None

            # Check if extracted CSV directory
            cmv_conv = p / "cmv_conversations.csv"
            if not cmv_conv.exists():
                cmv_conv = p / "csv" / "cmv_conversations.csv"
            if cmv_conv.exists():
                return None, cmv_conv.parent

        raise FileNotFoundError(
            f"Could not locate CMV dataset in '{self.data_path}'. "
            f"Expected 'corpora_csv.zip' or directory with 'cmv_conversations.csv' and 'cmv_utterances.csv'."
        )

    def load_raw_data(self, sample_limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load conversations and utterances data into pandas DataFrames.

        Args:
            sample_limit: Optional limit on the number of utterances to read.

        Returns:
            Tuple of (conversations_df, utterances_df).
        """
        zf, extracted_dir = self._resolve_data_source()

        try:
            if zf is not None:
                conv_file = "csv/cmv_conversations.csv"
                utt_file = "csv/cmv_utterances.csv"

                with zf.open(conv_file) as f:
                    self.conversations_df = pd.read_csv(f)

                with zf.open(utt_file) as f:
                    self.utterances_df = pd.read_csv(f, nrows=sample_limit)
            else:
                assert extracted_dir is not None
                conv_path = extracted_dir / "cmv_conversations.csv"
                utt_path = extracted_dir / "cmv_utterances.csv"

                self.conversations_df = pd.read_csv(conv_path)
                self.utterances_df = pd.read_csv(utt_path, nrows=sample_limit)

        finally:
            if zf is not None:
                zf.close()

        # Capture schema info
        self.schema_info = {
            "conversations_columns": list(self.conversations_df.columns),
            "utterances_columns": list(self.utterances_df.columns),
            "total_conversations_meta": len(self.conversations_df),
            "total_utterances_loaded": len(self.utterances_df),
        }

        self._validate_and_clean()
        return self.conversations_df, self.utterances_df

    def _validate_and_clean(self) -> None:
        """Validate required fields, sanitize nulls, remove duplicates, and ensure data integrity."""
        assert self.utterances_df is not None
        assert self.conversations_df is not None

        # 1. Deduplicate by utterance ID if any duplicate exists
        initial_len = len(self.utterances_df)
        self.utterances_df = self.utterances_df.drop_duplicates(subset=["id"]).copy()
        dedup_diff = initial_len - len(self.utterances_df)

        # 2. Standardize missing text safely
        self.utterances_df["text"] = self.utterances_df["text"].fillna("").astype(str)

        # 3. Standardize timestamps
        if "timestamp" in self.utterances_df.columns:
            self.utterances_df["timestamp"] = pd.to_numeric(
                self.utterances_df["timestamp"], errors="coerce"
            ).fillna(0.0)

        # 4. Standardize reply_to (root comments have NaN, empty string, or self-reference)
        def clean_reply_to(row: pd.Series) -> Optional[str]:
            val = row.get("reply_to")
            if pd.isna(val) or val is None or str(val).strip() == "" or str(val).lower() == "nan":
                return None
            val_str = str(val).strip()
            # If reply_to points to itself or equals conversation_id while id == conversation_id
            if val_str == str(row["id"]):
                return None
            return val_str

        self.utterances_df["cleaned_reply_to"] = self.utterances_df.apply(clean_reply_to, axis=1)

        self.schema_info["duplicates_removed"] = dedup_diff
        self.schema_info["unique_conversations_in_utterances"] = int(
            self.utterances_df["conversation_id"].nunique()
        )

    def reconstruct_thread(self, conversation_id: str) -> Optional[ConversationThread]:
        """Reconstruct a single conversation thread with its hierarchical reply tree and order.

        Args:
            conversation_id: The conversation ID to reconstruct.

        Returns:
            Reconstructed ConversationThread object or None if not found.
        """
        if self.utterances_df is None:
            raise RuntimeError("Data not loaded. Call load_raw_data() first.")

        conv_rows = self.utterances_df[self.utterances_df["conversation_id"] == conversation_id]
        if conv_rows.empty:
            return None

        # Sort chronologically by timestamp
        conv_rows = conv_rows.sort_values(by=["timestamp", "id"]).copy()

        # Extract conversation-level split and flag
        split = "train"
        if "conv_split" in conv_rows.columns:
            split_val = conv_rows["conv_split"].dropna().values
            if len(split_val) > 0:
                split = str(split_val[0])

        has_removed = None
        if "conv_has_removed_comment" in conv_rows.columns:
            flag_val = conv_rows["conv_has_removed_comment"].dropna().values
            if len(flag_val) > 0:
                has_removed = bool(flag_val[0])

        # Build tree structure
        comment_dict: Dict[str, Dict[str, Any]] = {}
        for _, row in conv_rows.iterrows():
            cid = str(row["id"])
            parent_id = row["cleaned_reply_to"]
            comment_dict[cid] = {
                "row": row,
                "parent_id": parent_id,
                "children": [],
            }

        # Populate children lists
        root_ids: List[str] = []
        for cid, data in comment_dict.items():
            parent_id = data["parent_id"]
            if parent_id and parent_id in comment_dict:
                comment_dict[parent_id]["children"].append(cid)
            else:
                root_ids.append(cid)

        # Depth calculation using BFS/DFS traversal
        ordered_records: List[CommentRecord] = []

        def traverse(node_id: str, current_depth: int) -> None:
            node = comment_dict[node_id]
            row = node["row"]
            is_root = (node["parent_id"] is None) or (node_id in root_ids and current_depth == 0)

            # Metadata dictionary
            meta_dict = {
                "score": row.get("meta_score", 0),
                "permalink": row.get("meta_permalink", ""),
                "conv_pair_id": row.get("conv_pair_id", ""),
            }

            record = CommentRecord(
                comment_id=str(row["id"]),
                conversation_id=str(row["conversation_id"]),
                speaker=str(row["speaker"]) if pd.notna(row.get("speaker")) else "[DELETED]",
                reply_to=node["parent_id"],
                timestamp=float(row["timestamp"]),
                original_text=str(row["text"]),
                is_root=is_root,
                depth=current_depth,
                split=split,
                metadata=meta_dict,
            )
            ordered_records.append(record)

            # Sort children by timestamp for chronological reply order
            children = node["children"]
            children_sorted = sorted(
                children,
                key=lambda x: (
                    float(comment_dict[x]["row"]["timestamp"]),
                    str(x),
                ),
            )
            for child_id in children_sorted:
                traverse(child_id, current_depth + 1)

        for rid in root_ids:
            traverse(rid, 0)

        # If any disconnected nodes remain due to missing parents
        visited_ids = {r.comment_id for r in ordered_records}
        remaining = [cid for cid in comment_dict if cid not in visited_ids]
        for cid in remaining:
            traverse(cid, 0)

        primary_root = root_ids[0] if root_ids else None
        return ConversationThread(
            conversation_id=conversation_id,
            split=split,
            has_removed_comment=has_removed,
            comments=ordered_records,
            root_id=primary_root,
        )

    def load_processed_records(
        self, sample_limit: Optional[int] = None
    ) -> List[CommentRecord]:
        """Load and reconstruct all conversations, returning a flat list of structured CommentRecords.

        Args:
            sample_limit: Maximum number of comments to return.

        Returns:
            List of structured CommentRecord instances with reply depths.
        """
        if self.utterances_df is None:
            self.load_raw_data()

        assert self.utterances_df is not None
        unique_convs = self.utterances_df["conversation_id"].unique()

        all_records: List[CommentRecord] = []
        for conv_id in unique_convs:
            thread = self.reconstruct_thread(str(conv_id))
            if thread:
                for rec in thread.comments:
                    all_records.append(rec)
                    if sample_limit is not None and len(all_records) >= sample_limit:
                        return all_records

        return all_records
