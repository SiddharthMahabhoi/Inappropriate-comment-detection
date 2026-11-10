"""Stage 2: Preprocessing Module - Context-Aware Inappropriate Comments Scanner.

This module implements deterministic, reusable text preprocessing tailored for
online conversational data. It cleans noise (URLs, mentions, excess whitespace)
while preserving critical conversational signals:
- Capitalization (shouting/emphasis)
- Negation ("not", "never", "n't")
- Emojis and Unicode characters
- Punctuation markers of emotion and sarcasm ("???", "!!!", quotes, "/s")
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class ConversationPreprocessor:
    """Deterministic, context-preserving text preprocessor for conversation comments."""

    # Matches Markdown links: [anchor text](http...) -> captures anchor and URL
    _MD_LINK_REGEX = re.compile(
        r"\[(?P<anchor>[^\]]+)\]\((?P<url>https?://[^\s\)]+|www\.[^\s\)]+)\)",
        re.IGNORECASE,
    )

    # Matches standalone URLs (http://, https://, ftp://, or www.)
    _URL_REGEX = re.compile(
        r"(?:https?://|ftp://|www\.)[^\s<>{}\[\]\^`\"']+",
        re.IGNORECASE,
    )

    # Matches user mentions: /u/username, u/username, or @username
    _USER_MENTION_REGEX = re.compile(
        r"(?:^|(?<=\s))(?:/u/|u/|@)[A-Za-z0-9_\-]+",
        re.IGNORECASE,
    )

    # Matches subreddit mentions: /r/subreddit or r/subreddit -> normalize to [SUBREDDIT]
    _SUBREDDIT_REGEX = re.compile(
        r"(?:^|(?<=\s))(?:/r/|r/)[A-Za-z0-9_\-]+",
        re.IGNORECASE,
    )

    # Matches HTML entities: &amp;, &gt;, &lt;, &quot;, &#39;
    _HTML_ENTITIES = {
        "&amp;": "&",
        "&gt;": ">",
        "&lt;": "<",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
    }

    def __init__(
        self,
        url_placeholder: str = "[URL]",
        user_placeholder: str = "[USER]",
        subreddit_placeholder: str = "[SUBREDDIT]",
        normalize_unicode: bool = True,
        preserve_casing: bool = True,
    ):
        """Configure the preprocessing rules.

        Args:
            url_placeholder: Placeholder token for URLs.
            user_placeholder: Placeholder token for user mentions.
            subreddit_placeholder: Placeholder token for subreddit mentions.
            normalize_unicode: Whether to apply NFKC Unicode normalization.
            preserve_casing: Whether to preserve original casing (True is recommended for NLP).
        """
        self.url_placeholder = url_placeholder
        self.user_placeholder = user_placeholder
        self.subreddit_placeholder = subreddit_placeholder
        self.normalize_unicode = normalize_unicode
        self.preserve_casing = preserve_casing

    def clean_text(self, text: Optional[str]) -> str:
        """Clean and normalize a single text comment deterministically.

        Args:
            text: Raw comment text or None.

        Returns:
            Cleaned and normalized text string.
        """
        # 1. Safe handling of null, empty, or non-string inputs
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        if not text.strip():
            return ""

        # 2. Decode common HTML entities (often present in Reddit dumps)
        for entity, char in self._HTML_ENTITIES.items():
            if entity in text:
                text = text.replace(entity, char)

        # 3. Unicode normalization (NFKC decomposes compatibility chars but preserves emojis & accents)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 4. Handle Markdown links: replace url with placeholder while keeping anchor text
        def md_link_sub(match: re.Match) -> str:
            anchor = match.group("anchor")
            return f"{anchor} {self.url_placeholder}"

        text = self._MD_LINK_REGEX.sub(md_link_sub, text)

        # 5. Replace remaining standalone URLs with URL placeholder
        text = self._URL_REGEX.sub(self.url_placeholder, text)

        # 6. Replace user mentions with USER placeholder
        text = self._USER_MENTION_REGEX.sub(self.user_placeholder, text)

        # 7. Replace subreddit references with SUBREDDIT placeholder
        text = self._SUBREDDIT_REGEX.sub(self.subreddit_placeholder, text)

        # 8. Normalize whitespace while preserving intentional paragraph structure
        # Replace carriage returns and excessive tabs
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace 3 or more consecutive newlines with exactly 2 (clean paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple horizontal whitespace characters (spaces, tabs) into a single space
        text = re.sub(r"[ \t]+", " ", text)
        # Clean leading/trailing spaces per line
        text = "\n".join(line.strip() for line in text.split("\n"))
        # Strip final outer whitespace
        text = text.strip()

        # 9. Casing: keep case as is if preserve_casing is True (preserves shouting/hostility)
        if not self.preserve_casing:
            text = text.lower()

        return text

    def preprocess_record(self, record_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single comment record dictionary, preserving all metadata.

        Args:
            record_dict: Dictionary containing 'original_text' or 'text', plus metadata.

        Returns:
            Dictionary with both 'original_text' and 'cleaned_text' preserved.
        """
        output = dict(record_dict)
        raw_text = output.get("original_text", output.get("text", ""))
        output["original_text"] = str(raw_text) if raw_text is not None else ""
        output["cleaned_text"] = self.clean_text(output["original_text"])
        return output

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        cleaned_column: str = "cleaned_text",
        original_column: str = "original_text",
    ) -> pd.DataFrame:
        """Preprocess a pandas DataFrame of comments.

        Preserves all existing columns, adds cleaned_text, and ensures original_text is preserved.

        Args:
            df: Input DataFrame.
            text_column: Column name containing raw text.
            cleaned_column: Column name for cleaned text output.
            original_column: Column name to ensure original text is stored.

        Returns:
            DataFrame containing original and cleaned text alongside all original metadata.
        """
        result_df = df.copy()

        # Ensure original text column exists
        if original_column not in result_df.columns:
            if text_column in result_df.columns:
                result_df[original_column] = result_df[text_column]
            else:
                result_df[original_column] = ""

        # Handle nulls safely before cleaning
        raw_series = result_df[original_column].fillna("").astype(str)
        result_df[cleaned_column] = raw_series.apply(self.clean_text)

        return result_df
