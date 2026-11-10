# Context-Aware Inappropriate Comments Scanner
## Phase 1 Technical Implementation & Verification Report

**Author:** NLP Project Research Team  
**Date:** October 27, 2026  
**Implementation Target:** Phase 1 Only (Workflow Stages 1 to 3)  
**Primary Dataset:** Reddit ChangeMyView (CMV) Corpus (`corpora_csv.zip`)  
**Pretrained Representation Model:** `sentence-transformers/all-distilroberta-v1` (RoBERTa / Sentence-BERT architecture)  
**Environment:** Windows 11 (PowerShell), Python 3.12.10, PyTorch 2.14.1 (CPU), Transformers / Sentence-Transformers  

---

## 1. Executive Summary & Project Objective

Online discussions on social platforms are vulnerable to toxicity, personal attacks, and hostile escalation. Existing moderation pipelines predominantly operate on individual comments in isolation, assigning binary toxic/non-toxic tags. This paradigm suffers from critical limitations:
1. **Semantic flattening:** Fails to distinguish between distinct abuse classes (insults, threats, obscenity, identity hate).
2. **Context blindness:** Ignores conversational dependencies such as sarcasm, quotations, and counter-speech.
3. **Reactive latency:** Detects hostility only after derailment has occurred.
4. **Lack of interpretability:** Provides no severity assessment or evidence highlights for human moderators.

The overall project aims to construct a **Context-Aware Inappropriate Comments Scanner** that integrates comment semantics, hierarchical conversation context, and derailment forecasting.

### Authoritative Phase 1 Scope
Following Slides 7 and 8 of the authoritative Phase 1 project presentation, the present implementation is strictly confined to **Phase 1 (Stages 1 to 3)**:
- **Stage 1 (Input: Conversation Thread):** Robust data ingestion, validation, and tree-structured conversational reconstruction preserving reply parent-child relationships and chronological ordering.
- **Stage 2 (Preprocessing):** Deterministic, pragmatic-preserving cleaning that normalizes noise (URLs, user mentions, subreddit tags, excessive whitespace) while preserving critical communicative cues (shouting/capitalization, negation tokens, repeated punctuation, emojis, quotes, and sarcasm markers).
- **Stage 3 (Text Representation):** Dense semantic representation generation using a pretrained RoBERTa / Sentence-BERT encoder with attention-mask-weighted mean pooling in batched inference mode.

**Explicit Scope Boundary:**
In strict compliance with project guidelines, **no classification heads, Bi-GRU context networks, CRAFT derailment forecasters, ordinal severity graders, or explainability heatmaps have been trained or evaluated**. Those components belong strictly to Phase 2 and Phase 3 (Future Work). No simulated or fabricated evaluation metrics (accuracy, F1-scores, ROC-AUC) are reported.

---

## 2. Dataset Selection & Schema Audit

### 2.1 Candidate Datasets Inspected
The local storage directory at `C:\Users\ASUS\Downloads\NLP Dataset` was directly inspected:
- `conversations-gone-awry-cmv-corpus-large.zip` (28.7 MB)
- `conversations-gone-awry-cmv-corpus.zip` (51.5 MB)
- `conversations-gone-awry-corpus.zip` (45.2 MB)
- `corpora_csv.zip` (43.8 MB)
- `movie-corpus.zip` (40.9 MB)

Inside `corpora_csv.zip`, the archive provides ready-to-use tabular representations of both the Reddit ChangeMyView (CMV) corpus and the Wikipedia Talk Pages corpus:
- `csv/cmv_conversations.csv` (549 KB)
- `csv/cmv_utterances.csv` (35.8 MB)
- `csv/cmv_speakers.csv` (177 KB)
- `csv/wiki_conversations.csv` (429 KB)
- `csv/wiki_utterances.csv` (17.2 MB)

### 2.2 Selected Primary Dataset: Reddit ChangeMyView (CMV)
`csv/cmv_conversations.csv` and `csv/cmv_utterances.csv` were selected as the primary dataset because:
1. They provide explicit comment reply links (`reply_to`), parent thread pointers, timestamps, and author IDs necessary for hierarchical reply tree reconstruction.
2. The corpus represents real multi-turn deliberations containing genuine rhetorical shifts, disagreements, and moderator removals (Chang & Danescu-Niculescu-Mizil, 2019).
3. The schema directly maps to the downstream requirements of Phase 2 context modeling.

### 2.3 Actual Dataset Schemas
Direct inspection of `csv/cmv_conversations.csv` and `csv/cmv_utterances.csv` yielded the following concrete schemas:

#### `cmv_conversations.csv` Schema (6,842 Conversations)
| Column Name | Data Type | Description |
|---|---|---|
| `conversation_id` | String | Unique Reddit submission identifier (thread root) |
| `has_removed_comment` | Boolean | Ground truth flag indicating whether a comment was removed by moderators |
| `pair_id` | String | Paired conversation ID for matched-pair experimental designs |
| `split` | String | Dataset split partition (`train`, `val`, `test`) |
| `summary_meta` | String | JSON metadata dictionary for conversation summary |

#### `cmv_utterances.csv` Schema (42,964 Utterances)
| Column Name | Data Type | Description |
|---|---|---|
| `id` | String | Unique comment/utterance identifier |
| `conversation_id` | String | Thread root ID this utterance belongs to |
| `speaker` | String | Reddit username of the comment author |
| `reply_to` | String / NaN | Parent comment ID (NaN or root ID for top-level comments) |
| `timestamp` | Float | Unix epoch timestamp of comment publication |
| `text` | String | Full raw text content of the comment |
| `conv_has_removed_comment` | Boolean | Inherited conversation-level moderator removal flag |
| `conv_pair_id` | String | Paired conversation reference |
| `conv_split` | String | Corpus partition (`train`, `val`, `test`) |
| `meta_score` | Integer | Reddit karma score received by the comment |
| `meta_permalink` | String | Web permalink to the original Reddit comment |
| `meta_top_level_comment` | String | ID of the direct top-level ancestor |

---

## 3. Detailed Stage Implementations

### 3.1 Stage 1: Input Conversation Thread Loading & Reconstruction (`src/data_loader.py`)

The data loader module reads directly from `corpora_csv.zip` (streaming from the zip file without requiring disk expansion):
1. **Schema Validation & Cleansing:**
   - Drops duplicate utterance rows based on unique `id`.
   - Replaces `NaN` text entries with empty strings (`""`).
   - Normalizes missing or malformed `reply_to` pointers (handles self-references and empty strings).
   - Coerces timestamps to float values, imputing 0.0 for corrupted timestamps.
2. **Conversation Tree Reconstruction (`ConversationDataLoader.reconstruct_thread`):**
   - Rebuilds the directed tree structure: assigns root posts (`is_root=True`, `depth=0`) and recursively links children comments to parents.
   - Computes conversation nesting depth ($d \ge 0$) for every utterance.
   - Sorts sibling comments chronologically by Unix timestamp.
   - Handles disconnected parent references by gracefully attaching orphan comments at root depth to avoid data loss.
3. **Structured Data Classes:**
   - `CommentRecord`: Standardized dataclass encapsulating `comment_id`, `conversation_id`, `speaker`, `reply_to`, `timestamp`, `original_text`, `is_root`, `depth`, `split`, and metadata.
   - `ConversationThread`: Container for thread-level properties with an ASCII visualization formatter (`format_tree()`).

### 3.2 Stage 2: Context-Preserving Preprocessing (`src/preprocessing.py`)

Standard NLP cleaning routines (lowercasing, stopword removal, punctuation stripping) severely compromise conversational abuse detection by destroying signals of aggression, emphasis, and irony. The implemented `ConversationPreprocessor` adopts a **pragmatic-preserving normalization policy**:
1. **URL Replacement:**
   - Markdown hyperlinks `[anchor](http...)` are converted to `anchor [URL]`, preserving semantic anchor text while removing noisy URLs.
   - Standalone URLs (`http://`, `https://`, `www.`) are replaced with `[URL]`.
2. **User & Subreddit Mentions:**
   - Reddit username mentions (`/u/username`, `u/username`, `@username`) are normalized to `[USER]`.
   - Subreddit links (`/r/subreddit`, `r/subreddit`) are normalized to `[SUBREDDIT]`.
3. **Unicode & HTML Sanitization:**
   - Unescapes standard HTML entities (`&amp;`, `&gt;`, `&lt;`, `&quot;`, `&#39;`).
   - Applies NFKC Unicode normalization to standardize character representations.
4. **Preservation of Critical Communicative Signals:**
   - **Capitalization:** Preserved intact (e.g., `"WHY ARE YOU SHOUTING"`) to retain vital cues of anger and hostility.
   - **Negation:** Preserved intact (`"not"`, `"never"`, `"n't"`, `"cannot"`) because negation fundamentally alters sentiment and toxicity polarity.
   - **Repeated Punctuation & Emotion Markers:** Retains question marks (`???`), exclamation points (`!!!`), combination marks (`?!`), and quotation marks (`"..."`) indicative of disbelief, mockery, or quoting.
   - **Emojis:** Unicode emojis (e.g., 😠, 👎, 👏) are fully preserved to allow semantic models to capture non-verbal affect.
   - **Sarcasm Markers:** Sarcasm tags such as `/s` and quoted phrasing are kept intact.
5. **Whitespace Normalization:**
   - Collapses multiple horizontal spaces to a single space.
   - Normalizes newline chains ($\ge 3$ consecutive newlines $\to 2$ newlines), preserving paragraph boundaries without excessive blank space.
6. **Separation of Original and Cleaned Text:**
   - Both `original_text` and `cleaned_text` are strictly retained side-by-side in separate DataFrame columns and exported CSVs.

### 3.3 Stage 3: Pretrained Text Representation (`src/embeddings.py`)

Following Paper 2 (Reimers & Gurevych, 2019) from the Phase 1 presentation, comment-level representations are generated using a pretrained Transformer encoder:
1. **Model Selection:**
   - Primary: `sentence-transformers/all-distilroberta-v1` (RoBERTa architecture, 768-dimensional output vector).
   - Configurable fallback: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional).
2. **Inference Architecture & Pooling:**
   - Operates in strict inference mode (`torch.no_grad()`, `model.eval()`) with no gradient computation.
   - Employs attention-mask-weighted mean pooling over output token embeddings:
     $$\mathbf{u} = \frac{\sum_{i=1}^{L} m_i \mathbf{h}_i}{\sum_{i=1}^{L} m_i}$$
     where $\mathbf{h}_i$ represents token hidden states and $m_i \in \{0, 1\}$ represents the attention mask.
   - Applies $L_2$ vector normalization to ensure standard cosine similarity properties:
     $$\mathbf{e} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}$$
3. **Batching & Resource Awareness:**
   - Configurable mini-batching (`batch_size=16` default) to guarantee bounded memory consumption on CPU and GPU.
   - Automatic device selection (`cuda` if available, otherwise `cpu`).
   - Built-in system resource auditing (`check_system_resources()`) reporting RAM, disk space, and PyTorch backend info.
4. **Resumability & Output Formats:**
   - Validates numerical integrity: asserts 100% finite float32 values ($\text{NaN} = 0$, $\text{Inf} = 0$).
   - Saves dense vectors as a binary NumPy file (`comment_embeddings.npy`).
   - Saves tabular index metadata (`embedding_metadata.csv`), ensuring a strict 1-to-1 index correspondence between row $i$ in `.npy` and comment metadata.
   - Writes `embedding_manifest.json` detailing model configuration, dimensions, device, and timestamp.
   - Implements automated cache resumption: skips redundant recomputation if matching embeddings exist, unless `--force-recompute` is specified.

---

## 4. Project Structure

```text
NLP_Full/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Stage 1: Load CMV threads & reconstruct reply trees
│   ├── preprocessing.py        # Stage 2: Context-preserving text normalizer
│   ├── embeddings.py           # Stage 3: RoBERTa / Sentence-BERT batched embedding generator
│   ├── pipeline.py             # CLI & programmatic orchestrator for all 3 stages
│   └── run_stage3.py           # Dedicated utility to run Stage 3 on existing cleaned CSV
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # 10 unit tests for cleaning edge cases
│   └── test_pipeline.py        # 4 integration tests for loader, tree & embeddings
├── outputs/                    # Output directory for Phase 1 artifacts
│   ├── cleaned_comments_sample.csv             # Cleaned vs original text CSV
│   ├── comment_embeddings.npy                  # Float32 NumPy embedding matrix (50x768)
│   ├── embedding_metadata.csv                  # 1-to-1 comment index metadata mapping
│   ├── embedding_manifest.json                 # Model and vector dimension manifest
│   ├── conversation_reconstruction_sample.txt   # ASCII visualization of reconstructed thread
│   ├── conversation_reconstruction_sample.json  # Hierarchical JSON tree representation
│   └── phase1_run_summary.json                 # Execution summary & verification audit
├── requirements.txt            # Minimal dependencies
├── README.md                   # Complete PowerShell setup & execution guide
└── PHASE1_REPORT.md            # Comprehensive Phase 1 technical submission report
```

---

## 5. Execution Commands & PowerShell Verification

All commands are validated for Windows 11 PowerShell.

### 5.1 Running the Automated Test Suite
```powershell
cd "C:\Users\ASUS\.gemini\antigravity\scratch\NLP_Full"
python -m pytest tests -v
```

### 5.2 Executing the Complete Phase 1 Pipeline (Sample of 50 Comments)
```powershell
cd "C:\Users\ASUS\.gemini\antigravity\scratch\NLP_Full"
python src/pipeline.py --sample-size 50 --force-recompute
```

### 5.3 Executing with Custom Batch Size or Model
```powershell
python src/pipeline.py --sample-size 50 --batch-size 16 --model-name "sentence-transformers/all-distilroberta-v1"
```

---

## 6. Actual Test Results & Verification Audit

### 6.1 Pytest Suite Execution
The automated test suite (`pytest tests -v`) executed across 14 test cases in 65.65 seconds:

| Test Identifier | Module | Status | Description |
|---|---|---|---|
| `test_comment_record_data_structures` | `test_pipeline.py` | **PASSED** | Dataclass initialization, dictionary conversion, tree formatting |
| `test_preprocessing_integration` | `test_pipeline.py` | **PASSED** | Pipeline integration with CommentRecord instances |
| `test_embedding_generation_synthetic` | `test_pipeline.py` | **PASSED** | SBERT model loading, shape `(3, 768)`, finite floats, metadata mapping |
| `test_missing_and_empty_text` | `test_preprocessing.py` | **PASSED** | Robustness against `None`, `""`, whitespace, and non-string inputs |
| `test_url_replacement` | `test_preprocessing.py` | **PASSED** | HTTP, HTTPS, WWW, and markdown link replacement with `[URL]` |
| `test_user_mentions_normalization` | `test_preprocessing.py` | **PASSED** | Reddit `/u/name`, `u/name`, `@name` normalization to `[USER]` |
| `test_punctuation_preservation` | `test_preprocessing.py` | **PASSED** | Preservation of quotes, repeated marks (`???`), and combos (`?!`) |
| `test_negation_preservation` | `test_preprocessing.py` | **PASSED** | Preservation of `"not"`, `"cannot"`, `"never"`, `"n't"` |
| `test_capitalization_preservation` | `test_preprocessing.py` | **PASSED** | Preservation of shouting/emphasis (ALL CAPS) |
| `test_unicode_and_emoji_preservation` | `test_preprocessing.py` | **PASSED** | Preservation of accented Latin characters and Unicode emojis |
| `test_sarcasm_and_quotes_preservation` | `test_preprocessing.py` | **PASSED** | Preservation of `/s` indicator and quotation marks |
| `test_whitespace_normalization` | `test_preprocessing.py` | **PASSED** | Clean multi-newline and horizontal whitespace normalization |
| `test_dataframe_preprocessing` | `test_preprocessing.py` | **PASSED** | Batch DataFrame cleaning preserving original and cleaned text |
| `test_determinism` | `test_preprocessing.py` | **PASSED** | Exact reproducibility across repeated runs on identical inputs |

**Result Summary:** 14 passed in 65.65 seconds. 0 failures, 0 warnings.

### 6.2 End-to-End Pipeline Verification
Running `python src/pipeline.py --sample-size 50 --force-recompute` produced the following verified outputs:
- **Corpus Ingestion:** 6,842 conversations and 42,964 utterances loaded from `corpora_csv.zip`.
- **Thread Selection:** 50 comments extracted across structured conversation threads.
- **Stage 1 Runtime:** 1.29 seconds (including schema validation, deduplication, tree reconstruction).
- **Stage 2 Runtime:** 0.02 seconds (50 comments cleaned deterministically).
- **Stage 3 Runtime:** 5.00 seconds (RoBERTa embedding inference on CPU, batch size 16).
- **Total Pipeline Runtime:** 6.31 seconds.
- **Embedding Matrix Shape:** `(50, 768)` float32.
- **Numerical Validity:** 100% finite values verified (`np.isfinite(embs).all() == True`).
- **Index Integrity:** 50 rows in `embedding_metadata.csv` strictly mapping row $i$ to `comment_id`, `conversation_id`, `speaker`, `timestamp`, `depth`, and `split`.

---

## 7. Reconstructed Conversation Example

From `outputs/conversation_reconstruction_sample.txt`, showing the reconstructed reply tree for thread `cue8y0b`:

```text
=== Conversation: cue8y0b (Split: train, Comments: 10) ===
+-- [OP] (cue8y0b) [RustyRook | t=1440446000]: "(Okay, I've seen this view come up a few times before and I've always been un..."
    \-- [Reply] (cuec5fs) [Canada_is_gay | t=1440450798]: "It's not just black and white America though. Sub-Saharan Africans and Caribb..."
        \-- [Reply] (cuect48) [aguafiestas | t=1440451823]: "Abstract reasoning is a skill that can be nurtured or hindered.  It doesn't s..."
            \-- [Reply] (cuedf8c) [ScholarlyVirtue | t=1440452797]: "Can we agree that genes account for about 50% of the variance ? (in the US; i..."
                \-- [Reply] (cuedywn) [Perpetualjoke | t=1440453690]: "Twin studies studies suggest that about 80 percent of variance in IQ is genetic."
                    \-- [Reply] (czb942p) [garnteller | t=1453734180]: "Sorry Perpetualjoke, your comment has been removed:   &gt; Comment Rule 5\. "..."
                        \-- [Reply] (czbbocu) [Perpetualjoke | t=1453738298]: "Are you kidding me?You decided to remove it after 5 months?"
                            \-- [Reply] (czbdh6q) [garnteller | t=1453740934]: "You decided to edit it after 5 months.  Why would we leave up an empty comment?"
                                \-- [Reply] (czbe470) [Perpetualjoke | t=1453741858]: "I just can't believe you are browsing 5 month  comments I made,I have a suspi..."
                                    \-- [Reply] (czbe8el) [garnteller | t=1453742029]: "No, we have an automoderator that checks for rule violations when a comment i..."
```

**Observation:** The reconstructed conversation vividly illustrates the escalation dynamic highlighted in Paper 3 (CRAFT): an initial debate on intelligence and genetics progresses through counter-arguments, an edit, a moderator removal under Comment Rule 5, and an escalating confrontation between user `Perpetualjoke` and moderator `garnteller`.

---

## 8. Preprocessing Demonstration: Original vs Cleaned Text

Sample extracted from `outputs/cleaned_comments_sample.csv`:

### Comment 1 (`cue8y0b` — Root Post)
- **Original Text:**
  > *(Okay, I've seen this view come up a few times before and I've always been unsuccessful in convincing people about why they're wrong. However, it seems that your view is based on studies, so maybe you'll respond well to evidence /u/Carlosriccy.)*  
  > *I cannot dispute the fact that there is a measurable IQ gap b/w white students and black students in the US. However, it is most certainly not because of genetics... You can find a comprehensive review [here](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3341646/).*  
  > *Edit: Proceed with caution. Members (or former members?) of r/Coontown are here with their copy-pasta arguments.*
- **Cleaned Text:**
  > *(Okay, I've seen this view come up a few times before and I've always been unsuccessful in convincing people about why they're wrong. However, it seems that your view is based on studies, so maybe you'll respond well to evidence [USER].)*  
  > *I cannot dispute the fact that there is a measurable IQ gap b/w white students and black students in the US. However, it is most certainly not because of genetics... You can find a comprehensive review here [URL].*  
  > *Edit: Proceed with caution. Members (or former members?) of [SUBREDDIT] are here with their copy-pasta arguments.*
- **Preserved Signals:**
  - `/u/Carlosriccy` replaced with `[USER]`
  - Markdown link `[here](http...)` replaced with `here [URL]`
  - `r/Coontown` replaced with `[SUBREDDIT]`
  - Negation tokens (`"cannot"`, `"not"`) fully retained
  - Casing, parentheses, and punctuation preserved

---

## 9. Limitations & Known Issues

1. **Dataset-Specific Reply Graph Incompleteness:**
   In online Reddit dumps, occasionally a comment's parent (`reply_to`) is missing because the parent was deleted or filtered out before corpus collection. Our loader gracefully attaches these orphan comments at root depth (`depth=0`) to prevent data dropping, but true genealogical depth may be underestimated for those specific nodes.
2. **Computational Footprint of Deep Encoders on CPU:**
   Encoding with `all-distilroberta-v1` requires approximately 0.10s per comment on an Intel/AMD CPU. Processing the entire 42,964-comment corpus on CPU would require approximately 70 minutes. For Phase 1 validation, a sample of 50 to 500 comments provides full numerical and structural verification without blocking development. CUDA GPU acceleration is supported and automatically utilized when available.
3. **Absence of Domain-Specific Toxic Embeddings in Phase 1:**
   `all-distilroberta-v1` is trained on general sentence similarity (SNLI, MultiNLI, STS) as described in Paper 2 (Reimers & Gurevych, 2019). While semantic vectors capture lexical meaning accurately, toxicity classification and context-dependent harm detection require the multi-label classification head and fine-tuning planned for Phase 2.

---

## 10. Future Work: Phase 2 and Phase 3 Roadmap

The following tasks are strictly reserved for subsequent phases as designated in Slide 8:

### Phase 2: Classification to Severity Grading (Workflow Stages 4 to 7)
- **Stage 4 (Multi-label Classification):**
  Implement a multi-label classification head on top of the RoBERTa encoder using sigmoid activation to independently score the five abuse dimensions: Toxic, Insult, Threat, Obscene, and Identity Hate. Train with focal/weighted binary cross-entropy loss to handle heavy class imbalance.
- **Stage 5 (Context Integration):**
  Implement a hierarchical conversation encoder (Sentence-BERT comment encoder followed by a Bi-GRU or Transformer context encoder) that sequentially processes preceding comments in the reply tree to disambiguate sarcasm and context-dependent hostility.
- **Stage 6 (Derailment Forecasting):**
  Implement a CRAFT-inspired recurrent forecasting network that consumes comment representations sequentially and estimates future derailment risk $\hat{y}_t \in [0, 1]$ several turns before moderator intervention.
- **Stage 7 (Severity Grading):**
  Construct an ordinal severity head categorizing comments into Low, Medium, High, and Critical risk tiers.

### Phase 3: Explainability, Evaluation & Moderator Dashboard (Workflow Stages 8 to 9)
- **Stage 8 (Explainability):**
  Implement Integrated Gradients and token attribution heatmaps to highlight abusive keywords and point moderators to earlier triggering comments in the conversation chain.
- **Stage 9 (Ablation Experiments & Benchmarking):**
  Execute baseline comparisons against TF-IDF + SVM, Logistic Regression, and Bi-LSTM + GloVe across macro-F1, PR-AUC, calibration error (ECE), and early-warning lead time.
- **Interactive Moderator Dashboard:**
  Develop a UI displaying flagged comments, predicted abuse labels, derailment probability trajectory, and highlighted rationales in thread context.

---

## 11. Conclusion & Phase 1 Sign-Off

Phase 1 of the **Context-Aware Inappropriate Comments Scanner** has been successfully implemented, executed, and verified:
- **Stage 1 (Conversation Thread Ingestion):** Reconstructs hierarchical Reddit CMV conversation trees with exact parent-child reply relationships, chronological ordering, and metadata preservation.
- **Stage 2 (Preprocessing):** Safely sanitizes noise while retaining vital communicative indicators (casing, negation, repeated punctuation, emojis).
- **Stage 3 (Text Representation):** Generates valid, 768-dimensional float32 RoBERTa embeddings with 100% finite values and 1-to-1 index-to-metadata alignment.
- **Testing & Verification:** 14/14 automated tests passed; end-to-end pipeline execution confirmed on Windows PowerShell.
- **Integrity Compliance:** No classification heads, fake labels, or simulated metrics are present; Phase 2 and 3 remain clearly demarcated future work.

