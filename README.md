# Context-Aware Early Detection of Conversational Escalation with Severity Grading and Explainability

An end-to-end NLP moderation system that analyses online conversation threads, reconstructs hierarchical reply trees, preprocesses text while preserving conversational pragmatics, generates RoBERTa sentence embeddings, performs multi-label abuse classification, integrates conversational context, forecasts conversation derailment (CRAFT-inspired), provides provisional severity grading, explains decisions via causal leave-one-out token occlusion, and provides an interactive Streamlit moderator dashboard.

This repository implements **Phases 1, 2, and 3** of the academic project as specified in the authoritative research presentation.

---

## 1. Project Architecture & Workflow Stages

| Phase | Stage | Description | Module | Implementation Status |
|---|---|---|---|:---:|
| **Phase 1** | **Stage 1: Input Thread** | Ingest Reddit CMV and Wikipedia talk pages from `corpora_csv.zip`, validate schemas, reconstruct hierarchical reply trees and timestamps. | `src/data_loader.py` | ✅ Complete |
| **Phase 1** | **Stage 2: Preprocessing** | Context-preserving normalization: URLs $\to$ `[URL]`, mentions $\to$ `[USER]`, whitespace cleaning. Preserves shouting, negations, emojis, repeated punctuation (`???`), quotes, and `/s`. | `src/preprocessing.py` | ✅ Complete |
| **Phase 1** | **Stage 3: Text Representation** | Pretrained RoBERTa sentence embeddings (`sentence-transformers/all-distilroberta-v1`, 768 dims) with attention-mask-weighted mean pooling. | `src/embeddings.py` | ✅ Complete |
| **Phase 2** | **Stage 4: Multi-Label Classification** | Co-occurring abuse detection (`toxic`, `insult`, `threat`, `obscene`, `identity_hate`, `severe_toxic`) using TF-IDF Baseline and RoBERTa Neural Head. | `src/classifier.py` | ✅ Complete |
| **Phase 2** | **Stage 5: Context Integration** | Causal preceding context aggregation ($[e_{\text{curr}}; e_{\text{ctx}}]$) compared against Comment-Only baseline on held-out conversation threads. | `src/context_encoder.py` | ✅ Complete |
| **Phase 2** | **Stage 6: Derailment Forecasting** | CRAFT-inspired sequential GRU predicting future conversation collapse prior to personal attack occurrence (1.45 turn lead time). | `src/forecasting.py` | ✅ Complete |
| **Phase 2** | **Stage 7: Severity Grading** | Supervised `severe_toxic` classification + rule-grounded Provisional Risk Estimation Engine (`Low`, `Medium`, `High`, `Critical`). | `src/severity.py` | ✅ Complete |
| **Phase 3** | **Stage 8: Explainability** | Causal Leave-One-Out token occlusion attribution, faithfulness verification tests, and contextual influence analysis. | `src/explainability.py` | ✅ Complete |
| **Phase 3** | **Stage 9 & 10: Evaluation & Transfer** | Comprehensive benchmark evaluation, domain-shift cross-dataset testing (Jigsaw $\to$ Wikipedia Talk Pages), and pragmatic stress tests. | `src/evaluator.py` | ✅ Complete |
| **Phase 3** | **Stage 11: Moderator Dashboard** | Interactive Streamlit dashboard for real-time single comment analysis, conversation trajectory tracking, and batch CSV moderation. | `src/dashboard.py` | ✅ Complete |

---

## 2. Directory Structure

```text
NLP_Full/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Stage 1: Load CMV & Wikipedia threads, reconstruct trees
│   ├── preprocessing.py        # Stage 2: Pragmatics-preserving text normalizer
│   ├── embeddings.py           # Stage 3: RoBERTa / SBERT 768-dim embedding generator
│   ├── classifier.py           # Stage 4: TF-IDF Baseline & RoBERTa Multi-Label Head
│   ├── context_encoder.py      # Stage 5: Comment-Only vs Context-Aware pipeline
│   ├── forecasting.py          # Stage 6: CRAFT-inspired sequential GRU forecaster
│   ├── severity.py             # Stage 7: Severity grading & provisional risk engine
│   ├── explainability.py       # Stage 8: Causal token occlusion & faithfulness testing
│   ├── evaluator.py            # Stage 9-10: Unified evaluation & cross-dataset validation
│   ├── pipeline.py             # Phase 1 standalone runner
│   ├── run_phase2_phase3.py    # Master runner for Phases 2 & 3
│   └── dashboard.py            # Stage 11: Streamlit moderator dashboard
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # 11 tests: URL, mentions, emoji, negation, casing
│   ├── test_pipeline.py        # 3 tests: Tree reconstruction & synthetic embeddings
│   ├── test_phase2_phase3.py   # 7 tests: Multi-label, context, forecasting, severity
│   └── test_end_to_end.py      # 1 test: Full inference pipeline integration
├── outputs/                    # Serialized models, metrics, and generated artifacts
│   ├── models/
│   │   ├── baseline_tfidf/     # TF-IDF vectorizer, classifier, thresholds
│   │   └── roberta_classifier/ # RoBERTa PyTorch weights, thresholds, config
│   ├── stage5_context/         # Context ablation models and results
│   ├── stage6_forecasting/     # CRAFT GRU weights and calibration metrics
│   ├── evaluation_results.json # Complete machine-readable experimental metrics
│   └── evaluation_metrics_summary.csv # Consolidated comparison table
├── requirements.txt            # Minimal, accurate dependencies
├── README.md                   # Setup, execution, and troubleshooting guide
├── NLP_concepts.md             # Comprehensive audit of 23 lab-sheet NLP concepts vs implementation
├── FINAL_PROJECT_COMPLIANCE_REPORT.md # Full compliance report against PPT & lab sheets
├── PHASE1_REPORT.md            # Phase 1 submission report
├── PHASE2_REPORT.md            # Phase 2 submission report
└── PHASE3_REPORT.md            # Phase 3 submission report
```

---

## 3. Environment & Prerequisites

- **OS**: Windows 10 / 11 (PowerShell)
- **Python**: 3.10, 3.11, or 3.12 (Tested on Python 3.12.10)
- **Hardware**: Compatible with standard laptops (CPU execution supported, 8 GB RAM tested)
- **Primary Datasets**:
  - `C:\Users\ASUS\Downloads\NLP Dataset\corpora_csv.zip`
  - `C:\Users\ASUS\Downloads\NLP_Project_Datasets\Jigsaw\train.csv.gz`

---

## 4. Setup & Installation (PowerShell)

### Step 1: Navigate to Project Directory

```powershell
cd "C:\Users\ASUS\.gemini\antigravity\scratch\NLP_Full"
```

### Step 2: Install Required Dependencies

```powershell
pip install -r requirements.txt
```

*(Note: For CPU execution on Windows, PyTorch can be installed via: `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`)*

---

## 5. Execution Commands

### A. Run Automated Test Suite (23 Tests)

Run all unit, integration, and end-to-end tests:

```powershell
python -m pytest tests -v
```

*Expected output: `23 passed in ~110s` (100% pass rate).*

---

### B. Run Complete Phase 2 & Phase 3 Training and Evaluation Pipeline

To train the multi-label classifiers, execute context ablation, train the CRAFT derailment forecaster, evaluate cross-dataset transfer, and export metrics:

```powershell
python src/run_phase2_phase3.py --sample-size 2500 --num-threads 60 --epochs 6 --batch-size 32
```

---

### C. Launch the Enterprise Trust & Safety Moderator Dashboard (Sing App Theme)

Launch the high-contrast light-themed moderation and escalation intelligence platform:

```powershell
streamlit run src/dashboard.py
```

The application opens at `http://localhost:8501` featuring:
1. **Three-Panel Layout:** Monitored Incident Queue (left), High-Contrast Turn-by-Turn Speech Stream with Turn-1 sensitivity (center), and Real-Time Threat Intelligence & LOO Explainability Panel (right).
2. **Interactive System Guide:** In-drawer 4-step workflow guide explaining context-aware NLP and early warning value.
3. **Executive Model Audit:** Instant modal showcasing verified test metrics (RoBERTa Micro-F1: 71.82%, CRAFT Lead Time: 1.45 turns, Cross-Domain AUC: 0.8887).

---

### D. Run Phase 1 Pipeline Alone (Sample Verification)

```powershell
python src/pipeline.py --sample-size 50
```

---

## 6. Empirical Evaluation Summary

| Stage | Model / Pipeline | Macro-F1 | Micro-F1 | Hamming Loss | Key Performance Indicator |
|---|---|:---:|:---:|:---:|---|
| **Stage 4** | TF-IDF Baseline | 0.5370 | 0.7098 | **0.0818** | Fast linear baseline |
| **Stage 4** | RoBERTa Neural Head | **0.5384** | **0.7182** | 0.0827 | **+14.85% Recall boost** on rare harm classes |
| **Stage 5** | Pipeline A (Comment-Only) | **0.5714** | — | — | ROC-AUC: 0.8291 |
| **Stage 5** | Pipeline B (Context-Aware) | 0.4596 | — | — | Context recovers subtle attacks; false alarms arise in heated debates ($\Delta F_1: -0.1607$) |
| **Stage 6** | CRAFT-Inspired GRU | 0.4583 | — | — | **1.45 turn early-warning lead time** prior to attacks; Brier: 0.2466, ECE: 0.1947 |
| **Stage 10** | Cross-Dataset Transfer | **0.5672** | — | — | **ROC-AUC: 0.8887** on Wikipedia Talk Pages |

---

## 7. Troubleshooting Guidance

- **HuggingFace Offline / Network Handshake:**
  `src/embeddings.py` is configured with `local_files_only=True` fallback to instantly load cached models from disk without slow internet checks.
- **CPU Memory Limits (8 GB RAM):**
  Embeddings are batched in micro-batches of 32 to prevent Out-Of-Memory exceptions.
- **Provisional Severity:**
  Raw datasets do not provide native 4-tier human labels (`Low`, `Medium`, `High`, `Critical`). Our system transparently designates this module as an explicitly documented, rule-grounded provisional estimator combining multi-label harm probabilities and derailment risk.
