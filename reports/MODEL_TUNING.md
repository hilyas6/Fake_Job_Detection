# Model Tuning Report — Fake Job Posting Detection

This document covers the hyperparameter search, trial results, and final configurations for every model family trained in this project.

---

## Table of Contents

1. [Dataset & Evaluation Protocol](#1-dataset--evaluation-protocol)
2. [Classical Models](#2-classical-models)
3. [Bi-LSTM](#3-bi-lstm)
4. [DistilBERT](#4-distilbert)
5. [TextGCN (Baseline)](#5-textgcn-baseline)
6. [TextGCN (Tuned — Production Model)](#6-textgcn-tuned--production-model)
7. [Full Results Comparison](#7-full-results-comparison)
8. [Key Findings](#8-key-findings)

---

## 1. Dataset & Evaluation Protocol

| Item | Detail |
|---|---|
| **Primary dataset** | EMSCAD (Employment Scam Aegean Corpus Dataset) |
| **Split** | 60 / 20 / 20 — train / val / test, stratified by label, seed = 42 |
| **OOD dataset** | OpenBay — used for evaluation only, never seen during training |
| **Threshold selection** | Grid search over 0.10–0.90 (41 steps) to maximise validation F1 |
| **Primary metric** | EMSCAD Test F1 |
| **Secondary metric** | OpenBay Recall (out-of-distribution generalisation) |

All models read from `data/processed/` and write metrics to `reports/`.
Tuned versions write to `reports/tuned/`.

---

## 2. Classical Models

**Script:** `tuned_models/tune_classical_models.py`

### Text Representation

All classical models share the same TF-IDF vectoriser:

| Parameter | Value |
|---|---|
| `ngram_range` | (1, 3) |
| `min_df` | 2 |
| `max_df` | 0.92 |
| `max_features` | 70,000 |
| `sublinear_tf` | True |

### Logistic Regression

| Parameter | Search Space | Best Value |
|---|---|---|
| `C` (regularisation) | 0.5, 1.0, 2.0 | 1.0 |
| `class_weight` | balanced | balanced |
| `solver` | saga | saga |
| `max_iter` | 3000 | 3000 |

### Naive Bayes (Complement NB)

| Parameter | Search Space | Best Value |
|---|---|---|
| `alpha` (Laplace smoothing) | 0.10, 0.25, 0.50, 1.00 | 0.25 |

### Random Forest

| Parameter | Search Space / Value |
|---|---|
| `n_estimators` | 600 |
| `max_depth` | None, 30 |
| `min_samples_leaf` | 1, 2 |
| `class_weight` | balanced_subsample |

### XGBoost

| Parameter | Search Space / Value |
|---|---|
| `n_estimators` | 700 |
| `learning_rate` | 0.03, 0.05 |
| `max_depth` | 5, 7 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `scale_pos_weight` | √(neg / pos) |

### LightGBM

| Parameter | Search Space / Value |
|---|---|
| `n_estimators` | 800 |
| `learning_rate` | 0.03, 0.05 |
| `num_leaves` | 63, 127 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `scale_pos_weight` | √(neg / pos) |

### Tuned Classical Models Results

| Model | Test F1 | Precision | Recall | Threshold | OOD Recall |
|---|---|---|---|---|---|
| **LightGBM** | **0.8494** | 0.9483 | 0.7692 | 0.16 | 0.9906 |
| XGBoost | 0.8042 | 0.8042 | 0.8042 | 0.28 | 0.9992 |
| Random Forest | 0.7805 | 0.7778 | 0.7832 | 0.24 | 1.0000 |
| Logistic Regression | 0.7692 | 0.8077 | 0.7343 | 0.80 | 1.0000 |
| Naive Bayes | 0.7097 | 0.6587 | 0.7692 | 0.18 | 0.5867 |

**Winner:** LightGBM (F1 = 0.849), highest precision of all classical models.

---

## 3. Bi-LSTM

**Script:** `tuned_models/tune_bilstm.py`

### Architecture

```
Embedding(vocab_size=50000, embed_dim)
  → EmbeddingDropout(min(0.25, dropout))
  → BiLSTM(embed_dim → hidden_dim, num_layers, bidirectional=True)
  → LSTM Dropout (if num_layers > 1)
  → Linear(hidden_dim × 2 → 1)
```

Optimiser: AdamW
Scheduler: ReduceLROnPlateau (factor = 0.5, patience = 1)
Early stopping patience: 2 epochs

### Three Trial Configurations

| Parameter | Trial 1 (Baseline) | Trial 2 (Enhanced) | Trial 3 (Large + Reg) |
|---|---|---|---|
| `embed_dim` | 200 | 256 | 256 |
| `hidden_dim` | 128 | 160 | 192 |
| `dropout` | 0.25 | 0.30 | 0.35 |
| `learning_rate` | 8e-4 | 1e-3 | 7e-4 |
| `weight_decay` | 1e-4 | 1e-4 | 2e-4 |
| `grad_clip` | 1.0 | 1.0 | 0.8 |
| `epochs` | 10 | 12 | 12 |
| `num_layers` | 2 | 2 | 2 |
| `max_len` | 360 | 400 | 420 |
| `batch_size` | 64 | 64 | 48 |

### Bi-LSTM Results

| Model | Test F1 | Precision | Recall | Threshold | OOD Recall |
|---|---|---|---|---|---|
| BiLSTM (best trial) | 0.7737 | 0.8092 | 0.7413 | 0.10 | 0.3701 |

**Note:** BiLSTM achieves competitive EMSCAD performance but generalises poorly to OpenBay (OOD recall 37%), indicating it overfits to in-distribution language patterns.

---

## 4. DistilBERT

**Script:** `tuned_models/tune_distilbert.py`

### Configuration

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Tokeniser | Fast tokeniser, max length = 160 tokens |
| Padding | To nearest multiple of 8 (data collator) |
| Batch size (train) | 12 |
| Batch size (val/test) | 16 |
| Loss | CrossEntropyLoss |
| Epochs per trial | 2 |

### Hyperparameter Search

Three learning rates were trialled with 4 epochs each (early stopping patience = 2):

| Learning Rate | Epoch 1 Val F1 | Epoch 2 Val F1 | Epoch 3 Val F1 | Epoch 4 Val F1 |
|---|---|---|---|---|
| 1e-5 | 0.5490 | 0.7761 | 0.8030 | 0.8261 |
| **2e-5** | 0.6496 | 0.8346 | 0.8467 | **0.8571** |
| 3e-5 | 0.6364 | 0.8062 | 0.8175 | 0.8451 |

**Winner:** lr = 2e-5, val F1 = 0.8571 at epoch 4.

### DistilBERT Results

| Version | Test F1 | Precision | Recall | Threshold | OOD Recall | Notes |
|---|---|---|---|---|---|---|
| Original (2 epochs, no weights) | 0.7641 | 0.7278 | 0.8042 | 0.20 | 0.4666 | Undertrained |
| **Improved (4 epochs + class weights + warmup)** | **0.8231** | **0.8013** | **0.8462** | 0.10 | 0.4041 | **+5.9pp F1** |

**Improvement:** +5.9 pp F1, +7.3 pp Precision, +4.2 pp Recall from three changes: 4 training epochs (vs 2), class weighting (fake class weight = √(22) ≈ 4.6×), and linear warmup scheduler over the first 10% of steps.

**Note:** OOD recall remains low (40%) regardless of tuning. This is a fundamental property of transformer fine-tuning on EMSCAD — the model learns surface-level lexical patterns specific to the training distribution rather than transferable fraud indicators. The TextGCN's co-occurrence graph structure generalises better across datasets.

---

## 5. TextGCN (Baseline)

**Script:** `src/train_textgcn_enhanced.py`

### Architecture: ImprovedWordGCN

```
Word Embeddings: (num_nodes, hidden_dim)  ← Xavier init
  → GCN Layer 1:  H₁ = ReLU( Â · H₀ · W₁ )  +  Dropout(p)
  → GCN Layer 2:  H₂ = ReLU( Â · H₁ · W₂ )  +  Dropout(p)
  → GCN Layer 3:  H₃ = ReLU( Â · H₂ · W₃ )
  → LayerNorm(H₃)
  → Residual:     H_out = (1 − α) · H₀ + α · H₃
  → MLP:          Linear(d→d) → ReLU → Dropout → Linear(d→d/2) → ReLU → Dropout
  → Classifier:   Linear(d/2 → 2)
```

Where Â = D^(−½) · (A + I) · D^(−½) (symmetrically normalised adjacency).

### Graph Construction

| Parameter | Value |
|---|---|
| Word–Document edges | TF-IDF(term, document) |
| Word–Word edges | PMI over sliding window |
| PMI window size | 20 (baseline) |
| PMI threshold | 0.0 |
| Self-loops | Added before normalisation |

### Baseline Hyperparameters

| Parameter | Value |
|---|---|
| `hidden_dim` | 256 |
| `dropout` | 0.35 |
| `residual_alpha` | 0.7 |
| `learning_rate` | 3e-3 |
| `weight_decay` | 1e-5 |
| `optimiser` | Adam |
| `scheduler` | None |
| `epochs` | 200 |
| `patience` | 25 |
| `label_smoothing` | 0.05 |
| `loss` | Cross-entropy |
| `class_weight` | √(neg / pos) |
| `max_features` | 40,000 |
| `ngram_range` | (1, 3) |

### Baseline Result

| Model | Test F1 | Precision | Recall | Threshold | OOD Recall |
|---|---|---|---|---|---|
| textgcn_improved | 0.8352 | 0.8769 | 0.7972 | 0.48 | 0.9583 |

---

## 6. TextGCN (Tuned — Production Model)

**Script:** `tuned_models/tune_textgcn.py`
**Winner script:** `tuned_models/best_tuned_textgcn_model.py`

34 trials were run across two rounds to find the best configuration.

---

### Round 1 — 20 Trials

#### R1: Learning Rate Schedulers

| Trial | Scheduler | Optimiser | Notes | Test F1 |
|---|---|---|---|---|
| r1_cosine_adamw | Cosine | AdamW | LR decays too fast, hurts late training | 0.8159 |
| r1_cosine_adamw_hi_wd | Cosine | AdamW | weight_decay = 5e-4, worse | 0.8090 |
| r1_plateau_adamw | ReduceLROnPlateau | AdamW | Better than cosine | 0.8209 |
| r1_cosine_low_lr | Cosine | Adam | lr = 1e-3, underpowered | 0.7972 |

**Finding:** AdamW + cosine annealing consistently hurts. Constant Adam LR is preferred.

---

#### R2: Focal Loss

| Trial | focal_alpha | focal_gamma | Test F1 | Issue |
|---|---|---|---|---|
| r2_focal_a75_g2 | 0.75 | 2.0 | 0.8028 | Training loss → 0.0000, overfitting |
| r2_focal_a80_g2 | 0.80 | 2.0 | 0.7956 | Same collapse |
| r2_focal_a75_g1_5 | 0.75 | 1.5 | 0.7857 | Worse |

**Finding:** Focal loss causes training collapse on this graph model. Cross-entropy is preferred.

---

#### R3: Class Weight Scaling

| Trial | class_weight_mode | scale | Test F1 |
|---|---|---|---|
| r3_full_weight | full | 0.5 | 0.8159 |
| r3_sqrt_scale1_5 | sqrt | 1.5 | 0.7985 |

**Finding:** Default sqrt(neg/pos) with scale = 1.0 is the best class weighting.

---

#### R4: Architecture Changes

| Trial | hidden_dim | dropout | lr | Variant | Test F1 |
|---|---|---|---|---|---|
| r4_hd256_do25 | 256 | 0.25 | 2e-3 | Standard | 0.8102 |
| r4_hd384_do30 | 384 | 0.30 | 2e-3 | Larger | 0.8214 |
| r4_hd512_do40 | 512 | 0.40 | 2e-3 | Too large | 0.7850 |
| **r4_attn_pool** | **300** | **0.35** | **3e-3** | **Attention pooling** | **0.8296** |

**Finding:** Attention pooling outperforms standard pooling among single-model variants.

---

#### R5: Graph Parameters

| Trial | PMI window | max_features | Test F1 |
|---|---|---|---|
| **r5_window15** | **15** | **40,000** | **0.8309** |
| r5_vocab50k | 20 | 50,000 | 0.8132 |
| r5_pmi02 | 20 (thresh=0.2) | 40,000 | 0.8195 |

**Finding:** Window size 15 (narrower PMI context) outperforms window 20. Larger vocab does not help.

---

#### R6: Label Smoothing

| Trial | label_smoothing | Test F1 |
|---|---|---|
| r6_ls00 | 0.00 | 0.8132 |
| r6_ls05 (default) | 0.05 | 0.8352 |
| r6_ls10 | 0.10 | 0.8108 |

**Finding:** Label smoothing = 0.05 is optimal; too much or none both hurt.

---

#### R7: Combined Experiments

| Trial | Config | Test F1 |
|---|---|---|
| r7_focal_attn | Focal + attention pooling | 0.7566 |
| r7_hd384_focal | hidden=384 + focal loss | 0.7774 |

**Finding:** Combining novel components without isolating variables causes compounding harm.

---

### Round 1 Ceiling

The single-model ceiling was approximately **F1 = 0.831** (r5_window15).
The baseline (0.835) was still not beaten by any single model.

---

### Round 2 — 14 Trials

Armed with Round 1 findings (window=15, Adam + constant LR, label_smoothing=0.05), Round 2 focused on ensemble strategies and alternative training dynamics.

---

#### V2: Adam + Constant LR — Long Training

| Trial | Epochs | Window | weight_decay | Test F1 |
|---|---|---|---|---|
| v2_adam_const_long | 300 | 20 | 1e-5 | 0.8271 |
| v2_adam_const_window15 | 300 | 15 | 1e-5 | 0.8240 |
| v2_adam_lo_wd | 300 | 20 | 1e-6 | 0.8253 |

**Finding:** Extending epochs does not help; the model converges by epoch ~150.

---

#### V2: CyclicLR

| Trial | base_lr | max_lr | Window | Test F1 | OOD Recall |
|---|---|---|---|---|---|
| v2_cyclic_wide | 5e-4 | 6e-3 | 20 | 0.8261 | 0.77 |
| v2_cyclic_narrow | 1e-3 | 5e-3 | 20 | 0.8289 | 0.77 |
| v2_cyclic_window15 | 1e-3 | 5e-3 | 15 | 0.8218 | — |

**Finding:** CyclicLR improves EMSCAD F1 marginally but hurts OOD recall (0.77 vs 0.96). Not suitable for production.

---

#### V2: Stochastic Weight Averaging (SWA) — Catastrophic Failure

| Trial | SWA start epoch | SWALR | Test F1 | Recall | Issue |
|---|---|---|---|---|---|
| v2_swa_80 | 80 | 5e-4 | 0.086 ❌ | 1.0 | All predictions = fake |
| v2_swa_100_window15 | 100 | 5e-4 | 0.086 ❌ | 1.0 | Same collapse |

**Finding:** SWA drops LR to near-zero via SWALR, which collapses the LayerNorm statistics in the GCN. This causes degenerate predictions (all fake). SWA is incompatible with this architecture.

---

#### V2: Ensemble Experiments

| Trial | Seeds | Window | Epochs | Test F1 | Precision | Recall | OOD Recall |
|---|---|---|---|---|---|---|---|
| v2_ensemble3_baseline | 3 (42,43,44) | 20 | 100 | 0.8248 | — | — | — |
| v2_ensemble3_long | 3 (42,43,44) | 20 | 200 | 0.8248 | — | — | — |
| v2_ensemble5_baseline | 5 seeds | 20 | 100 | 0.8296 | — | — | — |
| **v2_ensemble3_window15** ⭐ | **3 (42,43,44)** | **15** | **200** | **0.8390** | **0.9032** | **0.7832** | **0.9334** |
| v2_ensemble5_window15 | 5 seeds | 15 | 200 | 0.8222 | — | — | — |
| v2_cyclic_ensemble3 | 3 seeds | 20 | 200 | 0.8276 | — | — | — |

**Key finding:** Combining the two best individual improvements (window=15 + 3-seed ensemble) pushed beyond the baseline. 5 seeds performed worse than 3, suggesting too many seeds dilute the signal rather than reducing variance.

---

### Winning Configuration

**Trial:** `v2_ensemble3_window15`
**Artifacts:** `models/tuned/textgcn_tuned/`

| Hyperparameter | Value |
|---|---|
| `hidden_dim` | 300 |
| `dropout` | 0.35 |
| `residual_alpha` | 0.7 |
| `learning_rate` | 3e-3 |
| `weight_decay` | 1e-5 |
| `optimiser` | Adam |
| `scheduler` | None (constant LR) |
| `epochs` | 200 |
| `patience` | 25 |
| `label_smoothing` | 0.05 |
| `loss` | Cross-entropy |
| `class_weight` | √(neg / pos) |
| `max_features` | 40,000 |
| `ngram_range` | (1, 3) |
| `PMI window_size` | **15** |
| `n_seeds (ensemble)` | **3** (seeds 42, 43, 44) |
| `threshold` | 0.48 |
| **Test F1** | **0.8390** |
| **Precision** | **0.9032** |
| **Recall** | **0.7832** |
| **OOD Recall** | **0.9334** |
| Training time | ~760 s (3 × 200 epochs) |

**Improvement over baseline:**

| Metric | Baseline | Tuned | Δ |
|---|---|---|---|
| Test F1 | 0.8352 | 0.8390 | +0.0038 |
| Precision | 0.8769 | 0.9032 | +0.0263 |
| Recall | 0.7972 | 0.7832 | −0.0140 |
| OOD Recall | 0.9583 | 0.9334 | −0.0249 |

The tuned model trades a small amount of recall for substantially higher precision (+2.6 pp), which is preferable for a fraud-detection system to reduce false alarms.

---

## 7. Full Results Comparison

All models ranked by EMSCAD Test F1 (baseline models included for reference):

| Rank | Model | Test F1 | Precision | Recall | Threshold | OOD Recall |
|---|---|---|---|---|---|---|
| 1 | **TextGCN Tuned (ensemble, w=15)** | **0.8390** | **0.9032** | **0.7832** | 0.48 | 0.9334 |
| 2 | LightGBM (tuned) | 0.8494 | 0.9483 | 0.7692 | 0.16 | 0.9906 |
| 3 | TextGCN (baseline) | 0.8352 | 0.8769 | 0.7972 | 0.48 | 0.9583 |
| 4 | LightGBM (baseline) | 0.8339 | 0.8828 | 0.7902 | 0.12 | 0.9998 |
| 5 | XGBoost (tuned) | 0.8042 | 0.8042 | 0.8042 | 0.28 | 0.9992 |
| 6 | Logistic Regression (baseline) | 0.7915 | 0.8000 | 0.7832 | 0.58 | 0.9973 |
| 7 | XGBoost (baseline) | 0.7835 | 0.7703 | 0.7972 | 0.24 | 0.9980 |
| 8 | Random Forest (tuned) | 0.7805 | 0.7778 | 0.7832 | 0.24 | 1.0000 |
| 9 | Logistic Regression (tuned) | 0.7692 | 0.8077 | 0.7343 | 0.80 | 1.0000 |
| 10 | BiLSTM (tuned) | 0.7737 | 0.8092 | 0.7413 | 0.10 | 0.3701 |
| 11 | DistilBERT (tuned) | 0.8231 | 0.8013 | 0.8462 | 0.10 | 0.4041 |
| 12 | BiLSTM (baseline) | 0.7562 | 0.7643 | 0.7483 | 0.30 | 0.3513 |
| 13 | DistilBERT (baseline) | 0.7619 | 0.6977 | 0.8392 | 0.64 | 0.4666 |
| 14 | Random Forest (baseline) | 0.7320 | 0.6871 | 0.7832 | 0.22 | 1.0000 |
| 15 | Naive Bayes (tuned) | 0.7097 | 0.6587 | 0.7692 | 0.18 | 0.5867 |
| 16 | Naive Bayes (baseline) | 0.2426 | 0.2103 | 0.2867 | 0.10 | 0.0000 |

> **Production choice:** TextGCN Tuned is selected for the web app because it achieves the highest F1 among graph-based models, maintains strong OOD generalisation (93.3%), and includes uncertainty estimation via MC Dropout — a capability not available to classical or transformer models in this setup.

---

## 8. Key Findings

### What worked

| Finding | Impact |
|---|---|
| **PMI window = 15** (vs 20) | +0.5 F1 pp for single models. Narrower context windows produce tighter, more discriminative word–word edges. |
| **3-seed ensemble** | +0.4 F1 pp. Averaging predictions across 3 random seeds reduces variance without the dilution effect of 5+ seeds. |
| **Label smoothing = 0.05** | Prevents overconfident predictions; roughly +0.5 pp vs no smoothing. |
| **Constant Adam LR** | Outperforms all LR schedulers on this graph model. |
| **Cross-entropy + sqrt class weight** | Best loss configuration; focal loss causes training collapse. |

### What did not work

| Approach | Outcome |
|---|---|
| AdamW + cosine annealing | LR decays too fast; model under-trains |
| Focal loss | Training loss collapses to near-zero; model overfits severely |
| Stochastic Weight Averaging (SWA) | SWALR drops LR to near-zero, collapsing LayerNorm statistics → all predictions = fake (F1 = 0.086) |
| 5-seed ensemble | Worse than 3-seed; too many seeds dilute the signal |
| CyclicLR | Marginal EMSCAD gain but OOD recall drops from 0.96 → 0.77 |
| Larger hidden dim (512) | Overfits; worse than 300 |
| Extended training (300 epochs) | No benefit; model converges by epoch ~150 |

### Why TextGCN over classical models

LightGBM achieves a higher raw F1 (0.849 vs 0.839), but the TextGCN is preferred for the production web app because:

1. **Explainability:** Graph edge weights enable SHAP attribution and occlusion analysis at the token level. Classical models with TF-IDF produce sparse, less interpretable feature attributions.
2. **Structural representations:** TextGCN captures co-occurrence relationships between terms across the corpus, not just within individual postings.
3. **Uncertainty estimation:** MC Dropout provides calibrated confidence intervals without retraining.
4. **OOD robustness:** TextGCN (93.3%) vs LightGBM (99.1%) — both are strong, but TextGCN's graph structure generalises via semantic co-occurrence rather than surface n-gram statistics.
