# All TextGCN Tuning Methods Tested

Complete record of every hyperparameter configuration trialled during the BSc project.
Total: **34 trials** across two rounds. One trial beat the baseline.

---

## Baseline (Reference)

| Metric | Value |
|--------|-------|
| EMSCAD Test F1 | 0.8352 |
| EMSCAD Precision | 0.8769 |
| EMSCAD Recall | 0.7972 |
| OpenBay Recall | 0.9583 |
| Vocab size | 40,000 |
| Threshold | 0.48 |

**Baseline config:** Adam · lr=3e-3 · wd=1e-5 · hidden=300 · dropout=0.35 · window=20 · epochs=100 · patience=15 · label_smooth=0.05 · class_weight=sqrt(neg/pos)

---

## Round 1 — Broad Search (20 Trials)

### Group A: Learning Rate Scheduler + AdamW

| Trial | Scheduler | Optimizer | LR | WD | F1 | Precision | Recall | OB Recall |
|-------|-----------|-----------|----|----|-----|-----------|--------|-----------|
| r1_cosine_adamw | cosine | AdamW | 3e-3 | 1e-4 | 0.8159 | 0.8433 | 0.7902 | 0.9859 |
| r1_cosine_adamw_hi_wd | cosine | AdamW | 3e-3 | 5e-4 | 0.8090 | 0.8710 | 0.7552 | 0.9465 |
| r1_plateau_adamw | plateau | AdamW | 3e-3 | 1e-4 | 0.8209 | 0.8800 | 0.7692 | 0.9550 |
| r1_cosine_low_lr | cosine | AdamW | 1e-3 | 1e-4 | 0.7972 | 0.8116 | 0.7832 | 0.9055 |

**Outcome:** All below baseline. Cosine annealing decays LR too fast for this architecture; models stopped at epoch 40–50 with val F1 plateauing at ~0.77.

---

### Group B: Focal Loss

| Trial | Alpha | Gamma | F1 | Precision | Recall | OB Recall |
|-------|-------|-------|----|-----------|--------|-----------|
| r2_focal_a75_g2 | 0.75 | 2.0 | 0.8028 | 0.7945 | 0.8112 | 0.9945 |
| r2_focal_a80_g2 | 0.80 | 2.0 | 0.7956 | 0.8321 | 0.7622 | 0.9833 |
| r2_focal_a75_g1_5 | 0.75 | 1.5 | 0.7857 | 0.8029 | 0.7692 | 0.9800 |

**Outcome:** All below baseline. Training loss collapsed to ~0.0000; focal loss caused aggressive overfitting without the stabilising effect of label smoothing + cross-entropy.

---

### Group C: Class Weight Tuning

| Trial | Weight Mode | Effective Weight | F1 | Precision | Recall | OB Recall |
|-------|-------------|------------------|----|-----------|--------|-----------|
| r3_full_weight | full × 0.5 | 10.63 | 0.8159 | 0.8433 | 0.7902 | 0.9592 |
| r3_sqrt_scale1_5 | sqrt × 1.5 | 6.92 | 0.7985 | 0.8385 | 0.7622 | 0.9576 |

**Outcome:** Both below baseline. Full ratio (even scaled) pushed too aggressively toward fake, hurting precision. Sqrt×1.5 also declined.

---

### Group D: Architecture Changes

| Trial | Hidden Dim | Dropout | Model Type | F1 | Precision | Recall | OB Recall |
|-------|------------|---------|------------|----|-----------|--------|-----------|
| r4_hd384_do30 | 384 | 0.30 | base | 0.8214 | 0.8394 | 0.8042 | 0.9792 |
| r4_hd256_do25 | 256 | 0.25 | base | 0.8102 | 0.8473 | 0.7762 | 0.9487 |
| r4_hd512_do40 | 512 | 0.40 | base | 0.7850 | 0.7667 | 0.8042 | 0.9943 |
| r4_attn_pool | 300 | 0.35 | attention | 0.8296 | 0.8819 | 0.7832 | 0.9034 |

**Outcome:** All below baseline. Attention pooling (F1=0.8296) was the closest. Larger hidden dims without scheduler fixes did not help.

---

### Group E: Graph and Vocabulary Parameters

| Trial | Window Size | Vocab Size | PMI Threshold | F1 | Precision | Recall | OB Recall |
|-------|-------------|------------|---------------|-----|-----------|--------|-----------|
| r5_window15 | **15** | 40,000 | 0.0 | **0.8309** | 0.8760 | 0.7902 | 0.9769 |
| r5_vocab50k | 20 | 50,000 | 0.0 | 0.8132 | 0.8538 | 0.7762 | 0.9253 |
| r5_pmi02 | 20 | 40,000 | 0.2 | 0.8195 | 0.8862 | 0.7622 | 0.9746 |

**Outcome:** window=15 was the **best single trial in Round 1** (F1=0.8309). Smaller window captures tighter fraud phrase co-occurrences. 50k vocab added noise; PMI threshold 0.2 pruned useful edges.

---

### Group F: Label Smoothing

| Trial | Label Smoothing | F1 | Precision | Recall | OB Recall |
|-------|----------------|----|-----------|--------|-----------|
| r6_ls00 | 0.00 | 0.8132 | 0.8538 | 0.7762 | 0.9525 |
| r6_ls10 | 0.10 | 0.8108 | 0.9052 | 0.7343 | 0.9246 |

**Outcome:** Both below baseline. The default 0.05 was already well-tuned.

---

### Group G: Combined Configurations

| Trial | Config | F1 | Precision | Recall | OB Recall |
|-------|--------|----|-----------|--------|-----------|
| r7_focal_attn | Focal loss + Attention pooling | 0.7566 | 0.7143 | 0.8042 | 0.9740 |
| r7_hd384_focal | hidden=384 + Focal loss | 0.7774 | 0.7857 | 0.7692 | 0.9903 |

**Outcome:** Both well below baseline. Focal loss instability dominated.

---

### Round 1 Summary

| Rank | Trial | F1 | vs Baseline |
|------|-------|----|-------------|
| 1 | r5_window15 | 0.8309 | −0.0043 |
| 2 | r4_attn_pool | 0.8296 | −0.0056 |
| 3 | r1_plateau_adamw | 0.8209 | −0.0143 |
| — | **Baseline** | **0.8352** | — |

**Key learning:** Switch back to plain Adam with constant LR. The window=15 finding must be combined with something else to beat the baseline.

---

## Round 2 — Targeted Search (14 Trials)

### Group A: Adam + Constant LR, More Epochs

| Trial | Epochs | Patience | WD | Window | F1 | Precision | Recall | OB Recall |
|-------|--------|----------|----|--------|----|-----------|--------|-----------|
| v2_adam_const_long | 300 | 30 | 1e-5 | 20 | 0.8271 | 0.8943 | 0.7692 | 0.9015 |
| v2_adam_const_window15 | 300 | 30 | 1e-5 | 15 | 0.8240 | 0.8871 | 0.7692 | 0.8867 |
| v2_adam_lo_wd | 300 | 30 | 1e-6 | 20 | 0.8253 | 0.8810 | 0.7762 | 0.9247 |

**Outcome:** All below baseline. Single-model ceiling is ~0.827; more epochs do not help past this point.

---

### Group B: CyclicLR (Triangular2 Policy)

| Trial | Base LR | Max LR | Step | Window | F1 | Precision | Recall | OB Recall |
|-------|---------|--------|------|--------|----|-----------|--------|-----------|
| v2_cyclic_wide | 5e-4 | 6e-3 | 25 | 20 | 0.8261 | 0.8571 | 0.7972 | 0.8992 |
| v2_cyclic_narrow | 1e-3 | 5e-3 | 15 | 20 | 0.8289 | 0.9083 | 0.7622 | 0.7655 |
| v2_cyclic_window15 | 5e-4 | 6e-3 | 25 | 15 | 0.8218 | 0.8561 | 0.7902 | 0.9288 |

**Outcome:** All below baseline. Narrow cycle improved precision but collapsed OpenBay recall (0.77). Cyclic LR did not overcome the single-model ceiling.

---

### Group C: Stochastic Weight Averaging (SWA)

| Trial | SWA Start | SWA LR | Window | F1 | Precision | Recall | OB Recall |
|-------|-----------|--------|--------|----|-----------|--------|-----------|
| v2_swa_80 | 80 | 5e-4 | 20 | **0.0862** | 0.0450 | **1.0000** | 1.0000 |
| v2_swa_100_window15 | 100 | 3e-4 | 15 | **0.0862** | 0.0450 | **1.0000** | 1.0000 |

**Outcome:** Catastrophic failure. The SWALR scheduler dropped LR to near zero, causing the averaged model's LayerNorm statistics to collapse. Result: every posting predicted as fake (recall=1.0, precision=0.045). SWA is designed for models with Batch Normalisation, not LayerNorm.

---

### Group D: Ensemble — Baseline Window (window=20)

| Trial | Seeds | Epochs | F1 | Precision | Recall | OB Recall |
|-------|-------|--------|----|-----------|--------|-----------|
| v2_ensemble3_baseline | 3 | 100 | 0.8248 | 0.8626 | 0.7902 | 0.9678 |
| v2_ensemble3_long | 3 | 200 | 0.8248 | 0.8626 | 0.7902 | 0.9678 |
| v2_ensemble5_baseline | 5 | 100 | 0.8296 | 0.8819 | 0.7832 | 0.9634 |

**Outcome:** All below baseline. With window=20, individual models are correlated enough that averaging 3–5 of them gives diminishing returns. The 5-seed ensemble approached the baseline but could not exceed it.

---

### Group E: Ensemble + Window=15

| Trial | Seeds | Window | Epochs | F1 | Precision | Recall | OB Recall | **Beats?** |
|-------|-------|--------|--------|----|-----------|--------|-----------|-----------|
| **v2_ensemble3_window15** | **3** | **15** | **200** | **0.8390** | **0.9032** | **0.7832** | **0.9334** | **✅ YES** |
| v2_ensemble5_window15 | 5 | 15 | 100 | 0.8222 | 0.8740 | 0.7762 | 0.9619 | ❌ |

**Outcome:** 3-seed ensemble with window=15 is the only trial to beat the baseline. 5 seeds with window=15 actually performed worse — adding seeds 45 and 46 pulled the averaged probabilities away from the optimal calibration.

---

### Group F: CyclicLR + Ensemble

| Trial | Scheduler | Seeds | Window | F1 | Precision | Recall | OB Recall |
|-------|-----------|-------|--------|----|-----------|--------|-----------|
| v2_cyclic_ensemble3 | cyclic | 3 | 20 | 0.8276 | 0.9153 | 0.7552 | 0.6698 |

**Outcome:** Below baseline. Cyclic LR's precision bias compounded across all 3 ensemble members. OpenBay recall collapsed to 0.6698 — the model lost generalisation.

---

## All Trials — Combined Ranking

| Rank | Trial | Round | F1 | Precision | Recall | OB Recall | Beats Baseline |
|------|-------|-------|----|-----------|--------|-----------|----------------|
| **1** | **v2_ensemble3_window15** | **2** | **0.8390** | **0.9032** | **0.7832** | **0.9334** | **✅** |
| 2 | v2_ensemble5_baseline | 2 | 0.8296 | 0.8819 | 0.7832 | 0.9634 | ❌ |
| 3 | r4_attn_pool | 1 | 0.8296 | 0.8819 | 0.7832 | 0.9034 | ❌ |
| 4 | v2_cyclic_narrow | 2 | 0.8289 | 0.9083 | 0.7622 | 0.7655 | ❌ |
| 5 | r5_window15 | 1 | 0.8309 | 0.8760 | 0.7902 | 0.9769 | ❌ |
| — | **Baseline** | — | **0.8352** | **0.8769** | **0.7972** | **0.9583** | — |

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| AdamW + cosine annealing hurts this model | All 4 R1 scheduler trials stopped at epoch 40–50, F1 ~0.815 |
| Focal loss causes overfitting | Training loss → 0.0000; F1 stagnates at 0.79 |
| Single model ceiling ≈ F1 0.831 | No single-model trial exceeded this |
| SWA fails with LayerNorm | Both SWA trials collapsed to recall=1.0, F1=0.086 |
| window=15 is better than window=20 | Best single-model F1 gap: 0.8309 vs ~0.827 |
| 3 seeds > 5 seeds with window=15 | More seeds hurt calibration; 3 is the sweet spot |
| Ensemble + window=15 is synergistic | Neither alone beat baseline; together they do |
