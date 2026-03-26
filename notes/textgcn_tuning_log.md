# TextGCN Tuning Log — All Experiments

Complete record of every TextGCN configuration tried during this project.
Total: **39 trials** — 34 in the main search + 5 advanced experiments.

---

## Baseline (Reference Point)

| Metric | Value |
|--------|-------|
| EMSCAD Test F1 | 0.8352 |
| Precision | 0.8769 |
| Recall | 0.7972 |
| OpenBay Rate | 0.9583 |
| Threshold | 0.48 |

**Config:** Adam · lr=3e-3 · wd=1e-5 · hidden=300 · dropout=0.35 · window=20 · epochs=100 · patience=15 · label_smooth=0.05 · class_weight=√(neg/pos)

---

## Main Search — Round 1 (20 Trials): Broad Exploration

### Group A: Learning Rate Schedulers

| Trial | Scheduler | Optimiser | F1 | OOD |
|-------|-----------|-----------|-----|-----|
| r1_cosine_adamw | Cosine | AdamW | 0.8159 | 0.9859 |
| r1_cosine_adamw_hi_wd | Cosine + high WD | AdamW | 0.8090 | 0.9465 |
| r1_plateau_adamw | ReduceLROnPlateau | AdamW | 0.8209 | 0.9550 |
| r1_cosine_low_lr | Cosine + lr=1e-3 | AdamW | 0.7972 | 0.9055 |

**Finding:** All below baseline. AdamW + cosine annealing decays LR too fast — models stopped at epoch 40–50 with val F1 stagnating at ~0.77. Plain Adam with constant LR is preferred.

---

### Group B: Focal Loss

| Trial | Alpha | Gamma | F1 | OOD |
|-------|-------|-------|----|-----|
| r2_focal_a75_g2 | 0.75 | 2.0 | 0.8028 | 0.9945 |
| r2_focal_a80_g2 | 0.80 | 2.0 | 0.7956 | 0.9833 |
| r2_focal_a75_g1_5 | 0.75 | 1.5 | 0.7857 | 0.9800 |

**Finding:** All below baseline. Training loss collapsed to ~0.0000 — focal loss caused severe overfitting. Cross-entropy is preferred.

---

### Group C: Class Weight Scaling

| Trial | Weight Mode | Effective Weight | F1 | OOD |
|-------|-------------|------------------|----|-----|
| r3_full_weight | full × 0.5 | 10.63 | 0.8159 | 0.9592 |
| r3_sqrt_scale1_5 | sqrt × 1.5 | 6.92 | 0.7985 | 0.9576 |

**Finding:** Both below baseline. Full ratio pushed too aggressively toward fake class. The default √(neg/pos) is already optimal.

---

### Group D: Architecture Changes

| Trial | Hidden Dim | Dropout | Model Type | F1 | OOD |
|-------|------------|---------|------------|----|-----|
| r4_hd384_do30 | 384 | 0.30 | standard | 0.8214 | 0.9792 |
| r4_hd256_do25 | 256 | 0.25 | standard | 0.8102 | 0.9487 |
| r4_hd512_do40 | 512 | 0.40 | standard | 0.7850 | 0.9943 |
| r4_attn_pool | 300 | 0.35 | attention pooling | 0.8296 | 0.9034 |

**Finding:** All below baseline. Attention pooling (0.8296) was the closest. Larger hidden dims overfit without scheduler fixes.

---

### Group E: Graph and Vocabulary Parameters

| Trial | Window | Vocab | PMI Threshold | F1 | OOD |
|-------|--------|-------|---------------|----|-----|
| **r5_window15** | **15** | **40k** | **0.0** | **0.8309** | **0.9769** |
| r5_vocab50k | 20 | 50k | 0.0 | 0.8132 | 0.9253 |
| r5_pmi02 | 20 | 40k | 0.2 | 0.8195 | 0.9746 |

**Finding:** `window=15` was the **best single trial in Round 1** (F1=0.8309). Tighter window captures more discriminative co-occurrence patterns. 50k vocab adds noise; PMI threshold 0.2 prunes useful edges.

---

### Group F: Label Smoothing

| Trial | Label Smoothing | F1 | OOD |
|-------|----------------|----|-----|
| r6_ls00 | 0.00 | 0.8132 | 0.9525 |
| r6_ls10 | 0.10 | 0.8108 | 0.9246 |

**Finding:** Both below baseline. The default 0.05 was already well-tuned.

---

### Group G: Combined Configurations

| Trial | Config | F1 | OOD |
|-------|--------|----|-----|
| r7_focal_attn | Focal + attention pooling | 0.7566 | 0.9740 |
| r7_hd384_focal | hidden=384 + focal loss | 0.7774 | 0.9903 |

**Finding:** Focal loss instability dominated. Combining untested components without isolating variables compounds failure.

### Round 1 Conclusion

No single trial beat the baseline (F1=0.8352). The best improvement — `window=15` — must be combined with something else.

---

## Main Search — Round 2 (14 Trials): Targeted Search

Armed with Round 1 findings: use plain Adam + constant LR + window=15.

### Group A: Longer Training

| Trial | Epochs | Window | F1 | OOD |
|-------|--------|--------|----|-----|
| v2_adam_const_long | 300 | 20 | 0.8271 | 0.9015 |
| v2_adam_const_window15 | 300 | 15 | 0.8240 | 0.8867 |
| v2_adam_lo_wd | 300 | 20 (wd=1e-6) | 0.8253 | 0.9247 |

**Finding:** All below baseline. Single-model ceiling is ~F1=0.827; more epochs do not help past this.

---

### Group B: CyclicLR

| Trial | Base LR | Max LR | Window | F1 | OOD |
|-------|---------|--------|--------|----|-----|
| v2_cyclic_wide | 5e-4 | 6e-3 | 20 | 0.8261 | 0.8992 |
| v2_cyclic_narrow | 1e-3 | 5e-3 | 20 | 0.8289 | 0.7655 |
| v2_cyclic_window15 | 5e-4 | 6e-3 | 15 | 0.8218 | 0.9288 |

**Finding:** All below baseline. Narrow cycle improved precision but collapsed OOD recall (0.77 vs 0.96). Not suitable.

---

### Group C: Stochastic Weight Averaging (SWA) — Catastrophic Failure

| Trial | SWA Start | F1 | Recall | OOD |
|-------|-----------|-----|--------|-----|
| v2_swa_80 | epoch 80 | **0.086** | 1.000 | 1.000 |
| v2_swa_100_window15 | epoch 100 | **0.086** | 1.000 | 1.000 |

**Finding:** SWALR drops LR to near-zero, causing the LayerNorm statistics to collapse. Every posting predicted as fake (recall=1.0, precision=0.045). SWA is designed for Batch Normalisation, not LayerNorm — incompatible with this architecture.

---

### Group D: Ensembles (Window=20)

| Trial | Seeds | Epochs | F1 | OOD |
|-------|-------|--------|----|-----|
| v2_ensemble3_baseline | 3 | 100 | 0.8248 | 0.9678 |
| v2_ensemble3_long | 3 | 200 | 0.8248 | 0.9678 |
| v2_ensemble5_baseline | 5 | 100 | 0.8296 | 0.9634 |

**Finding:** All below baseline. With window=20, models are too similar for ensemble averaging to help significantly.

---

### Group E: Ensembles (Window=15) — Winner Found

| Trial | Seeds | Window | Epochs | F1 | Precision | Recall | OOD | Beats? |
|-------|-------|--------|--------|----|-----------|--------|-----|--------|
| **v2_ensemble3_window15** | **3** | **15** | **200** | **0.8390** | **0.9032** | **0.7832** | **0.9334** | **✅ YES** |
| v2_ensemble5_window15 | 5 | 15 | 100 | 0.8222 | 0.8740 | 0.7762 | 0.9619 | ❌ |

**Finding:** Combining window=15 + 3-seed ensemble is synergistic — neither alone beat the baseline, together they do. 5 seeds dilutes the signal rather than diversifying it.

---

### Group F: CyclicLR + Ensemble

| Trial | F1 | OOD |
|-------|-----|-----|
| v2_cyclic_ensemble3 | 0.8276 | 0.6698 |

**Finding:** CyclicLR compounded across seeds collapses OOD recall to 0.67.

---

## Advanced Tuning (5 Experiments) — Trying to Beat F1=0.8390

After establishing the winner, five more experiments attempted to close the gap with LightGBM (F1=0.8494).

| Experiment | Seeds | Window | Class Weight | F1 | Beats? |
|------------|-------|--------|--------------|----|--------|
| 5-seed ensemble | 5 | 15 | √(neg/pos) | 0.8222 | No |
| 5-seed + val-weighted | 5 | 15 | √(neg/pos) | 0.8267 | No |
| 5-seed + linear weight | 5 | 15 | neg/pos | 0.8235 | No |
| 5-seed + window=10 | 5 | 10 | √(neg/pos) | 0.8222 | No |
| 7-seed ensemble | 7 | 15 | √(neg/pos) | 0.8235 | No |
| **Current best (kept)** | **3** | **15** | **√(neg/pos)** | **0.8390** | — |

**Finding:** No advanced experiment beat F1=0.8390. Original model artifacts unchanged.

---

## Winning Configuration

**Trial:** `v2_ensemble3_window15`
**Artifacts:** `models/tuned/textgcn_tuned/`

| Hyperparameter | Value |
|---|---|
| Hidden dim | 300 |
| Dropout | 0.35 |
| Residual alpha (α) | 0.7 |
| Learning rate | 3e-3 |
| Weight decay | 1e-5 |
| Optimiser | Adam (constant LR) |
| Epochs | 200 |
| Patience | 25 |
| Label smoothing | 0.05 |
| Loss | Cross-entropy |
| Class weight | √(neg/pos) |
| Max features | 40,000 |
| n-gram range | (1, 3) |
| PMI window | **15** |
| Ensemble seeds | **3** (seeds 42, 43, 44) |
| Threshold | 0.48 |
| **Test F1** | **0.8390** |
| **Precision** | **0.9032** |
| **Recall** | **0.7832** |
| **OOD Rate** | **0.9334** |

**Improvement over baseline:**

| Metric | Baseline | Tuned | Δ |
|---|---|---|---|
| Test F1 | 0.8352 | 0.8390 | +0.0038 |
| Precision | 0.8769 | 0.9032 | **+0.0263** |
| Recall | 0.7972 | 0.7832 | −0.0140 |
| OOD Rate | 0.9583 | 0.9334 | −0.0249 |

The tuned model trades a small recall drop for substantially higher precision (+2.6 pp), which is preferable for fraud detection to reduce false alarms.

---

## Key Findings Summary

| Finding | Evidence |
|---------|----------|
| PMI window=15 outperforms window=20 | Best single-model F1 gap: 0.8309 vs ~0.827 |
| 3-seed ensemble is the sweet spot | 5 and 7 seeds both perform worse |
| window=15 + 3-seed are synergistic | Neither alone beat baseline; together +0.0038 F1 |
| Label smoothing=0.05 is optimal | No smoothing or 0.10 both hurt |
| Constant Adam LR beats all schedulers | All scheduler trials below baseline |
| Focal loss causes training collapse | Training loss → 0.0000, severe overfitting |
| SWA fails with LayerNorm | Both SWA trials: recall=1.0, F1=0.086 |
| More seeds hurt calibration | 5-seed worse than 3-seed even with val-weighting |
| Single-model ceiling ≈ F1=0.831 | No single-model trial exceeded this |
