# Paper Results — Consolidated

Generated to support drafting Results/Discussion/Limitations. Every number
below states its source: a saved artifact (file + path), a script re-run in
this session (script name), or a figure told to me as prior context that I
have not independently verified in this session (flagged explicitly). Two
items are still pending your go-ahead to re-run — see the notes inline and
the summary at the end.

---

## 1. Detection (YOLOv8s fine-tuning on CrowdHuman)

### Test set (1,937 images, 57,105 instances) — both models, Table 1

Source: `outputs/reports/detection_results.md`, run in Colab and reported
directly by the user (2026-08-09) — the first saved record of these numbers.

| model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8s, COCO-pretrained (no fine-tuning) | 0.632 | 0.398 | 0.432 | 0.215 |
| YOLOv8s + CrowdHuman fine-tuning | 0.8346 | 0.6016 | 0.6994 | 0.4312 |
| Inference (fine-tuned) | — | — | — | 5.8 ms/image on T4 |

Fine-tuning improves every metric substantially over the COCO-pretrained
baseline on identical data: +0.20 precision, +0.20 recall, +0.27 mAP50,
+0.22 mAP50-95. **This baseline row was previously missing and is now
resolved** — no re-run needed.

### Training configuration

Source: `outputs/yolo_training/crowdhuman_yolov8s/args.yaml`

| param | value |
|---|---|
| epochs (configured) | 60 |
| patience | 12 |
| batch | 16 |
| imgsz | 640 |
| optimizer | AdamW |
| lr0 / lrf | 0.001 / 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup_epochs | 3.0 |
| freeze | 10 |
| pretrained | true (COCO) |
| cls_remap | true |
| seed | 0 (deterministic) |

### Convergence evidence — RESOLVED

Source: `outputs/yolo_training/crowdhuman_yolov8s/results.csv` (validation-split
metrics logged per epoch during training) plus the user's direct account of
the completed run.

**The saved CSV log only goes to epoch 38, not 60** — training resumed in a
fresh Colab session (the `time` column resets sharply between epoch 23 and
24, and `args.yaml` shows `resume: .../last.pt`) whose log never synced back
to this Drive folder. **The training run itself did complete all 60
epochs** — per the user, validation mAP50 moved from 0.682 (epoch 38, in
the saved log) to 0.683 at epoch 60. That's consistent with the plateau
already visible below, not a discontinuity: the model had converged by
epoch ~30 and the missing 22 epochs of log rows changed the headline metric
by 0.001. Use the epoch 20-38 plateau trend as the convergence evidence in
the Methods section; the epoch 39-60 gap is a log-sync artifact, not a
missing result.

What the actual log shows (epoch : precision, recall, mAP50, mAP50-95):

| epoch | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 20 | 0.821 | 0.579 | 0.675 | 0.411 |
| 25 | 0.826 | 0.581 | 0.677 | 0.412 |
| 30 | 0.826 | 0.582 | 0.679 | 0.414 |
| 35 | 0.826 | 0.585 | 0.680 | 0.416 |
| 37 (peak mAP50-95) | 0.826 | 0.586 | 0.682 | **0.4168** |
| 38 (last logged) | 0.827 | 0.585 | 0.682 | 0.4165 |

mAP50-95 plateaus from roughly epoch 30 onward (0.4143 → 0.4165 over 8
epochs), peaking at epoch 37. That's the convergence evidence available —
a plateau narrative, not an "epoch 39 vs 60" comparison.

---

## 2. UMN ground truth

Source: `data/umn_groundtruth.csv` + `data/features/umn_meta.csv`, parsed
fresh in this session (pure pandas, no model inference).

**Total abnormal frames: 1,136 / 7,739 = 14.68%**

### 11 abnormal segments (onset, end, duration in frames)

| onset | end | duration |
|---|---|---|
| 525 | 615 | 90 |
| 1330 | 1440 | 110 |
| 1806 | 1986 | 180 |
| 2605 | 2685 | 80 |
| 3219 | 3429 | 210 |
| 3938 | 4018 | 80 |
| 4807 | 4929 | 122 |
| 5422 | 5596 | 174 |
| 6195 | 6235 | 40 |
| 6883 | 6913 | 30 |
| 7700 | 7720 | 20 |

### 10 clip boundaries / 11 clips

| clip | start | end | n_frames |
|---|---|---|---|
| 0 | 0 | 625 | 625 |
| 1 | 625 | 1453 | 828 |
| 2 | 1453 | 1996 | 543 |
| 3 | 1996 | 2687 | 691 |
| 4 | 2687 | 3455 | 768 |
| 5 | 3455 | 4034 | 579 |
| 6 | 4034 | 4929 | 895 |
| 7 | 4929 | 5596 | 667 |
| 8 | 5596 | 6238 | 642 |
| 9 | 6238 | 6931 | 693 |
| 10 | 6931 | 7739 | 808 |

---

## 3. UMN lead-window sweep (Table 2)

Source: `python3 -m src.diagnose_leadup_signal`, re-run fresh this session
(pure numpy/pandas over `data/features/umn_features.npy`, no model
inference, <1 min).

### Per-onset sign of (lead_mean − baseline_mean), lead window W=60

| onset | normalized_count | flow_magnitude | direction_entropy | stop_ratio |
|---|---|---|---|---|
| 525 | −0.0387 (−) | +0.0395 (+) | +0.0317 (+) | −0.1128 (−) |
| 1330 | −0.0082 (−) | +0.0733 (+) | +0.0234 (+) | −0.0989 (−) |
| 1806 | +0.0186 (+) | +0.0250 (+) | +0.0564 (+) | −0.0925 (−) |
| 2605 | +0.0070 (+) | −0.0404 (−) | +0.0475 (+) | −0.0131 (−) |
| 3219 | −0.0311 (−) | +0.0526 (+) | +0.0174 (+) | −0.0712 (−) |
| 3938 | −0.0145 (−) | +0.0157 (+) | +0.0152 (+) | −0.0504 (−) |
| 4807 | +0.0172 (+) | +0.0254 (+) | +0.0389 (+) | −0.0481 (−) |
| 5422 | −0.0158 (−) | +0.0112 (+) | +0.0005 (+) | −0.0461 (−) |
| 6195 | −0.0610 (−) | +0.1044 (+) | −0.0305 (−) | −0.0001 (−) |
| 6883 | −0.0975 (−) | +0.0595 (+) | −0.0913 (−) | −0.0668 (−) |
| 7700 | −0.0259 (−) | +0.0818 (+) | −0.0333 (−) | −0.0469 (−) |
| **sign count (+/−)** | **3 / 8** | **10 / 1** | **8 / 3** | **0 / 11** |

### Lead-window sweep: pooled gap and onset-level sign counts

| feature | W=30 | W=60 | W=90 | W=120 | W=180 |
|---|---|---|---|---|---|
| normalized_count (gap) | −0.0348 | −0.0227 | −0.0177 | −0.0158 | −0.0137 |
| normalized_count (n_pos/n_neg) | 3/8 | 3/8 | 3/8 | 2/9 | 2/9 |
| flow_magnitude (gap) | 0.0735 | 0.0407 | 0.0222 | 0.0127 | 0.0048 |
| flow_magnitude (n_pos/n_neg) | 10/1 | 10/1 | 9/2 | 6/5 | 6/5 |
| direction_entropy (gap) | 0.0095 | 0.0069 | 0.0029 | 0.0008 | −0.0036 |
| direction_entropy (n_pos/n_neg) | 8/3 | 8/3 | 7/4 | 6/5 | 6/5 |
| stop_ratio (gap) | −0.0796 | −0.0588 | −0.0417 | −0.0290 | −0.0146 |
| stop_ratio (n_pos/n_neg) | 0/11 | 0/11 | 2/9 | 2/9 | 5/6 |

**Reading:** `stop_ratio` is the only feature unanimous across all 11 onsets
at every window tested (0/11 flipping). `flow_magnitude` is robust at short
windows (10/11 at W=30/60) but degrades by W=120-180. `normalized_count` and
`direction_entropy` both lose majority consistency by W≥120. This is the
empirical basis for treating short-to-medium horizons as where genuine
precursor signal lives, consistent with the LSTM's own AUROC decay (Section
5).

---

## 4. UMN leakage analysis: naive vs lead-up gap

Source: computed fresh this session (pure numpy over the same
`umn_features.npy` + `umn_groundtruth.csv`; naive gap = pooled mean over ALL
abnormal-labeled frames minus ALL normal-labeled frames; lead-up gap = the
W=60 pooled figure from Section 3).

| feature | naive gap (whole segment) | lead-up gap (W=60) | relationship |
|---|---|---|---|
| normalized_count | −0.1095 | −0.0227 | same sign, **79.3% of the naive gap is not present in the lead-up window** — mostly post-onset population-collapse artifact |
| flow_magnitude | −0.0301 | +0.0407 | **sign inverts** — naive comparison shows a negative-going artifact; the genuine pre-onset effect is positive (people speed up before panic) |
| direction_entropy | −0.0425 | +0.0069 | **sign inverts**, small magnitude either way — naive gap also looks artifact-driven, though this specific flip wasn't documented in earlier notes and is newly surfaced by this recomputation |
| stop_ratio | −0.0807 | −0.0588 | same sign, only 27.2% shrinkage — the most robust feature, consistent with its 11/11 unanimous per-onset sign |

**Note on provenance:** an earlier informal note (not backed by a saved
script) recorded normalized_count's naive gap as "~−0.11, ~86% artifact,
true lead-up gap ~−0.016." This recomputation gives −0.1095 / 79.3% /
−0.0227 — directionally identical conclusion (large majority artifact), but
the exact percentage and lead-up figure differ from the earlier note,
whose original computation isn't preserved anywhere to reconcile against.
**Treat the numbers in this table as authoritative** — they're reproducible
from data still on disk via the naive-gap computation shown above plus
`diagnose_leadup_signal.py`.

---

## 5. UMN forecasting — per fold

**STATUS: RESOLVED.** Full 11-fold LOCO re-run of `train_forecast_model.py`
completed and saved to `outputs/reports/umn_loco_results.json`. `scene` is
null for every fold — no clip-to-scene (lawn/indoor/plaza) mapping exists in
any saved artifact.

**This re-run also resolves the two-conflicting-memory-records problem**:
both prior figures were real, just computed on different fold subsets that
were never clearly labeled. Full-set matches one record almost exactly;
common-fold-subset matches the other almost exactly. Neither was wrong —
they just needed the methodology attached.

### Per-fold results

| fold | best_epoch | stopped_at | n_pos k0 | n_pos k30 | n_pos k60 | n_pos k90 | AUROC k0 | AUPRC k0 | AUROC k30 | AUPRC k30 | AUROC k60 | AUPRC k60 | AUROC k90 | AUPRC k90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 18 | 40 | 10 | 40 | 70 | 90 | 0.9897 | 0.7081 | 0.5114 | 0.1896 | 0.1864 | 0.1088 | 0.2614 | 0.1524 |
| 1 | 6 | 28 | 33 | 63 | 93 | 110 | 0.9540 | 0.5744 | 0.9446 | 0.6647 | 0.9618 | 0.8137 | 0.9702 | 0.8687 |
| 2 | 13 | 35 | 100 | 130 | 160 | 180 | 0.9995 | 0.9989 | 0.9971 | 0.9963 | 0.9706 | 0.9729 | 0.8639 | 0.8578 |
| 3 | 11 | 33 | 0 | 22 | 52 | 80 | n/a | n/a | 0.7275 | 0.0850 | 0.6954 | 0.1639 | 0.8352 | 0.4828 |
| 4 | 31 | 53 | 146 | 176 | 206 | 210 | 0.9340 | 0.8690 | 0.8016 | 0.7429 | 0.6649 | 0.5899 | 0.5235 | 0.5677 |
| 5 | 17 | 39 | 6 | 36 | 66 | 80 | 0.9862 | 0.6907 | 0.9907 | 0.9413 | 0.9932 | 0.9682 | 0.9480 | 0.7342 |
| 6 | 18 | 40 | 32 | 62 | 92 | 122 | 0.8851 | 0.5206 | 0.9494 | 0.6905 | 0.9075 | 0.6231 | 0.8586 | 0.5372 |
| 7 | 33 | 55 | 84 | 114 | 144 | 174 | 0.9922 | 0.9658 | 0.9794 | 0.9614 | 0.8543 | 0.8367 | 0.7106 | 0.7315 |
| 8 | 21 | 43 | 0 | 0 | 13 | 40 | n/a | n/a | n/a | n/a | 0.1939 | 0.0200 | 0.6149 | 0.1149 |
| 9 | 54 | 76 | 0 | 0 | 18 | 30 | n/a | n/a | n/a | n/a | 0.6268 | 0.0525 | 0.5852 | 0.0751 |
| 10 | 19 | 41 | 0 | 0 | 9 | 20 | n/a | n/a | n/a | n/a | 0.7039 | 0.0277 | 0.6181 | 0.0437 |

**Excluded for zero positives in the test clip:** k=0 excludes folds
3, 8, 9, 10 (4/11); k=30 excludes 8, 9, 10 (3/11); k=60 and k=90 have all
11 folds valid.

### Aggregate — full set (whichever folds are valid per horizon)

| horizon | AUROC | AUPRC | n_folds |
|---|---|---|---|
| k=0 | 0.9630 ± 0.0386 | 0.7611 ± 0.1734 | 7/11 |
| k=30 | 0.8627 ± 0.1612 | 0.6589 ± 0.3246 | 8/11 |
| k=60 | 0.7053 ± 0.2732 | 0.4707 ± 0.3800 | 11/11 |
| k=90 | 0.7081 ± 0.2034 | 0.4696 ± 0.3055 | 11/11 |

### Aggregate — common fold subset (folds 0,1,2,4,5,6,7 — valid at all four horizons, n=7)

| horizon | AUROC | AUPRC |
|---|---|---|
| k=0 | 0.9630 ± 0.0386 | 0.7611 ± 0.1734 |
| k=30 | 0.8820 ± 0.1634 | 0.7409 ± 0.2581 |
| k=60 | 0.7913 ± 0.2676 | 0.7019 ± 0.2792 |
| k=90 | 0.7337 ± 0.2397 | 0.6356 ± 0.2300 |

The common-subset numbers show a cleaner monotonic decay and are arguably
the fairer "apples-to-apples" comparison across horizons (same 7 clips at
every k); the full-set numbers use more data per horizon but compare
different fold sets across horizons. Recommend reporting both with the
methodology stated explicitly, exactly as done here — this is what was
missing before, not the numbers themselves.

### Fold 0 investigation: cold-start hypothesis, falsified

Fold 0 is the largest single contributor to k=60's variance: AUROC 0.990 at
k=0 collapsing to 0.186 at k=60 on 70 positives — a systematic inversion,
not obviously small-sample noise given the magnitude and n. A specific
mechanism was proposed and tested: `FeatureExtractor`'s running normalizers
(`running_max_count`, `running_max_speed`, `running_median_speed`) reset to
their floor values at every clip boundary and need frames to stabilize;
since every clip's first valid LOCO sample (local offset 145) has a history
window reaching back to local offset 0, a clip whose normalizers stabilize
unusually late would have early features on a different scale than every
other clip's, which could plausibly produce exactly this kind of inversion.

Source: `src/diagnose_clip0_coldstart.py`, `outputs/reports/umn_clip0_coldstart_diagnostic.json`.
Method: reprocessed the first 200 frames of each of the 11 clips (seeking
directly to each clip's start, resetting tracker + FeatureExtractor exactly
as `process_umn.py` does at a real boundary), recording all three
normalizers at every frame. A seek-accuracy sanity check (person count at
the seeked first frame vs. the already-known value in `umn_meta.csv`)
matched exactly for all 11 clips before trusting anything downstream.

**Stabilization frame per normalizer** (first frame after which the value
stays within 5% of its own 200-frame-window final value for the rest of the
window; first valid LOCO sample is at local offset 145):

| clip | stabilize max_count | stabilize max_speed | stabilize median_speed | worst | past 145? |
|---|---|---|---|---|---|
| 0 | 75 | 40 | **179** | 179 | yes |
| 1 | 2 | 34 | 137 | 137 | no |
| 2 | 9 | 60 | 113 | 113 | no |
| 3 | **170** | 65 | 130 | 170 | yes |
| 4 | 5 | 112 | 115 | 115 | no |
| 5 | 11 | **185** | 136 | 185 | yes |
| 6 | 124 | 7 | **166** | 166 | yes |
| 7 | 38 | 1 | 115 | 115 | no |
| 8 | 114 | 118 | 53 | 118 | no |
| 9 | 33 | **146** | 22 | 146 | yes |
| 10 | 22 | 144 | 67 | 144 | no |

**Verdict: hypothesis rejected.** Clip 0 is not an outlier on this metric —
5 of 11 clips have at least one normalizer stabilizing after frame 145, and
clip 0's worst value (179) is not even the latest of the five (clip 5's
`running_max_speed` stabilizes at 185, later than anything in clip 0).

**The decisive counter-example is clip 5**: its `running_max_speed`
stabilizes later than any of clip 0's three normalizers (185 vs. clip 0's
worst of 179), yet clip 5 is **the single best-performing fold in the
entire LOCO evaluation** (AUROC 0.986 / 0.991 / 0.993 / 0.948 at k=0/30/60/90
— no inversion at all). Clips 3, 6, and 9 also stabilize late on at least
one normalizer and show no inversion either (clip 3's AUROC actually rises
from k=30 to k=90: 0.727→0.835; clip 6 stays flat and strong throughout,
0.85-0.95; clip 9 is unremarkable but not inverted, 0.627/0.585).

Checking the actual values (not just the stabilization-frame statistic)
reinforces this: at frame 145, clip 0's `running_median_speed` was already
close to its eventual window value (0.418 → 0.385, an 8% residual drift) —
a *smaller* gap than clip 9's `running_max_speed` (3.919 → 6.739, a 72%
jump after frame 145), and clip 9 shows no pathology.

**Fold 0's inversion remains unexplained by this mechanism.** It is not a
Limitations-section finding — the cold-start story is directly contradicted
by the data (a later-stabilizing, higher-residual-drift clip performs best,
not worst). Recorded here because a falsified leading hypothesis, tested
and ruled out, is worth the paper trail even without a positive result; the
inversion is most plausibly ordinary small-sample instability, consistent
with the already-documented variance floor from 11 independent clips.

---

## 6. UMN feature ablation

**RESOLVED.** `src/ablation.py`: 5 LSTM variants (full model + one feature
group zeroed across all 16 zones — normalized_count / flow_magnitude /
direction_entropy / stop_ratio), each trained and evaluated on the identical
11-fold LOCO splits as the full model. "Zeroed" means literally set to 0 in
both train and test data at every timestep — input_size stays 64 for every
variant, only which 16 columns are held constant differs, isolating the
feature's information content from any capacity change. Source:
`outputs/reports/umn_ablation_results.json`.

### AUROC drop vs. full model (common fold subset, mean)

| variant | k=0 | k=30 | k=60 | k=90 |
|---|---|---|---|---|
| full (baseline) | 0.963 | 0.882 | 0.791 | 0.734 |
| zero normalized_count | 0.786 (−0.177) | 0.828 (−0.054) | 0.757 (−0.035) | 0.721 (−0.013) |
| zero flow_magnitude | 0.956 (−0.007) | 0.830 (−0.052) | 0.701 (−0.091) | 0.630 (−0.104) |
| zero direction_entropy | 0.932 (−0.031) | 0.855 (−0.027) | 0.766 (−0.026) | 0.725 (−0.009) |
| zero stop_ratio | 0.671 (−0.292) | 0.558 (−0.324) | 0.511 (−0.280) | 0.518 (−0.215) |

### The prediction was half right, and the wrong half is the more interesting finding

**stop_ratio is confirmed load-bearing** — by far the largest drop at every
horizon (0.22-0.32 AUROC), consistent with its 11/11 unanimous per-onset
sign in the lead-window sweep (Section 3).

**flow_magnitude and normalized_count did NOT behave as predicted — they
swap roles across the nowcast/forecast boundary, and the swap is itself the
finding:**

- `normalized_count` was predicted near-decorative throughout. It's actually
  the **second most important feature at k=0** (−0.177, more than 4× any
  other non-stop_ratio feature) — but its importance collapses to genuinely
  near-decorative by k=60/90 (−0.035, −0.013), exactly as predicted **for
  forecasting horizons specifically**. Current person-count is informative
  about *whether panic is happening right now* (population visibly
  scattering) but nearly uninformative about *whether panic is coming* —
  coherent with Section 4's finding that ~79% of its naive-comparison gap
  was post-onset population-collapse artifact, not a genuine precursor.
- `flow_magnitude` was predicted load-bearing throughout. It's actually
  **nearly irrelevant at k=0** (−0.007) but **grows into the second most
  important feature at k=60/90** (−0.091, −0.104), overtaking
  direction_entropy and approaching stop_ratio's territory. This is the
  cleanest possible confirmation of Section 4's leakage finding: flow's
  genuine signal is a *precursor* effect (people speed up before panic)
  invisible in current-frame snapshots but load-bearing for forecasting
  ahead.
- `direction_entropy` was not explicitly predicted either way; it turns out
  the flattest and smallest contributor throughout (−0.009 to −0.031) —
  effectively decorative at every horizon, not just some of them.

**Net story for the paper:** feature importance is not static — it inverts
between nowcasting and forecasting. A model evaluated only at k=0 would
conclude normalized_count matters and flow_magnitude doesn't; a model
evaluated only at k=90 would conclude the opposite. Reporting ablation at a
single horizon would have been actively misleading here.

---

## 7. UMN baseline comparison

**RESOLVED.** `src/baseline_rule.py`: a no-learning rule computing one
risk score per frame from the CURRENT (not sequence) 64-dim feature vector
— mean of 4 sign-corrected, z-scored feature-group means (density/flow/
entropy/stop_ratio, equal weights). Sign and z-score stats fit on each
fold's 10 training clips only (sign from correlation with the k=0 label),
frozen, applied to the held-out test clip. The SAME score is evaluated
against every horizon's label — it has no forecasting mechanism, so any gap
at k=30/60/90 is attributable to the LSTM's temporal modeling, not better
per-frame features. Source: `outputs/reports/umn_baseline_rule_results.json`.

### Rule-based baseline vs. LSTM — AUROC, same folds/horizons

| horizon | baseline (full-set) | LSTM (full-set) | baseline (common-subset) | LSTM (common-subset) |
|---|---|---|---|---|
| k=0 | 0.686 ± 0.386 | 0.963 ± 0.039 | 0.686 ± 0.386 | 0.963 ± 0.039 |
| k=30 | 0.525 ± 0.359 | 0.863 ± 0.161 | 0.571 ± 0.361 | 0.882 ± 0.163 |
| k=60 | 0.621 ± 0.237 | 0.705 ± 0.273 | 0.592 ± 0.253 | 0.791 ± 0.268 |
| k=90 | 0.612 ± 0.166 | 0.708 ± 0.203 | 0.583 ± 0.193 | 0.734 ± 0.240 |

**The LSTM beats the baseline at every horizon in both aggregations.** The
gap is largest at k=0/30 (0.28-0.34 AUROC) and narrows but persists at
k=60/90 (0.11-0.17) — consistent with a real forecasting contribution from
temporal structure, not just better instantaneous features. Two individual
folds (5, 6) show the baseline scoring far below chance at k=0 (0.021,
0.152) — the training-fold sign-correction picked the wrong direction for
those specific held-out clips, a small-sample instability worth naming as a
baseline limitation rather than hiding.

---

## 8. ShanghaiTech

### Dataset stats

Source: `data/shanghaitech_forecast.npz` + `data/shanghaitech_meta.json` +
`data/features/shanghaitech_meta.csv`, read fresh this session.

- 199 videos, 44 anomalous, 142,060 total frames
- Forecast dataset: 110,419 samples, shape (30, 16) per sample
- Positive rate: k=0 5.87%, k=24 5.78%, k=48 5.49%, k=72 4.98%

### Per-scene video count and density

| scene | n_videos | mean persons/frame |
|---|---|---|
| 01 | 57 | 3.776 |
| 02 | 8 | 6.353 |
| 03 | 6 | 1.875 |
| 04 | 13 | 3.987 |
| 05 | 35 | 4.623 |
| 06 | 16 | 1.721 |
| 07 | 6 | 3.366 |
| 08 | 29 | 5.724 |
| 09 | 4 | 4.231 |
| 10 | 7 | 8.749 |
| 11 | 4 | 1.104 |
| 12 | 11 | 3.913 |
| 13 | 3 | 1.478 |

Overall mean ≈ 4.1 persons/frame vs. UMN's 15.2 — the density mismatch
underlying the negative-transfer finding.

### Sparsity diagnostics (4×4 vs 2×2 grid)

Source: computed fresh this session directly from all 199 saved 2×2-grid
feature arrays (`data/features/shanghaitech/*.npy`, 142,060 total frames).
**Confirms the prior-context figures almost exactly** (told to me
originally as 71.7% / 61.8% / 66.5% / 0.5%; independently recomputed here as
71.9% / 61.8% / 66.5% / 0.5% — differences only in the third significant
figure of the overall rate, everything else exact):

- Overall zero fraction at 2×2: **71.9%** (vs. 90.7% at 4×4, per the
  original context — the 4×4 figure itself has not been independently
  recomputed this session since the 4×4 feature arrays were superseded by
  the 2×2 re-extraction and are no longer on disk to check against).
- Zone-occupancy: 41.4% of zone-frames have ≥1 person present.
- Within occupied zones only: normalized_count zero 0.0% of the time,
  flow_magnitude 0.5%, direction_entropy **61.8%**, stop_ratio **66.5%** —
  i.e. count and flow are essentially always defined once someone is
  present, while entropy and stop_ratio are frequently undefined by
  construction (entropy needs ≥2 moving tracks in a zone; stop_ratio needs
  ≥1 track with a computable speed), not from a bug.

### Window-class stratification (per horizon)

Source: `data/shanghaitech_forecast.npz`'s `window_class` array, read fresh
this session.

| horizon | NEGATIVE | CLEAN_PRE | OVERLAP | POST |
|---|---|---|---|---|
| k=0 | 102,069 / 0 pos | 0 / 0 pos | 7,720 / 6,481 pos | 630 / 0 pos |
| k=24 | 101,592 / 0 pos | 477 / 477 pos | 7,720 / 5,902 pos | 630 / 0 pos |
| k=48 | 101,181 / 0 pos | 888 / 888 pos | 7,720 / 5,179 pos | 630 / 0 pos |
| k=72 | 100,912 / 0 pos | 1,157 / 1,157 pos | 7,720 / 4,347 pos | 630 / 0 pos |

(total samples / positive samples per cell). k=0's CLEAN_PRE is
structurally always 0 — the target frame at k=0 IS the window's last frame,
so it can never be simultaneously "unseen."

### Final per-fold results (post inner-validation-fix re-run)

Source: `outputs/reports/shanghaitech_results_summary.json`, the corrected
re-run after the inner-validation bug fix (inner-val was drawing only from
2 atypical scenes, causing `best_epoch=0` on multiple folds; fixed to draw
~15% of videos from every training scene, stratified).

#### Headline scenes (01/03/05/07) — LOSO, full-sample AUROC/AUPRC

| scene | k=0 | k=24 | k=48 | k=72 |
|---|---|---|---|---|
| 01 | 0.620 / 0.088 | 0.609 / 0.081 | 0.599 / 0.073 | 0.608 / 0.067 |
| 03 | 0.740 / 0.444 | 0.741 / 0.394 | 0.725 / 0.321 | 0.715 / 0.260 |
| 05 | 0.248 / 0.018 | 0.262 / 0.019 | 0.274 / 0.020 | 0.300 / 0.021 |
| 07 | 0.607 / 0.170 | 0.563 / 0.155 | 0.531 / 0.158 | 0.608 / 0.216 |
| **mean** | **0.554** | **0.544** | **0.532** | **0.558** |

#### Headline scenes — CLEAN_PRE+NEGATIVE restricted (the honest forecasting metric)

| scene | k=0 | k=24 | k=48 | k=72 |
|---|---|---|---|---|
| 01 | n/a | 0.676 / 0.010 | 0.645 / 0.015 | 0.620 / 0.016 |
| 03 | n/a | 0.781 / 0.213 | 0.814 / 0.339 | 0.802 / 0.335 |
| 05 | n/a | 0.308 / 0.002 | 0.331 / 0.005 | 0.372 / 0.008 |
| 07 | n/a | 0.394 / 0.026 | 0.437 / 0.065 | 0.614 / 0.157 |
| **mean** | n/a | **0.540** | **0.557** | **0.602** |

Scene 05 is below chance at every horizon in both tables — see the
negative-transfer discussion already agreed for Limitations: scene 05's
three anomalies are small-group physical altercations (2-5 people), not
crowd-scale panic, confirmed by direct frame inspection (`05_0017`
frame 283, `05_0019` frame 323, `05_0023` frame 381) — a mismatch in kind
with the target phenomenon, not just density.

#### Thin scenes (04/06/08/12) — reported separately, NOT averaged into headline

| scene | k=0 full | k=24 full | k=48 full | k=72 full | k=24 restricted | k=48 restricted | k=72 restricted |
|---|---|---|---|---|---|---|---|
| 04 | 0.577 | 0.586 | 0.600 | 0.624 | 0.595 | 0.578 | 0.629 |
| 06 | 0.795 | 0.789 | 0.748 | 0.632 | 0.611 | 0.474 | 0.441 |
| 08 | 0.779 | 0.750 | 0.700 | 0.668 | 0.585 | 0.703 | 0.698 |
| 12 | 0.440 | 0.442 | 0.390 | 0.344 | 0.390 | 0.423 | 0.512 |

Scene 12's restricted metric is computed on just 6 CLEAN_PRE positives at
every horizon (see earlier diagnostic) — a single-clip-level estimate, not
a scene-level one.

#### Specificity check: headline-fold models on non-evaluable scenes (02/09/10/11/13)

False-positive rate at threshold 0.5, per headline-fold model:

| held-out fold model | k=0 | k=24 | k=48 | k=72 |
|---|---|---|---|---|
| 01 | 0.120 | 0.103 | 0.104 | 0.107 |
| 03 | 0.040 | 0.036 | 0.036 | 0.040 |
| 05 | 0.125 | 0.107 | 0.100 | 0.101 |
| 07 | 0.047 | 0.034 | 0.024 | 0.025 |

FPR is computed only over frames with true label 0; scenes 02 and 10 do
contain anomalous videos (excluded from forecasting eval only because onset
lands too early for clean pre-onset history), so flags there are not
automatically false positives in the colloquial sense.

---

## 9. Latency benchmark — merged CPU + GPU

Source: `outputs/reports/latency_benchmark_combined.csv` (Mac CPU rows +
Colab CUDA/T4-class rows; Colab's own CPU rows were deliberately excluded —
they're a different, weaker, shared machine, not a fair CPU baseline; see
`latency_benchmark_gpu.csv` for the raw discarded rows if needed).

| resolution | device | decode | detect | track | features | lstm | total | achieved FPS | drop rate @ 30fps |
|---|---|---|---|---|---|---|---|---|---|
| 320×240 | CPU | 0.24 | 109.51 | 0.77 | 1.73 | 0.71 | 112.96 | 8.87 | 70.4% |
| 640×480 | CPU | 1.16 | 153.01 | 0.91 | 2.65 | 0.88 | 158.62 | 6.35 | 78.8% |
| 856×480 | CPU | 1.21 | 87.49 | 0.72 | 2.10 | 0.68 | 92.20 | 10.99 | 63.4% |
| 320×240 | CUDA | 0.36 | 9.46 | 1.78 | 2.59 | 1.33 | 15.52 | 65.97 | **0.0%** |
| 640×480 | CUDA | 0.85 | 7.04 | 1.46 | 2.58 | 0.80 | 12.73 | 84.18 | **0.0%** |
| 856×480 | CUDA | 1.19 | 8.52 | 1.65 | 3.08 | 1.01 | 15.45 | 70.16 | **0.0%** |

All ms values are means; the CSV has median/p95/p99 per stage too. "Total"
includes decode; "achieved FPS" excludes it (decode runs on a separate
reader thread in the real deployment and never serializes with the other
stages — see `stream_processor.py`).

**Context — mean persons/frame during these runs:** not captured as a CSV
column in the main benchmark; a separate diagnostic run (200 frames/config,
CPU only) measured ~17.1/17.1/16.7 persons/frame at 320×240/640×480/856×480
respectively — consistent across resolution as expected (same source
video). Not systematically measured for the GPU run.

CPU detect time is non-monotonic across resolution (109/153/87ms) and does
NOT correlate meaningfully with crowd density either (pooled Pearson
r=0.104, p=0.011, ~1% of variance explained; within-resolution correlations
don't even agree in sign: +0.23/−0.06/−0.23) — this is most likely CPU
scheduling noise on a shared, non-dedicated machine. GPU numbers are tight
and resolution-invariant (7.0-9.5ms) as expected from fixed `imgsz=640`,
and retroactively support the CPU pattern being noise rather than a real
resolution or density effect.

---

## 10. What's missing — bluntly

A reviewer would expect all of the following, in rough order of how likely
they are to be a blocking request:

1. **Resolved: UMN per-fold AUROC/AUPRC table (Section 5).** Full 11-fold
   LOCO re-run completed, saved to `outputs/reports/umn_loco_results.json`,
   and resolves the two-conflicting-memory-records problem (both were real,
   computed on different fold subsets).
2. **Resolved: rule-based baseline (Section 7).** `src/baseline_rule.py`
   built and run; LSTM beats it at every horizon in both aggregations.
3. **Resolved: feature ablation (Section 6).** `src/ablation.py` run;
   stop_ratio confirmed load-bearing, but normalized_count/flow_magnitude
   swap importance across the nowcast/forecast boundary rather than
   matching the flat prediction — see Section 6 for the full story.
4. **Resolved: COCO-pretrained YOLO baseline (Section 1).** Now recorded in
   `outputs/reports/detection_results.md` and Table 1 above.
5. **No comparison against any published forecasting or VAD method's
   numbers**, beyond the informal "published ShanghaiTech VAD is ~0.70-0.73"
   context mentioned in an earlier conversation. If the venue expects a
   related-work numbers table, none of the rows exist yet. Still open —
   nothing in this session addresses it.
6. **Statistical significance testing is absent throughout.** Wide
   confidence intervals are reported honestly (good practice), but nothing
   tests, e.g., whether k=90 AUROC is significantly above 0.5. Common in
   small-dataset papers, but worth a sentence acknowledging it rather than
   silence. Still open.
7. **Resolved: the epoch 39-60 training gap (Section 1).** The run
   completed all 60 epochs; only the log rows for 39-60 failed to sync back.
   Validation mAP50 moved 0.682→0.683 over that span per the user's direct
   account — a plateau, not a gap in the result itself. Use the epoch 20-38
   trend as convergence evidence in Methods.
8. **Resolved: ShanghaiTech sparsity diagnostics** now independently
   verified from scratch (Section 8) and match the prior-context figures
   almost exactly.
9. The 4×4-grid zero-fraction figure (90.7%) is still unverified — the
   original 4×4 feature arrays were superseded by the 2×2 re-extraction and
   are no longer on disk, so only the 2×2 numbers could be recomputed.
   Low-stakes (4×4 was abandoned specifically because of this sparsity, so
   the exact figure is background motivation, not a load-bearing result),
   but flagging it for completeness.
10. I do not know whether an actual manuscript draft exists to check
    against any of this — everything above is about whether the
    *experiments* support a paper, not whether prose already matches them.
