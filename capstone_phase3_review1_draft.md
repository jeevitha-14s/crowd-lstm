# UE23CS441A -- Capstone Project Phase 3, Project Progress Review #1
Draft content for the existing Google Slides template. Copy each section into
its corresponding slide. Anything in `[BRACKETS]` needs your input -- I don't
have it and won't guess (team names, guide, dates). Everything else is pulled
directly from files already in the repo (`outputs/reports/PAPER_RESULTS.md`,
`README.md`, `models/umn_forecast_lstm.README.md`) -- no numbers invented.

---

## Slide 1 -- Title

- **Project Title**: [PROJECT TITLE -- suggested: "Crowd Risk Forecasting:
  From Anomaly Detection to Multi-Horizon Panic Prediction Using LSTMs"]
- **Project ID**: [PROJECT ID]
- **Project Guide**: [GUIDE NAME]
- **Project Team**: [TEAM MEMBER NAMES]

---

## Slide 2 -- Outline
(Keep as-is from the template -- it's the agenda, not content.)

---

## Slide 3 -- Abstract and Scope

Most crowd-monitoring systems **detect** panic only after it is already
visible in a frame. This project reframes the problem as **forecasting**:
predicting whether panic will be occurring 0-3 seconds ahead (k=0/30/60/90
frames at 30fps), using only features observable before onset. This is a
genuine forecast, not just fast detection -- the rigorous, held-out evidence
is the feature-ablation study: precursor-type features (flow_magnitude:
people speeding up) become load-bearing specifically at the longer horizons
while current-frame features (raw person count) fade out, on the 11-fold
leave-one-clip-out evaluation where every test clip is unseen during
training. (The live dashboard demo shows the same qualitative pattern --
k=30/60/90 crossing alert threshold ~10-12 frames, ~0.3-0.4s, before the
ground-truth onset -- but that demo uses the deployment checkpoint trained
on all 11 UMN clips, so it illustrates the *mechanism*, not independent
generalization evidence; see the Demo section's caveat.) The horizon is
intentionally short, not a conservative design
choice: a lead-window sweep found genuine precursor signal in this dataset
degrades sharply past ~90-120 frames (3-4s), so k=90 sits near the practical
ceiling of what UMN supports. That makes the system's realistic value
triggering automated responses (alarms, access control) at machine speed --
not giving human authorities minutes of advance warning to intervene.

**Scope:**
- Detection + tracking: YOLOv8s (fine-tuned on CrowdHuman) + ByteTrack
- Feature extraction: dataset-agnostic 4x4 spatial-grid descriptor (64-dim)
- Forecasting model: shared-backbone LSTM with 4 independent heads predicting
  panic at k=0/30/60/90 frames ahead (0/1/2/3s at 30fps)
- Evaluation: leave-one-clip-out (UMN) and leave-one-scene-out (ShanghaiTech)
  -- never row-level splits, since adjacent forecast samples share most of
  their history window via the sliding-window construction
- Deployment: a real-time pipeline (frame-dropping, not queueing, to keep
  "real-time" an honest claim) with three-level alerting and a live dashboard

---

## Slide 4-5 -- Architecture / Pipeline (max 2 slides)

**Offline (training/eval) pipeline:**

```
Raw video (UMN / ShanghaiTech)
  -> YOLOv8s (CrowdHuman fine-tuned) person detection
  -> ByteTrack multi-object tracking
  -> FeatureExtractor: 4x4 spatial grid, 64-dim
       (per zone: normalized_count, flow_magnitude, direction_entropy, stop_ratio)
  -> build_forecast_dataset.py / build_shanghaitech_forecast.py:
       sliding-window sequences (seq_len=30, stride=5) + multi-horizon labels
  -> train_forecast_model.py: shared LSTM (hidden=64, 1 layer, dropout=0.2)
       + 4 linear heads (k=0/30/60/90), BCEWithLogitsLoss w/ per-head pos_weight
  -> Leave-one-clip-out (UMN, 11 folds) / leave-one-scene-out (ShanghaiTech)
       evaluation -> AUROC / AUPRC per horizon
```

**Deployment (live) pipeline** (`src/stream_processor.py`):

```
Live source (file / RTSP) --frame drop, never queue-->
  YOLOv8s + ByteTrack -> FeatureExtractor -> SequenceBuffer
       (resampled onto the trained model's frame_id grid, forward-filled
        under drops -- fixed 2026-08-20 to remove train/deploy distribution
        shift under sustained drop)
  -> LSTM forecast (deployment checkpoint, trained on all 11 UMN clips)
  -> alerting.py: 3-level GREEN/AMBER/RED, driven by k=60 head, asymmetric
       rise/fall persistence
  -> live_dashboard.py: OpenCV view or headless annotated video output
```

Ask me to render an actual diagram image (block diagram) separately if the
panel wants a graphic rather than this text pipeline -- I can generate one.

---

## Slide 6 -- Summary of Work Done in Phase 2

[Confirm this split against your actual Phase-2 submission date -- based on
file history, everything below predates Aug 8, 2026 and looks like the
Phase-2 boundary:]

- UMN dataset fully processed: detection+tracking, feature extraction,
  forecast-dataset construction
- LSTM forecasting model trained and evaluated via 11-fold leave-one-clip-out
- Headline result: AUROC 0.963 -> 0.882 -> 0.791 -> 0.734 for k=0/30/60/90
  (common 7-fold subset, apples-to-apples across horizons) -- a clean
  monotonic decay as forecast horizon lengthens
- ShanghaiTech pipeline built and validated on a 2-video test subset

---

## Slide 7-11 -- Detailed Modules Developed Post-Phase-2 (max 5 slides)

### Module inventory

| Module | File | Purpose |
|---|---|---|
| ShanghaiTech forecast builder | `build_shanghaitech_forecast.py` | Per-video sliding-window sample builder, matched horizons (k=0/24/48/72 frames = 0/1/2/3s at 24fps) |
| ShanghaiTech LOSO training | `train_shanghaitech_forecast_model.py` | Leave-one-scene-out training/eval across 13 scenes |
| Rule-based baseline | `baseline_rule.py` | No-learning per-frame risk score (z-scored, sign-corrected feature-group means) -- isolates the LSTM's temporal-modeling contribution from raw feature quality |
| Feature ablation | `ablation.py` | 5 LSTM variants, one feature group zeroed at a time, same LOCO splits as the full model |
| Cold-start diagnostic | `diagnose_clip0_coldstart.py` | Hypothesis test on FeatureExtractor normalizer warm-up as a source of fold-0 variance (see Results) |
| Real-time stream processor | `stream_processor.py` | Live pipeline: mandatory frame-dropping (single-slot buffer), `SequenceBuffer` resampled on video frame_id (bug fixed 2026-08-20) |
| Alerting | `alerting.py` | 3-level GREEN/AMBER/RED, driven by k=60 head, asymmetric persistence |
| Live dashboard | `live_dashboard.py` | Interactive OpenCV view / headless annotated video output |
| Latency benchmark | `benchmark_latency.py` | Per-stage latency across resolutions, CPU + GPU |

### Tech stack

- **Detection**: YOLOv8s (Ultralytics), fine-tuned on CrowdHuman
- **Tracking**: ByteTrack
- **Modeling**: PyTorch (LSTM), NumPy/Pandas for feature pipeline
- **Streaming test infra**: OpenCV `VideoCapture`, mediamtx + ffmpeg for
  genuine RTSP-stream testing (not just file replay)
- All open-source.

### Testing

- `tests/test_feature_extractor.py` -- FeatureExtractor unit tests
- `tests/test_stream_processor.py` -- SequenceBuffer resampling/forward-fill
  behavior under simulated frame drops (added alongside the 2026-08-20 fix)

### Individual contribution table

[TEAM MEMBER NAME] | [Modules assigned] | [LOC] | [Time spent]
--- fill in per member ---

---

## Slide 12 -- Results, Demo, and Baseline Comparison

**(This is the strongest slide in the deck -- lead with it.)**

### 1. Detection backbone: fine-tuning vs. COCO-pretrained baseline

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8s, COCO-pretrained (no fine-tuning) | 0.632 | 0.398 | 0.432 | 0.215 |
| YOLOv8s + CrowdHuman fine-tuning | **0.835** | **0.602** | **0.699** | **0.431** |

Fine-tuning improves every metric substantially (+0.20 precision, +0.20
recall, +0.27 mAP50, +0.22 mAP50-95) on the identical 1,937-image test set.
Inference: 5.8 ms/image on a T4 GPU.

### 2. UMN forecasting vs. a no-learning rule-based baseline

The rule baseline computes one risk score per frame from *current* features
only (no sequence, no learning) -- any gap at k=30/60/90 is attributable to
the LSTM's temporal modeling, not better per-frame features.

| Horizon | Rule baseline (AUROC) | LSTM (AUROC) | Gap |
|---|---|---|---|
| k=0 | 0.686 ± 0.386 | **0.963 ± 0.039** | +0.277 |
| k=30 | 0.571 ± 0.361 | **0.882 ± 0.163** | +0.311 |
| k=60 | 0.592 ± 0.253 | **0.791 ± 0.268** | +0.199 |
| k=90 | 0.583 ± 0.193 | **0.734 ± 0.240** | +0.151 |

The LSTM beats the baseline at every horizon; the gap narrows but persists
as the horizon lengthens -- consistent with genuine forecasting value from
temporal structure.

### 3. Feature ablation -- the nowcast/forecast importance swap

| Variant (zeroed) | k=0 | k=30 | k=60 | k=90 |
|---|---|---|---|---|
| full model | 0.963 | 0.882 | 0.791 | 0.734 |
| normalized_count | 0.786 (**-0.177**) | 0.828 (-0.054) | 0.757 (-0.035) | 0.721 (-0.013) |
| flow_magnitude | 0.956 (-0.007) | 0.830 (-0.052) | 0.701 (-0.091) | 0.630 (**-0.104**) |
| direction_entropy | 0.932 (-0.031) | 0.855 (-0.027) | 0.766 (-0.026) | 0.725 (-0.009) |
| stop_ratio | 0.671 (**-0.292**) | 0.558 (**-0.324**) | 0.511 (**-0.280**) | 0.518 (**-0.215**) |

**Finding**: `stop_ratio` is load-bearing at every horizon. But
`normalized_count` and `flow_magnitude` swap roles between nowcasting and
forecasting -- current person-count matters for detecting panic *now* but
is nearly decorative for forecasting it; flow (people speeding up) is the
opposite. A single-horizon ablation would have missed this entirely.

### 4. ShanghaiTech cross-dataset generalization (leave-one-scene-out)

| Scene | k=0 | k=24 | k=48 | k=72 |
|---|---|---|---|---|
| 01 | 0.620 | 0.609 | 0.599 | 0.608 |
| 03 | 0.740 | 0.741 | 0.725 | 0.715 |
| 05 | 0.248 | 0.262 | 0.274 | 0.300 |
| 07 | 0.607 | 0.563 | 0.531 | 0.608 |
| **mean** | **0.554** | **0.544** | **0.532** | **0.558** |

Weaker than UMN. The explanation is a mismatch in the **kind** of anomaly,
not in crowd density -- see Sect. 10, where correlating per-scene density
against per-scene AUROC across all 8 evaluable scenes returns r=-0.481
(p=0.228), i.e. no significant relationship, and negative in sign. Scene 05
is the clearest case: its "anomalies" are small-group altercations, not
crowd-scale panic (confirmed by direct frame inspection, not just the
metric), and it scores below chance at every horizon. Reported honestly as a
negative-transfer finding, not hidden.

*Metric note*: the table above is the full-sample AUROC. `PAPER_RESULTS.md`
also reports a stricter CLEAN_PRE+NEGATIVE-restricted version (excludes
windows that already overlap the anomalous segment) that it labels "the
honest forecasting metric" -- k=24/48/72 mean 0.540/0.557/0.602 (k=0 n/a by
construction). It's *higher*, not lower, so this isn't a case of hiding a
worse number -- but if asked which is authoritative, the restricted one is,
and it's worth having ready.

### 5. Deployment latency (per-stage, CPU vs. GPU)

| Resolution | Device | Total (ms) | Achieved FPS | Drop rate @ 30fps |
|---|---|---|---|---|
| 640x480 | CPU | 158.62 | 6.35 | 78.8% |
| 640x480 | CUDA (T4) | 12.73 | 84.18 | **0.0%** |

GPU deployment is real-time with zero drop; CPU-only is not viable at 30fps
input -- a concrete deployment-hardware requirement for the demo/discussion.

### 6. Positioning against the base paper -- Sabih & Vishwakarma (2022) [4]

[4] is a strong methodological relative -- both approaches extract
spatial-patch motion features per frame and feed a fixed-length window into
an LSTM -- which makes the comparison direct rather than a stretch. But it
solves a structurally different problem: it is **bidirectional**, meaning it
needs all 50 frames of a window, including those *after* the frame being
labeled, before it can classify anything. Their own paper states the
resulting ~1.7s delay is because a full window must arrive before *any*
frame in it can be labeled -- this is retrospective classification of an
already-captured clip, not prediction of a not-yet-observed one. It also
reports a single 75/25 train/test split per dataset with no cross-validation
and no variance reported.

| | Sabih & Vishwakarma (2022) [4] | This work |
|---|---|---|
| Causality | Bidirectional -- needs future frames in-window | Strictly causal |
| Task | Frame-level classification within an observed window | Multi-horizon forecasting (k=0/30/60/90 frames ahead) |
| Evaluation | Single 75/25 split, no variance reported | 11-fold leave-one-clip-out, per-fold + variance reported |
| Features | Optical-flow patch histograms only | Zone-based motion features + track-derived (stop_ratio, count) |
| Reported metric | AUC 0.9483 (UCSD Ped-1), 0.9650 (Ped-2) | AUROC 0.963/0.882/0.791/0.734 at k=0/30/60/90 (UMN, common-fold) |

**Why these numbers aren't put head-to-head**: different datasets and a
different task definition (retrospective classification vs. forecasting) --
a same-metric leaderboard comparison would be apples-to-oranges and a
reviewer would flag it. The honest claim is structural: our model predicts
before the frames it's scoring exist; theirs requires them already decoded.
Combined with the feature-ablation and rule/logreg/LSTM ladder results
above, this also shows *why* the gap holds -- stop_ratio and flow_magnitude
(precursor-type signals) carry the forecasting horizons, not just raw
person-count, which is exactly the kind of feature a bidirectional,
single-window classifier has no structural reason to need.

### 7. Positioning against the base paper -- DeepROD, Ammar & Cherif (2021) [1]

[1] is genuinely causal/online (unlike [4]) -- it predicts the next single
Gradient-of-Motion (GoM) scalar from d=5 past values via LSTM, then flags an
anomaly if the prediction error is statistically atypical. Two properties
shape how it can be compared. First, it looks ~1/30s ahead (the next frame),
not a multi-second horizon. Second, it is evaluated *per video*: the model
trains online on each test video's own first 300 frames before scoring the
rest of that same video, so performance is measured with the scene already
calibrated.

Because this project targets cross-scene deployment -- a camera the model
has never seen -- we evaluate under leave-one-clip-out instead. To compare
like-for-like rather than against a figure measured under a different
protocol, we reimplemented [1]'s method (dense optical-flow GoM feature,
single-step LSTM regressor, Box-Cox/Mahalanobis threshold) and ran it under
*our* protocol, on the identical frames and labels:

| Method | Protocol | AUC |
|---|---|---|
| DeepROD, our reimplementation | Leave-one-clip-out (identical folds to ours) | **0.702 +/- 0.110** |
| This work (baseline LSTM) | Leave-one-clip-out | **0.963** (k=0) / 0.882 / 0.791 / 0.734 |

To keep that row fair, the reimplementation was given [1]'s full published
hyperparameter search (their Sect. 4.2: layers 1-4, hidden units in
{32,50,64,100}, epochs 5-30, d in {5,15,30} -- 504 configurations per fold,
a space containing every architecture their Table 2 reports as a winner),
their 80/20 train/validation split (their Sect. 3.2), and their own
selection rule (minimum validation MSE). Two sanity checks: every selected
model beats a mean-predicting constant on validation MSE, and 10 of 11 folds
independently select d=5 -- the same value their Table 2 reports as best for
every scene they tested.

Two findings:
1. **Under a matched protocol, our model scores higher**: 0.963 vs. 0.702 at
   k=0, the horizon [1]'s single-frame lookahead is closest to in spirit.
   Note this comparison does not survive Holm correction across the full
   family of baseline tests (Sect. 9) -- it is the largest observed gap, but
   11 folds cannot establish it at alpha=0.05.
2. **DeepROD has no mechanism for multi-second forecasting** -- it
   structurally cannot answer "will this be abnormal in 1/2/3 seconds,"
   which is what this project's k=30/60/90 heads address.

We do not compare against [1]'s reported figure (0.992 AUC under their
per-video protocol), as the two protocols measure different things: theirs
quantifies detection on a calibrated scene, ours generalization to an unseen
one. Deviations forced by the paper leaving details unstated -- the dense
optical-flow method (Farneback here), mini-batch size (32), and input
standardization (required: raw GoM sums are O(10^2) and saturate an LSTM's
gates without it) -- are recorded in `src/deeprod_gridsearch.py`.

Full per-fold results: `outputs/reports/deeprod_gridsearch_loco.json`.
(`deeprod_protocol_diagnostic.json` and `deeprod_loco_comparison.json` are
superseded -- they predate both the input-standardization fix and the
hyperparameter search, and their numbers should not be used.)

### 8. Zero-shot external validation -- Crowd Panic Dataset (Kaggle)

A separate check of generalization: the UMN-trained pipeline (unmodified,
no fine-tuning) run end-to-end -- YOLOv8s -> ByteTrack -> FeatureExtractor
-> SequenceBuffer -> LSTM k=0 head -- on 1,330 short phone-recorded clips
from an unrelated, informally-labeled dataset (No Panic / Normal / Crowdy /
Panic), completely unseen during training. Clips shorter than the model's
145-frame warmup span are front-padded (frozen first frame) rather than
excluded, after an initial run showed exclusion disproportionately dropped
the shortest, most chaotic Panic clips (55% excluded) -- a real bias, not
just a technicality (see `outputs/reports/crowd_panic_transfer.json` for
the diagnostic).

| class | n clips | mean risk (k=0) | median | % clips >0.5 |
|---|---|---|---|---|
| No Panic | 364 | 0.219 | 0.016 | 21.4% |
| Normal | 526 | 0.136 | 0.006 | 11.0% |
| Crowdy | 159 | 0.202 | 0.003 | 22.0% |
| **Panic** | 281 | **0.530** | **0.647** | **55.5%** |

**Panic vs. pooled non-panic**: median 0.647 vs. 0.007, Mann-Whitney U
one-sided p = 1.3x10^-35. This is deliberately NOT reported as AUROC --
these are clip-level categorical labels from an unvetted source (no
description, unknown license, informal WhatsApp-sourced footage), not
frame-level onset ground truth, so a proper ROC isn't well-posed here. Treat
this as a supplementary sanity check on generalization, not benchmark-grade
evidence: the ordering among the three non-panic classes doesn't track
density cleanly (No Panic scores above Crowdy), only the panic/non-panic
split is clean and strongly significant.

### 9. Statistical significance of the headline results

Every table above reports point estimates with fold-to-fold spread. This
section tests them. Each method was evaluated on the *identical* 11
leave-one-clip-out folds, so the folds pair exactly and support paired
tests. Resampling is at the fold level, not the frame level -- the held-out
clip is the independent unit, and frames within a clip are heavily
autocorrelated, so a frame bootstrap would badly understate variance.
Source: `src/significance_tests.py` ->
`outputs/reports/significance_tests.json`.

**Q1 -- is the model above chance at each horizon?** (10,000-resample
percentile bootstrap over folds; one-sided sign test)

| horizon | AUROC | 95% CI | sign test | verdict |
|---|---|---|---|---|
| k=0 | 0.963 | [0.932, 0.988] | 7/7, p=0.008 | **above chance** |
| k=30 | 0.882 | [0.746, 0.978] | 7/7, p=0.008 | **above chance** |
| k=60 | 0.791 | [0.573, 0.950] | 6/7, p=0.063 | CI excludes 0.5, sign test not significant |
| k=90 | 0.734 | [0.545, 0.891] | 6/7, p=0.063 | CI excludes 0.5, sign test not significant |

Forecasting ahead is statistically supported out to k=30 (1 second) by both
measures. At k=60/90 the bootstrap CI excludes chance but the stricter sign
test does not reach alpha=0.05 -- the evidence weakens with horizon rather
than vanishing, and should be reported as suggestive, not established.

**Q2 -- does the LSTM beat the baselines?** (Wilcoxon signed-rank on paired
fold AUROCs, Holm-Bonferroni corrected across all 16 comparisons)

**No comparison survives correction.** The largest raw effect is vs. the
DeepROD reimplementation at k=0 (7/7 folds favour the LSTM, median gap
+0.270, p=0.016 raw), followed by the rule-based baseline at k=30 (7/8
folds, +0.236, p=0.055 raw). Against logistic regression the median gap is
+-0.001 at every horizon -- the two are indistinguishable, consistent with
Sect. 7's "learning matters more than recurrence" finding.

This is a genuine limitation and is reported as one: with 7-11 folds these
tests are severely underpowered, so a non-significant result means *this
experiment cannot distinguish the methods*, not that they are equivalent.
The point estimates consistently favour the LSTM; the sample size cannot
confirm it. A larger clip corpus is the only fix, and is named as future
work rather than argued around.

### 10. Operating envelope -- where this works and where it does not

Source: `src/operating_envelope.py` ->
`outputs/reports/operating_envelope.json`.

The UMN-vs-ShanghaiTech gap was previously attributed to crowd density
(UMN ~15 persons/frame vs. ShanghaiTech ~4). **Testing that directly does
not support it.** Correlating per-scene mean density against per-scene AUROC
across the 8 evaluable ShanghaiTech scenes gives Pearson r=-0.481 (p=0.228),
Spearman rho=-0.524 (p=0.183) -- statistically indistinguishable from zero,
and *negative* in sign (sparser scenes scored slightly better, not worse).
Density does not explain the ordering.

The better-supported explanation is a mismatch in the **kind** of anomaly.
Scene 05 scores below chance at every horizon (0.248-0.300), and direct
frame inspection confirms its three anomalies are small-group physical
altercations involving 2-5 people -- events that do not move a crowd-level
aggregate and can push it the wrong way. UMN's anomaly is always collective
escape, where a large fraction of visible people accelerate together.

**Suited to:** collective, crowd-scale events; dense scenes (validated at
~15 persons/frame); fixed cameras; lead times to k=30 established and
k=60/90 suggestive.

**Not suited to:** individual or small-group anomalies (a lone cyclist, a
2-5 person scuffle); sparse scenes, where direction_entropy and stop_ratio
are undefined 62%/67% of the time even within occupied zones; moving or
newly-installed cameras, since the per-clip running normalizers need time to
stabilise; any deployment needing a calibrated probability rather than a
ranking, as all reported metrics are threshold-free.

### Demo

**Working, verified end-to-end**: `src/live_dashboard.py` on the UMN video
(`data/raw/umn/Crowd-Activity-All.avi`), 700 frames, headless render to
`outputs/figures/live_dashboard_demo.mp4` (320x240, 15fps output, ~47s).
Confirmed real detections, tracked IDs, and a genuine GREEN -> RED alert
escalation matching the ground-truth panic onset -- not a placeholder run:

- Frame 10 (calm): GREEN banner, near-zero per-zone probabilities.
- Frame 350 (building): still GREEN overall, but the k=90 ("3s") horizon
  bar has gone red and one zone hits 0.84 -- the forecast head reacting
  ahead of the overall alert level, as designed.
- Frame 690 (post-onset, now into clip 1's indoor scene): full RED, all
  four horizon bars maxed (1.00/1.00/1.00/0.96), sparkline spiking,
  tracked person boxes with IDs visible.

**Caveat, worth having a ready answer for if asked**: this demo runs
`models/umn_forecast_lstm.pt`, which per its own README is trained on
*all 11* UMN clips with no held-out fold -- clips 0 and 1, shown in this
demo, were both in its training set. So the alert-timing pattern (RED
firing ~10-12 frames before the ground-truth onset) shows the *pipeline
mechanism* working correctly end-to-end, not independent proof the
forecast generalizes to unseen crowds -- that evidence is the LOCO AUROC
decay (subsection 2/3) and the feature-ablation nowcast/forecast swap
(subsection 3), where every test clip is genuinely unseen. If time allows
before the review, a stronger version of this demo would run a specific
LOCO fold's checkpoint (trained on the other 10 clips) against its one
held-out clip -- ask me if you want that built.

**For the live presentation**: two options.
1. **Play the pre-rendered video** (`outputs/figures/live_dashboard_demo.mp4`)
   -- zero risk of a live technical failure, already verified to look right.
2. **Run it live**: `python3 -m src.live_dashboard data/raw/umn/Crowd-Activity-All.avi --realtime`
   (drop `--headless`/`--output` to open an interactive window instead of
   rendering to file). On this CPU-only machine, detection runs at
   ~103-118ms/frame against a 33ms/frame budget at 30fps, so ~74% of
   frames get dropped during real-time playback -- the dashboard still
   updates coherently (`SequenceBuffer` forward-fills by video frame
   position, not by how many frames were actually processed, so drops
   don't desync the alert timeline), but the visible feed will look choppy,
   not smooth. Worth rehearsing once before presenting live; option 1 is
   the safer default if presentation time is tight or a smooth feed matters
   more than "it's happening live."

---

## Slide 13 -- Remaining Modules

- **Resolved: DeepROD controlled baseline (Slide 12, subsection 7).**
  Reimplemented with their full published hyperparameter search and
  evaluated under the same LOCO protocol as our model -- 0.702 AUC vs. our
  0.963 (k=0), a directly comparable row measured on identical frames and
  labels.
- **Resolved: statistical significance testing (Slide 12, subsection 9).**
  Bootstrap CIs and paired Wilcoxon tests with Holm correction, via
  `src/significance_tests.py`. Outcome is mixed and reported as such:
  above-chance forecasting is established to k=30, and no baseline
  comparison survives correction at n=7-11 folds.
- **Resolved: operating envelope (Slide 12, subsection 10).** The previous
  density-mismatch explanation for the ShanghaiTech gap was tested and is
  not supported (r=-0.481, p=0.228); anomaly *kind* is the better-supported
  explanation.
- **Resolved: live-pipeline hardening for continuous feeds.**
  `stream_processor.py` previously terminated permanently on any failed
  read -- correct for a file at EOF, fatal for a camera after one network
  blip. Live sources (rtsp/http/camera-index) now reconnect with capped
  exponential backoff (1s doubling to 30s), and all scene-dependent state
  (tracker IDs, running normalizers, sequence buffer) is reset on
  reconnect, since a window spanning an outage would pair frames separated
  by an unknown gap. 14 tests pass.
- **Open: comparison against recent published methods (2023-2026).** No
  numbers table yet against current work in crowd anomaly detection or
  video anomaly *anticipation* -- the latter is an established task name
  (Cao et al., CVPR 2023) that this project's multi-horizon framing maps
  onto directly, and citing it would ground the forecasting contribution
  in existing literature rather than presenting it as novel framing.
- ShanghaiTech thin-scene reporting refinement / possible additional scene
  coverage
- Homography-based persons/m^2 feature (on backlog, not started)

---

## Slide 14 -- Gantt Chart (Phase-3 work only)

[Needs your actual dates. Suggested rows based on what's open per Slide 13:]

| Task | Start | End |
|---|---|---|
| DeepROD-style controlled baseline reimplementation | done | 2026-08-23 |
| Statistical significance testing | [DATE] | [DATE] |
| ShanghaiTech reporting refinement | [DATE] | [DATE] |
| Manuscript drafting | [DATE] | [DATE] |
| Final review prep | [DATE] | [DATE] |

---

## Slide 15 -- Repository

- **GitHub**: https://github.com/jeevitha-14s/crowd-lstm
  (confirm visibility is set to Private with only team + guide added)
- Key contents: `src/` (all pipeline modules), `tests/`, `data/features/`
  (checked-in extracted features), `models/` (checkpoints + provenance
  READMEs), `outputs/reports/` (all result artifacts, incl.
  `PAPER_RESULTS.md` -- the consolidated source of every number in this
  deck), `README.md` (setup + deployment instructions)

---

## Slide 16 -- References (IEEE format)

[1] H. Ammar and A. Cherif, "DeepROD: a deep learning approach for
real-time and online detection of a panic behavior in human crowds,"
*Machine Vision and Applications*, vol. 32, no. 3, Art. no. 57, 2021.

[2] M. H. Sharif, L. Jiao, and C. W. Omlin, "Deep crowd anomaly detection
by fusing reconstruction and prediction networks," *Electronics*, vol. 12,
no. 7, p. 1517, 2023.

[3] W. Liu, W. Luo, D. Lian, and S. Gao, "Future frame prediction for
anomaly detection -- a new baseline," in *Proc. IEEE/CVF Conf. Computer
Vision and Pattern Recognition (CVPR)*, Salt Lake City, UT, USA, 2018,
pp. 6536-6545.

[4] M. Sabih and D. K. Vishwakarma, "Crowd anomaly detection with LSTMs
using optical features and domain knowledge for improved inferring," *The
Visual Computer*, vol. 38, pp. 1719-1730, 2022.

[5] M. H. Sharif, L. Jiao, and C. W. Omlin, "Deep crowd anomaly detection:
state-of-the-art, challenges, and future research directions," *Artificial
Intelligence Review*, 2025.

[6] S. Shao et al., "CrowdHuman: A benchmark for detecting human in a
crowd," *arXiv preprint arXiv:1805.00123*, 2018.

[7] Ultralytics, "YOLOv8," 2023. [Online]. Available:
https://github.com/ultralytics/ultralytics

[8] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating
every detection box," in *Proc. European Conf. Computer Vision (ECCV)*,
2022, pp. 1-21.

[9] "Unusual Crowd Activity Dataset," University of Minnesota. [Online].
Available: http://mha.cs.umn.edu/proj_events.shtml#crowd

---

## Slide 17 -- Thank You
(Keep as-is.)

---

## What I need from you to finish this

1. Project Title (or approve the suggested one), Project ID, Guide name,
   team member names.
2. Per-member module assignment, LOC, and time spent for the contribution
   table (Slide 7-11) -- I can pull LOC per file with `wc -l` if you tell me
   who owns which files, but I can't attribute names/effort myself.
3. Confirm the Phase-2/Phase-3 boundary (I inferred it from file
   modification dates -- Aug 8 vs. later -- but that's a guess, not a fact).
4. Actual dates for the Gantt chart.
5. Whether you want an actual rendered architecture diagram image instead of
   the text pipeline above -- I can produce one.
6. Whether you want the "proper" held-out demo built (a specific LOCO fold's
   checkpoint against its one held-out clip) before the review -- the
   current demo (Slide 12, Demo section) is genuine and working, but uses
   the all-clips deployment checkpoint, so it's not independent
   generalization evidence -- see that section's caveat.
7. Watch the video I sent (`live_dashboard_demo.mp4`) yourself before
   presenting it -- I verified it frame-by-frame, but you should see it too.
