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

Most crowd-monitoring systems **detect** panic after it is already visible in
a frame. This project reframes the problem as **forecasting**: predicting a
panic onset *k* seconds before it happens, using only features observable
before onset, so the output is early enough to act on rather than a
confirmation of what a human could already see.

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

Weaker than UMN -- attributed to a density mismatch (ShanghaiTech ~4.1
persons/frame vs. UMN ~15.2) and, for scene 05 specifically, a mismatch in
kind (its "anomalies" are small-group altercations, not crowd-scale panic --
confirmed by direct frame inspection, not just the metric). Reported
honestly as a negative-transfer finding, not hidden.

### 5. Deployment latency (per-stage, CPU vs. GPU)

| Resolution | Device | Total (ms) | Achieved FPS | Drop rate @ 30fps |
|---|---|---|---|---|
| 640x480 | CPU | 158.62 | 6.35 | 78.8% |
| 640x480 | CUDA (T4) | 12.73 | 84.18 | **0.0%** |

GPU deployment is real-time with zero drop; CPU-only is not viable at 30fps
input -- a concrete deployment-hardware requirement for the demo/discussion.

### Demo
[Insert screenshots/GIF of `live_dashboard.py` running on the UMN clip here
-- I can help capture these if you run it locally and share the output.]

---

## Slide 13 -- Remaining Modules

- Controlled baseline: re-implement a single-step-ahead LSTM (DeepROD-style
  prediction) under the *same* LOCO protocol, to get one directly comparable
  row against related-work numbers (currently those are each paper's own
  self-reported number on their own dataset/protocol, not a controlled
  comparison)
- Statistical significance testing on AUROC (currently reported as wide
  confidence intervals only, per Section 6/10 of `PAPER_RESULTS.md`)
- ShanghaiTech thin-scene reporting refinement / possible additional scene
  coverage
- [ANY OTHER PLANNED WORK -- fill in]

---

## Slide 14 -- Gantt Chart (Phase-3 work only)

[Needs your actual dates. Suggested rows based on what's open per Slide 13:]

| Task | Start | End |
|---|---|---|
| DeepROD-style controlled baseline reimplementation | [DATE] | [DATE] |
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
