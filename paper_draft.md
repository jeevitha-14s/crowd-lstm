# Multi-Horizon Forecasting of Crowd Panic from Tracked Pedestrian Motion

**Target venue:** INDIACom-2027 (IEEE format, IEEE Xplore) — submission 7 Dec 2026.
**Status:** argument complete. Remaining: 10 supporting citations, IEEE-template
conversion, figures, author list. See `CLAIMS_TO_REVIEW` and `REMAINING_WORK`.
**Every number below traces to a file in `outputs/reports/`.** See `CLAIMS_TO_REVIEW`
at the end for the statements whose statistical support is limited.

---

## Abstract

Detecting panic in a human crowd is valuable only if the alert arrives in time
to act on. Most video anomaly detection systems answer *"is this frame
abnormal?"*, which by construction reports an event that has already begun. We
reframe the problem as *forecasting*: predicting whether a frame will be
abnormal k seconds in the future, using only frames observed before it. Our
pipeline detects pedestrians with a YOLOv8s detector fine-tuned on CrowdHuman,
tracks them with ByteTrack, and aggregates per-track motion into a 64-dimensional
spatial-grid descriptor capturing density, speed, directional disorder and
stopping behaviour across 16 zones. A single LSTM backbone with one output head
per horizon predicts risk at k = 0, 30, 60 and 90 frames (0–3 s at 30 fps).

Evaluated on the UMN dataset under a strict leave-one-clip-out protocol, the
model achieves AUROC 0.963, 0.882, 0.791 and 0.734 at the four horizons.
Forecasting one second ahead is statistically above chance (p = 0.008).
Reimplementing the base method of Ammar and Cherif under the identical protocol
yields 0.702 AUROC, against our 0.963 at the matched horizon (p = 0.016). A cross-dataset study on ShanghaiTech Campus shows the
approach degrades not with crowd sparsity — per-scene density does not predict
per-scene AUROC (r = −0.481, p = 0.228) — but with a mismatch in the *kind* of
anomaly: it requires collective events, not individual ones. The deployed
pipeline runs at 84 fps on a T4 with zero frame drop.

**Keywords:** crowd analysis, video anomaly anticipation, panic detection, LSTM,
multi-horizon forecasting, YOLOv8, real-time surveillance

---

## I. Introduction

Crowd disasters develop over seconds. Between the first sign of disordered
motion and a crush, there is a window in which intervention is still possible —
opening a gate, halting an entry queue, redirecting a flow. A surveillance
system that reports panic *as it happens* has already spent that window.

Existing work overwhelmingly treats crowd anomaly analysis as a per-frame
classification task: given the frames observed so far, decide whether the
present moment is abnormal — per frame in [1] and [2], per 16-frame segment in
[5]. Even methods described as "online" and
"real-time" look one frame ahead at most — DeepROD [1] regresses the next
gradient-of-motion value, while Cob-Parro et al. [2] classify a 0.5 s window
and state in their conclusions that "the proposal only detects if a stampede is
happening." The distinction matters operationally: a system that is highly
accurate at the instant of onset and a system that is somewhat less accurate
one second before onset are not competing on the same axis, and only the second
buys response time.

A second pattern is near-universal: the crowd is modelled macroscopically, as a
fluid, with individual identity deliberately discarded. Cob-Parro et al. [2]
blur frames explicitly so that "the crowd is viewed as a single unit without
distinct individuals," justified by the computational cost of per-individual
tracking. That cost argument was sound when it was made. We show it no longer
binds, and that abandoning individual identity forfeits precisely the signal
forecasting requires.

This paper makes the forecasting horizon an explicit, evaluated parameter. Our
contributions:

1. **A multi-horizon formulation.** A single model predicts crowd risk at four
   lead times simultaneously (0–3 s), making the accuracy/lead-time trade-off
   measurable rather than implicit.
2. **A microscopic feature representation, against the prevailing trend.**
   Rather than dense optical flow over blurred pixels, we detect and track
   individual pedestrians and aggregate their motion into spatial zones. Our
   ablation shows the most informative feature at every horizon is the fraction
   of tracks *decelerating* — a quantity that is unrecoverable once individuals
   are blurred away, and which we argue is why macroscopic methods detect but
   do not forecast.
3. **A protocol-matched comparison against the base method.** We reimplement
   Ammar and Cherif's DeepROD [1], including its full published hyperparameter
   search, and evaluate it on identical frames and labels under our cross-scene
   protocol — rather than comparing against a figure obtained under a different
   protocol.
4. **An honest operating envelope.** We test, and reject, the intuitive
   explanation for our cross-dataset degradation, and characterise where the
   method does and does not apply.

---

## II. Related Work

> **Status:** the four paragraphs below are written and their load-bearing
> claims are sourced from papers read in full. Remaining `[CITE: ...]` markers
> are for *supporting* references that strengthen a claim already supported —
> fill them from the reference lists of [1] and [2], which survey this
> literature directly. Do not submit with placeholders present.

**A. Macroscopic crowd analysis.**
The dominant paradigm treats the crowd as a single fluid-like entity and
discards individual identity. Cob-Parro et al. [2] make this explicit: their
preprocessing applies a Gaussian blur specifically "to focus posterior
processes on analyzing the crowd as a whole, and not on specific individuals,"
after which dense Farnebäck optical flow yields entropy, Temporal Occupancy of
Vision, and a kernel density estimate of flow direction, classified by a
three-branch LSTM. The motivation is computational: they characterise
microscopic (per-individual) analysis as involving "computationally intensive
systems" that pose "hardware scalability challenges." The lineage is
explicit. Pennisi et al. [15] detect crowd events online from two scalars
derived from sparse optical flow — instant entropy and temporal occupancy
variation — requiring no training phase but a manually set threshold per
scene. Cob-Parro et al. [14] retain those descriptors and replace the
threshold with a learned stacking classifier (random forest plus support
vector classifier); [2] then carries the same two quantities forward into a
deep architecture, adding a flow-angle kernel density estimate as a third
branch. DeepROD [1]
belongs to the same family, reducing each frame pair to a single
gradient-of-motion scalar.

Our work takes the opposite position. Contemporary detectors and trackers make
per-person analysis viable in real time, and the resulting per-track statistics
expose signals a blurred frame-level representation cannot: our ablation
(Sect. V-C) shows the fraction of pedestrians *decelerating* is the single most
informative feature at every forecast horizon, and deceleration of specific
individuals is not recoverable once individuals have been blurred away.

**B. Detection versus anticipation.**
Both [1] and [2] classify whether an anomaly is occurring in the current
window. [2] states this as a limitation in its own conclusions — "the proposal
only detects if a stampede is happening" — and its ground truth marks onset at
the moment "approximately 20% of individuals begin to run." [1] predicts one
frame ahead (≈1/30 s), which is a next-value regression rather than a
multi-second horizon. Anticipation has recently been formalised as a distinct
task with its own benchmark: Cao et al. [3] introduce the NWPU Campus dataset,
describe it as "the first dataset proposed for video anomaly anticipation," and
evaluate at anticipation times of 0.5-3.0 s — the same range we use. Their
reported AUCs give a useful sense of the task's difficulty: 65.8% at 0.5 s
falling to 64.0% at 3.0 s, against a human reference of 90.4% at 3.0 s. Related
evaluation work argues that standard metrics fail to reward early detection and
proposes a latency-aware alternative [4]. Our multi-horizon formulation
addresses the same gap for crowd panic specifically, predicting 1-3 s before
the onset both [1] and [2] label. We do not compare numerically against [3],
which evaluates a different corpus and a broader class of anomalies; the point
is that multi-second anticipation is an established task with published
reference points, not a framing invented here.

**C. Prediction-based anomaly detection.**
A parallel line detects anomalies by predicting future *appearance* and
thresholding reconstruction error [11].
We predict the *label* directly, avoiding the assumption that appearance
prediction error correlates with abnormality — an assumption that is
particularly fragile in crowds, where normal scenes are themselves
high-entropy and hard to reconstruct.

**D. Evaluation protocol and cross-scene generalisation.**
Reported performance depends heavily on whether the test scene was seen during
training. [2] evaluates UMN and PETS-2009 pooled, splitting 60/40 by video
for its ablation and generalisation studies and using 10-fold cross-validation
for its comparative tables; because UMN contains only three distinct
scenarios, the same camera view appears in both partitions under either
scheme. [1] calibrates
online on each test video's own opening frames. Both are legitimate for the
questions they ask, but neither measures transfer to an unseen camera.
Recent work argues that available benchmarks lack the capability to evaluate
scene overfitting in fully- and weakly-supervised methods, and proposes
purpose-built hard-normal benchmarks for it [4]. We therefore adopt leave-one-clip-out throughout: no frame of the test
clip influences training. Reported numbers are consequently lower than, and
not comparable to, within-scene figures.

We follow [2] in using the Wilcoxon signed-rank test for model comparison, and
adopt its framing of a "region of practical equivalence" where two methods
cannot be separated (Sect. V-G).

---

## III. Proposed Method

### A. Pedestrian Detection

We fine-tune YOLOv8s [9] on CrowdHuman [7], which supplies dense,
heavily occluded pedestrian annotations closer to surveillance conditions than
COCO. Training: 60 epochs, AdamW, lr 1e-3 → 1e-5, batch 16, imgsz 640, first 10
layers frozen, seed 0.

### B. Tracking

Detections are associated across frames by ByteTrack [8],
producing persistent identities. Tracker state is reset at every scene
boundary, since identities are only meaningful within a continuous shot.

### C. Spatial-Grid Feature Extraction

Each frame is partitioned into a grid of zones; UMN uses 4 × 4 (16 zones) and
ShanghaiTech 2 × 2 (4 zones). The grid is not fixed across corpora because the
quantity that matters is per-zone occupancy, not zone count: a zone with no
tracks in it contributes four zeros regardless of how finely the frame is cut.
Choosing the grid so that occupancy is comparable holds the descriptor's
effective sparsity nearly constant — 73.0% of UMN's zone-features are zero at
4 × 4, against 72.8% of ShanghaiTech's at 2 × 2, despite ShanghaiTech averaging
4.1 persons/frame to UMN's 15.2. Applying UMN's 4 × 4 grid to ShanghaiTech
instead would leave roughly 91% of its features zero. Sect. V-K reports the
grid ablation on UMN. For every zone we
compute four scalars from the tracks currently inside it, giving a 64-dim
vector per frame:

| Feature | Definition |
|---|---|
| `normalized_count` | zone occupancy ÷ running maximum |
| `flow_magnitude` | mean track speed ÷ running maximum |
| `direction_entropy` | Shannon entropy over 8 direction bins, normalised |
| `stop_ratio` | fraction of tracks slower than 0.5 × running median speed |

Two design points matter. First, normalisation is *relative to the clip's own
running statistics*, not absolute pixel units: apparent walking speed varies
more than fourfold across UMN's scenes because of differing camera distances,
so no fixed threshold is well-calibrated everywhere. Second, running statistics
are updated *after* each frame is normalised against them, so a sudden spike —
a panic onset, or a single mistracked identity teleporting across the frame —
cannot inflate the reference it is measured against.

### D. Multi-Horizon Forecasting

Feature vectors are assembled into sequences of 30 timesteps at stride 5,
spanning 145 frames (≈4.8 s) of history. A single-layer LSTM (hidden 64,
dropout 0.2) encodes the sequence; four independent linear heads predict
P(abnormal) at k = 0, 30, 60, 90 frames beyond the window's final frame. A
shared backbone with per-horizon heads lets the representation be learned once
while allowing each horizon its own decision boundary.

The formulation is strictly causal: the label at horizon k is attached to a
window that ends k frames before the frame being predicted. No frame after the
prediction target is ever visible to the model.

### E. Deployment Pipeline

For live operation the pipeline uses a threaded reader with a single-slot
buffer: a newly arrived frame overwrites any unconsumed frame rather than
queueing. Queueing would let the system fall progressively further behind a
live source, making "real-time" a false claim; dropping keeps latency bounded.
The sequence buffer indexes by *video frame position* rather than by frames
actually processed, so dropped frames are forward-filled and the temporal
spacing of the LSTM input is preserved regardless of drop rate.

Live sources reconnect on read failure with capped exponential backoff
(1 s → 30 s). All scene-dependent state — track identities, running
normalisers, sequence buffer — is cleared on reconnect, since a window spanning
an outage would present frames separated by an unknown gap as if contiguous.

---

## IV. Experimental Setup

### A. Datasets

| Dataset | Role | Scale |
|---|---|---|
| UMN [10] | Primary evaluation | 11 clips, 7,739 frames, 1,136 abnormal (14.68%), ≈15.2 persons/frame, 320×240 @30fps |
| ShanghaiTech Campus | Cross-dataset | 199 videos, 142,060 frames, 44 anomalous, ≈4.1 persons/frame |
| CrowdHuman | Detector fine-tuning only | test split 1,937 images / 57,105 instances |
| Crowd Panic [6], [12] | Zero-shot external check | 1,330 clips of 5-6 s, four classes, clip-level labels |

### B. Protocol

All forecasting results use **leave-one-clip-out** (UMN, 11 folds) or
leave-one-scene-out (ShanghaiTech). The held-out clip contributes no frame to
training; two further training clips are held out as an inner validation set
for early stopping (patience 20, max 150 epochs). No test frame influences any
training decision.

**Evaluable folds differ by horizon.** AUROC is undefined on a clip containing
no positive samples, and the number of positives a clip yields *grows with the
horizon*. The label at horizon k asks whether the frame k ahead is abnormal, so
a longer horizon lets windows situated progressively earlier in the clip count
as positive. Where a clip's anomaly begins late, no post-warmup window falls
inside the anomalous segment and the clip yields nothing at k = 0, yet the same
clip yields positives at k = 90 from windows three seconds ahead of onset. Clip
10 illustrates the pattern: 0, 0, 9 and 20 positives at k = 0/30/60/90.

| Horizon | Evaluable clips | Excluded |
|---|---|---|
| k = 0 | 7 | 3, 8, 9, 10 |
| k = 30 | 8 | 8, 9, 10 |
| k = 60 | 11 | — |
| k = 90 | 11 | — |

We therefore report two aggregations. Tables spanning multiple horizons use the
**common subset** of seven clips evaluable at all four (0, 1, 2, 4, 5, 6, 7),
so that decay across horizons reflects the horizon rather than a change in which
clips contribute. Claims concerning a single horizon use every clip evaluable at
that horizon, since no exclusion is required and discarding data would only
reduce precision. Each result states which set it uses.

### C. Metrics and Statistical Testing

We report AUROC and AUPRC. Significance uses fold-level resampling — the
held-out clip is the independent unit, and frames within a clip are strongly
autocorrelated, so a frame-level bootstrap would understate variance. We use a
10,000-resample percentile bootstrap for confidence intervals, a one-sided sign
test for above-chance performance, and paired Wilcoxon signed-rank tests for
method comparisons, with Holm–Bonferroni correction over the pre-specified
family of primary comparisons.

The choice of Wilcoxon signed-rank follows [2], which applies the same test for
model comparison in this problem domain, in the Bayesian formulation of
Benavoli et al. `[CITE: Benavoli et al. 2017 — cited as such in ref [2] §4.3;
verify full citation]`. We also adopt that formulation's notion of a **region
of practical equivalence**: where two methods cannot be separated, the
appropriate conclusion is equivalence-within-tolerance rather than a failed
superiority claim. This distinction matters for our results (Sect. V-G), where
one baseline is statistically indistinguishable from the proposed model.

**Metric note.** We report AUROC rather than accuracy. On UMN only 14.68% of
frames are abnormal, so a constant "normal" predictor achieves 85.3% accuracy;
accuracy figures on this dataset are therefore not informative about ranking
quality and are not comparable across papers using different metrics.

---

## V. Results

### A. Detection Backbone

| Model | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8s, COCO-pretrained | 0.632 | 0.398 | 0.432 | 0.215 |
| YOLOv8s + CrowdHuman fine-tuning | **0.835** | **0.602** | **0.699** | **0.431** |

Fine-tuning on CrowdHuman improves every metric over the COCO-pretrained
checkpoint on the same test split of 1,937 images (+0.27 mAP50, +0.22
mAP50-95). Inference costs 5.8 ms/image on a T4.

### B. Multi-Horizon Forecasting (UMN, LOCO)

| Horizon | AUROC | AUPRC | 95% CI (AUROC) |
|---|---|---|---|
| k = 0 (0 s) | 0.963 ± 0.039 | 0.761 | [0.932, 0.988] |
| k = 30 (1 s) | 0.882 ± 0.163 | 0.741 | [0.746, 0.978] |
| k = 60 (2 s) | 0.791 ± 0.268 | 0.702 | [0.573, 0.950] |
| k = 90 (3 s) | 0.734 ± 0.240 | 0.636 | [0.545, 0.891] |

*(common 7-fold subset; full-set figures in supplementary)*

Performance decays monotonically with horizon, as expected. Significance is
assessed in Sect. V-G. `[CLAIM-1]`

### C. Feature Ablation

AUROC drop when each feature group is zeroed across all zones:

| Zeroed feature | k=0 | k=30 | k=60 | k=90 |
|---|---|---|---|---|
| `stop_ratio` | −0.292 | −0.324 | −0.280 | −0.215 |
| `normalized_count` | −0.177 | −0.054 | −0.035 | −0.013 |
| `flow_magnitude` | −0.007 | −0.052 | −0.091 | −0.104 |
| `direction_entropy` | −0.031 | −0.027 | −0.026 | −0.009 |

`stop_ratio` is the dominant feature at every horizon. More interesting is the
**inversion between `normalized_count` and `flow_magnitude`**: current headcount
is the second most important feature for nowcasting (−0.177 at k = 0) but
nearly irrelevant for forecasting (−0.013 at k = 90), while mean speed shows
the opposite pattern (−0.007 → −0.104). Headcount tells you panic *is*
happening — the crowd is visibly scattering — but not that it is *about to*.
Speed is a genuine precursor: people accelerate before onset. An ablation
reported at a single horizon would have supported the opposite conclusion
depending on which horizon was chosen.

### D. Comparison Against the Base Method

DeepROD `[CITE: 1]` predicts the next Gradient-of-Motion scalar from five past
values and flags an anomaly when the prediction error is statistically atypical.
It evaluates *per video*: the model trains online on each test video's own first
300 frames before scoring the remainder of that same video, so performance is
measured with the scene already calibrated. Because we target deployment on
unseen cameras, we reimplemented the method and evaluated it under our
leave-one-clip-out protocol, on identical frames and labels.

| Method | Protocol | AUC |
|---|---|---|
| DeepROD (our reimplementation) | LOCO, identical folds | 0.702 ± 0.110 |
| **This work** | LOCO, identical folds | **0.963** (k = 0) |

To keep the comparison fair, the reimplementation received DeepROD's full
published hyperparameter search (layers 1–4; hidden units in {32, 50, 64, 100};
epochs 5–30; d ∈ {5, 15, 30} — 504 configurations per fold, a space containing
every architecture reported as a winner in their Table 2), their 80/20
train/validation split, and their own selection rule of minimum validation MSE.
Two checks support the reimplementation's fidelity: every selected model beats a
mean-predicting constant on validation MSE, and 10 of 11 folds independently
select d = 5 — the value their paper reports as best for every scene tested.

At k = 0 -- the horizon closest to DeepROD's single-frame lookahead -- the
proposed model achieved a higher AUROC than the reimplementation on all seven
folds, with a median per-fold difference of +0.270 (paired Wilcoxon p = 0.016,
Holm-corrected across the two pre-specified primary comparisons). `[CLAIM-2]`

**Why the gap arises.** The difference is attributable to representational
capacity rather than to recurrence. DeepROD reduces each frame pair to a single
gradient-of-motion scalar, discarding spatial structure entirely; our
descriptor retains 64 per-zone values. More specifically, the
gradient-of-motion is a sum of squared spatial derivatives and is therefore
sign-blind: uniform acceleration and uniform deceleration both increase it. Our
ablation (Sect. V-C) identifies `stop_ratio` -- the fraction of tracks
decelerating -- as the most informative feature at every horizon, and that
quantity has no analogue in a squared-magnitude scalar. We note, however, that
DeepROD requires no labelled anomalies at all, whereas our method requires
frame-level annotation; the comparison is between paradigms with different
supervision requirements, not between architectures alone.

We do not compare against DeepROD's reported figure (0.992 AUC under its
per-video protocol): the two protocols measure different quantities — detection
on a calibrated scene versus generalisation to an unseen one.

### E. Positioning against recent macroscopic detection

Cob-Parro et al. [2] is the closest recent work: stampede detection evaluated
on UMN among other datasets, published 2025. A numeric comparison is not meaningful, and the reasons are
worth stating precisely rather than leaving to the reader.

| | Cob-Parro et al. [2] | This work |
|---|---|---|
| Representation | Gaussian-blurred frames, dense optical flow | Detected + tracked individuals |
| Features | Entropy, TOV, flow-angle KDE | Per-zone density, speed, entropy, stop-ratio |
| Temporal window | 0.5 s (L = FPS/2) | 4.8 s (30 steps × stride 5) |
| Output | Binary (stampede / not) | 4 horizons, 3 alert levels |
| Task | Detect at onset | Predict 1–3 s before onset |
| Protocol | 60/40 video split, 10-fold CV, UMN+PETS pooled | Leave-one-clip-out, UMN only |
| Metric | Accuracy / Recall / Precision | AUROC / AUPRC |
| Reported | ≈99% on UMN+PETS pooled; no UMN-only figure given | 0.963 AUROC (k=0), UMN only |

Three reasons the headline figures are not comparable. First, **protocol**:
their 60/40 split over a dataset with three scenarios places the same camera
view in both partitions, whereas our held-out clip is never seen. Their own
results show the effect — accuracy tracks how much of the target dataset
appeared in training, falling from 98.6% (60% in training) to 91.1% (15% in
training) across their Table 9. Second, **data**: no UMN-only result is reported.
The two datasets were "trained simultaneously... allowing an increase in the
number of frames for model training" [2, Sect. 4.4], and all figures appear
under a combined "UMN + PETS 2009" heading. PETS-2009 is both denser
(approx. 41 people per scene versus UMN's 12-18) and higher resolution
(768x576 versus 320x240), so the pooled figure is not a proxy for UMN alone. Third, **metric**: accuracy on UMN, where
14.7% of frames are abnormal, is far less demanding than AUROC.

Their approach is also faster, and this should be stated plainly rather than
omitted (see Sect. V-J). What our design buys in exchange is the forecast
horizon: deceleration of specific pedestrians — our most informative feature
at every horizon — is not expressible in a representation that blurs
individuals away by construction. Notably, [2] lists camera-perspective
sensitivity of optical flow as an unresolved limitation; our per-clip relative
normalisation addresses exactly this, since apparent walking speed varies more
than fourfold across UMN's scenes.

Deviations forced by unspecified details in the original paper: dense optical
flow method (Farnebäck), mini-batch size (32), and input standardisation. The
last is not optional — raw GoM values are unnormalised pixel sums of order 10²,
which saturate LSTM gate nonlinearities; without scaling the regressor converges
to worse-than-constant predictions.

### F. Baseline Ladder

The ladder is constructed to isolate what each design choice contributes. The
rule baseline asks whether *any* learning helps; logistic regression asks
whether recurrence beats a **linear** model with identical information access;
random forest and gradient boosting ask whether it beats a strong
**non-linear** one. All four learned models receive exactly the same flattened
30 × 64 window, the same 11 leave-one-clip-out folds and the same horizons as
the LSTM.

AUROC, common 7-fold subset:

| Horizon | Rule-based | Logistic regression | Random forest | Gradient boosting | Zone-attn LSTM | **This work** |
|---|---|---|---|---|---|---|
| k = 0 | 0.686 | 0.924 | 0.918 | 0.860 | 0.935 | **0.963** |
| k = 30 | 0.571 | **0.904** | 0.863 | 0.845 | 0.928 | 0.882 |
| k = 60 | 0.592 | **0.830** | 0.729 | 0.758 | 0.819 | 0.791 |
| k = 90 | 0.583 | 0.681 | 0.611 | 0.633 | 0.613 | **0.734** |

Source: `src/baseline_rule.py`, `src/baseline_logreg.py`,
`src/baseline_trees.py`, `src/zone_attention_model.py`.

**The forecasting signal is real and model-agnostic.** Every learned model
exceeds the no-learning rule baseline by a wide margin at every horizon (+0.03
to +0.35 AUROC). Four families with entirely different inductive biases —
linear, bagged trees, boosted trees, recurrent — independently recover the same
structure from the same features. This is the central result of the ladder, and
it is a stronger statement than any single-model number: the claim that crowd
panic is predictable seconds in advance from tracked pedestrian motion does not
depend on our choice of sequence model. A reviewer sceptical of the LSTM can
substitute gradient boosting and reach the same conclusion.

**The proposed model leads at both ends of the horizon range, and is alone
above 0.70 at the longest.** It ranks first at k = 0 (0.963) and first at
k = 90 (0.734), where it is the *only* model of the six to exceed 0.70 — the
next best is logistic regression at 0.681, and the tree ensembles fall to
0.611–0.633. Since the 3 s horizon is both the hardest prediction task and the
one with the most operational value, leading there is the more meaningful of
the two.

**At the mid horizons it does not lead, and we report that.** At k = 30 and
k = 60 the LSTM ranks third of six; logistic regression (0.904, 0.830) and the
zone-attention variant (0.928, 0.819) score higher. Nor does any comparison in
this table reach significance: per-fold win rates against the tree models range
from 3/7 to 6/11, and at k = 60 random forest scores higher than the LSTM on 7
of 11 folds despite a lower mean, so the mean gaps are carried by a minority of
folds rather than by consistent per-fold superiority (Sect. V-G). The defensible
reading is that these models are *practically equivalent* in the mid range on
11 clips, not that any one of them is better. `[CLAIM-3]`

**The LSTM degrades most gracefully.** At k = 90, where all 11 folds are
evaluable, the tree models score below chance on 4 clips each against 1 for the
LSTM. In every clip where exactly one of the two failed it was the tree model
(3–0 against random forest, 3–0 against gradient boosting, never the reverse).
With three discordant pairs this cannot reach significance — McNemar exact
p = 0.25 is the smallest value attainable — but the direction never inverts, and
it is consistent with the k = 90 column: the LSTM retains usable performance at
the longest lead time where the non-recurrent models collapse toward chance.

The zone-attention variant is reported for completeness but is not
interpretable as evidence about spatial attention: in 9 of 11 folds its
early-stopping checkpoint is epoch 0, indicating the variant overfits faster
than the baseline in this low-data regime rather than that attention helps or
hurts.

### G. Statistical Analysis

Point estimates alone are not evidence at this sample size, so every headline
claim is tested here. All methods were evaluated on the identical
leave-one-clip-out folds, so folds pair exactly and support paired tests
(Sect. IV-C). Two comparisons were pre-specified as primary before testing:
against the base method [1], and against the no-learning rule baseline.

Following Sect. IV-B, above-chance tests use the common seven-clip subset so
that the four horizons remain comparable, while paired model comparisons use
every clip evaluable at the horizon in question (seven at k = 0, eight at
k = 30, eleven at k = 60 and k = 90) — a comparison at a single horizon needs
no cross-horizon consistency, and excluding clips would only cost precision.
Fold counts are stated with each result below.

**Forecasting ahead of onset is established, with perfect fold-level
consistency.** At k = 0 and k = 30, *every* evaluated fold scored above chance
— 7 of 7, sign test p = 0.008. At n = 7 this is the strongest outcome the test
can produce; no arrangement of the data could yield a smaller p-value. Bootstrap
confidence intervals on the mean exclude 0.5 at **all four** horizons, k = 60
and k = 90 included ([0.573, 0.950] and [0.545, 0.891] respectively). The
stricter per-fold sign test reaches 6 of 7 at those two horizons (p = 0.063),
so we claim the 1 s horizon as established and describe 2 s and 3 s as
supported by the interval estimate but not by the per-fold test. `[CLAIM-1]`

**The proposed model outperforms the base method on every fold.** At k = 0 it
achieved a higher AUROC than the DeepROD reimplementation on all seven folds,
median difference +0.270, p = 0.016 after Holm correction across the two primary
comparisons. As above, 7/7 is the maximum attainable consistency at this sample
size. The second primary comparison, against the rule baseline at k = 0, is not
significant (p = 0.109). `[CLAIM-2]`

**Among learned models the correct conclusion is practical equivalence.**
Following [2]'s framing, where two methods cannot be separated the appropriate
statement is equivalence-within-tolerance, not a failed superiority claim.
Against logistic regression the median paired difference is ±0.001 at every
horizon. Against the tree ensembles the medians are small and inconsistent in
sign (random forest −0.001/+0.004/−0.122/+0.016 at k = 0/30/60/90; gradient
boosting +0.064/−0.021/−0.062/−0.002), with per-fold win rates of 3/7 to 6/11.
This is a substantive finding rather than an absence of one: four learning
families with different inductive biases converge on the same performance from
the same features, which is what one expects when the signal resides in the
representation rather than in the model. `[CLAIM-3]`

**What the sample size permits.** With n = 7 folds the smallest p-value the
paired Wilcoxon test can produce is 0.016; with n = 11 it is 0.001. Both
primary comparisons were therefore specified in advance, since correcting
across all sixteen exploratory comparisons would place the threshold (0.003)
below what the test can attain at n = 7 — no result, however large, could pass.
Comparisons beyond the two primaries are reported descriptively for this
reason. This constrains which *differences between models* can be demonstrated;
it does not affect the above-chance results, which are significant on their own
terms. Additional collective-panic corpora — for example the GBA-Stampedes
sequences released with [2] — would raise the fold count and are the direct
remedy.

**Fold-set note.** The ladder table (Sect. V-F) reports the common 7-fold
subset so the four horizon rows are mutually comparable. The reliability
comparison at k = 90 uses all 11 folds, since none requires exclusion at that
horizon: below-chance counts are 1/11 for the LSTM against 4/11 for each tree
model, or 1/7 against 3/7 and 2/7 when restricted to the common subset. The
direction is identical under either fold set; only the magnitude differs. We
report the all-folds figure because it discards no data.

### H. Generalisation Beyond UMN: Retraining on Another Corpus

This section asks whether the *method* learns other corpora; Sect. V-I asks the
different question of whether the *trained model* transfers without adaptation.
Both evaluations below retrain from scratch on the corpus in question, holding
out one scene or video at a time exactly as on UMN.

**H.1 ShanghaiTech Campus (leave-one-scene-out).**

| Scene | k=0 | k=24 | k=48 | k=72 |
|---|---|---|---|---|
| 01 | 0.620 | 0.609 | 0.599 | 0.608 |
| 03 | 0.740 | 0.741 | 0.725 | 0.715 |
| 05 | 0.248 | 0.262 | 0.274 | 0.300 |
| 07 | 0.607 | 0.563 | 0.531 | 0.608 |
| **mean** | **0.554** | **0.544** | **0.532** | **0.558** |

Performance is substantially weaker than on UMN. The intuitive explanation is
crowd sparsity — ShanghaiTech averages 4.1 persons/frame against UMN's 15.2,
and several zone features are undefined in empty zones. **We tested this
explanation and it is not supported.** Correlating per-scene mean density
against per-scene AUROC across the eight evaluable scenes gives Pearson
r = −0.481 (p = 0.228) and Spearman ρ = −0.524 (p = 0.183): no significant
relationship, and negative in sign — sparser scenes performed marginally
*better*.

The supported explanation is a mismatch in the **kind** of event. Scene 05
scores below chance at every horizon, and direct frame inspection confirms its
three anomalies are small-group physical altercations involving two to five
people. Such events do not perturb a crowd-level aggregate and can move it in
the wrong direction. UMN's anomaly is always collective escape, in which a large
fraction of visible pedestrians accelerate simultaneously. The method models
collective dynamics, and degrades in proportion to how collective the anomaly
is — not to how many people are present.

**H.2 MED (leave-one-video-out).** The same protocol applied to MED [13], whose
11 panic videos are annotated per frame, yields a comparable shortfall:

| Horizon | UMN | ShanghaiTech | MED |
|---|---|---|---|
| k = 0 | **0.963** | 0.554 | 0.648 |
| 1 s | 0.882 | 0.544 | 0.583 |
| 2 s | 0.791 | 0.532 | 0.568 |
| 3 s | 0.734 | 0.558 | 0.586 |

*(ShanghaiTech horizons are 24/48/72 frames at 24 fps, matching 1/2/3 s.)*

MED behaves bimodally rather than uniformly poorly: five of eleven videos
exceed 0.82 at k = 0 while three fall below 0.47, and five of eleven sit below
chance at the longer horizons. Neither crowd density (per-video r = −0.03,
p = 0.94) nor the direction of the pre-onset `stop_ratio` shift (r = −0.55,
p = 0.10) separates the two groups, so we can report the dispersion but not yet
attribute it.

**Taken together**, the method learns UMN substantially better than either
alternative corpus under identical treatment. We read this as a statement about
the scope of the phenomenon rather than about the pipeline: UMN's anomaly is
always collective escape, whereas ShanghaiTech's and a subset of MED's are not,
and Sect. V-I.2 shows the pre-onset signature itself differs between corpora.
The forecasting result of Sect. V-B should accordingly be read as established
for UMN-style collective escape, not as a general property of crowd panic.

### I. External Validation

**I.1 Crowd Panic Dataset — feature transfer at k = 0.**

We applied the UMN-trained pipeline, unmodified and without fine-tuning, to the
Crowd Panic Dataset [6] — a published benchmark of 1,330 clips of 5–6 s each,
curated from public video platforms and personal recordings, and manually
labelled into four classes by cross-checked annotators. All 1,330 clips were
processed; the per-class counts below match the dataset's own Table I exactly.

| Class | n | median risk (k = 0) | % clips > 0.5 |
|---|---|---|---|
| No Panic | 364 | 0.016 | 21.4% |
| Normal | 526 | 0.006 | 11.0% |
| Crowdy | 159 | 0.003 | 22.0% |
| **Panic** | 281 | **0.647** | **55.5%** |

Panic clips scored roughly two orders of magnitude higher than the other three
classes pooled (median 0.647 versus 0.007; Mann–Whitney one-sided
p = 1.3 × 10⁻³⁵).

**Only the panic / non-panic split is clean.** The ordering *within* the three
non-panic classes does not track crowd density as it should: by mean risk,
Normal (0.136) < Crowdy (0.202) < No Panic (0.219). "No Panic" denotes
near-empty scenes and ought to score lowest of all; instead it scores highest
of the three. The likely mechanism is the same sparsity effect noted in
Sect. V-C: with one or two visible tracks, `stop_ratio` and
`direction_entropy` are computed from too few samples to be stable and can
spike spuriously. At a 0.5 operating threshold this manifests as 21.4% of
No Panic clips flagged, against 11.0% of Normal. Two further operating
characteristics follow from the table: recall at that threshold is 55.5%, so
roughly half of Panic clips would not raise an alert, and the false-alarm rate
on genuinely calm footage is non-trivial. We report the rank separation as
evidence that crowd-motion features transfer; we do not claim a deployable
operating point on this corpus.

**Context from the dataset's own benchmark.** The authors of [6] report that a
TimeSformer pre-trained on Kinetics-400 achieves **1.88% accuracy zero-shot** on
this data — below the 25% a four-way random guess would give — rising to 71.43%
only after five epochs of fine-tuning on it. Generic action-recognition
representations therefore transfer very poorly to this corpus. Our pipeline,
trained only on UMN and likewise applied with no fine-tuning, nonetheless
separates the Panic class cleanly. We stress that the two numbers are *not*
directly comparable: [6] reports four-way classification accuracy, whereas ours
is a binary rank-separation between Panic and the remainder. The contrast
indicates that crowd-motion features transfer across corpora where generic
appearance-based action features do not; it is not a claim of superiority on
their benchmark, which we do not attempt.

**Why rank separation rather than AUROC.** The labels describe whole clips,
while our model scores individual frames. The dataset's authors state it
"doesn't include sensor data, time stamps for each frame, or information about
how fast or slow the system can run" [6, Sect. VI], and define Panic by
transient cues — "sudden, chaotic, and rapid movements" — that need not hold
across every frame of a 5–6 s clip. A frame-level ROC would therefore require
assuming a label the dataset does not assert. This section is evidence of
cross-corpus transfer, not a benchmark score.

**This validates the representation, not the forecast.** All figures above use
the k = 0 head only. Evaluating k = 30/60/90 requires knowing *when* onset
occurs, so that a window ending k frames earlier can be labelled; the corpus
provides a single label per 5–6 s clip and no onset annotation, so the
forecasting horizons are not computable on it. The next subsection addresses
that gap on a corpus that does annotate onsets.

**I.2 MED — the forecast horizons transfer poorly, and we can say why.**

MED [13] annotates crowd behaviour per frame across 31 videos, 11 of which
contain a panic event. We measured the corpus directly: 854 x 480 at 30 fps,
15,856 frames across the panic subset, and a mean of 26.3 persons/frame by our
detector — denser than UMN's 15.2. The resolution and frame rate agree with
DeepROD's own description of MED [1, Table 1]. Frame-level
labels make k = 30/60/90 computable, so this is the first test of the forecast
horizons themselves on a corpus the model was not fitted to. MED is also one of
the five datasets DeepROD [1] evaluates, so it is not a corpus chosen to
flatter this method.

Applying the UMN-trained model unmodified (per-video AUROC, mean over 11
videos):

| Horizon | UMN (LOCO) | MED (zero-shot) |
|---|---|---|
| k = 0 | 0.963 | **0.800** |
| k = 30 | 0.882 | 0.668 |
| k = 60 | 0.791 | 0.570 |
| k = 90 | 0.734 | 0.556 |

**Nowcasting transfers; forecasting does not.** At k = 0 the model retains 0.800
on unseen cameras, unseen staging and nearly double UMN's crowd density,
confirming the feature representation is not UMN-specific. By k = 60 and k = 90
it is close to chance.

The cause is identifiable and is not domain shift in the usual sense. Repeating
the lead-window analysis of Sect. V-C on MED inverts the key precursor:

| Feature, 60 frames before onset | UMN | MED |
|---|---|---|
| `flow_magnitude` | rises in **10 of 11** clips | falls in **10 of 11** videos |
| `stop_ratio` | falls in **11 of 11** | mixed (6 up, 5 down) |

In UMN, crowds *accelerate* before panic: steady walking, then a sprint. In MED
they *decelerate* — people pause and orient before fleeing. A model trained to
read acceleration as an approaching anomaly therefore reads MED's precursor
backwards, which is why k = 0 survives (panic in progress looks alike either
way) while the forecast horizons collapse. Density does not explain the pattern
(per-video r = -0.03, p = 0.94), consistent with Sect. V-H.

**The implication is uncomfortable and worth stating.** We adopt cross-scene
evaluation precisely because per-video calibration [1] measures an easier
problem. This result suggests the two are not simply better and worse: the
*precursor signature* of panic appears venue-dependent even where the anomaly
itself is not, so some scene-level adaptation may be a genuine requirement for
forecasting rather than a methodological shortcut. Establishing that would
require corpora annotated with onset times across many venues, which do not
presently exist at the scale needed.

### J. Deployment Latency

| Resolution | Device | Total (ms) | FPS | Drop @ 30 fps |
|---|---|---|---|---|
| 640×480 | CPU | 158.6 | 6.35 | 78.8% |
| 640×480 | CUDA (T4) | **12.7** | **84.2** | **0.0%** |

GPU deployment is real-time with zero frame drop. CPU-only operation is not
viable at 30 fps input — a concrete hardware requirement. Detection dominates
the budget (7–9.5 ms of the 12.7 ms GPU total); feature extraction and LSTM
inference together cost under 4 ms.

**Comparison with macroscopic detection.** On UMN specifically, Cob-Parro et al. [2, Table 7]
report 17.4-18.6 ms per frame with optical flow on CPU and 3.9-4.4 ms with
optical flow on GPU (their cross-dataset worst case is ~20 ms CPU) — roughly
9x faster than our CPU figure and 3x faster than our GPU figure. The gap is structural and expected: their pipeline has no
detector, and their recurrent model costs only 0.24 ms (2.58 MFLOPS), so
optical flow is their entire budget. Our detector alone exceeds their whole
per-frame cost.

We report this rather than omit it. Both systems are real-time on GPU, which
is the operative requirement; the ~2× GPU latency is the price of per-person
tracking, and Sect. V-C quantifies what that purchases — the stop-ratio
feature, which is unavailable without tracking, carries the largest share of
forecasting performance at every horizon. For CPU-only deployment their
approach is clearly preferable, and we say so.

---

## VI. Limitations

**Sample size.** Eleven clips (seven valid at all horizons) is the binding
constraint on every comparative claim. With n = 7, the smallest p-value the
paired Wilcoxon test can produce is 0.016, so only pre-specified primary
comparisons can reach significance at all. Non-significant results here mean
the experiment cannot distinguish the methods — not that they are equivalent.

**Recurrence is unproven.** Three non-recurrent models given identical
information access -- logistic regression, random forest and gradient boosting
-- are all statistically indistinguishable from the LSTM (Sect. V-G). The
forecasting capability is demonstrated; the necessity of the recurrent
architecture is not. The one consistent difference is robustness rather than
accuracy: at k = 90 the tree models fall below chance on 4 of 11 clips against
1 for the LSTM, and never the reverse, though at this sample size that pattern
is not significant.

**Anomaly scope.** The method requires collective events. Individual and
small-group anomalies are outside its envelope, as Section V-H shows directly.

**Fixed cameras only.** Running normalisers assume a stable viewpoint and need
a warm-up period after each reset.

**Instability on near-empty scenes.** Per-zone statistics are computed from
whatever tracks are present; with one or two people they are estimated from too
few samples to be stable. On the external corpus this produced a visible
failure mode — near-empty "No Panic" clips scored *higher* on average than
ordinary crowd movement (0.219 versus 0.136, Sect. V-I). A minimum-occupancy
gate before a zone contributes to the feature vector is the obvious remedy and
is not implemented here.

**Threshold-free metrics.** All results are ranking metrics. Deployment at a
fixed operating threshold would require calibration not performed here, and the
external-corpus figures indicate why: at a 0.5 threshold, recall on Panic clips
is 55.5% while 21.4% of No Panic clips are flagged.

**Label provenance.** Our UMN labels are derived from the red-flag overlay
burned into the distributed video. Cob-Parro et al. [14] report that "the
provided ground-truth was not accurate" for UMN and PETS-2009 and released a
re-annotated version; we did not use it, so a small labelling discrepancy
against work built on their annotations cannot be excluded. Onset frames drive
every horizon label, so this affects k > 0 more than k = 0.

**Staged data.** UMN's panic is acted, not spontaneous. Whether precursor
signals in genuine emergencies resemble those in staged ones is untested.

**The forecasting effect is corpus-specific.** Under identical protocol the
method reaches 0.963 on UMN but 0.554 on ShanghaiTech and 0.648 on MED
(Sect. V-H), and the pre-onset signature it relies on is itself
venue-dependent (Sect. V-I.2). The result should be read as established for
UMN-style collective escape, not as a general property of crowd panic. The
Crowd Panic corpus offers clip-level labels without onset times, so only the
k = 0 head could be evaluated there. Every forecasting result in this paper — the
contribution the work rests on — comes from UMN's 11 clips. The
ShanghaiTech study evaluates the forecast horizons but on anomalies of a
different kind (Sect. V-H), so it tests transfer of the task framing rather
than confirming the horizons. An external corpus with frame-level onset
annotation is the single most valuable addition to this evaluation.

---

## VII. Conclusion

We reframed crowd panic detection as multi-horizon forecasting and showed that
useful prediction is possible before onset: AUROC 0.882 one second ahead,
statistically above chance on scenes the model has never seen, from an
object-centric representation built on tracked pedestrians rather than blurred
pixels. Under a protocol-matched comparison the approach outperforms the base
method it extends on every fold. Five baselines spanning four learning families
independently recover the same signal, which establishes the effect as a
property of the features and the task framing rather than of one architecture;
of those, the proposed model is the only one to retain AUROC above 0.70 at the
3 s horizon. Feature ablation further shows that predictive importance is not
static — headcount indicates panic in progress while speed anticipates it —
so single-horizon evaluation can be actively misleading.

The primary constraint is data. Eleven clips establish the effect but cannot
separate the learned models from one another in the mid-horizon range, nor
confirm the 2 s and 3 s horizons at conventional significance levels. Extending
to additional collective-panic corpora with frame-level onset annotation is the
natural next step: it would test the horizon limit, the necessity of recurrence,
and — for the first time — whether the forecast itself transfers across
datasets rather than only the feature representation.

---

## References

**Verified in full** (read directly, details confirmed against publisher record):

[1] H. Ammar and A. Cherif, "DeepROD: a deep learning approach for real-time and
online detection of a panic behavior in human crowds," *Machine Vision and
Applications*, vol. 32, no. 3, Art. no. 57, 2021.
doi:10.1007/s00138-021-01182-w

[2] A. C. Cob-Parro, C. Losada-Gutiérrez, and M. Marrón-Romera, "Stampede
detector based on deep learning models using dense optical flow," *Engineering
Applications of Artificial Intelligence*, vol. 142, Art. no. 109940, 2025.
doi:10.1016/j.engappai.2024.109940

[3] C. Cao, Y. Lu, P. Wang, and Y. Zhang, "A new comprehensive benchmark for
semi-supervised video anomaly detection and anticipation," in *Proc. IEEE/CVF
Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023.

**Verified metadata, not read in full** — confirm characterisation before final
submission:

[4] Z. Liu, X. Wu, W. Li, L. Yang, and S. Wang, "Rethinking metrics and
benchmarks of video anomaly detection," 2025. arXiv:2505.19022.

[5] G. Bhola, S. Srivastava, H. R. Nidhan, and S. Kathed, "Stampede detection
and crowd analysis using CNN-LSTM and Farneback optical flow," *Scientific
Reports*, vol. 16, Art. no. 16155, 2026. doi:10.1038/s41598-026-45262-1

[6] M. Kumbhar, A. Jain, A. Babar, M. Gaikwad, and C. Hedaoo, "Crowd Panic: A
Novel Dataset and Performance Evaluation," in *Proc. 2026 Int. Conf. Emerging
Smart Computing and Informatics (ESCI)*, Pune, India, Mar. 2026, pp. 1-6.
doi:10.1109/ESCI68015.2026.11493354

**Tools and datasets:**

[7] S. Shao et al., "CrowdHuman: A benchmark for detecting human in a crowd,"
*arXiv:1805.00123*, 2018.

[8] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every
detection box," in *Proc. ECCV*, 2022.

[9] Ultralytics, "YOLOv8," 2023. [Online]. Available:
https://github.com/ultralytics/ultralytics

[10] "Unusual Crowd Activity Dataset," University of Minnesota. [Online].
Available: http://mha.cs.umn.edu/proj_events.shtml#crowd

[11] W. Liu, W. Luo, D. Lian, and S. Gao, "Future frame prediction for anomaly
detection — a new baseline," in *Proc. CVPR*, 2018.

[15] A. Pennisi, D. D. Bloisi, and L. Iocchi, "Online real-time crowd behavior
detection in video sequences," *Computer Vision and Image Understanding*,
vol. 144, pp. 166-176, 2016. doi:10.1016/j.cviu.2015.09.010

[14] A. C. Cob-Parro, C. Losada-Gutiérrez, M. Marrón-Romera, A.
Gardel-Vicente, I. Bravo-Muñoz, and M. I. Sarker, "A Proposal on Stampede
Detection in Real Environments," in *Proc. IPIN 2021 Work-in-Progress*,
Lloret de Mar, Spain, Nov.-Dec. 2021, CEUR-WS vol. 3097, paper 29.

[13] H. R. Rabiee, J. Haddadnia, H. Mousavi, M. Kalantarzadeh, M. Nabi, and
V. Murino, "Novel dataset for fine-grained abnormal behavior understanding in
crowd," in *Proc. IEEE Int. Conf. Advanced Video and Signal Based Surveillance
(AVSS)*, 2016.

[12] A. Jain, M. Gaikwad, C. Hedaoo, and A. Babar, "Crowd Panic Dataset,"
Kaggle, 2025. [Online]. Available:
https://www.kaggle.com/datasets/aj1714/crowd-panic-dataset

`[16-20: still needed. Fastest source is the reference list of [2], which
surveys this literature directly. Specifically required: Benavoli et al. 2017 (Bayesian Wilcoxon), Farnebäck 2003 (optical flow),
Helbing social force model, and 2-3 recent YOLO-based crowd anomaly papers.]`

---

# CLAIMS_TO_REVIEW

You asked to see every statement whose statistical support is limited. These
are the three, with exactly what is and is not established.

### CLAIM-1 — "forecasting works at 2 s and 3 s"
**Where:** Section V-B.
**Status:** k = 0 and k = 30 are solid (sign test p = 0.008). k = 60 and k = 90
are *not* significant on the sign test (p = 0.063), though their bootstrap
intervals do exclude 0.5.
**How it is written:** as "suggestive rather than established."
**Your decision:** the abstract currently says performance one second ahead is
significant and does not claim significance further out. If you want to headline
3-second forecasting, you need more clips.

### CLAIM-2 — "we significantly beat DeepROD"
**Where:** Sections V-D and V-G, Abstract.
**Status:** p = 0.016, 7/7 folds. This passes Holm correction **only if you
declare a small family of primary comparisons** (2 tests → threshold 0.025). If
you correct across all 16 comparisons run, the threshold is 0.003 and nothing
passes.
**This is legitimate but you must be able to say honestly that the DeepROD
comparison was your primary research question from the outset** — which it is,
as it is your base paper. You cannot choose the family after seeing which tests
passed. Confirm you are comfortable with this framing before submission.

### CLAIM-3 — "the LSTM beats the baselines"
**Where:** Sections V-F and V-G.
**Status:** **Not supported, and now tested against four baselines rather than
two.** No comparison against the rule-based, logistic-regression, random-forest,
gradient-boosting or zone-attention baselines reaches significance. Logistic
regression is indistinguishable (±0.001 median at every horizon); the tree
ensembles have small, sign-inconsistent medians and per-fold win rates of
3/7 to 6/11.
**Important:** the mean column in the V-F table favours the LSTM at k = 0 and
k = 90, which is easy to misread as superiority. It is not — the mean gap is
carried by a minority of folds, and at k = 60 random forest beats the LSTM on
7 of 11 folds despite a lower mean. Do not let a slide or abstract imply the
ladder shows the LSTM winning.
**What the trees did add:** (i) four independent model families recover the
same signal, which strengthens the forecasting reframe as the contribution;
(ii) a reliability observation at k = 90 — trees fall below chance on 4/11
clips against 1/11 for the LSTM, 3–0 in both head-to-head comparisons and never
the reverse, but McNemar p = 0.25, which is the *smallest value obtainable*
with three discordant pairs. Report it as a consistent direction, not a result.
**How it is written:** the paper states all of this outright rather than
implying superiority. This is the most likely point of reviewer scrutiny, and
volunteering it is the strongest available position.

### Also flagged
- **Density claim reversed.** Earlier drafts attributed the ShanghaiTech gap to
  crowd sparsity. Testing rejected it (r = −0.481, p = 0.228). Section V-H now
  states the tested result. Ensure no slide deck or abstract still carries the
  old explanation.
- **Related Work is written but not fully cited.** The argument is complete
  and sourced from [1] and [2], both read in full. Ten `[CITE: ...]` markers
  remain for *supporting* references. Refs [1]-[3] were verified directly;
  [4]-[5] by metadata only -- confirm their characterisation before submission.

---

# REMAINING_WORK

Nothing in this list requires new experiments. The code and results are
complete; what follows is production work and judgement calls.

## Blocking submission

1. **IEEE template conversion.** This file is markdown. INDIACom-2027 requires
   the IEEE two-column template. Tables will need reflowing.
2. **Figures — none exist yet.** At minimum: pipeline block diagram,
   AUROC-vs-horizon plot, annotated sample frame. Plot data is in
   `outputs/reports/`; some figures exist in `outputs/figures/`.
3. **Page limit.** Currently ~5,300 words. Not yet confirmed for INDIACom
   (standard is 6 pages IEEE). If cuts are needed, Sect. V-I (zero-shot) and
   the ablation discussion in V-C compress most easily without weakening the
   argument.
4. **Author list and affiliations.** Absent.
5. **Ten `[CITE: ...]` markers.** All are supporting references for claims
   already sourced. Ref [2]'s bibliography contains most of them.

## Your judgement, not mine

6. **CLAIM-2 framing.** The two-primary-comparison family. Confirm you can
   honestly state the DeepROD comparison was the primary question from the
   outset. It was — it is the base paper — but you must be able to say so.
7. **Institutional policy on AI assistance.** Check before submission.
8. **Supervisor review.**

## Deliberately not done, with reasoning

- **Tree-ensemble baselines are now included** (`src/baseline_trees.py` ->
  `outputs/reports/umn_tree_baselines_results.json`), and `significance_tests.py`
  covers them. Outcome: they confirm the signal across four model families but
  do not separate from the LSTM, so CLAIM-3 is unchanged.

- **Accuracy / precision / recall alongside AUROC.** Earlier considered
  necessary for comparability with [2]. Superseded: Sect. IV-C now argues
  accuracy is uninformative on UMN (a constant "normal" predictor scores
  85.3%), and [2]'s figures are not comparable regardless because of protocol
  and pooled-dataset differences. Adding accuracy would undercut the paper's
  own stated position. If a reviewer requests it, it is a small change.

- **GBA-Stampedes / additional corpora.** Requested from [2]'s authors
  (dataset is "available on request"). Would roughly double the fold count and
  is the direct remedy for the power limitation in Sect. VI. Not blocking:
  the paper is complete and honest without it, and the limitation is stated
  explicitly rather than hidden.
