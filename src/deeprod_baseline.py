"""Diagnostic reimplementation of DeepROD (Ammar & Cherif, 2021, Machine
Vision and Applications) to test a specific claim before it goes in the
paper: that DeepROD's near-perfect self-reported UMN numbers (AUC up to
0.999) come from an evaluation protocol that is fundamentally easier than
leave-one-clip-out, not from a better model.

DeepROD's actual protocol, reproduced here: for each video, train an
LSTM regressor ONLINE on that same video's own first 300 frames (assumed
normal), predicting the next Gradient-of-Motion (GoM) scalar from the
previous d=5 values (MSE loss). Fit a Box-Cox-transformed normal reference
distribution N(mu_ref, v_ref) from that training window's prediction
errors. Then score the REST of that SAME video by Mahalanobis distance of
each new prediction error from the reference, flagging anomalies past the
95%-confidence chi-squared(1) threshold (d > 1.9600). This is a next-frame
prediction-error detector, not a classifier, and the model is calibrated on
each video it is later tested against -- the video is never truly held out.

GoM feature (paper Eqs. 1-3): dense optical flow per consecutive frame pair
(Farneback here -- the paper doesn't pin down which dense method; Farneback
is the standard practical choice for a genuinely per-pixel field, unlike the
sparse point-tracked flow this project's own pipeline uses elsewhere), then
sum over all pixels of (Ux^2 + Uy^2 + Vx^2 + Vy^2) -- a single scalar per
frame pair, no spatial zones, no tracking.

Simplification from the paper: one fixed architecture (2-layer LSTM,
hidden=32, d=5) across all clips, rather than their per-scene grid search
(Table 2) -- this diagnostic is testing the *protocol's* effect on the
reported numbers, not chasing their exact figures, so re-grid-searching
would not change the question being answered.

--loco mode: the actual base-paper comparison row (not the diagnostic
above). The paper never covers cross-video generalization, so one
deliberate choice was made here and is worth stating plainly: for each
held-out clip, the regressor and the Box-Cox reference distribution are
fit ONLY from windows drawn from the OTHER 10 clips' NORMAL-labeled
frame-pairs (pooled, never crossing a clip boundary), then scored against
the held-out clip's ENTIRE sequence (normal and abnormal both) -- exactly
mirroring how this project's own baseline LSTM is evaluated under
leave-one-clip-out, so the two rows in the final table are measured the
same way.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

VIDEO_PATH = "data/raw/umn/Crowd-Activity-All.avi"
GROUNDTRUTH_PATH = "data/umn_groundtruth.csv"
META_PATH = "data/features/umn_meta.csv"
OUT_JSON_PATH = Path("outputs/reports/deeprod_protocol_diagnostic.json")

TRAIN_FRAMES = 300  # DeepROD's UMN protocol: first 300 frames of EACH video, online
D = 5  # history length fed to the LSTM regressor, per DeepROD's UMN configuration
HIDDEN_SIZE = 32
NUM_LAYERS = 2
EPOCHS = 30
LR = 1e-3
CHI2_95_DF1 = 3.8415  # threshold on d^2; paper's Eq. 7, 95% confidence, 1 degree of freedom
SEED = 42


class GoMPredictor(nn.Module):
    """Single-feature-in, single-feature-out LSTM regressor: predicts the
    next GoM scalar from the previous D values. Matches DeepROD's Fig. 2/3
    role (not their exact per-scene depth/width -- see module docstring)."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        return self.head(hn[-1]).squeeze(-1)


def compute_gom_sequence(frames: List[np.ndarray]) -> np.ndarray:
    """One GoM scalar per consecutive frame pair -> length len(frames)-1."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    values = np.empty(len(frames) - 1, dtype=np.float64)
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        u, v = flow[..., 0], flow[..., 1]
        ux, uy = np.gradient(u, axis=1), np.gradient(u, axis=0)
        vx, vy = np.gradient(v, axis=1), np.gradient(v, axis=0)
        s = ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2
        values[i] = float(s.sum())
    return values


def make_windows(seq: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """(n, d, 1) inputs and (n,) targets: seq[i:i+d] -> seq[i+d]."""
    n = len(seq) - d
    X = np.empty((n, d, 1), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    for i in range(n):
        X[i, :, 0] = seq[i : i + d]
        y[i] = seq[i + d]
    return X, y


def process_clip(clip_id: int, frames: List[np.ndarray], labels: np.ndarray) -> Dict:
    """labels: abnormal flag per source FRAME (not per GoM value), same
    length as frames. GoM value i (between frame i and i+1) is associated
    with frame i+1's label -- by the time that error is computable, frame
    i+1 has arrived, matching DeepROD's online step (Sect. 3.4, step 7)."""
    gom = compute_gom_sequence(frames)
    gom_labels = labels[1:]  # align: gom[i] <-> frame i+1

    n_train_gom = TRAIN_FRAMES - 1  # first TRAIN_FRAMES frames -> TRAIN_FRAMES-1 GoM values
    if len(gom) <= n_train_gom + D + 5:
        return {"clip_id": clip_id, "excluded": True, "exclude_reason": "clip_too_short"}

    train_gom = gom[:n_train_gom]
    X_train, y_train = make_windows(train_gom, D)

    torch.manual_seed(SEED)
    model = GoMPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).numpy()
    train_errors = np.abs(y_train - train_pred)

    # Box-Cox requires strictly positive values; errors can be ~0, so offset slightly
    train_errors_pos = train_errors + 1e-6
    transformed, lam_ref = sp_stats.boxcox(train_errors_pos)
    mu_ref, v_ref = float(transformed.mean()), float(transformed.var())
    if v_ref <= 0:
        v_ref = 1e-8

    # Detection phase: everything after the training window, walked forward
    # one real step at a time (real history, not the model's own predictions).
    test_start = n_train_gom
    X_test, y_test = make_windows(gom[test_start - D :], D)
    test_labels = gom_labels[test_start:]
    assert len(X_test) == len(test_labels), (len(X_test), len(test_labels))

    with torch.no_grad():
        test_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    test_errors = np.abs(y_test - test_pred) + 1e-6
    test_transformed = sp_stats.boxcox(test_errors, lmbda=lam_ref)
    distances = np.abs(test_transformed - mu_ref) / np.sqrt(v_ref)

    threshold = np.sqrt(CHI2_95_DF1)
    predicted_anomaly = (distances > threshold).astype(int)
    true_anomaly = test_labels.astype(int)

    tp = int(((predicted_anomaly == 1) & (true_anomaly == 1)).sum())
    fp = int(((predicted_anomaly == 1) & (true_anomaly == 0)).sum())
    tn = int(((predicted_anomaly == 0) & (true_anomaly == 0)).sum())
    fn = int(((predicted_anomaly == 0) & (true_anomaly == 1)).sum())
    k = len(true_anomaly)
    accuracy = (tp + tn) / k
    pf = (fp + fn) / k
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    auc = roc_auc_score(true_anomaly, distances) if len(np.unique(true_anomaly)) > 1 else float("nan")

    return {
        "clip_id": clip_id, "excluded": False,
        "n_train_gom": n_train_gom, "n_test": k,
        "n_test_positive": int(true_anomaly.sum()),
        "accuracy": accuracy, "pf": pf, "precision": precision, "recall": recall, "auc": auc,
    }


def fit_regressor(X_train: np.ndarray, y_train: np.ndarray) -> GoMPredictor:
    torch.manual_seed(SEED)
    model = GoMPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def score_windows(model: GoMPredictor, X: np.ndarray, y: np.ndarray, lam_ref: float, mu_ref: float, v_ref: float):
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
    errors = np.abs(y - pred) + 1e-6
    transformed = sp_stats.boxcox(errors, lmbda=lam_ref)
    return np.abs(transformed - mu_ref) / np.sqrt(v_ref)


def normal_windows_for_clip(gom: np.ndarray, gom_labels: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding d-windows that never cross into an abnormal-labeled frame
    pair, so pooled training data across clips stays "normal-only" per
    DeepROD's own assumption -- just generalized across clips instead of
    confined to one clip's first 300 frames."""
    X_all, y_all = make_windows(gom, d)
    span_labels = np.array([gom_labels[i : i + d + 1].max() for i in range(len(gom) - d)])
    keep = span_labels == 0
    return X_all[keep], y_all[keep]


def run_loco(clip_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """clip_data: {clip_id: (gom_sequence, gom_labels)}."""
    clip_ids = sorted(clip_data.keys())
    records = []
    for held_out in clip_ids:
        X_train_parts, y_train_parts = [], []
        for cid in clip_ids:
            if cid == held_out:
                continue
            gom, labels = clip_data[cid]
            Xc, yc = normal_windows_for_clip(gom, labels, D)
            X_train_parts.append(Xc)
            y_train_parts.append(yc)
        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)

        model = fit_regressor(X_train, y_train)
        with torch.no_grad():
            train_pred = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
        train_errors = np.abs(y_train - train_pred) + 1e-6
        transformed, lam_ref = sp_stats.boxcox(train_errors)
        mu_ref, v_ref = float(transformed.mean()), float(transformed.var())
        if v_ref <= 0:
            v_ref = 1e-8

        gom_test, labels_test = clip_data[held_out]
        X_test, y_test = make_windows(gom_test, D)
        test_labels = labels_test[D:]  # gom index i+D aligned to gom_labels[i+D] == source-frame label
        distances = score_windows(model, X_test, y_test, lam_ref, mu_ref, v_ref)

        threshold = np.sqrt(CHI2_95_DF1)
        predicted = (distances > threshold).astype(int)
        true = test_labels.astype(int)
        tp = int(((predicted == 1) & (true == 1)).sum())
        fp = int(((predicted == 1) & (true == 0)).sum())
        tn = int(((predicted == 0) & (true == 0)).sum())
        fn = int(((predicted == 0) & (true == 1)).sum())
        k = len(true)
        rec = {
            "held_out_clip": held_out, "n_train_windows": len(X_train), "n_test": k,
            "n_test_positive": int(true.sum()),
            "accuracy": (tp + tn) / k, "pf": (fp + fn) / k,
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
            "auc": roc_auc_score(true, distances) if len(np.unique(true)) > 1 else float("nan"),
        }
        records.append(rec)
        print(
            f"fold held_out={held_out}: n_train_windows={rec['n_train_windows']} "
            f"acc={rec['accuracy']:.4f} Pf={rec['pf']:.4f} auc={rec['auc']:.4f}"
        )

    avg = {m: float(np.nanmean([r[m] for r in records])) for m in ["accuracy", "pf", "precision", "recall", "auc"]}
    return {"folds": records, "average": avg}


def decode_all_frames() -> List[np.ndarray]:
    cap = cv2.VideoCapture(VIDEO_PATH)
    all_frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()
    print(f"Decoded {len(all_frames)} frames from {VIDEO_PATH}")
    return all_frames


def load_clip_frames_and_labels() -> Dict[int, Tuple[List[np.ndarray], np.ndarray]]:
    gt = pd.read_csv(GROUNDTRUTH_PATH)
    meta = pd.read_csv(META_PATH)
    df = meta.merge(gt, on="frame")
    all_frames = decode_all_frames()
    out = {}
    for clip_id, g in df.groupby("clip_id"):
        frame_idx = g["frame"].to_numpy()
        assert np.array_equal(frame_idx, np.arange(frame_idx[0], frame_idx[-1] + 1)), "non-contiguous clip"
        out[int(clip_id)] = (all_frames[frame_idx[0] : frame_idx[-1] + 1], g["abnormal"].to_numpy())
    return out


def main_loco() -> None:
    clips = load_clip_frames_and_labels()
    clip_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for clip_id, (frames, labels) in clips.items():
        print(f"clip {clip_id}: {len(frames)} frames, computing GoM sequence...")
        gom = compute_gom_sequence(frames)
        clip_data[clip_id] = (gom, labels[1:])  # gom[i] <-> frame i+1's label

    result = run_loco(clip_data)
    print("\n=== DeepROD-method, leave-one-clip-out (fair comparison to baseline LSTM) ===")
    for k, v in result["average"].items():
        print(f"  {k}: {v:.4f}")
    print(
        "\nThis project's own baseline LSTM, same protocol, same dataset (common-fold AUROC): "
        "k=0 0.963, k=30 0.882, k=60 0.791, k=90 0.734. DeepROD's method has no horizon concept "
        "(~1/30s lookahead only), so it produces one row, not four."
    )

    out_path = Path("outputs/reports/deeprod_loco_comparison.json")
    payload = {
        "description": (
            "DeepROD's method (GoM feature + single-step LSTM regressor + Box-Cox/Mahalanobis "
            "anomaly threshold) reimplemented and evaluated under this project's own leave-one-clip-out "
            "protocol -- regressor and reference distribution fit ONLY on the 10 training clips' "
            "normal-labeled windows, scored against the ENTIRE held-out clip. This is the row directly "
            "comparable to the baseline LSTM's LOCO numbers; the per-clip self-calibrated diagnostic in "
            "outputs/reports/deeprod_protocol_diagnostic.json is a different, easier protocol (matching "
            "the paper's own setup) and is not this row."
        ),
        "model_config": {
            "d": D, "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "epochs": EPOCHS, "chi2_95_df1_threshold": CHI2_95_DF1,
        },
        **result,
        "baseline_lstm_loco_comparison": {
            "k0": 0.963, "k30": 0.882, "k60": 0.791, "k90": 0.734,
            "note": "common-fold AUROC, outputs/reports/umn_loco_results.json",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")


def main() -> None:
    import sys

    if "--loco" in sys.argv:
        main_loco()
        return

    gt = pd.read_csv(GROUNDTRUTH_PATH)
    meta = pd.read_csv(META_PATH)
    df = meta.merge(gt, on="frame")

    cap = cv2.VideoCapture(VIDEO_PATH)
    all_frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()
    print(f"Decoded {len(all_frames)} frames from {VIDEO_PATH}")

    records = []
    for clip_id, g in df.groupby("clip_id"):
        frame_idx = g["frame"].to_numpy()
        assert np.array_equal(frame_idx, np.arange(frame_idx[0], frame_idx[-1] + 1)), "non-contiguous clip"
        clip_frames = all_frames[frame_idx[0] : frame_idx[-1] + 1]
        clip_labels = g["abnormal"].to_numpy()
        print(f"clip {clip_id}: {len(clip_frames)} frames, computing GoM + online train/detect...")
        rec = process_clip(int(clip_id), clip_frames, clip_labels)
        records.append(rec)
        if not rec["excluded"]:
            print(
                f"  acc={rec['accuracy']:.4f} Pf={rec['pf']:.4f} "
                f"precision={rec['precision']:.4f} recall={rec['recall']:.4f} auc={rec['auc']:.4f}"
            )
        else:
            print(f"  EXCLUDED ({rec['exclude_reason']})")

    included = [r for r in records if not r["excluded"]]
    avg = {
        metric: float(np.nanmean([r[metric] for r in included]))
        for metric in ["accuracy", "pf", "precision", "recall", "auc"]
    }
    print("\n=== DeepROD-protocol (per-clip online self-calibration) average over UMN's 11 clips ===")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")
    print(
        "\nCompare against this project's own leave-one-clip-out baseline LSTM AUROC "
        "(outputs/reports/umn_loco_results.json, aggregate_common_subset): "
        "k=0 0.963, k=30 0.882, k=60 0.791, k=90 0.734 -- same dataset, held-out-clip protocol."
    )

    payload = {
        "description": (
            "Diagnostic reimplementation of DeepROD (Ammar & Cherif 2021) to test whether its "
            "near-perfect self-reported UMN numbers come from an easier evaluation protocol "
            "(per-video online self-calibration: train on that video's own first 300 frames, "
            "test on the rest of that SAME video) rather than a better model. See module docstring."
        ),
        "model_config": {
            "train_frames": TRAIN_FRAMES, "d": D, "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS, "epochs": EPOCHS, "chi2_95_df1_threshold": CHI2_95_DF1,
        },
        "per_clip": records,
        "average_over_included_clips": avg,
        "comparison_note": (
            "This project's own LOCO-evaluated baseline LSTM on the identical UMN dataset scores "
            "AUROC 0.963/0.882/0.791/0.734 at k=0/30/60/90 (common-fold subset) -- a held-out-clip "
            "protocol, never seeing any frame of the test clip during training."
        ),
    }
    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
