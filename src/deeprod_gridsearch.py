"""Faithful reimplementation of DeepROD (Ammar & Cherif, 2021) including the
two pieces `deeprod_baseline.py` deliberately skipped: their per-scene
hyperparameter grid search (their Sect. 4.2 / Table 2) and their 80/20
train/validation split (their Sect. 3.2).

Why this file exists separately from `deeprod_baseline.py`: that module was
written to answer "does the *protocol* explain their numbers?" and says so --
it fixes one architecture across all clips and trains without a validation
split, which is fine for a protocol diagnostic but makes any
"their result does not reproduce" claim attackable (we would have handed
their method worse hyperparameters than they used, then reported the
shortfall as theirs). This module removes that objection by giving DeepROD
its full published search space and reporting two numbers per clip:

  - `selected_*`  -- config chosen the way THEY choose it, by minimum
    validation MSE (their Sect. 4.2: the search "yields the minimum mean
    squared error"). This is the faithful reproduction number.
  - `oracle_*`    -- the best AUC achievable over the entire grid, selected
    using the test labels. This is NOT a legitimate result and must never be
    reported as one; it is an upper bound. If even the oracle falls short of
    their reported figure, no hyperparameter objection can explain the gap.

Search space (their Sect. 4.2): layers 1-4, hidden units per layer drawn
from {32, 50, 64, 100}, epochs 5-30 in steps of 5, d in {5, 15, 30}. Their
Table 2 caption ("[32,64] denotes ...") shows widths may differ per layer,
so widths are enumerated freely for 1- and 2-layer nets and held uniform for
3- and 4-layer nets to keep the grid tractable. That covers every
architecture they actually report as a winner in Table 2 ([32], [32,64],
[50,50], [32,32,32], [100,100,100,100]) -- the restriction only prunes
non-uniform deep nets, none of which they selected.

Deviations that remain, all forced by the paper leaving them unstated:
  - Dense optical flow method: Farneback (they say only "an OF technique").
  - Mini-batch size: 32 (they say weights update "for every batch", no size).
  - Validation split is the last 20% of the training window in temporal
    order, matching Keras's `validation_split` semantics -- they used
    TensorFlow 2.2 (their Sect. 4.5), where that argument takes the final
    fraction rather than a random subset.
  - Per-SCENE search is impossible here: no clip-to-scene (Lawn/Indoor/Plaza)
    mapping survives in any saved artifact, so the search is run per CLIP.
    That is strictly more generous than theirs -- 11 independent searches
    instead of 3 shared ones.

Usage:
    python3 -m src.deeprod_gridsearch            # their per-video protocol
    python3 -m src.deeprod_gridsearch --loco     # leave-one-clip-out
"""

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

from src.deeprod_baseline import (
    CHI2_95_DF1,
    GROUNDTRUTH_PATH,
    META_PATH,
    compute_gom_sequence,
    decode_all_frames,
    make_windows,
    normal_windows_for_clip,
)

GOM_CACHE_PATH = Path("data/features/umn_gom.npz")
OUT_PERVIDEO_PATH = Path("outputs/reports/deeprod_gridsearch_pervideo.json")
OUT_LOCO_PATH = Path("outputs/reports/deeprod_gridsearch_loco.json")

TRAIN_FRAMES = 300  # their UMN protocol: first 300 frames of each video
HIDDEN_CHOICES = (32, 50, 64, 100)  # their Sect. 4.2
D_CHOICES = (5, 15, 30)  # their Sect. 4.2
EPOCH_CHECKPOINTS = (5, 10, 15, 20, 25, 30)  # their Sect. 4.2, 5..30 step 5
MAX_EPOCHS = max(EPOCH_CHECKPOINTS)
BATCH_SIZE = 32
LR = 1e-3
VAL_FRAC = 0.2  # their Sect. 3.2: 80% train / 20% validation
SEED = 42


def architecture_grid() -> List[Tuple[int, ...]]:
    """Layer-width tuples: free widths at depth 1-2, uniform at depth 3-4."""
    archs: List[Tuple[int, ...]] = []
    for depth in (1, 2):
        archs.extend(itertools.product(HIDDEN_CHOICES, repeat=depth))
    for depth in (3, 4):
        archs.extend((h,) * depth for h in HIDDEN_CHOICES)
    return archs


class StackedGoMPredictor(nn.Module):
    """DeepROD's Fig. 2 network: stacked LSTM layers with (possibly) differing
    widths, then a dense layer collapsing the final hidden state to the single
    predicted GoM scalar. Separate nn.LSTM modules per layer rather than one
    multi-layer nn.LSTM because the latter forces a uniform hidden size.
    """

    def __init__(self, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers = []
        in_size = 1
        for h in hidden_sizes:
            layers.append(nn.LSTM(in_size, h, batch_first=True))
            in_size = h
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(in_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        hn = None
        for lstm in self.layers:
            out, (hn, _) = lstm(out)
        return self.head(hn[-1]).squeeze(-1)


def temporal_val_split(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Last VAL_FRAC of the window sequence held out, Keras-style."""
    n_val = max(1, int(round(len(X) * VAL_FRAC)))
    n_train = len(X) - n_val
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def fit_standardizer(X_tr: np.ndarray) -> Tuple[float, float]:
    """Mean/std of the 80% training split's GoM values.

    The paper never mentions input scaling, but raw GoM is an unnormalized
    sum over every pixel of a 320x240 frame -- O(10^2) on UMN and larger
    elsewhere. Fed directly into an LSTM those magnitudes saturate the gate
    nonlinearities and the network converges to something *worse than
    predicting the training mean* (verified: val MSE 1.03e4 vs 1.52e3 for a
    constant on clip 0). Any implementation that reproduces their reported
    accuracy must be scaling somewhere, so it is standardized here. Fitting
    on the 80% split only (not the full training window) keeps the
    validation-MSE model selection free of leakage.
    """
    mu, sd = float(X_tr.mean()), float(X_tr.std())
    return mu, (sd if sd > 0 else 1.0)


def train_with_checkpoints(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: Sequence[int],
) -> Dict[int, Tuple[Dict[str, torch.Tensor], float]]:
    """Train once to MAX_EPOCHS, snapshotting weights + val MSE at each
    checkpoint in EPOCH_CHECKPOINTS. Treating the epoch count as a grid
    dimension (as they do) rather than as early stopping means one training
    run serves every epoch value, instead of len(EPOCH_CHECKPOINTS) reruns.
    """
    torch.manual_seed(SEED)
    model = StackedGoMPredictor(hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    snapshots: Dict[int, Tuple[Dict[str, torch.Tensor], float]] = {}
    n = len(X_tr_t)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
        if epoch in EPOCH_CHECKPOINTS:
            model.eval()
            with torch.no_grad():
                val_mse = float(loss_fn(model(X_val_t), y_val_t))
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            snapshots[epoch] = (state, val_mse)
    return snapshots


def fit_reference(errors: np.ndarray) -> Tuple[float, float, float]:
    """Box-Cox the normal-situation prediction errors, return (lambda, mu, v).
    Their Sect. 3.3, Eqs. 5-6."""
    transformed, lam = sp_stats.boxcox(errors + 1e-6)
    mu, v = float(transformed.mean()), float(transformed.var())
    return lam, mu, (v if v > 0 else 1e-8)


def mahalanobis_scores(y_true_vals: np.ndarray, preds: np.ndarray, lam: float, mu: float, v: float) -> np.ndarray:
    """Their Eq. 7: |boxcox(|error|) - mu_ref| / sqrt(v_ref)."""
    errors = np.abs(y_true_vals - preds) + 1e-6
    transformed = sp_stats.boxcox(errors, lmbda=lam)
    return np.abs(transformed - mu) / np.sqrt(v)


def classification_metrics(distances: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Their Eqs. 8-9. Note Pf is (FP+FN)/K exactly as they define it, i.e.
    the total error rate, not a conventional false-alarm rate FP/(FP+TN);
    both are reported so the comparison to their tables stays like-for-like
    while the standard quantity is also available."""
    predicted = (distances > np.sqrt(CHI2_95_DF1)).astype(int)
    true = labels.astype(int)
    tp = int(((predicted == 1) & (true == 1)).sum())
    fp = int(((predicted == 1) & (true == 0)).sum())
    tn = int(((predicted == 0) & (true == 0)).sum())
    fn = int(((predicted == 0) & (true == 1)).sum())
    k = len(true)
    return {
        "accuracy": (tp + tn) / k,
        "pf_as_defined_in_paper": (fp + fn) / k,
        "false_alarm_rate_standard": fp / (fp + tn) if (fp + tn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "auc": roc_auc_score(true, distances) if len(np.unique(true)) > 1 else float("nan"),
    }


def evaluate_config(
    train_gom: np.ndarray,
    test_gom_windows: Tuple[np.ndarray, np.ndarray],
    test_labels: np.ndarray,
    hidden_sizes: Sequence[int],
    d: int,
) -> Dict[int, Dict]:
    """Train one architecture on `train_gom`, score the test windows at every
    epoch checkpoint. Returns {epoch: {val_mse, metrics...}}."""
    X_all, y_all = make_windows(train_gom, d)
    X_tr, y_tr, X_val, y_val = temporal_val_split(X_all, y_all)
    mu_s, sd_s = fit_standardizer(X_tr)
    snapshots = train_with_checkpoints(
        (X_tr - mu_s) / sd_s, (y_tr - mu_s) / sd_s, (X_val - mu_s) / sd_s, (y_val - mu_s) / sd_s, hidden_sizes
    )

    X_test, y_test_raw = test_gom_windows
    # Everything downstream (errors, Box-Cox reference, Mahalanobis) stays in
    # standardized units; the detection rule is a comparison of test errors
    # against train errors, so both sides only need to share one scale.
    y_all = (y_all - mu_s) / sd_s
    y_test = (y_test_raw - mu_s) / sd_s
    X_all_t = torch.tensor((X_all - mu_s) / sd_s, dtype=torch.float32)
    X_test_t = torch.tensor((X_test - mu_s) / sd_s, dtype=torch.float32)

    # MSE of the trivial "predict the training mean" forecaster on the same
    # validation split. Recorded alongside every config so a non-learning
    # model (val_mse >= this) is visible in the saved artifact rather than
    # silently producing plausible-looking AUCs.
    yv_n = (y_val - mu_s) / sd_s
    const_baseline = float(((yv_n - ((y_tr - mu_s) / sd_s).mean()) ** 2).mean())

    results: Dict[int, Dict] = {}
    model = StackedGoMPredictor(hidden_sizes)
    for epoch, (state, val_mse) in snapshots.items():
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            # Reference distribution over the WHOLE normal training window
            # (their Fig. 5 builds it from the normal situation, not from the
            # 80% subset), then score the held-out portion.
            train_pred = model(X_all_t).numpy()
            test_pred = model(X_test_t).numpy()
        lam, mu, v = fit_reference(np.abs(y_all - train_pred))
        distances = mahalanobis_scores(y_test, test_pred, lam, mu, v)
        results[epoch] = {
            "val_mse": val_mse,
            "val_mse_constant_baseline": const_baseline,
            "beats_constant": bool(val_mse < const_baseline),
            **classification_metrics(distances, test_labels),
        }
    return results


def search_one_split(
    train_gom: np.ndarray,
    test_gom: np.ndarray,
    test_labels_by_d: Dict[int, np.ndarray],
    label: str,
) -> Dict:
    """Run the full grid for one train/test split. `test_labels_by_d` maps d
    to the label vector aligned with make_windows(test_gom, d)."""
    archs = architecture_grid()
    records: List[Dict] = []
    # process_time, not wall clock: a laptop that sleeps mid-run inflates
    # wall-clock elapsed by hours of doing nothing, which makes any
    # "seconds per config" progress estimate wildly pessimistic.
    t0, c0 = time.time(), time.process_time()
    total = len(archs) * len(D_CHOICES)
    done = 0
    for d in D_CHOICES:
        test_windows = make_windows(test_gom, d)
        labels_d = test_labels_by_d[d]
        assert len(test_windows[0]) == len(labels_d), (len(test_windows[0]), len(labels_d))
        for hidden_sizes in archs:
            per_epoch = evaluate_config(train_gom, test_windows, labels_d, hidden_sizes, d)
            for epoch, res in per_epoch.items():
                records.append({"d": d, "hidden_sizes": list(hidden_sizes), "epochs": epoch, **res})
            done += 1
            if done % 20 == 0:
                cpu = time.process_time() - c0
                print(
                    f"  [{label}] {done}/{total} configs, {cpu:.0f}s cpu "
                    f"({time.time() - t0:.0f}s wall), ~{cpu / done * (total - done) / 60:.0f} min cpu left",
                    flush=True,
                )

    # Their selection rule: minimum validation MSE (Sect. 4.2).
    selected = min(records, key=lambda r: r["val_mse"])
    # Upper bound only -- selected using test labels, never a reportable result.
    scored = [r for r in records if not np.isnan(r["auc"])]
    oracle = max(scored, key=lambda r: r["auc"]) if scored else None
    return {
        "n_configs_evaluated": len(records),
        "search_seconds_cpu": time.process_time() - c0,
        "search_seconds_wall": time.time() - t0,
        "selected_by_val_mse": selected,
        "oracle_best_auc_UPPER_BOUND_NOT_A_RESULT": oracle,
    }


def build_gom_cache() -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """{clip_id: (gom_sequence, gom_labels)}, cached to disk -- dense Farneback
    flow over all 7,739 frames is the slow part and never changes."""
    if GOM_CACHE_PATH.exists():
        blob = np.load(GOM_CACHE_PATH)
        clip_ids = sorted({int(k.split("_")[1]) for k in blob.files})
        print(f"Loaded cached GoM for {len(clip_ids)} clips from {GOM_CACHE_PATH}")
        return {c: (blob[f"gom_{c}"], blob[f"lab_{c}"]) for c in clip_ids}

    gt = pd.read_csv(GROUNDTRUTH_PATH)
    meta = pd.read_csv(META_PATH)
    df = meta.merge(gt, on="frame")
    all_frames = decode_all_frames()

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for clip_id, g in df.groupby("clip_id"):
        idx = g["frame"].to_numpy()
        assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)), "non-contiguous clip"
        frames = all_frames[idx[0] : idx[-1] + 1]
        print(f"clip {clip_id}: computing GoM over {len(frames)} frames...", flush=True)
        gom = compute_gom_sequence(frames)
        out[int(clip_id)] = (gom, g["abnormal"].to_numpy()[1:])  # gom[i] <-> frame i+1

    GOM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GOM_CACHE_PATH,
        **{f"gom_{c}": v[0] for c, v in out.items()},
        **{f"lab_{c}": v[1] for c, v in out.items()},
    )
    print(f"Cached GoM to {GOM_CACHE_PATH}")
    return out


def run_pervideo(clip_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Their protocol: train on this video's own first 300 frames, test on the
    rest of the same video."""
    n_train_gom = TRAIN_FRAMES - 1
    folds = []
    for clip_id in sorted(clip_data):
        gom, gom_labels = clip_data[clip_id]
        if len(gom) <= n_train_gom + max(D_CHOICES) + 5:
            print(f"clip {clip_id}: EXCLUDED (too short)")
            folds.append({"clip_id": clip_id, "excluded": True})
            continue
        train_gom = gom[:n_train_gom]
        test_gom_full = gom[n_train_gom:]
        # For each d the test windows start d values in, so labels shift too.
        labels_by_d = {d: gom_labels[n_train_gom + d :] for d in D_CHOICES}
        assert all(
            len(make_windows(test_gom_full, d)[0]) == len(labels_by_d[d]) for d in D_CHOICES
        ), "window/label misalignment"
        print(f"clip {clip_id}: grid search ({len(test_gom_full)} test GoM values)...", flush=True)
        res = search_one_split(train_gom, test_gom_full, labels_by_d, f"clip {clip_id}")
        sel = res["selected_by_val_mse"]
        print(
            f"  selected {sel['hidden_sizes']} d={sel['d']} ep={sel['epochs']} "
            f"-> auc={sel['auc']:.4f} acc={sel['accuracy']:.4f} | "
            f"oracle auc={res['oracle_best_auc_UPPER_BOUND_NOT_A_RESULT']['auc']:.4f}"
        )
        folds.append({"clip_id": clip_id, "excluded": False, **res})
    return summarize(folds, "per-video online self-calibration (their Sect. 4.5 protocol)")


def run_loco(clip_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Leave-one-clip-out: fit on the other 10 clips' normal-labeled windows,
    score the entire held-out clip. Mirrors this project's own LOCO protocol
    so the two rows in the paper's table are measured the same way."""
    clip_ids = sorted(clip_data)
    folds = []
    for held_out in clip_ids:
        print(f"fold held_out={held_out}: grid search...", flush=True)
        gom_test, labels_test = clip_data[held_out]
        labels_by_d = {d: labels_test[d:] for d in D_CHOICES}
        res = search_one_split_pooled(clip_data, held_out, gom_test, labels_by_d, f"fold {held_out}")
        sel = res["selected_by_val_mse"]
        print(
            f"  selected {sel['hidden_sizes']} d={sel['d']} ep={sel['epochs']} "
            f"-> auc={sel['auc']:.4f} acc={sel['accuracy']:.4f} | "
            f"oracle auc={res['oracle_best_auc_UPPER_BOUND_NOT_A_RESULT']['auc']:.4f}"
        )
        folds.append({"clip_id": held_out, "excluded": False, **res})
    return summarize(folds, "leave-one-clip-out (fair comparison to this project's LSTM)")


def search_one_split_pooled(
    clip_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    held_out: int,
    test_gom: np.ndarray,
    test_labels_by_d: Dict[int, np.ndarray],
    label: str,
) -> Dict:
    """Same grid as search_one_split, but training windows are pooled across
    the 10 non-held-out clips' normal-labeled spans (never crossing a clip
    boundary), so windows must be built per-d before training rather than
    from a single contiguous sequence."""
    archs = architecture_grid()
    records: List[Dict] = []
    # process_time, not wall clock: a laptop that sleeps mid-run inflates
    # wall-clock elapsed by hours of doing nothing, which makes any
    # "seconds per config" progress estimate wildly pessimistic.
    t0, c0 = time.time(), time.process_time()
    total = len(archs) * len(D_CHOICES)
    done = 0
    for d in D_CHOICES:
        X_parts, y_parts = [], []
        for cid, (gom_c, lab_c) in clip_data.items():
            if cid == held_out:
                continue
            Xc, yc = normal_windows_for_clip(gom_c, lab_c, d)
            X_parts.append(Xc)
            y_parts.append(yc)
        X_all = np.concatenate(X_parts, axis=0)
        y_all = np.concatenate(y_parts, axis=0)
        X_tr, y_tr, X_val, y_val = temporal_val_split(X_all, y_all)
        mu_s, sd_s = fit_standardizer(X_tr)
        X_tr, y_tr = (X_tr - mu_s) / sd_s, (y_tr - mu_s) / sd_s
        X_val, y_val = (X_val - mu_s) / sd_s, (y_val - mu_s) / sd_s

        X_test, y_test_raw = make_windows(test_gom, d)
        labels_d = test_labels_by_d[d]
        assert len(X_test) == len(labels_d), (len(X_test), len(labels_d))
        y_all = (y_all - mu_s) / sd_s
        y_test = (y_test_raw - mu_s) / sd_s
        X_all_t = torch.tensor((X_all - mu_s) / sd_s, dtype=torch.float32)
        X_test_t = torch.tensor((X_test - mu_s) / sd_s, dtype=torch.float32)

        const_baseline = float(((y_val - y_tr.mean()) ** 2).mean())

        for hidden_sizes in archs:
            snapshots = train_with_checkpoints(X_tr, y_tr, X_val, y_val, hidden_sizes)
            model = StackedGoMPredictor(hidden_sizes)
            for epoch, (state, val_mse) in snapshots.items():
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    train_pred = model(X_all_t).numpy()
                    test_pred = model(X_test_t).numpy()
                lam, mu, v = fit_reference(np.abs(y_all - train_pred))
                distances = mahalanobis_scores(y_test, test_pred, lam, mu, v)
                records.append(
                    {
                        "d": d,
                        "hidden_sizes": list(hidden_sizes),
                        "epochs": epoch,
                        "val_mse": val_mse,
                        "val_mse_constant_baseline": const_baseline,
                        "beats_constant": bool(val_mse < const_baseline),
                        **classification_metrics(distances, labels_d),
                    }
                )
            done += 1
            if done % 10 == 0:
                cpu = time.process_time() - c0
                print(
                    f"  [{label}] {done}/{total} configs, {cpu:.0f}s cpu "
                    f"({time.time() - t0:.0f}s wall), ~{cpu / done * (total - done) / 60:.0f} min cpu left",
                    flush=True,
                )

    selected = min(records, key=lambda r: r["val_mse"])
    scored = [r for r in records if not np.isnan(r["auc"])]
    oracle = max(scored, key=lambda r: r["auc"]) if scored else None
    return {
        "n_configs_evaluated": len(records),
        "search_seconds_cpu": time.process_time() - c0,
        "search_seconds_wall": time.time() - t0,
        "selected_by_val_mse": selected,
        "oracle_best_auc_UPPER_BOUND_NOT_A_RESULT": oracle,
    }


def summarize(folds: List[Dict], protocol: str) -> Dict:
    included = [f for f in folds if not f.get("excluded")]
    metrics = ["accuracy", "pf_as_defined_in_paper", "false_alarm_rate_standard", "precision", "recall", "auc"]
    avg_sel = {
        m: float(np.nanmean([f["selected_by_val_mse"][m] for f in included])) for m in metrics
    }
    avg_oracle = {
        m: float(np.nanmean([f["oracle_best_auc_UPPER_BOUND_NOT_A_RESULT"][m] for f in included]))
        for m in metrics
    }
    return {
        "protocol": protocol,
        "folds": folds,
        "average_selected_by_val_mse": avg_sel,
        "average_oracle_UPPER_BOUND_NOT_A_RESULT": avg_oracle,
    }


def main() -> None:
    loco = "--loco" in sys.argv
    clip_data = build_gom_cache()
    result = run_loco(clip_data) if loco else run_pervideo(clip_data)

    out_path = OUT_LOCO_PATH if loco else OUT_PERVIDEO_PATH
    payload = {
        "description": (
            "DeepROD (Ammar & Cherif 2021) reimplemented WITH their published "
            "hyperparameter grid search (Sect. 4.2) and 80/20 train/validation "
            "split (Sect. 3.2), which src/deeprod_baseline.py deliberately omits. "
            "'selected_by_val_mse' is the faithful reproduction (their own "
            "selection rule); 'oracle' is a test-label-selected upper bound that "
            "is NOT a reportable result and exists only to rule out the objection "
            "that a shortfall is a hyperparameter artifact."
        ),
        "search_space": {
            "architectures": [list(a) for a in architecture_grid()],
            "d": list(D_CHOICES),
            "epochs": list(EPOCH_CHECKPOINTS),
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "val_frac": VAL_FRAC,
            "seed": SEED,
        },
        "paper_reported_umn": {"accuracy": 0.994, "pf": 0.005, "auc": 0.992},
        **result,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=== DeepROD w/ full grid search -- {result['protocol']} ===")
    print("  selected by val MSE (the faithful number):")
    for k, v in result["average_selected_by_val_mse"].items():
        print(f"    {k}: {v:.4f}")
    print("  oracle upper bound (NOT a result):")
    for k, v in result["average_oracle_UPPER_BOUND_NOT_A_RESULT"].items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  paper reports on UMN: accuracy 0.994, Pf 0.005, AUC 0.992")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
