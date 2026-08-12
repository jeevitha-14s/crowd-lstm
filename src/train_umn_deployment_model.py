"""Trains the ONE deployment checkpoint used by stream_processor.py,
benchmark_latency.py, and live_dashboard.py -- models/umn_forecast_lstm.pt.

This is NOT an evaluation artifact. Every AUROC/AUPRC number in the paper
comes from the 11-fold leave-one-clip-out evaluation in
train_forecast_model.py, on models that never see their own test clip. This
script trains on all 11 UMN clips at once, so its output has seen every clip
during training -- there is no valid held-out measurement to compute from
it, and none is computed here. See the generated
models/umn_forecast_lstm.README.md, written next to the checkpoint so this
can't be forgotten later.

Training on all 11 clips means there is no held-out fold to early-stop
against. Guessing an epoch count would be arbitrary, so instead this script
re-runs the exact 11-fold LOCO training from train_forecast_model.py (same
code, same seeding, imported directly rather than duplicated) purely to
harvest each fold's best_epoch, and trains the final model for a fixed
number of epochs equal to the mean of those 11 values, rounded to the
nearest integer. That number is derived fresh every time this script runs,
never hardcoded, and printed for the audit trail.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.train_forecast_model import (
    BATCH_SIZE,
    DATASET_PATH,
    HORIZONS,
    LR,
    SEED,
    WEIGHT_DECAY,
    CrowdRiskLSTM,
    compute_pos_weight,
    train_one_fold,
)

MODEL_OUT_PATH = Path("models/umn_forecast_lstm.pt")
README_PATH = Path("models/umn_forecast_lstm.README.md")


def derive_fixed_epoch_count(
    X: np.ndarray, clip_ids: np.ndarray, y: Dict[int, np.ndarray], device: torch.device
) -> Tuple[int, List[int], float, List[int]]:
    unique_clips = sorted(np.unique(clip_ids).tolist())
    best_epochs: List[int] = []
    print(f"{'fold(clip)':>10}{'best_epoch':>12}")
    for fold_clip in unique_clips:
        train_mask = clip_ids != fold_clip
        test_mask = clip_ids == fold_clip
        X_train, X_test = X[train_mask], X[test_mask]
        y_train = {k: y[k][train_mask] for k in HORIZONS}
        train_clip_ids = clip_ids[train_mask]

        _, _, best_epoch, _ = train_one_fold(X_train, y_train, train_clip_ids, X_test, fold_clip, device)
        best_epochs.append(best_epoch)
        print(f"{fold_clip:>10}{best_epoch:>12}")

    mean_best_epoch = float(np.mean(best_epochs))
    fixed_epochs = int(round(mean_best_epoch))
    print(f"\nmean best_epoch across {len(unique_clips)} folds = {mean_best_epoch:.2f}  ->  fixed_epochs = {fixed_epochs}")
    return fixed_epochs, best_epochs, mean_best_epoch, unique_clips


def train_final_model(X: np.ndarray, y: Dict[int, np.ndarray], fixed_epochs: int, device: torch.device) -> CrowdRiskLSTM:
    torch.manual_seed(SEED)
    model = CrowdRiskLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fns = {
        k: nn.BCEWithLogitsLoss(pos_weight=torch.tensor(compute_pos_weight(y[k]), device=device))
        for k in HORIZONS
    }

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = {k: torch.tensor(y[k], dtype=torch.float32, device=device) for k in HORIZONS}
    n = X_t.shape[0]

    for epoch in range(fixed_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            preds = model(X_t[idx])
            loss = sum(loss_fns[k](preds[str(k)], y_t[k][idx]) for k in HORIZONS)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def write_readme(
    fixed_epochs: int, best_epochs: List[int], mean_best_epoch: float, unique_clips: List[int]
) -> None:
    fold_rows = "\n".join(f"| {c} | {e} |" for c, e in zip(unique_clips, best_epochs))
    text = f"""# umn_forecast_lstm.pt -- deployment checkpoint, NOT an evaluation artifact

This model is trained on ALL 11 UMN clips, with no held-out fold. It exists
only for the latency benchmark (`src/benchmark_latency.py`) and the live
demo (`src/stream_processor.py`, `src/live_dashboard.py`).

**Every AUROC/AUPRC number reported in the paper comes from the 11-fold
leave-one-clip-out evaluation in `src/train_forecast_model.py`, never from
this checkpoint.** This model has seen every clip during training, so its
predictions on any UMN clip are not a valid held-out measurement of
anything. Do not compute or report any metric from this checkpoint.

## How the epoch count was chosen

Training on all 11 clips leaves no held-out validation signal for early
stopping. The epoch count below is the mean `best_epoch` across the 11 LOCO
folds from `train_forecast_model.py` (re-derived fresh by this script's
`derive_fixed_epoch_count()` every time it runs, never hardcoded), used as a
fixed, non-adaptive training budget:

| fold (held-out clip) | best_epoch |
|---|---|
{fold_rows}

mean best_epoch = {mean_best_epoch:.2f} -> fixed_epochs used = {fixed_epochs}

## Architecture

Shared LSTM backbone (hidden=64, 1 layer, dropout=0.2), 4 linear heads at
k=0/30/60/90 frames (0/1/2/3s at 30fps), input_size=64 (UMN's 4x4 spatial
grid FeatureExtractor output).
"""
    README_PATH.parent.mkdir(parents=True, exist_ok=True)
    README_PATH.write_text(text)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(DATASET_PATH)
    X = data["X"]
    clip_ids = data["clip_ids"]
    y = {k: data[f"y_{k}"] for k in HORIZONS}

    print(f"Device: {device}")
    print("=== Step 1: deriving fixed epoch count from 11-fold LOCO best_epochs ===")
    fixed_epochs, best_epochs, mean_best_epoch, unique_clips = derive_fixed_epoch_count(
        X, clip_ids, y, device
    )

    print(f"\n=== Step 2: training deployment model on all {len(unique_clips)} clips, {fixed_epochs} fixed epochs ===")
    model = train_final_model(X, y, fixed_epochs, device)

    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT_PATH)
    write_readme(fixed_epochs, best_epochs, mean_best_epoch, unique_clips)
    print(f"\nSaved {MODEL_OUT_PATH}")
    print(f"Saved {README_PATH}")


if __name__ == "__main__":
    main()
