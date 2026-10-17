"""Zero-shot cross-dataset test of the FORECAST horizons: the UMN-trained model,
unmodified, evaluated on MED.

This is the experiment the paper could not previously run. Sect. V-I's external
check (Crowd Panic Dataset) used the k=0 head only, because that corpus labels
whole clips and supplies no onset times -- without knowing when an anomaly
begins, a window ending k frames before it cannot be labelled. MED annotates
behaviour per frame, so k=30/60/90 are computable and the central claim of the
paper (panic is predictable seconds ahead) can be tested on a corpus it was not
fitted to.

On the checkpoint. models/umn_forecast_lstm.pt carries an explicit warning not
to report metrics from it, because it was trained on all 11 UMN clips with no
held-out fold; any UMN number it produces is contaminated. That warning does
not apply here and the distinction is the point: this model has seen no MED
frame, so MED is genuinely held out for it. It is in fact the correct
checkpoint for a transfer test, since it uses all available UMN training data
rather than the 10/11 a LOCO fold would. No UMN metric is computed below.

Two protocols are reported, because they answer different questions:

  pooled   -- all MED windows at once, one AUROC per horizon. Answers "does the
              model rank panic above normal across this corpus".
  per-video -- AUROC computed within each video, then averaged. Answers "does
              it rank correctly *within* a scene", which is the operationally
              relevant question for a fixed camera and is also the form
              directly comparable to the paper's leave-one-clip-out numbers.

The pooled figure is typically the more flattering of the two when videos
differ in base rate, so both are given.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.train_forecast_model import HORIZONS, CrowdRiskLSTM

DATASET_PATH = Path("data/med_forecast_dataset.npz")
CHECKPOINT = Path("models/umn_forecast_lstm.pt")
OUT_PATH = Path("outputs/reports/med_zeroshot_transfer.json")

UMN_LOCO_REFERENCE = {"0": 0.963, "30": 0.882, "60": 0.791, "90": 0.734}


def evaluate(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "n_pos": int(y.sum()), "excluded": True,
                "auroc": float("nan"), "auprc": float("nan")}
    return {"n": int(len(y)), "n_pos": int(y.sum()), "excluded": False,
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p))}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(DATASET_PATH)
    X = data["X"]
    video_ids = data["clip_ids"]
    y = {k: data[f"y_{k}"] for k in HORIZONS}

    model = CrowdRiskLSTM(input_size=X.shape[2]).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # Batched to keep peak memory bounded; the model is tiny but X is ~11k x 30 x 64.
    probs: Dict[int, np.ndarray] = {k: np.empty(len(X), dtype=np.float64) for k in HORIZONS}
    with torch.no_grad():
        for start in range(0, len(X), 2048):
            xb = torch.tensor(X[start : start + 2048], dtype=torch.float32, device=device)
            out = model(xb)
            for k in HORIZONS:
                probs[k][start : start + 2048] = torch.sigmoid(out[str(k)]).cpu().numpy()

    print(f"Checkpoint: {CHECKPOINT} (trained on UMN only; MED unseen)")
    print(f"MED windows: {X.shape}, videos: {len(np.unique(video_ids))}\n")

    pooled = {str(k): evaluate(y[k], probs[k]) for k in HORIZONS}
    print("=== Pooled across all MED videos ===")
    print(f"{'horizon':>8}{'AUROC':>9}{'AUPRC':>9}{'n_pos':>8}{'  UMN LOCO ref':>16}")
    for k in HORIZONS:
        e = pooled[str(k)]
        print(f"{k:>8}{e['auroc']:>9.4f}{e['auprc']:>9.4f}{e['n_pos']:>8}"
              f"{UMN_LOCO_REFERENCE[str(k)]:>16.3f}")

    per_video: Dict[str, Dict[str, Dict]] = {}
    for v in sorted(np.unique(video_ids).tolist()):
        m = video_ids == v
        per_video[str(v)] = {str(k): evaluate(y[k][m], probs[k][m]) for k in HORIZONS}

    print("\n=== Per-video AUROC ===")
    print(f"{'video':>6}" + "".join(f"{'k=' + str(k):>9}" for k in HORIZONS))
    for v in sorted(per_video, key=int):
        row = f"{int(v) + 1:>6}"
        for k in HORIZONS:
            e = per_video[v][str(k)]
            row += f"{'n/a':>9}" if e["excluded"] else f"{e['auroc']:>9.4f}"
        print(row)

    agg: Dict[str, Dict[str, float]] = {}
    print(f"\n{'horizon':>8}{'mean AUROC':>12}{'std':>8}{'n_videos':>10}{'  UMN LOCO ref':>16}")
    for k in HORIZONS:
        vals = [per_video[v][str(k)]["auroc"] for v in per_video
                if not per_video[v][str(k)]["excluded"]]
        arr = np.array(vals, dtype=float)
        agg[str(k)] = {"auroc_mean": float(arr.mean()), "auroc_std": float(arr.std()),
                       "n_videos": int(len(arr))}
        print(f"{k:>8}{arr.mean():>12.4f}{arr.std():>8.4f}{len(arr):>10}"
              f"{UMN_LOCO_REFERENCE[str(k)]:>16.3f}")

    payload = {
        "description": (
            "Zero-shot cross-dataset transfer: models/umn_forecast_lstm.pt (trained on UMN "
            "only, no MED frame seen) evaluated on MED's 11 panic videos at all four forecast "
            "horizons. First external test of the forecast horizons themselves; the Crowd "
            "Panic external check could only test k=0 because it lacks onset annotation."
        ),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_note": (
            "This checkpoint must not be used for UMN metrics (it trained on all 11 UMN "
            "clips). It is valid here because MED is entirely unseen by it."
        ),
        "n_windows": int(len(X)),
        "pooled": pooled,
        "per_video": per_video,
        "per_video_aggregate": agg,
        "umn_loco_reference": UMN_LOCO_REFERENCE,
        "reference_note": "UMN common-fold LOCO AUROC, from outputs/reports/umn_loco_results.json",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
