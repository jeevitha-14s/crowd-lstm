"""Zero-shot external-validity check: run the deployed UMN-trained pipeline
(YOLOv8s+ByteTrack -> FeatureExtractor -> SequenceBuffer -> CrowdRiskLSTM
k=0 head), completely unmodified and un-fine-tuned, against an unrelated
crowd-sourced dataset of short phone-recorded clips labeled by density/panic
level (No Panic / Normal / Crowdy / Panic). No retraining, no frame-level
ground truth alignment -- this is a rank-order transfer test: does the
model's mean per-clip risk score trend in the expected direction across
classes on footage it has never seen, from a different camera, country, and
crowd type than UMN. AUROC is deliberately not computed -- these are
clip-level categorical labels, not frame-level binary onset labels, so a
proper ROC isn't well-posed here.

Deliberately does NOT reuse stream_processor.StreamProcessor: that class's
FrameSlot/threaded-reader architecture exists specifically to simulate a
live camera's single-slot-overwrite constraint (see its module docstring),
which drops most or all frames on a short file processed without --realtime
pacing, and takes real wall-clock playback time with --realtime pacing. Both
are wrong for this job -- these clips are 4-7s already, and the model needs
145 continuous video frames of history before its first prediction, so full
frame coverage (fast, synchronous, no drops) is what makes rank-order
comparison meaningful at all. This script reuses the exact same building
blocks (FeatureExtractor, SequenceBuffer, CrowdRiskLSTM, YOLO+ByteTrack) but
loads YOLO/LSTM once and drives every clip through a plain sequential loop.

Clips shorter than the (seq_len-1)*stride = 145-frame warmup span never
reach SequenceBuffer.is_ready(). Rather than excluding them outright --
which turned out to disproportionately drop the Panic class (55%, since
real panic footage is plausibly shot in shorter, more chaotic bursts than
calm scenes) -- short clips are front-padded with their own first frame,
repeated, up to PAD_TARGET_FRAMES. A frozen first frame reads as "nothing
happening yet" (zero motion/flow for any tracked person), a much smaller
distortion than looping the clip's real content, which would create an
artificial jump-cut at the seam and risk a spurious motion spike. Clips
still too degenerate to open/dimension at all remain excluded (reported
per class, not hidden).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.feature_extractor import FeatureExtractor
from src.stream_processor import SequenceBuffer
from src.train_forecast_model import CrowdRiskLSTM

DEFAULT_YOLO_PATH = Path("models/finetuned_yolov8s_cctv.pt")
DEFAULT_LSTM_PATH = Path("models/umn_forecast_lstm.pt")
DEFAULT_DATA_ROOT = Path(
    "/private/tmp/claude-501/-Users-jeevithasmacbook-Library-CloudStorage-GoogleDrive-sjeevitha679-gmail-com-My-Drive-crowd-risk-lstm/"
    "c23455dd-1865-48a5-9dd2-525792c1e69e/scratchpad/crowd_panic_dataset/Dataset/Dataset"
)
OUT_JSON_PATH = Path("outputs/reports/crowd_panic_transfer.json")
CLASSES = ["No Panic", "Normal", "Crowdy", "Panic"]
GRID_ROWS, GRID_COLS = 4, 4
SEQ_LEN, STRIDE = 30, 5
REQUIRED_SPAN = (SEQ_LEN - 1) * STRIDE  # 145
MIN_TOTAL_FRAMES = REQUIRED_SPAN + 1  # 146, guarantees >=1 prediction
PAD_TARGET_FRAMES = 150  # matches the dataset-wide median clip length -- gives ~5 predictions like most natural clips
IMGSZ, CONF = 640, 0.25


def process_clip(
    path: Path, yolo: YOLO, lstm: CrowdRiskLSTM, device: torch.device, imgsz: int, conf: float
) -> Dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"filename": path.name, "excluded": True, "exclude_reason": "could_not_open"}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return {"filename": path.name, "excluded": True, "exclude_reason": "bad_dimensions"}

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    n_source = len(frames)
    if n_source == 0:
        return {"filename": path.name, "n_frames_source": 0, "excluded": True, "exclude_reason": "no_frames"}

    n_pad = 0
    if n_source < MIN_TOTAL_FRAMES:
        n_pad = max(0, PAD_TARGET_FRAMES - n_source)
        frames = [frames[0]] * n_pad + frames

    feature_extractor = FeatureExtractor(width, height, GRID_ROWS, GRID_COLS)
    seq_buffer = SequenceBuffer(SEQ_LEN, STRIDE, GRID_ROWS * GRID_COLS * 4)
    if hasattr(yolo.predictor, "trackers"):
        yolo.predictor.trackers[0].reset()

    risk_scores: List[float] = []
    t0 = time.perf_counter()
    for frame_id, frame in enumerate(frames, start=1):
        results = yolo.track(
            frame, imgsz=imgsz, conf=conf, classes=[0], tracker="bytetrack.yaml",
            persist=True, verbose=False, device=str(device),
        )
        detections = []
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), track_id in zip(xyxy, track_ids):
                detections.append({
                    "track_id": int(track_id),
                    "cx": float((x1 + x2) / 2),
                    "cy": float((y1 + y2) / 2),
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                })
        features = feature_extractor.extract(detections)
        seq_buffer.push(frame_id, features)
        if seq_buffer.is_ready():
            x = torch.tensor(seq_buffer.get_sequence(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(lstm(x)["0"]).item()
            risk_scores.append(float(prob))
    elapsed_s = time.perf_counter() - t0

    if not risk_scores:
        return {
            "filename": path.name, "n_frames_source": n_source, "n_pad_frames": n_pad, "excluded": True,
            "exclude_reason": "below_145_frame_warmup_span_even_after_padding", "elapsed_s": elapsed_s,
        }
    return {
        "filename": path.name, "n_frames_source": n_source, "n_pad_frames": n_pad,
        "n_predictions": len(risk_scores), "mean_risk_k0": float(np.mean(risk_scores)),
        "median_risk_k0": float(np.median(risk_scores)), "excluded": False, "elapsed_s": elapsed_s,
    }


def load_models(device: torch.device) -> tuple:
    yolo = YOLO(str(DEFAULT_YOLO_PATH))
    lstm = CrowdRiskLSTM(input_size=GRID_ROWS * GRID_COLS * 4)
    lstm.load_state_dict(torch.load(DEFAULT_LSTM_PATH, map_location="cpu"))
    lstm.eval()
    return yolo, lstm


def compute_aggregate(per_clip_by_class: Dict[str, List[Dict]]) -> Dict:
    aggregate = {}
    for cls in CLASSES:
        records = per_clip_by_class[cls]
        included = [r for r in records if not r["excluded"]]
        excluded = [r for r in records if r["excluded"]]
        means = np.array([r["mean_risk_k0"] for r in included])
        aggregate[cls] = {
            "n_total_clips": len(records),
            "n_included": len(included),
            "n_excluded": len(excluded),
            "exclusion_rate": len(excluded) / len(records) if records else float("nan"),
            "mean_risk_k0": float(means.mean()) if len(means) else float("nan"),
            "std_risk_k0": float(means.std()) if len(means) else float("nan"),
        }
    return aggregate


def compute_supplementary(per_clip_by_class: Dict[str, List[Dict]]) -> Dict:
    from scipy import stats

    per_class_stats = {}
    for cls in CLASSES:
        arr = np.array([r["mean_risk_k0"] for r in per_clip_by_class[cls] if not r["excluded"]])
        per_class_stats[cls] = {
            "n": len(arr), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
            "frac_above_0.5": float(np.mean(arr > 0.5)),
        }
    non_panic = np.concatenate([
        np.array([r["mean_risk_k0"] for r in per_clip_by_class[c] if not r["excluded"]])
        for c in ["No Panic", "Normal", "Crowdy"]
    ])
    panic = np.array([r["mean_risk_k0"] for r in per_clip_by_class["Panic"] if not r["excluded"]])
    u, p = stats.mannwhitneyu(panic, non_panic, alternative="greater")
    return {
        "per_class_median_iqr": per_class_stats,
        "panic_vs_pooled_nonpanic": {
            "non_panic_pooled_n": int(len(non_panic)), "non_panic_pooled_mean": float(non_panic.mean()),
            "non_panic_pooled_median": float(np.median(non_panic)),
            "panic_n": int(len(panic)), "panic_mean": float(panic.mean()), "panic_median": float(np.median(panic)),
            "mann_whitney_u": float(u), "p_value_one_sided_panic_greater": float(p),
            "test_description": (
                "Mann-Whitney U, one-sided (panic > pooled non-panic), on per-clip mean k=0 risk scores."
            ),
        },
    }


def print_summary(aggregate: Dict) -> None:
    print("\n=== Mean +/- std risk score (k=0) by class ===")
    for cls in CLASSES:
        a = aggregate[cls]
        print(
            f"  {cls:10s}: {a['mean_risk_k0']:.4f} +/- {a['std_risk_k0']:.4f}  "
            f"(n_included={a['n_included']}/{a['n_total_clips']}, excluded={a['exclusion_rate']:.0%})"
        )


def save_payload(per_clip_by_class: Dict[str, List[Dict]], device: torch.device) -> None:
    aggregate = compute_aggregate(per_clip_by_class)
    print_summary(aggregate)
    payload = {
        "description": (
            "Zero-shot transfer test: UMN-trained pipeline (unmodified, no fine-tuning) run on "
            "https://www.kaggle.com/datasets/aj1714/crowd-panic-dataset. Rank-order comparison only "
            "-- AUROC deliberately not computed, see module docstring. Short clips are front-padded "
            "(frozen first frame) rather than excluded -- see module docstring."
        ),
        "source_dataset": "kaggle.com/datasets/aj1714/crowd-panic-dataset",
        "model_config": {
            "yolo_path": str(DEFAULT_YOLO_PATH), "lstm_path": str(DEFAULT_LSTM_PATH),
            "seq_len": SEQ_LEN, "stride": STRIDE, "required_warmup_frames": REQUIRED_SPAN,
            "pad_target_frames": PAD_TARGET_FRAMES, "imgsz": IMGSZ, "conf": CONF, "device": str(device),
        },
        "aggregate_by_class": aggregate,
        "supplementary_analysis": compute_supplementary(per_clip_by_class),
        "per_clip": per_clip_by_class,
    }
    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON_PATH}")


def rerun_excluded(args: argparse.Namespace, device: torch.device) -> None:
    """Reprocess only clips previously excluded for being below the warmup
    span, now with front-padding enabled, and merge results back into the
    existing per-clip data (clips that already succeeded are left as-is --
    they didn't need padding and reprocessing them would just burn an
    unnecessary ~4.3h re-run)."""
    with open(OUT_JSON_PATH) as f:
        existing = json.load(f)
    per_clip_by_class: Dict[str, List[Dict]] = existing["per_clip"]

    yolo, lstm = load_models(device)
    for cls in CLASSES:
        records = per_clip_by_class[cls]
        to_rerun = [
            (i, r) for i, r in enumerate(records)
            if r["excluded"] and r.get("exclude_reason") == "below_145_frame_warmup_span"
        ]
        print(f"\n=== class: {cls}  ({len(to_rerun)} previously-excluded clips to reprocess with padding) ===")
        for j, (i, old_rec) in enumerate(to_rerun):
            f = args.data_root / cls / old_rec["filename"]
            rec = process_clip(f, yolo, lstm, device, IMGSZ, CONF)
            records[i] = rec
            status = (
                f"mean_risk_k0={rec['mean_risk_k0']:.3f} (n_pred={rec['n_predictions']}, pad={rec['n_pad_frames']})"
                if not rec["excluded"] else f"STILL EXCLUDED ({rec['exclude_reason']})"
            )
            print(f"  [{j+1}/{len(to_rerun)}] {rec['filename']}  {status}  ({rec.get('elapsed_s', 0):.1f}s)")

    save_payload(per_clip_by_class, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crowd Panic Dataset zero-shot transfer test")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="max clips per class, for validation runs")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--rerun-excluded", action="store_true",
        help="reprocess (with padding) only clips excluded in the existing outputs/reports/crowd_panic_transfer.json",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if args.rerun_excluded:
        rerun_excluded(args, device)
        return

    yolo, lstm = load_models(device)
    per_clip_by_class: Dict[str, List[Dict]] = {}
    for cls in CLASSES:
        files = sorted((args.data_root / cls).glob("*.mp4"))
        if args.limit:
            files = files[: args.limit]
        print(f"\n=== class: {cls}  ({len(files)} clips) ===")
        records = []
        for i, f in enumerate(files):
            rec = process_clip(f, yolo, lstm, device, IMGSZ, CONF)
            records.append(rec)
            status = (
                f"mean_risk_k0={rec['mean_risk_k0']:.3f} (n_pred={rec['n_predictions']})"
                if not rec["excluded"] else f"EXCLUDED ({rec['exclude_reason']})"
            )
            print(f"  [{i+1}/{len(files)}] {rec['filename']}  {status}  ({rec.get('elapsed_s', 0):.1f}s)")
        per_clip_by_class[cls] = records

    save_payload(per_clip_by_class, device)


if __name__ == "__main__":
    main()
