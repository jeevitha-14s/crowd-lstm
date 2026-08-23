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
reach SequenceBuffer.is_ready() and are excluded with a reason -- reported
per class, not hidden, since exclusion rates differ sharply by class (see
outputs/reports/PAPER_RESULTS.md for the diagnostic that found this).
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

    feature_extractor = FeatureExtractor(width, height, GRID_ROWS, GRID_COLS)
    seq_buffer = SequenceBuffer(SEQ_LEN, STRIDE, GRID_ROWS * GRID_COLS * 4)
    if hasattr(yolo.predictor, "trackers"):
        yolo.predictor.trackers[0].reset()

    frame_id = 0
    risk_scores: List[float] = []
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
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
    cap.release()
    elapsed_s = time.perf_counter() - t0

    if not risk_scores:
        return {
            "filename": path.name, "n_frames_source": frame_id, "excluded": True,
            "exclude_reason": "below_145_frame_warmup_span", "elapsed_s": elapsed_s,
        }
    return {
        "filename": path.name, "n_frames_source": frame_id, "n_predictions": len(risk_scores),
        "mean_risk_k0": float(np.mean(risk_scores)), "excluded": False, "elapsed_s": elapsed_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Crowd Panic Dataset zero-shot transfer test")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="max clips per class, for validation runs")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
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
    yolo = YOLO(str(DEFAULT_YOLO_PATH))
    lstm = CrowdRiskLSTM(input_size=GRID_ROWS * GRID_COLS * 4)
    lstm.load_state_dict(torch.load(DEFAULT_LSTM_PATH, map_location="cpu"))
    lstm.eval()

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

    print("\n=== Mean +/- std risk score (k=0) by class ===")
    for cls in CLASSES:
        a = aggregate[cls]
        print(
            f"  {cls:10s}: {a['mean_risk_k0']:.4f} +/- {a['std_risk_k0']:.4f}  "
            f"(n_included={a['n_included']}/{a['n_total_clips']}, excluded={a['exclusion_rate']:.0%})"
        )

    payload = {
        "description": (
            "Zero-shot transfer test: UMN-trained pipeline (unmodified, no fine-tuning) run on "
            "https://www.kaggle.com/datasets/aj1714/crowd-panic-dataset. Rank-order comparison only "
            "-- AUROC deliberately not computed, see module docstring."
        ),
        "source_dataset": "kaggle.com/datasets/aj1714/crowd-panic-dataset",
        "model_config": {
            "yolo_path": str(DEFAULT_YOLO_PATH), "lstm_path": str(DEFAULT_LSTM_PATH),
            "seq_len": SEQ_LEN, "stride": STRIDE, "required_warmup_frames": REQUIRED_SPAN,
            "imgsz": IMGSZ, "conf": CONF, "device": str(device),
        },
        "aggregate_by_class": aggregate,
        "per_clip": per_clip_by_class,
    }
    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
