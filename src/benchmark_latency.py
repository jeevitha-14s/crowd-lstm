"""Per-stage latency benchmark for the paper: mean/median/p95/p99 ms for
decode/detect/track/features/lstm/render/total, across three input
resolutions and on every available device (CPU always, CUDA if available).

METHODOLOGY, read before trusting the numbers:

Runs a SEQUENTIAL, unthrottled loop (every frame processed, nothing dropped)
rather than replaying stream_processor.StreamProcessor's live threaded
pipeline. The Task-A realtime test showed GIL contention between the reader
thread and CPU-bound inference inflates *measured* decode latency by >100x
(406ms vs 0.5ms for the identical operation, paced vs unthrottled/contended)
-- a threaded harness would bake that same noise into a paper table. This
loop isolates per-stage compute cost cleanly instead.

"achieved FPS" and "dropped-frame %" are therefore DERIVED from these clean
per-stage numbers, not measured by re-running the actual FrameSlot-dropping
mechanism: achieved_fps = 1000 / (detect+track+features+lstm+render mean ms)
-- decode is excluded from this sum because in the real StreamProcessor it
runs on a separate reader thread and never serializes with the consumer
stages (see stream_processor.py). drop_rate_pct assumes a 30fps source
(NOMINAL_SOURCE_FPS) and is reported as max(0, 1 - achieved_fps/30). The
"total" row/column, by contrast, IS the literal per-frame sum of all six
stages including decode -- a fully-sequential worst-case bound, reported
because the task spec asks for a total column, not because it's the
deployment-relevant number.

YOLO's imgsz is held FIXED at 640 (the model's fine-tuning resolution)
across all three input resolutions. Ultralytics resizes internally to imgsz
regardless of input frame size, so inference cost is expected to be roughly
resolution-invariant -- what actually varies with resolution here is
decode/preprocess cost. This is the realistic choice (a deployment doesn't
change model imgsz based on camera resolution), documented so a flat
"detect" number across resolutions in the output isn't misread as a bug.

640x480 and 856x480 are produced by resizing UP from the UMN source's native
320x240 -- this measures compute cost at that pixel count, not real captured
high-resolution detail, since upsampling can't recover detail the camera
never captured.
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.feature_extractor import FeatureExtractor
from src.stream_processor import SEQ_LEN, STRIDE, DEFAULT_LSTM_PATH, DEFAULT_YOLO_PATH, SequenceBuffer
from src.train_forecast_model import HORIZONS, CrowdRiskLSTM

STAGES = ["decode", "detect", "track", "features", "lstm", "render"]
CONSUMER_STAGES = ["detect", "track", "features", "lstm", "render"]  # excludes decode -- see docstring
WARMUP_FRAMES = 50
DEFAULT_MEASURED_FRAMES = 300
IMGSZ = 640  # fixed across resolutions -- see module docstring
RESOLUTIONS: List[Tuple[int, int]] = [(320, 240), (640, 480), (856, 480)]
GRID_ROWS, GRID_COLS = 4, 4  # UMN deployment model's grid; input_size = 4*4*4 = 64
NOMINAL_SOURCE_FPS = 30.0
OUT_CSV_PATH = Path("outputs/reports/latency_benchmark.csv")


def percentiles(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n": len(arr),
    }


def run_one_config(
    video_path: Path,
    width: int,
    height: int,
    device: torch.device,
    yolo_path: Path,
    lstm_path: Path,
    total_frames: int,
) -> Dict[str, List[float]]:
    yolo = YOLO(str(yolo_path))
    input_size = GRID_ROWS * GRID_COLS * 4
    lstm = CrowdRiskLSTM(input_size=input_size).to(device)
    lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm.eval()

    fe = FeatureExtractor(width, height, grid_rows=GRID_ROWS, grid_cols=GRID_COLS)
    seq_buf = SequenceBuffer(SEQ_LEN, STRIDE, input_size)
    cap = cv2.VideoCapture(str(video_path))
    if hasattr(yolo.predictor, "trackers"):
        yolo.predictor.trackers[0].reset()

    timings: Dict[str, List[float]] = {s: [] for s in STAGES}
    timings["total"] = []
    timings["n_persons"] = []  # detections per frame -- not a latency stat, kept for density analysis
    frame_id = 0

    while frame_id < total_frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the source if it's shorter than total_frames
            ok, frame = cap.read()
            if not ok:
                break
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height))
        decode_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        results = yolo.track(
            frame,
            imgsz=IMGSZ,
            conf=0.25,
            classes=[0],
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            device=str(device),
        )
        if device.type == "cuda":
            # Defensive, not strictly required: ultralytics' own speed dict
            # should already reflect fully-synced GPU work (it has to move
            # results off-device to build the Results object), and the
            # .cpu()/.item() calls below force sync incidentally anyway.
            # Explicit here so correctness doesn't silently depend on those
            # calls never being refactored away.
            torch.cuda.synchronize()
        speed = results[0].speed
        detect_ms = speed.get("preprocess", 0.0) + speed.get("inference", 0.0)
        track_ms = speed.get("postprocess", 0.0)

        t1 = time.perf_counter()
        detections = []
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(xyxy, track_ids):
                detections.append(
                    {
                        "track_id": int(tid),
                        "cx": float((x1 + x2) / 2),
                        "cy": float((y1 + y2) / 2),
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    }
                )
        track_ms += (time.perf_counter() - t1) * 1000

        t0 = time.perf_counter()
        features = fe.extract(detections)
        features_ms = (time.perf_counter() - t0) * 1000

        seq_buf.push(features)
        t0 = time.perf_counter()
        if seq_buf.is_ready():
            x = torch.tensor(seq_buf.get_sequence(), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                preds = lstm(x)
                _ = {k: float(torch.sigmoid(preds[str(k)]).item()) for k in HORIZONS}
        if device.type == "cuda":
            torch.cuda.synchronize()  # redundant with .item() above, kept explicit -- see detect/track note
        lstm_ms = (time.perf_counter() - t0) * 1000

        render_ms = 0.0  # no renderer attached in the benchmark loop

        frame_id += 1
        if frame_id <= WARMUP_FRAMES:
            continue
        timings["decode"].append(decode_ms)
        timings["detect"].append(detect_ms)
        timings["track"].append(track_ms)
        timings["features"].append(features_ms)
        timings["lstm"].append(lstm_ms)
        timings["render"].append(render_ms)
        timings["total"].append(decode_ms + detect_ms + track_ms + features_ms + lstm_ms + render_ms)
        timings["n_persons"].append(float(len(detections)))

    cap.release()
    return timings


def available_devices() -> List[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-stage latency benchmark")
    parser.add_argument("video", type=Path, help="fixed video file to benchmark against")
    parser.add_argument("--yolo-path", type=Path, default=DEFAULT_YOLO_PATH)
    parser.add_argument("--lstm-path", type=Path, default=DEFAULT_LSTM_PATH)
    parser.add_argument("--measured-frames", type=int, default=DEFAULT_MEASURED_FRAMES)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV_PATH)
    args = parser.parse_args()

    total_frames = WARMUP_FRAMES + args.measured_frames
    devices = available_devices()
    print(f"Devices: {[str(d) for d in devices]}  resolutions: {RESOLUTIONS}")
    print(f"warmup={WARMUP_FRAMES} frames, measured={args.measured_frames} frames per config, imgsz={IMGSZ} (fixed)\n")

    rows: List[Dict[str, object]] = []
    for device in devices:
        for width, height in RESOLUTIONS:
            print(f"--- device={device}  resolution={width}x{height} ---")
            t_start = time.perf_counter()
            timings = run_one_config(
                args.video, width, height, device, args.yolo_path, args.lstm_path, total_frames
            )
            wall_s = time.perf_counter() - t_start

            consumer_total_mean = sum(
                percentiles(timings[s])["mean"] for s in CONSUMER_STAGES if timings[s]
            )
            achieved_fps = 1000.0 / consumer_total_mean if consumer_total_mean else float("nan")
            drop_rate_pct = max(0.0, 1.0 - achieved_fps / NOMINAL_SOURCE_FPS) * 100.0

            for stage in STAGES + ["total"]:
                if not timings[stage]:
                    continue
                p = percentiles(timings[stage])
                row = {
                    "width": width,
                    "height": height,
                    "device": str(device),
                    "stage": stage,
                    "mean_ms": round(p["mean"], 4),
                    "median_ms": round(p["median"], 4),
                    "p95_ms": round(p["p95"], 4),
                    "p99_ms": round(p["p99"], 4),
                    "n": p["n"],
                    "achieved_fps": round(achieved_fps, 3) if stage == "total" else "",
                    "drop_rate_pct_at_30fps": round(drop_rate_pct, 2) if stage == "total" else "",
                }
                rows.append(row)
                print(
                    f"  {stage:>8}: mean={p['mean']:>8.3f}  median={p['median']:>8.3f}  "
                    f"p95={p['p95']:>8.3f}  p99={p['p99']:>8.3f}  (n={p['n']})"
                )
            print(
                f"  -> achieved {achieved_fps:.2f} FPS (consumer stages only, decode excluded -- "
                f"see module docstring), ~{drop_rate_pct:.1f}% dropped against a {NOMINAL_SOURCE_FPS:.0f}fps source"
            )
            mean_persons = float(np.mean(timings["n_persons"])) if timings["n_persons"] else float("nan")
            print(f"  mean persons/frame: {mean_persons:.2f} (context for detect-time variation, see diagnose_detect_latency_vs_density.py)")
            print(f"  (config wall time: {wall_s:.1f}s)\n")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "width", "height", "device", "stage", "mean_ms", "median_ms",
                "p95_ms", "p99_ms", "n", "achieved_fps", "drop_rate_pct_at_30fps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
