"""Diagnostic: distribution of raw per-track pixel speeds in the UMN video,
to calibrate FeatureExtractor's stop_threshold for this resolution (320x240).
Not part of the main pipeline -- run once, read the numbers, decide the
threshold, then update feature_extractor.py's default if warranted.
"""

import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_PATH = Path("data/raw/umn/Crowd-Activity-All.avi")
MODEL_PATH = Path("models/finetuned_yolov8s_cctv.pt")
GROUNDTRUTH_PATH = Path("data/umn_groundtruth.csv")
HISTORY_LEN = 5
# Same boundaries validated for process_umn.py, so track history resets at
# the same points the real pipeline resets at (keeps this diagnostic honest).
CLIP_BOUNDARIES = {625, 1453, 1996, 2687, 3455, 4034, 4929, 5596, 6238, 6931}
CANDIDATE_THRESHOLDS = [0.5, 0.8, 1.0, 1.5, 2.0]


def load_labels() -> np.ndarray:
    labels = []
    with open(GROUNDTRUTH_PATH, newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["abnormal"]))
    return np.array(labels, dtype=np.int64)


def report(name: str, arr: np.ndarray) -> None:
    if len(arr) == 0:
        print(f"{name}: no samples")
        return
    p5, p25, p50, p75, p90, p95, p99 = np.percentile(arr, [5, 25, 50, 75, 90, 95, 99])
    print(
        f"{name}: n={len(arr)} mean={arr.mean():.3f} "
        f"p5={p5:.3f} p25={p25:.3f} median={p50:.3f} "
        f"p75={p75:.3f} p90={p90:.3f} p95={p95:.3f} p99={p99:.3f}"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels()
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    track_history = {}
    speeds_normal = []
    speeds_abnormal = []
    # (speed, clip_id, label) for every sample, to break down by clip afterward
    speeds_by_clip = {cid: {"normal": [], "abnormal": []} for cid in range(len(CLIP_BOUNDARIES) + 1)}

    clip_id = 0
    frame_idx = 0
    with tqdm(total=total_frames, desc="speed distribution scan") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx in CLIP_BOUNDARIES:
                clip_id += 1
                track_history.clear()
                model.predictor.trackers[0].reset()

            results = model.track(
                frame,
                imgsz=640,
                conf=0.25,
                classes=[0],
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                device=str(device),
            )
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if tid not in track_history:
                        track_history[tid] = deque(maxlen=HISTORY_LEN)
                    track_history[tid].append((cx, cy))
                    hist = track_history[tid]
                    if len(hist) >= 2:
                        first = np.array(hist[0])
                        last = np.array(hist[-1])
                        speed = float(np.linalg.norm(last - first) / (len(hist) - 1))
                        bucket = "abnormal" if labels[frame_idx] == 1 else "normal"
                        (speeds_abnormal if bucket == "abnormal" else speeds_normal).append(speed)
                        speeds_by_clip[clip_id][bucket].append(speed)

            frame_idx += 1
            pbar.update(1)
    cap.release()

    speeds_normal = np.array(speeds_normal)
    speeds_abnormal = np.array(speeds_abnormal)

    print("\n=== Per-track speed distribution (px/frame), global ===")
    report("normal-labeled frames", speeds_normal)
    report("abnormal-labeled frames", speeds_abnormal)

    print("\nFraction of NORMAL-frame speeds below candidate stop_threshold values:")
    for thresh in CANDIDATE_THRESHOLDS:
        frac = float((speeds_normal < thresh).mean()) if len(speeds_normal) else float("nan")
        print(f"  < {thresh}: {frac:.3f}")

    print("\n=== Per-clip speed medians (checks whether a single global threshold makes sense) ===")
    print(f"{'clip':>4}  {'n_normal':>8}  {'median_normal':>13}  {'n_abnormal':>10}  {'median_abnormal':>15}")
    for cid in sorted(speeds_by_clip):
        normal_arr = np.array(speeds_by_clip[cid]["normal"])
        abnormal_arr = np.array(speeds_by_clip[cid]["abnormal"])
        med_n = f"{np.median(normal_arr):.3f}" if len(normal_arr) else "n/a"
        med_a = f"{np.median(abnormal_arr):.3f}" if len(abnormal_arr) else "n/a"
        print(f"{cid:>4}  {len(normal_arr):>8}  {med_n:>13}  {len(abnormal_arr):>10}  {med_a:>15}")

    normal_medians = np.array(
        [np.median(speeds_by_clip[cid]["normal"]) for cid in speeds_by_clip if speeds_by_clip[cid]["normal"]]
    )
    if len(normal_medians) > 1:
        print(
            f"\nSpread of per-clip normal-speed medians: min={normal_medians.min():.3f} "
            f"max={normal_medians.max():.3f} ratio={normal_medians.max() / max(normal_medians.min(), 1e-6):.2f}x"
        )


if __name__ == "__main__":
    main()
