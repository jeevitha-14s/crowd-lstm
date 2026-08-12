"""Detect + track persons frame-by-frame in the UMN video and extract 64-dim
spatial-grid features per frame, resetting tracker/feature state at clip
boundaries.

Clip boundaries are located using ground-truth abnormal-segment ends plus a
person-count signal, NOT pixel-diff: pixel-diff confounds real scene splices
with the panic sprint itself (a large frame-to-frame pixel delta), which
lands "boundaries" almost exactly on the abnormal onsets we're trying to
forecast. See detect_clip_boundaries_from_counts().
"""

import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.feature_extractor import FeatureExtractor

VIDEO_PATH = Path("data/raw/umn/Crowd-Activity-All.avi")
MODEL_PATH = Path("models/finetuned_yolov8s_cctv.pt")
GROUNDTRUTH_PATH = Path("data/umn_groundtruth.csv")
FEATURES_OUT = Path("data/features/umn_features.npy")
META_OUT = Path("data/features/umn_meta.csv")
PERSON_COUNTS_CACHE = Path("data/features/umn_person_counts_cache.npy")

N_KNOWN_CLIPS = 11
SEARCH_WINDOW = 120  # frames to look forward from each segment end for the count minimum
MAX_EXTENSION = 400  # give up extending the search window past segment_end + this
EXTEND_STEP = 60
MIN_EXPECTED_CLIP_LEN = 200
MAX_EXPECTED_CLIP_LEN = 1200
FEATURE_NAMES = ["normalized_count", "flow_magnitude", "direction_entropy", "stop_ratio"]


def load_abnormal_segments(groundtruth_path: Path) -> Tuple[List[int], List[int]]:
    """Read frame-level abnormal labels and return (segment_starts, segment_ends)."""
    frames, abnormal = [], []
    with open(groundtruth_path, newline="") as f:
        for row in csv.DictReader(f):
            frames.append(int(row["frame"]))
            abnormal.append(int(row["abnormal"]))

    starts, ends = [], []
    in_segment = False
    for i, is_abnormal in enumerate(abnormal):
        if is_abnormal and not in_segment:
            starts.append(frames[i])
            in_segment = True
        if not is_abnormal and in_segment:
            ends.append(frames[i - 1])
            in_segment = False
    if in_segment:
        ends.append(frames[-1])
    return starts, ends


def run_person_count_pass(video_path: Path, model_path: Path, device: torch.device) -> np.ndarray:
    """Pass 1: run detection+tracking over the whole video with NO resets, just
    to obtain a per-frame person count. Tracker resets only affect track-ID
    continuity, not how many boxes YOLO detects, so this count is valid
    regardless of where the (not-yet-known) clip boundaries actually are.
    Cached to disk since it's a full ~15min CPU pass.
    """
    if PERSON_COUNTS_CACHE.exists():
        print(f"Using cached person counts from {PERSON_COUNTS_CACHE}")
        return np.load(PERSON_COUNTS_CACHE)

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    counts = np.zeros(total_frames, dtype=np.int64)
    frame_idx = 0
    with tqdm(total=total_frames, desc="Pass 1/2 (person counts)") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
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
            counts[frame_idx] = 0 if boxes is None else len(boxes)
            frame_idx += 1
            pbar.update(1)
    cap.release()

    counts = counts[:frame_idx]
    PERSON_COUNTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(PERSON_COUNTS_CACHE, counts)
    return counts


def detect_clip_boundaries_from_counts(
    n_persons: np.ndarray, segment_ends: List[int]
) -> List[int]:
    """For each of the first N-1 abnormal-segment ends (the last clip has no
    trailing splice), find where the person count bottoms out shortly after
    the panic ends, then place the boundary at the first frame where the
    count recovers above that minimum -- the new clip repopulating.
    """
    total = len(n_persons)
    boundaries = []
    for seg_end in segment_ends[:-1]:
        search_end = min(seg_end + SEARCH_WINDOW, total - 1)
        boundary = None
        while boundary is None:
            window = n_persons[seg_end : search_end + 1]
            min_frame = seg_end + int(np.argmin(window))
            min_count = n_persons[min_frame]
            for f in range(min_frame + 1, search_end + 1):
                if n_persons[f] > min_count:
                    boundary = f
                    break
            if boundary is None:
                if search_end >= min(seg_end + MAX_EXTENSION, total - 1):
                    boundary = min(min_frame + 1, total - 1)
                    print(
                        f"  WARNING: no count recovery found after segment_end={seg_end}; "
                        f"falling back to min_frame+1={boundary}"
                    )
                    break
                search_end = min(search_end + EXTEND_STEP, total - 1)
        boundaries.append(boundary)
    return boundaries


def validate_boundaries(
    boundaries: List[int],
    segment_starts: List[int],
    segment_ends: List[int],
    total_frames: int,
) -> None:
    expected = N_KNOWN_CLIPS - 1
    if len(boundaries) != expected:
        print(f"WARNING: expected exactly {expected} boundaries, got {len(boundaries)}.")

    for b in boundaries:
        for s, e in zip(segment_starts, segment_ends):
            if s <= b <= e:
                print(f"WARNING: boundary {b} falls INSIDE abnormal segment [{s}, {e}]!")

    edges = [0] + boundaries + [total_frames]
    lengths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    print("Clip lengths:")
    for i, length in enumerate(lengths):
        flag = ""
        if length < MIN_EXPECTED_CLIP_LEN or length > MAX_EXPECTED_CLIP_LEN:
            flag = "  <<< OUT OF EXPECTED RANGE"
        print(f"  clip {i}: [{edges[i]}, {edges[i + 1]}) length={length}{flag}")


def process_video() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    segment_starts, segment_ends = load_abnormal_segments(GROUNDTRUTH_PATH)
    print(f"Loaded {len(segment_starts)} abnormal segments from ground truth.")

    n_persons_pass1 = run_person_count_pass(VIDEO_PATH, MODEL_PATH, device)

    boundary_frames = detect_clip_boundaries_from_counts(n_persons_pass1, segment_ends)
    boundary_set = set(boundary_frames)
    print(f"Detected {len(boundary_frames)} clip boundaries at frames: {boundary_frames}")
    validate_boundaries(boundary_frames, segment_starts, segment_ends, len(n_persons_pass1))

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fe = FeatureExtractor(frame_width, frame_height)
    features = np.zeros((total_frames, 64), dtype=np.float32)
    meta_rows = []
    clip_id = 0

    print("Pass 2/2: detection + tracking + feature extraction (with correct resets)...")
    frame_idx = 0
    with tqdm(total=total_frames, desc="Pass 2/2 (features)") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx in boundary_set:
                clip_id += 1
                fe.reset()
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

            detections = []
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), track_id in zip(xyxy, track_ids):
                    detections.append(
                        {
                            "track_id": int(track_id),
                            "cx": float((x1 + x2) / 2),
                            "cy": float((y1 + y2) / 2),
                            "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        }
                    )

            features[frame_idx] = fe.extract(detections)
            meta_rows.append((frame_idx, len(detections), clip_id))

            frame_idx += 1
            pbar.update(1)
    cap.release()

    n_frames = frame_idx
    features = features[:n_frames]

    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_OUT, features)
    with open(META_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "n_persons", "clip_id"])
        writer.writerows(meta_rows)

    n_persons = np.array([row[1] for row in meta_rows])
    print("\n=== Summary ===")
    print(f"Total frames processed: {n_frames}")
    print(f"Mean persons/frame: {n_persons.mean():.3f}")
    print(f"Detected clip boundaries ({len(boundary_frames)}): {boundary_frames}")
    print(f"Number of clips (segments): {clip_id + 1}")
    print(f"Feature array shape: {features.shape}")
    print("\nPer-feature-type stats across all frames/zones:")
    for i, name in enumerate(FEATURE_NAMES):
        vals = features[:, i::4]
        print(f"  {name:20s} min={vals.min():.4f} mean={vals.mean():.4f} max={vals.max():.4f}")


if __name__ == "__main__":
    process_video()
