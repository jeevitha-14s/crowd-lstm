"""Detect + track + extract 4x4 spatial-grid features from the Motion Emotion
Dataset (MED), producing per-frame features and Panic labels for the 11 videos
that contain a panic event.

MED (Rabiee et al., AVSS 2016) provides per-frame behaviour labels for 31
videos: 1=Panic, 2=Fight, 3=Congestion, 4=Obstacle, 5=Neutral, 0=unlabelled.
Only label 1 is a collective crowd event of the kind this method targets;
Fight and Congestion are small-group or static phenomena and are excluded here
for the same reason ShanghaiTech scene 05 falls outside the operating envelope
(see the paper's Sect. V-H). Videos 1-11 (0-indexed 0-10 in annotation.mat) are
the panic subset.

Two dataset quirks, both handled below rather than silently:

  1. The annotation arrays are always an exact multiple of 30 frames -- they
     were made at whole-second granularity -- while the decoded videos run 1-2
     frames shorter. Labels are therefore truncated to the decoded frame count.
     Verified: every video's annotation length is within 2 frames of its decode
     length, so this discards at most 2 trailing labels and never misaligns the
     interior.
  2. Trailing label-0 (unlabelled) runs appear at the end of most videos. Those
     frames are marked valid=0 and excluded downstream rather than being
     treated as normal, since "not annotated" is not the same as "not panic".

The grid is fixed at 4x4 (64-dim output) because the deployed UMN checkpoint
takes input_size=64; a different grid would make the zero-shot transfer test in
build_med_forecast.py impossible without retraining. This differs from
process_shanghaitech.py, which uses 2x2 because that corpus is too sparse to
populate 16 zones -- MED at 854x480 is expected to be denser, but the choice
here is driven by checkpoint compatibility, not density.

Tracker and FeatureExtractor state is reset at every video boundary: track IDs
are only meaningful within a continuous shot, and the running normalisers are
calibrated per scene.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.feature_extractor import FeatureExtractor

MED_DIR = Path("(Copy) Motion_Emotion_Dataset")
ANNOTATION_PATH = MED_DIR / "Motion Emotion Dataset(MED) annotation" / "annotation.mat"
MODEL_PATH = Path("models/finetuned_yolov8s_cctv.pt")

FEATURES_OUT = Path("data/features/med_features.npy")
META_OUT = Path("data/features/med_meta.csv")

PANIC_LABEL = 1
UNLABELLED = 0
PANIC_VIDEO_INDICES = list(range(11))  # 0-indexed; files are 001.mp4 .. 011.mp4
GRID_ROWS = GRID_COLS = 4
IMGSZ = 640
CONF = 0.25


def load_labels() -> Dict[int, np.ndarray]:
    """{video_index: per-frame behaviour label array} for the panic subset."""
    mat = sio.loadmat(str(ANNOTATION_PATH))["behavelabels"][0]
    return {i: np.asarray(mat[i]).ravel().astype(int) for i in PANIC_VIDEO_INDICES}


def align_labels(labels: np.ndarray, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate (or pad) labels to the decoded frame count.

    Returns (abnormal, valid) as 0/1 arrays of length n_frames, where
    abnormal = label is Panic and valid = label was annotated at all.
    """
    if len(labels) >= n_frames:
        lab = labels[:n_frames]
    else:
        # Never observed on this dataset, but pad defensively as unlabelled
        # rather than assuming the tail is normal.
        lab = np.concatenate([labels, np.full(n_frames - len(labels), UNLABELLED)])
    return (lab == PANIC_LABEL).astype(np.uint8), (lab != UNLABELLED).astype(np.uint8)


def detections_from_result(result) -> List[Dict]:
    boxes = result.boxes
    out: List[Dict] = []
    if boxes is None or boxes.id is None:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), tid in zip(xyxy, ids):
        out.append(
            {
                "track_id": int(tid),
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            }
        )
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    labels_by_video = load_labels()
    model = YOLO(str(MODEL_PATH))

    all_features: List[np.ndarray] = []
    meta_rows: List[Tuple[int, int, int, int, int, int]] = []
    global_frame = 0

    for vid in PANIC_VIDEO_INDICES:
        path = MED_DIR / f"{vid + 1:03d}.mp4"
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {path}")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        abnormal, valid = align_labels(labels_by_video[vid], n_frames)
        onset = int(np.argmax(abnormal)) if abnormal.any() else -1

        # Per-video reset: track IDs and running normalisers are scene-scoped.
        fe = FeatureExtractor(w, h, GRID_ROWS, GRID_COLS)
        if getattr(model, "predictor", None) is not None and hasattr(model.predictor, "trackers"):
            model.predictor.trackers[0].reset()

        feats = np.zeros((n_frames, GRID_ROWS * GRID_COLS * 4), dtype=np.float32)
        idx = 0
        desc = f"vid {vid + 1:03d} ({n_frames}f, onset {onset})"
        with tqdm(total=n_frames, desc=desc) as pbar:
            while idx < n_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                res = model.track(frame, imgsz=IMGSZ, conf=CONF, classes=[0],
                                  tracker="bytetrack.yaml", persist=True,
                                  verbose=False, device=str(device))
                dets = detections_from_result(res[0])
                feats[idx] = fe.extract(dets)
                meta_rows.append((global_frame, len(dets), vid, idx,
                                  int(abnormal[idx]), int(valid[idx])))
                idx += 1
                global_frame += 1
                pbar.update(1)
        cap.release()

        all_features.append(feats[:idx])
        print(f"  vid {vid + 1:03d}: {idx} frames, {int(abnormal[:idx].sum())} panic, "
              f"{int((valid[:idx] == 0).sum())} unlabelled")

    features = np.concatenate(all_features, axis=0)
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_OUT, features)
    with open(META_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "n_persons", "video_id", "local_frame", "abnormal", "valid"])
        writer.writerows(meta_rows)

    print(f"\nSaved {FEATURES_OUT} {features.shape}")
    print(f"Saved {META_OUT} ({len(meta_rows)} rows)")


if __name__ == "__main__":
    main()
