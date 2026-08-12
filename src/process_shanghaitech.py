"""Detect + track persons across all 199 ShanghaiTech Campus test videos,
extracting a 64-dim feature vector per frame with the same FeatureExtractor
used for UMN (unchanged -- same 4x4 grid, same relative stop_ratio).

Unlike UMN, each video is an independent unit: no boundary detection is
needed, tracker and FeatureExtractor are simply reset between videos.

Frames are read from JPGs on local disk (NOT the Drive-mounted project
path -- 142k small files over Drive is far too slow). Set SHANGHAITECH_ROOT
to wherever the zip was extracted this session, e.g.:
    export SHANGHAITECH_ROOT=/content/shanghaitech/SHANGHAI/SHANGHAI_Test   # Colab
    export SHANGHAITECH_ROOT=/path/to/local/extraction/SHANGHAI/SHANGHAI_Test
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.feature_extractor import FeatureExtractor

SHANGHAITECH_ROOT = Path(
    os.environ.get("SHANGHAITECH_ROOT", "/content/shanghaitech/SHANGHAI/SHANGHAI_Test")
)
FRAMES_DIR = SHANGHAITECH_ROOT / "frames"
VIDEO_LIST_PATH = Path("data/shanghaitech_meta.json")
MODEL_PATH = Path("models/finetuned_yolov8s_cctv.pt")
FEATURES_DIR = Path("data/features/shanghaitech")
META_OUT = Path("data/features/shanghaitech_meta.csv")

FRAME_WIDTH = 856
FRAME_HEIGHT = 480
FEATURE_NAMES = ["normalized_count", "flow_magnitude", "direction_entropy", "stop_ratio"]


def load_video_list() -> List[Dict[str, Any]]:
    with open(VIDEO_LIST_PATH) as f:
        return json.load(f)


def load_existing_meta() -> List[List[Any]]:
    """Resume support: reload previously-written summary rows so a resumed
    run's final summary still covers videos processed in earlier sessions.
    """
    rows: List[List[Any]] = []
    if META_OUT.exists():
        with open(META_OUT, newline="") as f:
            for row in csv.DictReader(f):
                rows.append([row["vid"], int(row["n_frames"]), float(row["mean_persons"]), row["scene"]])
    return rows


def process_one_video(
    vid: str, model: YOLO, fe: FeatureExtractor, device: torch.device
) -> tuple[np.ndarray, float] | tuple[None, None]:
    video_dir = FRAMES_DIR / vid
    frame_files = sorted(video_dir.glob("*.jpg"))
    if not frame_files:
        print(f"WARNING: no frames found for {vid} at {video_dir}, skipping")
        return None, None

    fe.reset()
    if hasattr(model.predictor, "trackers"):
        model.predictor.trackers[0].reset()

    n_frames = len(frame_files)
    features = np.zeros((n_frames, 64), dtype=np.float32)
    n_persons_per_frame = np.zeros(n_frames, dtype=np.int64)

    for i, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"WARNING: failed to read {frame_path}, treating as empty frame")
            features[i] = fe.extract([])
            continue

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
        features[i] = fe.extract(detections)
        n_persons_per_frame[i] = len(detections)

    return features, float(n_persons_per_frame.mean())


def process_all_videos() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Reading frames from: {FRAMES_DIR}")
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"{FRAMES_DIR} does not exist -- set SHANGHAITECH_ROOT to the extraction directory"
        )

    videos = load_video_list()
    print(f"{len(videos)} videos in manifest")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    meta_rows = load_existing_meta()
    already_done = {row[0] for row in meta_rows}
    if already_done:
        print(f"Resuming: {len(already_done)} videos already processed, will skip them")

    model = YOLO(str(MODEL_PATH))
    fe = FeatureExtractor(FRAME_WIDTH, FRAME_HEIGHT)

    pbar = tqdm(videos, desc="videos")
    for video in pbar:
        vid = video["vid"]
        out_path = FEATURES_DIR / f"{vid}.npy"
        if vid in already_done or out_path.exists():
            continue

        features, mean_persons = process_one_video(vid, model, fe, device)
        if features is None:
            continue

        np.save(out_path, features)
        scene = vid.split("_")[0]
        meta_rows.append([vid, features.shape[0], mean_persons, scene])
        with open(META_OUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vid", "n_frames", "mean_persons", "scene"])
            writer.writerows(meta_rows)

        pbar.set_postfix(vid=vid, n_frames=features.shape[0])
        tqdm.write(f"{vid}: {features.shape[0]} frames, scene={scene}")

    print("\n=== Summary ===")
    print(f"Videos with saved features: {len(meta_rows)}/{len(videos)}")

    scene_stats: Dict[str, List[float]] = {}
    for vid, n_frames, mean_persons, scene in meta_rows:
        scene_stats.setdefault(scene, []).append(mean_persons)
    print("\nMean persons/frame per scene:")
    for scene in sorted(scene_stats):
        vals = np.array(scene_stats[scene])
        print(f"  scene {scene}: n_videos={len(vals):>3}  mean_persons={vals.mean():.3f}")

    total_frames = sum(row[1] for row in meta_rows)
    overall_mean_persons = np.average(
        [row[2] for row in meta_rows], weights=[row[1] for row in meta_rows]
    )
    print(f"\nTotal frames processed: {total_frames}")
    print(f"Overall mean persons/frame: {overall_mean_persons:.3f}")


if __name__ == "__main__":
    process_all_videos()
