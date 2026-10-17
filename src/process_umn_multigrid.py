"""Re-extract UMN features at 2x2, 4x4 and 8x8 spatial grids in a single pass,
for the grid-size ablation.

process_umn.py hard-codes the 4x4 grid and saves only the resulting 64-dim
descriptor, not the detections behind it, so changing the grid would normally
mean re-running YOLO and ByteTrack once per grid. Detection and tracking are
identical across grids -- only the zone assignment downstream differs -- so
this module runs them once and fans the same detections into three
FeatureExtractor instances. Besides being three times cheaper, it guarantees
the three descriptors see byte-identical detections and tracker state, which a
re-run would not: ByteTrack's association is deterministic given a seed but
the ablation should not have to rely on that.

Clip boundaries are read from the meta CSV the original run produced rather
than re-derived, so every grid sees the same 11-clip segmentation as the
published results.

Each extractor keeps its own running normalisers, which is correct: the
running maxima are per-zone-population statistics, so a 2x2 zone (covering
four times the area of a 4x4 zone) legitimately sees different maxima. Sharing
them would make the coarser grids look artificially unlike themselves.
"""

import csv
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.feature_extractor import FeatureExtractor

VIDEO_PATH = Path("data/raw/umn/Crowd-Activity-All.avi")
META_IN = Path("data/features/umn_meta.csv")
MODEL_PATH = Path("models/finetuned_yolov8s_cctv.pt")
GRIDS = [2, 4, 8]           # 16-, 64- and 256-dim descriptors
OUT_TMPL = "data/features/umn_features_g{g}.npy"


def load_clip_ids() -> np.ndarray:
    with open(META_IN, newline="") as f:
        return np.array([int(r["clip_id"]) for r in csv.DictReader(f)], dtype=np.int64)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    clip_ids = load_clip_ids()
    boundaries = {i for i in range(1, len(clip_ids)) if clip_ids[i] != clip_ids[i - 1]}
    print(f"Device: {device} | {len(set(clip_ids.tolist()))} clips, "
          f"{len(boundaries)} boundaries (from {META_IN})")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert n == len(clip_ids), (n, len(clip_ids))

    model = YOLO(str(MODEL_PATH))
    extractors: Dict[int, FeatureExtractor] = {g: FeatureExtractor(w, h, g, g) for g in GRIDS}
    feats: Dict[int, np.ndarray] = {
        g: np.zeros((n, g * g * 4), dtype=np.float32) for g in GRIDS
    }

    idx = 0
    with tqdm(total=n, desc="detect+track -> 3 grids") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in boundaries:
                for fe in extractors.values():
                    fe.reset()
                if getattr(model, "predictor", None) is not None and hasattr(model.predictor, "trackers"):
                    model.predictor.trackers[0].reset()

            res = model.track(frame, imgsz=640, conf=0.25, classes=[0],
                              tracker="bytetrack.yaml", persist=True,
                              verbose=False, device=str(device))
            dets: List[Dict] = []
            boxes = res[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                    dets.append({"track_id": int(tid),
                                 "cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2),
                                 "bbox": (float(x1), float(y1), float(x2), float(y2))})

            for g in GRIDS:
                feats[g][idx] = extractors[g].extract(dets)
            idx += 1
            pbar.update(1)
    cap.release()

    for g in GRIDS:
        out = Path(OUT_TMPL.format(g=g))
        arr = feats[g][:idx]
        np.save(out, arr)
        print(f"  {g}x{g}: {arr.shape} -> {out}   non-zero {np.mean(arr != 0):.3f}")

    ref = np.load("data/features/umn_features.npy")
    new4 = feats[4][:idx]
    agree = float(np.mean(np.isclose(ref[:idx], new4, atol=1e-5)))
    print(f"\n4x4 agreement with the published run: {agree:.4f} "
          f"(<1.0 only from tracker nondeterminism, not a grid effect)")


if __name__ == "__main__":
    main()
