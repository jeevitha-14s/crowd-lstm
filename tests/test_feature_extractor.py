"""Tests for src.feature_extractor.FeatureExtractor using synthetic detections."""

from typing import Any, Dict, List

import numpy as np

from src.feature_extractor import FeatureExtractor

FRAME_W, FRAME_H = 320, 240  # matches UMN resolution
GRID_ROWS, GRID_COLS = 4, 4
ZONE_W, ZONE_H = FRAME_W / GRID_COLS, FRAME_H / GRID_ROWS


def _det(track_id: int, cx: float, cy: float) -> Dict[str, Any]:
    return {"track_id": track_id, "cx": cx, "cy": cy, "bbox": (cx - 5, cy - 5, cx + 5, cy + 5)}


def _zone_center(row: int, col: int) -> tuple:
    return (col * ZONE_W + ZONE_W / 2, row * ZONE_H + ZONE_H / 2)


def _make_extractor() -> FeatureExtractor:
    return FeatureExtractor(FRAME_W, FRAME_H, grid_rows=GRID_ROWS, grid_cols=GRID_COLS)


def test_all_persons_in_one_zone_only_that_zone_nonzero():
    fe = _make_extractor()
    row, col = 1, 2
    cx, cy = _zone_center(row, col)
    dets = [_det(tid, cx + tid, cy) for tid in range(5)]  # small jitter, same zone

    features = fe.extract(dets)

    target_zone = row * GRID_COLS + col
    for zone_idx in range(GRID_ROWS * GRID_COLS):
        base = zone_idx * 4
        if zone_idx == target_zone:
            assert features[base + 0] > 0.0
            # single observation per track -> no motion features yet
            assert features[base + 1] == 0.0
            assert features[base + 2] == 0.0
            assert features[base + 3] == 0.0
        else:
            assert np.all(features[base : base + 4] == 0.0)


def test_tracks_moving_same_direction_low_entropy():
    fe = _make_extractor()
    row, col = 0, 0
    cx0, cy0 = _zone_center(row, col)

    dets_frame1 = [_det(tid, cx0 + tid, cy0 + tid) for tid in range(4)]
    fe.extract(dets_frame1)

    # all tracks move +10px in x, 0 in y (same direction)
    dets_frame2 = [_det(tid, cx0 + tid + 10, cy0 + tid) for tid in range(4)]
    features = fe.extract(dets_frame2)

    target_zone = row * GRID_COLS + col
    entropy_val = features[target_zone * 4 + 2]
    assert entropy_val < 0.1


def test_tracks_moving_in_all_8_directions_high_entropy():
    fe = _make_extractor()
    row, col = 2, 2
    cx0, cy0 = _zone_center(row, col)

    angles = [i * (2 * np.pi / 8) for i in range(8)]
    dets_frame1 = [_det(tid, cx0, cy0) for tid in range(8)]
    fe.extract(dets_frame1)

    step = 10.0
    dets_frame2 = [
        _det(tid, cx0 + step * np.cos(a), cy0 + step * np.sin(a))
        for tid, a in enumerate(angles)
    ]
    features = fe.extract(dets_frame2)

    target_zone = row * GRID_COLS + col
    entropy_val = features[target_zone * 4 + 2]
    assert entropy_val > 0.9


def test_all_stationary_tracks_stop_ratio_one_flow_zero():
    fe = _make_extractor()
    row, col = 3, 1
    cx0, cy0 = _zone_center(row, col)

    dets = [_det(tid, cx0, cy0) for tid in range(5)]
    fe.extract(dets)
    features = fe.extract(dets)  # same positions again -> zero speed

    target_zone = row * GRID_COLS + col
    flow_val = features[target_zone * 4 + 1]
    stop_ratio_val = features[target_zone * 4 + 3]
    assert flow_val == 0.0
    assert stop_ratio_val == 1.0


def test_running_max_speed_lags_by_one_frame():
    """A same-frame spike should not be able to normalize itself away."""
    fe = _make_extractor()
    row, col = 0, 3
    cx0, cy0 = _zone_center(row, col)
    tid = 0

    fe.extract([_det(tid, cx0, cy0)])
    fe.extract([_det(tid, cx0 + 2, cy0)])  # small speed, establishes a low ceiling
    ceiling_before_spike = fe.running_max_speed

    # sudden large jump (panic onset)
    features = fe.extract([_det(tid, cx0 + 60, cy0)])
    target_zone = row * GRID_COLS + col
    flow_val = features[target_zone * 4 + 1]

    # divisor used for this frame must still be the pre-spike ceiling
    assert fe.running_max_speed > ceiling_before_spike
    assert flow_val == 1.0  # spike clipped at ceiling that hadn't yet absorbed it


def test_stop_ratio_uses_relative_threshold():
    """A track moving well below the clip's typical pace is flagged as stopped;
    a track moving at the typical pace is not, once a baseline is established.
    """
    fe = _make_extractor()
    row, col = 1, 1
    cx0, cy0 = _zone_center(row, col)
    typical_ids = [0, 1, 2]
    slow_id = 3

    fe.extract([_det(tid, cx0, cy0) for tid in typical_ids + [slow_id]])
    # typical tracks move 10px/frame; slow track moves 1px/frame
    fe.extract(
        [_det(tid, cx0 + 10, cy0) for tid in typical_ids]
        + [_det(slow_id, cx0 + 1, cy0)]
    )
    # baseline (median speed ~10) is now established but not yet applied
    # (lagged); one more frame of the same pace lets it take effect
    features = fe.extract(
        [_det(tid, cx0 + 20, cy0) for tid in typical_ids]
        + [_det(slow_id, cx0 + 2, cy0)]
    )

    target_zone = row * GRID_COLS + col
    stop_ratio_val = features[target_zone * 4 + 3]
    # only the slow track (1 of 4) should register as stopped
    assert abs(stop_ratio_val - 0.25) < 1e-6


def test_running_median_speed_robust_to_single_outlier():
    """A single mistracked/teleported detection should barely move the
    median-based reference, unlike a max-based one.
    """
    fe = _make_extractor()
    row, col = 2, 1
    cx0, cy0 = _zone_center(row, col)
    typical_ids = list(range(5))
    outlier_id = 5
    slow_id = 6

    all_ids = typical_ids + [outlier_id, slow_id]
    fe.extract([_det(tid, cx0, cy0) for tid in all_ids])
    # frame 2: everyone (including the future outlier) moves at a normal pace
    fe.extract(
        [_det(tid, cx0 + 10, cy0) for tid in typical_ids]
        + [_det(outlier_id, cx0 + 10, cy0)]
        + [_det(slow_id, cx0 + 0.5, cy0)]
    )
    # frame 3: typical tracks continue normally; outlier teleports (ID switch)
    # so far across the frame it lands in a different zone -- a realistic
    # symptom of a mistracked ID, not a test artifact; slow track continues
    # at its slow pace
    features = fe.extract(
        [_det(tid, cx0 + 20, cy0) for tid in typical_ids]
        + [_det(outlier_id, cx0 + 310, cy0)]  # huge jump: speed = 310/2 = 155
        + [_det(slow_id, cx0 + 1.0, cy0)]
    )

    target_zone = row * GRID_COLS + col
    stop_ratio_val = features[target_zone * 4 + 3]

    # median stayed anchored near the typical pace (~10), nowhere near the
    # outlier's speed (155) -- a max-based statistic would have jumped there
    assert fe.running_median_speed < 15.0
    assert fe.running_max_speed > 20.0  # contrast: the (zone-mean) max DID get pulled up (~29.4)
    # the outlier's jump moved it out of this zone (clamped by frame bounds),
    # leaving 6 tracks here; only the slow one should register as stopped
    assert abs(stop_ratio_val - 1.0 / 6.0) < 1e-6
