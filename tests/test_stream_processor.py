"""Tests for src.stream_processor.SequenceBuffer's frame_id-based resampling."""

import numpy as np

from src.stream_processor import SequenceBuffer


def _feat(value: float) -> np.ndarray:
    return np.array([value], dtype=np.float32)


def test_no_drops_matches_fixed_stride_positions():
    # seq_len=3, stride=5 -> span=10. Pushing every consecutive frame_id
    # 1..11 (no drops) must reproduce the old position-based arr[::stride]
    # result exactly: samples at frame_id 1, 6, 11.
    buf = SequenceBuffer(seq_len=3, stride=5, feature_dim=1)
    for frame_id in range(1, 12):
        buf.push(frame_id, _feat(frame_id))
        if frame_id < 11:
            assert not buf.is_ready()
    assert buf.is_ready()
    np.testing.assert_array_equal(buf.get_sequence(), [[1], [6], [11]])


def test_dropped_frames_forward_fill_instead_of_compressing_window():
    # Same seq_len/stride, but frames 6-9 are dropped (never pushed). The old
    # position-based scheme would have used whatever the 6th successfully
    # pushed entry was (frame 10) as if it were 5 video-frames after frame 1
    # -- silently compressing 9 video-frames of real gap into a 5-frame-wide
    # sample slot. The fix must instead forward-fill target frame_id 6 from
    # the last known state (frame 5), keeping sample spacing tied to actual
    # video-frame indices.
    buf = SequenceBuffer(seq_len=3, stride=5, feature_dim=1)
    for frame_id in [1, 2, 3, 4, 5, 10, 11]:
        buf.push(frame_id, _feat(frame_id))
    assert buf.is_ready()
    np.testing.assert_array_equal(buf.get_sequence(), [[1], [5], [11]])


def test_is_ready_reflects_frame_id_span_not_push_count():
    # Under drops, readiness must depend on video-frame span elapsed, not on
    # how many pushes happened -- otherwise warmup would wait for successful
    # pushes that a high drop-rate stream may take far longer than the real
    # video-frame span to produce.
    buf = SequenceBuffer(seq_len=3, stride=5, feature_dim=1)
    buf.push(1, _feat(1))
    buf.push(11, _feat(11))  # only 2 pushes, but frame_id span is exactly 10
    assert buf.is_ready()


def test_reset_clears_state():
    buf = SequenceBuffer(seq_len=3, stride=5, feature_dim=1)
    for frame_id in range(1, 12):
        buf.push(frame_id, _feat(frame_id))
    assert buf.is_ready()
    buf.reset()
    assert not buf.is_ready()
