"""Tests for src.ops_console's Hub coalescing and camera-spec parsing."""

from src.ops_console import Hub, parse_cameras


def test_hub_returns_only_latest_frame_per_camera():
    # A slow SSE client must get each camera's newest packet, never a backlog.
    hub = Hub()
    for i in range(5):
        hub.publish_frame("cam01", {"frame_id": i})
    hub.publish_frame("cam02", {"frame_id": 99})
    version, frames, alerts = hub.wait(0, timeout=0.1)
    assert sorted(f["frame_id"] for f in frames) == [4, 99]
    assert alerts == []

    hub.publish_alert({"cam": "cam01", "new": "RED"})
    _, frames, alerts = hub.wait(version, timeout=0.1)
    assert frames == []
    assert [a["new"] for a in alerts] == ["RED"]


def test_hub_wait_times_out_with_nothing_new():
    hub = Hub()
    hub.publish_frame("cam01", {"frame_id": 1})
    version, _, _ = hub.wait(0, timeout=0.1)
    assert hub.wait(version, timeout=0.05) == (version, [], [])


def test_parse_cameras_named_and_bare_specs():
    assert parse_cameras(["Gate A=rtsp://host:8554/umn"]) == [("Gate A", "rtsp://host:8554/umn")]
    # '=' inside a URL query string is not a NAME= prefix
    assert parse_cameras(["rtsp://host/s?a=b"]) == [("s?a=b", "rtsp://host/s?a=b")]
    assert parse_cameras(["clips/001.mp4"]) == [("001", "clips/001.mp4")]
