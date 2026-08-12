"""Three-level alerting (GREEN/AMBER/RED) with temporal persistence, driven
by one forecast head's probability stream.

Persistence exists because an instantaneous threshold crossing on a single
noisy frame is a false-alarm generator, not a signal. Rise and fall use
separate, asymmetric windows on purpose: rising escalates fast (default 15
frames = 0.5s at 30fps) so a real onset isn't missed waiting for
confirmation, while falling is deliberately slower (default 45 frames) so
the alert doesn't flicker off mid-event just because probability dipped
below threshold for a few noisy frames.
"""

import csv
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.train_forecast_model import HORIZONS

DEFAULT_CSV_PATH = Path("outputs/reports/alert_log.csv")
WEBHOOK_URL_ENV_VAR = "CROWD_RISK_ALERT_WEBHOOK_URL"  # unset by default; never hardcode a URL here


class AlertLevel(IntEnum):
    GREEN = 0
    AMBER = 1
    RED = 2


@dataclass
class AlertEvent:
    timestamp: float
    frame_id: int
    old_level: AlertLevel
    new_level: AlertLevel
    probabilities: Dict[int, float] = field(default_factory=dict)


Sink = Callable[[AlertEvent], None]


class AlertManager:
    def __init__(
        self,
        horizon: int = 60,
        amber_threshold: float = 0.4,
        red_threshold: float = 0.6,
        rise_persistence: int = 15,
        fall_persistence: int = 45,
        csv_path: Path = DEFAULT_CSV_PATH,
        webhook_url: Optional[str] = "unset",
        extra_sinks: Optional[List[Sink]] = None,
    ) -> None:
        assert horizon in HORIZONS, f"horizon {horizon} not in {HORIZONS}"
        assert 0.0 <= amber_threshold < red_threshold <= 1.0, "require 0 <= amber < red <= 1"
        self.horizon = horizon
        self.amber_threshold = amber_threshold
        self.red_threshold = red_threshold
        self.rise_persistence = rise_persistence
        self.fall_persistence = fall_persistence
        self.csv_path = Path(csv_path)
        # "unset" sentinel (not None) distinguishes "caller didn't pass
        # anything, read from config/env" from "caller explicitly disabled
        # the webhook" -- both leave it off, but only the former checks env.
        self.webhook_url = os.environ.get(WEBHOOK_URL_ENV_VAR) if webhook_url == "unset" else webhook_url

        self.current_level = AlertLevel.GREEN
        self._candidate_level: Optional[AlertLevel] = None
        self._candidate_count = 0
        self.events: List[AlertEvent] = []

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._csv_file = open(self.csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if write_header:
            self._csv_writer.writerow(
                ["timestamp", "frame_id", "old_level", "new_level"] + [f"prob_k{k}" for k in HORIZONS]
            )
            self._csv_file.flush()

        self._sinks: List[Sink] = [self._log_csv]
        if self.webhook_url:
            self._sinks.append(self._fire_webhook)
        if extra_sinks:
            self._sinks.extend(extra_sinks)

    def add_sink(self, sink: Sink) -> None:
        self._sinks.append(sink)

    def _level_for_probability(self, p: float) -> AlertLevel:
        if p >= self.red_threshold:
            return AlertLevel.RED
        if p >= self.amber_threshold:
            return AlertLevel.AMBER
        return AlertLevel.GREEN

    def update(self, frame_id: int, probabilities: Dict[int, float]) -> AlertLevel:
        """Feed one frame's full multi-horizon probability dict. Returns the
        current level (unchanged unless persistence was just satisfied)."""
        p = probabilities.get(self.horizon)
        if p is None:
            return self.current_level  # e.g. SequenceBuffer not warmed up yet

        target_level = self._level_for_probability(p)
        if target_level == self.current_level:
            self._candidate_level = None
            self._candidate_count = 0
            return self.current_level

        required = (
            self.rise_persistence if target_level > self.current_level else self.fall_persistence
        )
        if target_level == self._candidate_level:
            self._candidate_count += 1
        else:
            self._candidate_level = target_level
            self._candidate_count = 1

        if self._candidate_count >= required:
            self._transition(frame_id, target_level, probabilities)
            self._candidate_level = None
            self._candidate_count = 0

        return self.current_level

    def _transition(self, frame_id: int, new_level: AlertLevel, probabilities: Dict[int, float]) -> None:
        event = AlertEvent(
            timestamp=time.time(),
            frame_id=frame_id,
            old_level=self.current_level,
            new_level=new_level,
            probabilities=dict(probabilities),
        )
        self.events.append(event)
        self.current_level = new_level
        for sink in self._sinks:
            sink(event)

    def _log_csv(self, event: AlertEvent) -> None:
        row = [event.timestamp, event.frame_id, event.old_level.name, event.new_level.name] + [
            event.probabilities.get(k, "") for k in HORIZONS
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _fire_webhook(self, event: AlertEvent) -> None:
        import requests  # imported lazily: only needed if a URL is actually configured

        payload = {
            "timestamp": event.timestamp,
            "frame_id": event.frame_id,
            "old_level": event.old_level.name,
            "new_level": event.new_level.name,
            "probabilities": {str(k): v for k, v in event.probabilities.items()},
        }
        try:
            requests.post(self.webhook_url, json=payload, timeout=2.0)
        except Exception as exc:  # webhook delivery must never crash the pipeline
            print(f"alerting: webhook POST to {self.webhook_url} failed: {exc}")

    def close(self) -> None:
        self._csv_file.close()

    def __enter__(self) -> "AlertManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


if __name__ == "__main__":
    # Synthetic self-test of the persistence state machine, no stream needed.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        am = AlertManager(csv_path=Path(tmp) / "alert_log.csv", webhook_url=None)
        # Ramp 0 -> 0.7 -> 0 over 200 frames; expect a rise to RED then a
        # slower fall back to GREEN.
        for t in range(200):
            if t < 20:
                p = 0.1
            elif t < 100:
                p = 0.7
            else:
                p = 0.1
            level = am.update(t, {k: p for k in HORIZONS})
        print(f"final level: {am.current_level.name}")
        for e in am.events:
            print(f"  frame {e.frame_id:>4}: {e.old_level.name} -> {e.new_level.name}")
