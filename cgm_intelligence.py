"""Deterministic, read-only CGM intelligence layer.

This module provides conservative signal processing and alerting helpers
for personal glucose monitoring. Logic is deterministic and explainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable, Optional, Sequence
import logging


@dataclass(frozen=True)
class GlucoseReading:
    """Immutable CGM data point with optional context signals."""

    timestamp: datetime
    glucose_mg_dl: float
    trend_mg_dl_per_min: Optional[float] = None
    heart_rate: Optional[int] = None
    motion: Optional[float] = None


@dataclass(frozen=True)
class DecisionResult:
    """Alert decision payload suitable for logging/auditing."""

    should_alert: bool
    reason: str


def ewma_smooth(values: Sequence[float], alpha: float = 0.3) -> list[float]:
    """Apply exponential weighted moving average smoothing.

    Args:
        values: Ordered numeric signal samples.
        alpha: Smoothing factor in range (0, 1].

    Returns:
        Smoothed series with same length as input.
    """
    if not values:
        return []
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in the range (0, 1]")

    smoothed = [float(values[0])]
    for current in values[1:]:
        smoothed.append(alpha * float(current) + (1 - alpha) * smoothed[-1])
    return smoothed


def _validate_recent_readings(readings: Sequence[GlucoseReading]) -> bool:
    if len(readings) < 4:
        return False

    recent = readings[-4:]
    timestamps = [r.timestamp for r in recent]
    return timestamps == sorted(timestamps)


def is_probable_compression_low(
    readings: Sequence[GlucoseReading],
    night_hours: tuple[int, int] = (23, 6),
    sharp_drop_mg_dl: float = 15.0,
    low_threshold: float = 75.0,
    motion_threshold: float = 0.05,
) -> bool:
    """Conservative heuristic for pressure-induced false lows."""
    if not _validate_recent_readings(readings):
        return False

    latest = readings[-1]
    recent = readings[-4:]

    if latest.glucose_mg_dl > low_threshold:
        return False

    # Derivatives over recent samples.
    deltas = [recent[i].glucose_mg_dl - recent[i - 1].glucose_mg_dl for i in range(1, len(recent))]
    sharp_drop = any(delta < -sharp_drop_mg_dl for delta in deltas)

    start_hour, end_hour = night_hours
    hour = latest.timestamp.hour
    is_night = hour >= start_hour or hour <= end_hour

    low_motion = latest.motion is not None and latest.motion < motion_threshold

    return sharp_drop and is_night and low_motion


def predict_low_within_minutes(
    readings: Sequence[GlucoseReading],
    horizon_min: int = 30,
    low_threshold: float = 70.0,
    max_reasonable_fall_rate: float = -8.0,
) -> bool:
    """Linear projection warning for upcoming hypoglycemia.

    Includes a sanity bound to avoid alerting from implausible trend spikes.
    """
    if len(readings) < 3:
        return False

    latest = readings[-1]
    if latest.trend_mg_dl_per_min is None:
        return False

    if latest.trend_mg_dl_per_min < max_reasonable_fall_rate:
        # Defensive: trend likely sensor artifact; avoid over-alerting.
        return False

    projected = latest.glucose_mg_dl + (latest.trend_mg_dl_per_min * horizon_min)
    return projected <= low_threshold


class AlertDecisionEngine:
    """Deterministic alert engine with conservative suppression and cooldown."""

    def __init__(self, cooldown_minutes: int = 15) -> None:
        self.last_alert_time: Optional[datetime] = None
        self.cooldown_minutes = cooldown_minutes

    def should_alert(self, readings: Sequence[GlucoseReading], now: datetime) -> DecisionResult:
        if not readings:
            return DecisionResult(False, "Suppressed: no readings available")

        latest = readings[-1]

        if is_probable_compression_low(readings):
            return DecisionResult(False, "Suppressed: probable compression low")

        if predict_low_within_minutes(readings):
            if self._in_cooldown(now):
                return DecisionResult(False, "Suppressed: predictive low during cooldown")
            self.last_alert_time = now
            return DecisionResult(True, "Predictive low in next 30 minutes")

        recent = [x.glucose_mg_dl for x in readings[-3:]]
        if len(recent) == 3 and all(v < 70 for v in recent):
            if self._in_cooldown(now):
                return DecisionResult(False, "Suppressed: sustained low during cooldown")
            self.last_alert_time = now
            return DecisionResult(True, "Sustained low glucose")

        # Optional noise labeling from robust local median.
        if len(readings) >= 5:
            local = [x.glucose_mg_dl for x in readings[-5:]]
            if abs(latest.glucose_mg_dl - median(local)) > 40:
                return DecisionResult(False, "Suppressed: likely transient sensor noise")

        return DecisionResult(False, "No alert condition met")

    def _in_cooldown(self, now: datetime) -> bool:
        if self.last_alert_time is None:
            return False
        delta = now - self.last_alert_time
        return delta.total_seconds() < self.cooldown_minutes * 60


class DecisionLogger:
    """Medical-style decision logging helper."""

    def __init__(self, log_path: str = "cgm_decisions.log") -> None:
        self.log_path = Path(log_path)
        self.logger = logging.getLogger(f"cgm_intelligence.{self.log_path}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self.log_path)
            formatter = logging.Formatter("%(asctime)s | %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_decision(self, reading: GlucoseReading, result: DecisionResult) -> None:
        self.logger.info(
            "Timestamp=%s | Glucose=%.1f | Trend=%s | HR=%s | Motion=%s | Decision=%s | Reason=%s",
            _normalize_timestamp(reading.timestamp),
            reading.glucose_mg_dl,
            reading.trend_mg_dl_per_min,
            reading.heart_rate,
            reading.motion,
            result.should_alert,
            result.reason,
        )


def _normalize_timestamp(ts: datetime) -> str:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc).isoformat()
    return ts.astimezone(timezone.utc).isoformat()


def process_new_reading(
    readings: Sequence[GlucoseReading],
    engine: AlertDecisionEngine,
    logger: DecisionLogger,
    alert_callback: Optional[Callable[[str], None]] = None,
) -> DecisionResult:
    """Evaluate most recent reading and optionally emit user alert callback."""
    if not readings:
        result = DecisionResult(False, "Suppressed: no readings available")
        return result

    now = readings[-1].timestamp
    result = engine.should_alert(readings, now)
    logger.log_decision(readings[-1], result)

    if result.should_alert and alert_callback is not None:
        alert_callback(result.reason)

    return result
