"""
Thread-safe holders for state shared between InferPipe threads.

SharedROIState — one ROI, applied to every DL branch that needs it.
  Only one InferPipe (the "master") calls ROIGenerator.step() per iteration
  and publishes here; other pipes ("followers") read the same value instead
  of stepping the generator themselves. This avoids advancing the mock CAN
  reader twice per frame and guarantees identical ROI geometry across models.
  A monotonically increasing seq is stamped on each publish; followers can
  block until seq advances (rough per-frame synchronisation).

SharedFeedbackState — cross-subflow feedback for the ROI generator.
  In the parallel-flow config the lane subflow (the ROI master) only
  produces lane_info; the object subflow (a follower) produces detections.
  The master's ROIGenerator.step() needs both — this class lets each
  subflow write what it produces and the master read the union.
"""

from __future__ import annotations

import threading

from .dynamic_roi import LaneInfo, ROIParameters


class SharedROIState:
    """One ROI + seq counter, updated by the master, read by followers."""

    def __init__(self, fallback: ROIParameters | None = None):
        # Full-frame fallback for frame 0 (before master has published).
        # set_hw_roi() clamps this into a per-subflow valid range if needed.
        self._roi = fallback or ROIParameters(
            x_left=0.0, y_top=0.0, width=1.0, height=1.0
        )
        self._seq = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def set(self, roi: ROIParameters) -> int:
        """Publish a new ROI. Returns the seq assigned to it."""
        with self._cond:
            self._seq += 1
            self._roi = roi
            self._cond.notify_all()
            return self._seq

    def get(self) -> tuple[ROIParameters, int]:
        """Non-blocking current-value read."""
        with self._lock:
            return self._roi, self._seq

    def wait_next(self, last_seq: int, timeout: float) -> tuple[ROIParameters, int]:
        """Block until seq > last_seq or timeout elapses.

        Returns the current (roi, seq) regardless of whether a fresh publish
        arrived — timing out just means the follower stays with what it has.
        """
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout=timeout)
            return self._roi, self._seq


class SharedFeedbackState:
    """Cross-subflow feedback used to feed the ROI generator.

    The lane master writes lane_info; the object follower writes objects.
    The master reads both in its next iteration when calling step().
    """

    def __init__(self):
        self._lane_info: LaneInfo | None = LaneInfo(
            center_norm=None, width_norm=None, confidence=0.0
        )
        self._objects: list = []
        self._lock = threading.Lock()

    def set_lane_info(self, lane_info: LaneInfo | None) -> None:
        if lane_info is None:
            return
        with self._lock:
            self._lane_info = lane_info

    def set_objects(self, objects) -> None:
        if objects is None:
            return
        with self._lock:
            self._objects = list(objects)

    def get_lane_info(self) -> LaneInfo:
        with self._lock:
            return self._lane_info

    def get_objects(self) -> list:
        with self._lock:
            return list(self._objects)
