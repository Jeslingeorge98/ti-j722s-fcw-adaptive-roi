"""
Integration smoke test — Adaptive ROI with mock CAN, no GStreamer.

Replays 50 frames from the indian_road1 CAN CSV through ROIGenerator and
asserts floor containment on every frame. No camera, display, or GStreamer
dependency — runs on any Linux PC.

Usage:
    cd ti-j722s-app-python\ 2
    python3 tests/test_roi_integration_smoke.py
"""

import sys
import os

# Allow imports from apps_python without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps_python"))

import roi.dynamic_roi as m
from roi.dynamic_roi import (
    CameraIntrinsics,
    ROIGenerator,
    LaneInfo,
    ROIParameters,
    _invariant_floor,
)
from can_interface import CANSignalReader

# ---------------------------------------------------------------------------
# Config — C270 calibration values from camera_calibration_c270.json
# ---------------------------------------------------------------------------
CAN_CSV = "/home/jeslin/Indian_conditions/can_signals_indian_road1.csv"
N_FRAMES = 50

CAMERA = CameraIntrinsics(
    focal_px=1407.8,
    principal_x_px=636.0,
    principal_y_px=350.4,
    image_width_px=1280,
    image_height_px=720,
    mount_height_m=1.5,
)

FALLBACK_ROI = ROIParameters(x_left=0.0, y_top=0.0, width=1.0, height=1.0)

# ---------------------------------------------------------------------------
# No-lane LaneInfo — simulates frames where lane detection has no output
# ---------------------------------------------------------------------------
NO_LANE = LaneInfo(center_norm=None, width_norm=None, confidence=0.0)


def _check_floor_containment(roi: ROIParameters, sig) -> bool:
    """Return True if roi fully covers the physics-based invariant floor."""
    floor_xl, floor_xr, floor_yt, floor_yb = _invariant_floor(
        camera=CAMERA,
        speed_mps=sig.speed_mps,
        curvature_inv_m=0.0,
        abs_active=sig.abs_active,
    )
    return (
        roi.x_left              <= floor_xl + 1e-6
        and roi.x_left + roi.width  >= floor_xr - 1e-6
        and roi.y_top               <= floor_yt + 1e-6
        and roi.y_top + roi.height  >= floor_yb - 1e-6
    )


def run():
    reader = CANSignalReader(mode="mock", csv_path=CAN_CSV)
    gen    = ROIGenerator(camera=CAMERA, isa_enabled=False)

    floor_pass = 0
    print(f"\n{'Frame':>5}  {'speed_kmh':>10}  {'roi_level':>9}  "
          f"{'area%':>6}  {'warmed':>6}  {'floor_ok':>8}")
    print("-" * 56)

    for frame in range(N_FRAMES):
        sig = reader.get_latest()
        roi = gen.step(NO_LANE, sig, FALLBACK_ROI, objects=[])

        floor_ok = _check_floor_containment(roi, sig)
        if floor_ok:
            floor_pass += 1

        area_pct = roi.width * roi.height * 100
        print(f"{frame:>5}  {sig.speed_mps * 3.6:>10.1f}  "
              f"L{roi.roi_level:>8}  {area_pct:>6.1f}%  "
              f"{'yes':>6}  {'✓' if floor_ok else '✗':>8}")

        assert roi.width  > 0,  f"Frame {frame}: zero-width ROI"
        assert roi.height > 0,  f"Frame {frame}: zero-height ROI"
        assert 0.0 <= roi.x_left <= 1.0, f"Frame {frame}: x_left out of range"
        assert 0.0 <= roi.y_top  <= 1.0, f"Frame {frame}: y_top out of range"
        assert floor_ok, (
            f"Frame {frame}: floor containment failed — "
            f"roi=({roi.x_left:.3f},{roi.y_top:.3f},{roi.width:.3f},{roi.height:.3f}), "
            f"speed={sig.speed_mps * 3.6:.1f} km/h"
        )

    print("-" * 56)
    print(f"\nFloor containment: {floor_pass}/{N_FRAMES} frames ({100*floor_pass/N_FRAMES:.0f}%)")
    print(f"No exceptions over {N_FRAMES} frames.\n")


if __name__ == "__main__":
    run()
