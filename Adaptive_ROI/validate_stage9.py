"""
Stage 9 Validation — Adaptive Shared ROI
=========================================
Produces the three measurements the manuscript's results section requires:

  1. Floor coverage recall  — does Final ROI ⊇ floor guarantee hold on real-ish scenarios?
  2. Baseline comparison    — adaptive vs. fixed static crop, same scenarios
  3. Region area distribution — how much of the image does each mode use?

Dataset: synthetic-but-defensible scenarios that cover the operating envelope
described in the paper (IRC 67 speed brackets, Indian road curvatures, truck
platform parameters). Defensibility argument: the scenario set is parameterized
from the paper's own stated operating domain, every parameter choice is
documented below, and the floor-coverage guarantee was verified exhaustively
at 10,584 combinations in Stage 8B — so what this validation adds is:
(a) realistic multi-frame sequences with dynamic objects, lane dropout, and
    degradation events, rather than single-frame synthetic checks; and
(b) the comparative fixed-crop baseline measurement, which is genuinely new.

All results are printed to stdout AND written to stage9_results.txt for
the review note and manuscript.
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import dynamic_roi as m

# ---------------------------------------------------------------------------
# Camera — representative truck/heavy vehicle platform
# ---------------------------------------------------------------------------
CAMERA = m.CameraIntrinsics(
    focal_px=1000.0,          # representative; real value from OEM calibration
    principal_x_px=960.0,
    principal_y_px=540.0,
    image_width_px=1920.0,
    image_height_px=1080.0,
    mount_height_m=2.5,       # truck camera mount height
)

# Fixed static crop baseline — centred, lower 60% of image, full width.
# Chosen as a reasonable, well-intentioned fixed crop a system designer
# might use without adaptive logic: covers the road ahead, avoids the sky,
# uses slightly less than full frame to reduce inference load.
STATIC_CROP = m.ROIParameters(x_left=0.0, y_top=0.4, width=1.0, height=0.6)

# Ground-truth "object within frame" minimum pixel requirement —
# below this size an object is considered undetectable regardless of ROI,
# so it's excluded from coverage recall rather than counted as a miss.
MIN_DETECTABLE_HEIGHT_NORM = 0.04  # ~43px at 1080p

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Scenario set
# ---------------------------------------------------------------------------

def make_can(speed_kmh, steer_deg=0.0, yaw_dps=0.0, abs_active=False, esc_active=False):
    return m.CanSignals(
        speed_mps=speed_kmh / 3.6,
        steering_angle_deg=steer_deg,
        yaw_rate_dps=yaw_dps,
        steering_valid=True,
        yaw_rate_valid=True,
        abs_active=abs_active,
        esc_active=esc_active,
    )

def make_lane(center=0.5, width=0.3, confidence=0.9, c2=None):
    return m.LaneInfo(
        center_norm=center,
        width_norm=width,
        confidence=confidence,
        c2_curvature=c2,
        c2_confidence=0.85 if c2 is not None else 0.0,
    )

def place_vehicle_at_range_m(range_m, lateral_offset_m=0.0):
    """
    Return a DetectedObject bbox for a vehicle at a given range,
    using the pinhole projection (h = fH/Z from the paper's own
    invariant floor formula), so the ground-truth positions are
    physically consistent with the same camera model used internally.
    """
    f = CAMERA.focal_px
    H_vehicle = 2.0   # typical passenger vehicle height, metres
    H_truck   = 3.8   # heavy vehicle
    W_vehicle = 1.8
    cx_px = CAMERA.principal_x_px + f * lateral_offset_m / range_m
    h_px  = f * H_vehicle / range_m
    w_px  = f * W_vehicle / range_m
    cx_n  = cx_px / CAMERA.image_width_px
    cy_n  = (CAMERA.principal_y_px + f * CAMERA.mount_height_m / range_m) / CAMERA.image_height_px
    h_n   = h_px / CAMERA.image_height_px
    w_n   = w_px / CAMERA.image_width_px
    return m.DetectedObject(
        category=m.ObjectCategory.VEHICLE,
        bbox=(
            max(0.0, cx_n - w_n/2),
            max(0.0, cy_n - h_n/2),
            min(1.0, cx_n + w_n/2),
            min(1.0, cy_n + h_n/2),
        ),
        confidence=0.9,
    )


SCENARIOS = {
    # ------------------------------------
    # HIGHWAY (100-120 km/h, straight)
    # ------------------------------------
    "highway_straight_far": {
        "description": "Highway 100km/h, vehicle at 120m (FCW envelope boundary)",
        "n_frames": 15,
        "can_params": {"speed_kmh": 100, "steer_deg": 0.0, "yaw_dps": 0.0},
        "lane_params": {"confidence": 0.9},
        "objects_fn": lambda _: [place_vehicle_at_range_m(120)],
        "degradation": None,
    },
    "highway_straight_closing": {
        "description": "Highway 120km/h, vehicle closing from 100m to 40m over 15 frames",
        "n_frames": 15,
        "can_params": {"speed_kmh": 120, "steer_deg": 0.0, "yaw_dps": 0.0},
        "lane_params": {"confidence": 0.9},
        "objects_fn": lambda i: [place_vehicle_at_range_m(max(40, 100 - i * 4))],
        "degradation": None,
    },
    "highway_lane_dropout": {
        "description": "Highway 100km/h, lane detection drops at frame 5 (CAN fallback test)",
        "n_frames": 12,
        "can_params": {"speed_kmh": 100, "steer_deg": 0.0, "yaw_dps": 0.0},
        "lane_params": {"confidence": 0.9},
        "objects_fn": lambda i: [place_vehicle_at_range_m(80)],
        "degradation": {"type": "lane_dropout", "start_frame": 5},
    },
    "highway_hard_braking": {
        "description": "Highway 100->60km/h hard braking, ABS active, vehicle at 60m",
        "n_frames": 10,
        "can_params": {"speed_kmh": 100, "steer_deg": 0.0, "yaw_dps": 0.0},
        "lane_params": {"confidence": 0.9},
        "objects_fn": lambda i: [place_vehicle_at_range_m(60)],
        "degradation": {"type": "braking", "start_frame": 3},
    },

    # ------------------------------------
    # STATE HIGHWAY / DISTRICT ROAD (60-80 km/h, mild curve)
    # ------------------------------------
    "state_road_mild_curve": {
        "description": "State road 80km/h, 150m radius curve (IRC 67 typical), vehicle at 70m",
        "n_frames": 12,
        "can_params": {"speed_kmh": 80, "steer_deg": 1.6, "yaw_dps": 3.8},
        "lane_params": {"confidence": 0.85, "c2": 1/150},
        "objects_fn": lambda _: [place_vehicle_at_range_m(70, lateral_offset_m=0.5)],
        "degradation": None,
    },
    "state_road_curvature_dropout": {
        "description": "State road 80km/h, curve, lane confidence drops mid-sequence",
        "n_frames": 12,
        "can_params": {"speed_kmh": 80, "steer_deg": 1.6, "yaw_dps": 3.8},
        "lane_params": {"confidence": 0.85, "c2": 1/150},
        "objects_fn": lambda i: [place_vehicle_at_range_m(60)],
        "degradation": {"type": "confidence_drop", "start_frame": 5, "low_conf": 0.3},
    },

    # ------------------------------------
    # URBAN / SLOW (below 40 km/h)
    # ------------------------------------
    "urban_slow_pedestrian_zone": {
        "description": "Urban 30km/h, near vehicle at 20m, slow approach",
        "n_frames": 10,
        "can_params": {"speed_kmh": 30, "steer_deg": 0.0, "yaw_dps": 0.0},
        "lane_params": {"confidence": 0.7},
        "objects_fn": lambda i: [place_vehicle_at_range_m(max(15, 20 - i))],
        "degradation": None,
    },
    "urban_can_dropout": {
        "description": "Urban 40km/h, CAN signals lost (both invalid), lane healthy",
        "n_frames": 10,
        "can_params": {"speed_kmh": 40, "steer_deg": 0.0, "yaw_dps": 0.0},
        "lane_params": {"confidence": 0.9},
        "objects_fn": lambda i: [place_vehicle_at_range_m(30)],
        "degradation": {"type": "can_dropout", "start_frame": 3},
    },

    # ------------------------------------
    # SIGN / ISA scenarios
    # ------------------------------------
    "highway_speed_sign": {
        "description": "Highway 100km/h, road sign at right edge (ISA scenario)",
        "n_frames": 10,
        "can_params": {"speed_kmh": 100, "steer_deg": 0.0, "yaw_dps": 0.0},
        "lane_params": {"confidence": 0.9},
        "objects_fn": lambda i: [
            place_vehicle_at_range_m(100),
            m.DetectedObject(category=m.ObjectCategory.SIGN_ROADSIDE,
                              bbox=(0.83, 0.42, 0.90, 0.55), confidence=0.85),
        ],
        "degradation": None,
    },
    "highway_sign_occluded": {
        "description": "Highway, large truck occludes roadside sign — sign memory test",
        "n_frames": 12,
        "can_params": {"speed_kmh": 90, "steer_deg": 0.0, "yaw_dps": 0.0},
        "lane_params": {"confidence": 0.9},
        "objects_fn": lambda i: (
            [m.DetectedObject(category=m.ObjectCategory.VEHICLE,
                               bbox=(0.35, 0.45, 0.65, 0.75), confidence=0.9)]
            if i >= 4 else
            [m.DetectedObject(category=m.ObjectCategory.SIGN_ROADSIDE,
                               bbox=(0.83, 0.42, 0.90, 0.55), confidence=0.85),
             m.DetectedObject(category=m.ObjectCategory.VEHICLE,
                               bbox=(0.35, 0.45, 0.65, 0.75), confidence=0.9)]
        ),
        "degradation": None,
    },

    # ------------------------------------
    # CUT-IN scenario
    # ------------------------------------
    "highway_cut_in": {
        "description": "Highway 100km/h, vehicle cuts in from outside corridor over 15 frames. "
                        "Frame 0: vehicle clearly outside corridor (cx=0.78) -- EXPECTED miss "
                        "since the system correctly withholds expansion before the vehicle enters. "
                        "Frames 1+: vehicle drifts in at IoU-trackable rate, expansion fires on entry.",
        "n_frames": 15,
        "can_params": {"speed_kmh": 100, "steer_deg": 0.0, "yaw_dps": 0.0},
        "lane_params": {"confidence": 0.9},
        "objects_fn": lambda i: [
            m.DetectedObject(
                category=m.ObjectCategory.VEHICLE,
                bbox=(
                    max(0, 0.78 - i * 0.008 - 0.03),
                    max(0, 0.54 - i * 0.008),
                    min(1, 0.78 - i * 0.008 + 0.03),
                    min(1, 0.54 - i * 0.008 + 0.05 + i * 0.01),
                ),
                confidence=0.9,
            )
        ],
        "degradation": None,
    },

    # ------------------------------------
    # FULLY DEGRADED (Level 3)
    # ------------------------------------
    "full_degradation_level3": {
        "description": "Dynamics failure (yaw mismatch), lane confidence zero — Level 3",
        "n_frames": 8,
        "can_params": {"speed_kmh": 80, "steer_deg": 0.0, "yaw_dps": 25.0},
        "lane_params": {"confidence": 0.0},
        "objects_fn": lambda i: [place_vehicle_at_range_m(60)],
        "degradation": None,
    },
}


# ---------------------------------------------------------------------------
# Run a single scenario, return per-frame records
# ---------------------------------------------------------------------------

def run_scenario(name, scenario):
    # IMPORTANT: fresh generator per scenario — smoothing state must not
    # bleed from one scenario into the next.
    gen = m.ROIGenerator(camera=CAMERA, isa_enabled=True)
    records = []

    for frame_i in range(scenario["n_frames"]):
        # Build signals, applying degradation if specified
        cp = dict(scenario["can_params"])
        lp = dict(scenario["lane_params"])
        deg = scenario.get("degradation")

        if deg and frame_i >= deg["start_frame"]:
            if deg["type"] == "lane_dropout":
                lp["confidence"] = 0.0
            elif deg["type"] == "confidence_drop":
                lp["confidence"] = deg["low_conf"]
            elif deg["type"] == "can_dropout":
                cp["steer_deg"] = None
                cp["yaw_dps"] = None
            elif deg["type"] == "braking":
                spd = max(60, cp["speed_kmh"] - (frame_i - deg["start_frame"]) * 8)
                cp["speed_kmh"] = spd
                cp["abs_active"] = True

        sig = m.CanSignals(
            speed_mps=cp["speed_kmh"] / 3.6,
            steering_angle_deg=cp.get("steer_deg"),
            yaw_rate_dps=cp.get("yaw_dps"),
            steering_valid=(cp.get("steer_deg") is not None),
            yaw_rate_valid=(cp.get("yaw_dps") is not None),
            abs_active=cp.get("abs_active", False),
            esc_active=cp.get("esc_active", False),
        )
        lane = m.LaneInfo(
            center_norm=lp.get("center", 0.5),
            width_norm=lp.get("width", 0.3),
            confidence=lp["confidence"],
            c2_curvature=lp.get("c2"),
            c2_confidence=0.85 if lp.get("c2") else 0.0,
        )

        objects = scenario["objects_fn"](frame_i)

        # --- Adaptive ROI ---
        roi = gen.step(lane, sig, STATIC_CROP, objects=objects)

        # --- Floor guarantee (ground truth) ---
        curvature_can = m._compute_curvature(sig)
        curvature, curvature_conf = m._fuse_curvature(curvature_can, lane, m._dynamics_confidence(sig))
        c0 = m._estimate_c0_m(lane, CAMERA)
        # Pass abs_active so the containment check uses the SAME floor
        # the ROIGenerator used internally -- ABS-active adds extra vertical
        # margin, so omitting it would make the check compare against a
        # smaller floor, causing false failures during hard-braking frames.
        effective_abs = sig.abs_active
        floor_xl, floor_xr, floor_yt, floor_yb = m._invariant_floor(
            sig.speed_mps, curvature, CAMERA,
            lane_c0_m=c0, confidence=curvature_conf,
            abs_active=effective_abs,
        )

        # --- Floor containment check ---
        floor_contained = (
            roi.x_left       <= floor_xl + 1e-6 and
            roi.x_left + roi.width  >= floor_xr - 1e-6 and
            roi.y_top        <= floor_yt + 1e-6 and
            roi.y_top + roi.height  >= floor_yb - 1e-6
        )

        # --- Object containment check ---
        # For the object recall metric, distinguish two kinds of adaptive miss:
        #   (a) EXPECTED: object outside corridor, system correctly withholds expansion
        #   (b) UNEXPECTED: object in-corridor and confirmed, but adaptive missed it
        obj_results = []
        for obj in objects:
            obj_cx = (obj.bbox[0] + obj.bbox[2]) / 2
            obj_cy = (obj.bbox[1] + obj.bbox[3]) / 2
            obj_h  = obj.bbox[3] - obj.bbox[1]
            if obj_h < MIN_DETECTABLE_HEIGHT_NORM:
                obj_results.append({"detectable": False, "contained_adaptive": None,
                                     "contained_static": None, "expected_miss": False})
                continue
            in_adaptive = (roi.x_left <= obj_cx <= roi.x_left + roi.width and
                           roi.y_top  <= obj_cy <= roi.y_top  + roi.height)
            in_static   = (STATIC_CROP.x_left <= obj_cx <= STATIC_CROP.x_left + STATIC_CROP.width and
                           STATIC_CROP.y_top   <= obj_cy <= STATIC_CROP.y_top   + STATIC_CROP.height)
            # Determine if a miss is "expected" (object outside the floor corridor,
            # so the system correctly does not expand for it)
            in_floor_corridor = (floor_xl <= obj_cx <= floor_xr)
            expected_miss = (not in_adaptive) and (not in_floor_corridor)
            obj_results.append({"detectable": True, "contained_adaptive": in_adaptive,
                                 "contained_static": in_static, "expected_miss": expected_miss})

        # --- Area (as fraction of full frame) ---
        adaptive_area = roi.width * roi.height
        static_area   = STATIC_CROP.width * STATIC_CROP.height

        records.append({
            "scenario": name,
            "frame": frame_i,
            "roi_level": roi.roi_level,
            "floor_contained": floor_contained,
            "adaptive_area": adaptive_area,
            "static_area": static_area,
            "objects": obj_results,
            "speed_kmh": sig.speed_mps * 3.6,
        })

    return records


# ---------------------------------------------------------------------------
# Aggregate and report
# ---------------------------------------------------------------------------

def run_all():
    all_records = []
    print("=" * 68)
    print("STAGE 9 VALIDATION — Adaptive Shared ROI")
    print("=" * 68)
    print()

    scenario_summaries = []
    for name, scenario in SCENARIOS.items():
        records = run_scenario(name, scenario)
        all_records.extend(records)

        n = len(records)
        floor_ok  = sum(1 for r in records if r["floor_contained"])
        level_dist = {}
        for r in records:
            level_dist[r["roi_level"]] = level_dist.get(r["roi_level"], 0) + 1
        avg_adaptive = sum(r["adaptive_area"] for r in records) / n
        avg_static   = sum(r["static_area"]   for r in records) / n

        # Object containment
        obj_detectable = [o for r in records for o in r["objects"] if o["detectable"]]
        adaptive_hits = sum(1 for o in obj_detectable if o["contained_adaptive"])
        static_hits   = sum(1 for o in obj_detectable if o["contained_static"])
        n_det = len(obj_detectable)

        summary = {
            "name": name,
            "description": scenario["description"],
            "n_frames": n,
            "floor_recall_pct": 100 * floor_ok / n,
            "level_dist": level_dist,
            "avg_adaptive_area_pct": 100 * avg_adaptive,
            "avg_static_area_pct": 100 * avg_static,
            "area_saving_pct": 100 * (avg_static - avg_adaptive) / avg_static,
            "obj_adaptive_recall_pct": 100 * adaptive_hits / n_det if n_det else None,
            "obj_static_recall_pct": 100 * static_hits / n_det if n_det else None,
            "n_detectable_objects": n_det,
        }
        scenario_summaries.append(summary)
        print(f"[{name}]")
        print(f"  {scenario['description']}")
        print(f"  Frames: {n}  |  Floor guarantee: {floor_ok}/{n} ({summary['floor_recall_pct']:.0f}%)")
        print(f"  Avg area — adaptive: {summary['avg_adaptive_area_pct']:.1f}%  "
              f"static: {summary['avg_static_area_pct']:.1f}%  "
              f"saving: {summary['area_saving_pct']:.1f}%")
        if n_det:
            print(f"  Object recall — adaptive: {adaptive_hits}/{n_det} "
                  f"({summary['obj_adaptive_recall_pct']:.0f}%)  "
                  f"static: {static_hits}/{n_det} "
                  f"({summary['obj_static_recall_pct']:.0f}%)")
        levels_str = "  ".join(f"L{k}:{v}" for k, v in sorted(level_dist.items()))
        print(f"  ROI levels: {levels_str}")
        print()

    # ------------------------------------
    # Global aggregates (the paper numbers)
    # ------------------------------------
    total_frames = len(all_records)
    floor_total  = sum(1 for r in all_records if r["floor_contained"])
    all_objs_det = [o for r in all_records for o in r["objects"] if o["detectable"]]
    n_det_total  = len(all_objs_det)
    # Adaptive recall: among all detectable objects, how many were covered
    adapt_hits_t = sum(1 for o in all_objs_det if o["contained_adaptive"])
    stat_hits_t  = sum(1 for o in all_objs_det if o["contained_static"])
    # In-corridor recall: exclude objects that were EXPECTED to be missed
    # (outside the floor corridor -- system correctly withholds expansion)
    in_corridor_objs = [o for o in all_objs_det if not o["expected_miss"]]
    n_incorr = len(in_corridor_objs)
    adapt_incorr = sum(1 for o in in_corridor_objs if o["contained_adaptive"])
    avg_adapt_t  = sum(r["adaptive_area"] for r in all_records) / total_frames
    avg_stat_t   = sum(r["static_area"]   for r in all_records) / total_frames

    level_all = {}
    for r in all_records:
        level_all[r["roi_level"]] = level_all.get(r["roi_level"], 0) + 1

    print("=" * 68)
    print("AGGREGATE RESULTS  (citable in manuscript)")
    print("=" * 68)
    print()
    print(f"Total frames evaluated:          {total_frames}")
    print(f"Total scenarios:                 {len(SCENARIOS)}")
    print()
    print("--- Measurement 1: Floor Coverage Guarantee ---")
    print(f"  Final ROI ⊇ floor:  {floor_total}/{total_frames} frames  "
          f"({100*floor_total/total_frames:.1f}%)")
    print()
    print("--- Measurement 2: Object Containment (Adaptive vs. Static Crop) ---")
    if n_det_total:
        print(f"  Total detectable object-frames:  {n_det_total}")
        print(f"  Adaptive ROI recall (all):        {adapt_hits_t}/{n_det_total}  "
              f"({100*adapt_hits_t/n_det_total:.1f}%)")
        print(f"  Static crop recall (all):         {stat_hits_t}/{n_det_total}  "
              f"({100*stat_hits_t/n_det_total:.1f}%)")
        if n_incorr:
            print(f"  Adaptive recall (in-corridor only, expected misses excluded): "
                  f"{adapt_incorr}/{n_incorr}  ({100*adapt_incorr/n_incorr:.1f}%)")
        expected_misses = n_det_total - n_incorr
        if expected_misses:
            print(f"  Expected misses (object outside corridor, correct behavior): "
                  f"{expected_misses}")
    print()
    print("--- Measurement 3: Region Area Distribution ---")
    print(f"  Adaptive ROI mean area:   {100*avg_adapt_t:.1f}% of frame")
    print(f"  Static crop area:         {100*avg_stat_t:.1f}% of frame")
    print(f"  Mean area reduction:      {100*(avg_stat_t-avg_adapt_t)/avg_stat_t:.1f}% "
          f"vs static baseline")
    print()
    print("--- ROI Level Distribution (all frames) ---")
    for lv in sorted(level_all):
        pct = 100 * level_all[lv] / total_frames
        label = {0: "L0 (full adaptive, high confidence)",
                 1: "L1 (CAN-only curvature, lane degraded)",
                 2: "L2 (confidence blend, partial data)",
                 3: "L3 (full-frame fallback, safety net)"}.get(lv, f"L{lv}")
        print(f"  {label}: {level_all[lv]} frames ({pct:.1f}%)")
    print()
    print("--- Per-scenario area saving summary ---")
    for s in scenario_summaries:
        print(f"  {s['name']:35s}  {s['area_saving_pct']:+6.1f}%")
    print()

    return {
        "total_frames": total_frames,
        "floor_recall_pct": 100 * floor_total / total_frames,
        "obj_adaptive_recall_pct": 100 * adapt_hits_t / n_det_total if n_det_total else None,
        "obj_static_recall_pct":   100 * stat_hits_t  / n_det_total if n_det_total else None,
        "mean_adaptive_area_pct":  100 * avg_adapt_t,
        "mean_static_area_pct":    100 * avg_stat_t,
        "mean_area_reduction_pct": 100 * (avg_stat_t - avg_adapt_t) / avg_stat_t,
        "level_distribution": level_all,
        "scenario_summaries": scenario_summaries,
    }


if __name__ == "__main__":
    results = run_all()

    # Save to file for review note / manuscript
    out_path = os.path.join(os.path.dirname(__file__), "stage9_results.txt")
    with open(out_path, "w") as f:
        f.write("Stage 9 Validation Results\n")
        f.write(f"Total frames: {results['total_frames']}\n")
        f.write(f"Floor coverage recall: {results['floor_recall_pct']:.1f}%\n")
        if results['obj_adaptive_recall_pct'] is not None:
            f.write(f"Adaptive object recall: {results['obj_adaptive_recall_pct']:.1f}%\n")
            f.write(f"Static crop recall: {results['obj_static_recall_pct']:.1f}%\n")
            f.write(f"Recall gain: +{results['obj_adaptive_recall_pct']-results['obj_static_recall_pct']:.1f} pp\n")
        f.write(f"Mean adaptive area: {results['mean_adaptive_area_pct']:.1f}%\n")
        f.write(f"Static crop area: {results['mean_static_area_pct']:.1f}%\n")
        f.write(f"Mean area reduction: {results['mean_area_reduction_pct']:.1f}%\n")
    print(f"Results written to {out_path}")
