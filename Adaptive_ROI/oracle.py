"""
Oracle File — Independently Derived Expected Values
=====================================================
This file is the GROUND TRUTH reference for validating dynamic_roi.py.

CRITICAL RULE (do not violate this when extending the file):
Every `expected` value below must be worked out BY HAND from the
formula, the cited standard, or the documented design rule — BEFORE
ever running dynamic_roi.py. Never populate `expected` by running the
code and copying its output. A value derived that way is not an
oracle entry; it is just a snapshot of whatever the code currently
does, bugs included, and proves nothing when re-run later.

This file is intentionally just DATA (a list of plain dictionaries),
kept separate from the code that runs and scores it
(run_oracle_analysis.py) and separate from the pytest-style assertions
in test_vectors.py. The separation matters: this file can be reviewed,
checked, and signed off by someone reading only arithmetic and cited
standards, without needing to read or trust any Python execution logic.

Each row's `target` field says which computation it exercises;
run_oracle_analysis.py's dispatcher maps that name to the actual call
into dynamic_roi.py.

expected_type values:
  "value"     — expected is a number (or short tuple of numbers),
                compared to the actual result within `tolerance`.
  "exception" — expected is the substring that must appear in the
                ValueError raised; the call is expected to raise.
  "bool"      — expected is True/False, for boolean-outcome checks
                (e.g. "does y_top end up less than y_bottom").

stage_dependency: None if this can be evaluated against the code as it
stands today (2026-08-05, Stage 1 complete). Otherwise, the stage that
must land before this row can be scored — such rows are still listed
here (the expected value can often already be worked out even before
the code exists) but the runner will mark them DEFERRED rather than
PASS/FAIL.
"""

ORACLE = [

    # ======================================================================
    # CATEGORY 1 — Baseline sanity
    # ======================================================================
    {
        "id": "C1-001",
        "category": "1_baseline",
        "description": "Z_max at 100 km/h, hand-calculated from ISO 15623 formula",
        "target": "z_max",
        "inputs": {"speed_kmh": 100.0},
        "expected_type": "value",
        "expected": 122.63,
        "tolerance": 0.05,
        "basis": (
            "v=27.778 m/s; Z_max = v*2.1 + v^2/(2*6.0) "
            "= 58.333 + 64.30 = 122.63 m. "
            "Constants: TTC_MIN_WARNING_S=2.1 (ISO 15623, stationary lead "
            "vehicle), MAX_BRAKING_DECEL_MPS2=6.0."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C1-002",
        "category": "1_baseline",
        "description": "Z_max at 0 km/h (raw formula, before minimum-depth clamp)",
        "target": "z_max",
        "inputs": {"speed_kmh": 0.0},
        "expected_type": "value",
        "expected": 0.0,
        "tolerance": 1e-9,
        "basis": "v=0 -> both terms of Z_max(v) = v*t_r + v^2/(2a) are zero.",
        "stage_dependency": None,
    },
    {
        "id": "C1-003",
        "category": "1_baseline",
        "description": "Z_max at 60 km/h, hand-calculated",
        "target": "z_max",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "value",
        "expected": 58.15,
        "tolerance": 0.05,
        "basis": (
            "v=16.667 m/s; Z_max = 16.667*2.1 + 16.667^2/12 "
            "= 35.00 + 23.15 = 58.15 m."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C1-004",
        "category": "1_baseline",
        "description": "Z_max at 120 km/h, hand-calculated",
        "target": "z_max",
        "inputs": {"speed_kmh": 120.0},
        "expected_type": "value",
        "expected": 162.59,
        "tolerance": 0.05,
        "basis": (
            "v=33.333 m/s; Z_max = 33.333*2.1 + 33.333^2/12 "
            "= 70.00 + 92.59 = 162.59 m."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C1-005",
        "category": "1_baseline",
        "description": "Stationary vehicle produces a well-ordered, non-degenerate ROI",
        "target": "base_roi_ordering_ok",
        "inputs": {"speed_kmh": 0.0, "steer_deg": 0.0, "yaw_dps": 0.0,
                    "lane_center": 0.5, "lane_width": 0.3, "lane_conf": 0.9},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Smoke test: x_left<x_right, y_top<y_bottom, all within [0,1].",
        "stage_dependency": None,
    },

    # ======================================================================
    # CATEGORY 2 — One-variable sweeps
    # ======================================================================
    {
        "id": "C2-001",
        "category": "2_sweep_speed",
        "description": "Region height at 20 km/h (straight road) — hand-projected",
        "target": "floor_height",
        "inputs": {"speed_kmh": 20.0, "curvature": 0.0},
        "expected_type": "value",
        "expected": 0.3472,
        "tolerance": 0.001,
        "basis": (
            "v=5.556 m/s; raw Z_max=14.24m < clamp floor of Z_near(5)+"
            "Z_MIN_FLOOR_DEPTH(15)=20m, so z_far=20m is used. "
            "v_top = cy + f*(2.5/20) = 540 + 125 = 665px -> 665/1080=0.6157. "
            "v_bottom = cy + f*(2.5/5) = 540+500=1040px -> 1040/1080=0.9630. "
            "height = 0.9630-0.6157 = 0.3472."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C2-002",
        "category": "2_sweep_speed",
        "description": "Region height at 60 km/h (straight road) — hand-projected",
        "target": "floor_height",
        "inputs": {"speed_kmh": 60.0, "curvature": 0.0},
        "expected_type": "value",
        "expected": 0.4232,
        "tolerance": 0.001,
        "basis": (
            "v=16.667 m/s; Z_max=58.15m (above clamp, used directly). "
            "v_top = cy + f*(2.5/58.15) = 540+42.99=582.99px -> 0.5398. "
            "v_bottom = 0.9630 (same near-field calc as C2-001). "
            "height = 0.9630-0.5398 = 0.4232."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C2-003",
        "category": "2_sweep_speed",
        "description": "Region height at 100 km/h (straight road) — hand-projected",
        "target": "floor_height",
        "inputs": {"speed_kmh": 100.0, "curvature": 0.0},
        "expected_type": "value",
        "expected": 0.4441,
        "tolerance": 0.001,
        "basis": (
            "Z_max=122.63m. v_top = 540 + 1000*(2.5/122.63) = 540+20.39=560.39px "
            "-> 0.5189. height = 0.9630-0.5189 = 0.4441."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C2-004",
        "category": "2_sweep_speed",
        "description": "Region height is non-decreasing across the full speed sweep",
        "target": "height_monotonic_nondecreasing",
        "inputs": {"speeds_kmh": [0, 10, 20, 40, 60, 80, 100, 120, 150]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Z_max(v) is a non-decreasing function of v for v>=0; combined with the minimum-depth clamp, height can plateau but never decrease.",
        "stage_dependency": None,
    },
    {
        "id": "C2-005",
        "category": "2_sweep_curvature",
        "description": "Lateral centre shift direction agrees between floor and pre-existing lane logic (+10 deg steer)",
        "target": "curvature_sign_agreement",
        "inputs": {"speed_kmh": 100.0, "steer_deg": 10.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "This is the exact sign-consistency bug found and fixed on "
            "2026-08-05 (Stage 1 verification). The pre-existing "
            "_lateral_offset_norm() and the new floor calculation must "
            "agree on shift direction for the same steering input."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C2-006",
        "category": "2_sweep_lane_confidence",
        "description": "UPDATED 2026-08-05: lane confidence 0.51 under Stage 2 blending",
        "target": "degradation_level",
        "inputs": {"speed_kmh": 60.0, "lane_conf": 0.51},
        "expected_type": "value",
        "expected": 1,
        "tolerance": None,
        "basis": (
            "SUPERSEDED by Stage 2: 0.51 falls inside the blend zone "
            "(CONF_BLEND_LOW=0.40 to CONF_BLEND_HIGH=0.65), giving "
            "blend_weight=(0.51-0.40)/(0.65-0.40)=0.44 -> neither 0 nor 1 "
            "-> Level 1 (blended). The old entry left this ambiguous "
            "('accept 0 or 1') because the pre-Stage-2 hard switch made an "
            "exact answer less interesting; Stage 2's exact blend formula "
            "now gives a single determinate answer worth pinning down."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C2-007",
        "category": "2_sweep_lane_confidence",
        "description": "RETIRED CHECK, updated 2026-08-05: lane confidence 0.49 under Stage 2 blending (was: hard switch to level 2)",
        "target": "degradation_level",
        "inputs": {"speed_kmh": 60.0, "lane_conf": 0.49},
        "expected_type": "value",
        "expected": 1,
        "tolerance": None,
        "basis": (
            "SUPERSEDED by Stage 2 (2026-08-05): the hard LANE_CONF_MIN=0.5 "
            "switch this entry originally tested has been removed and "
            "replaced by continuous confidence blending (CONF_BLEND_LOW=0.40, "
            "CONF_BLEND_HIGH=0.65). 0.49 now falls INSIDE the blend zone, not "
            "below a hard cutoff, so the correct expectation is Level 1 "
            "(blended), not Level 2. This entry is kept (not deleted) "
            "specifically to document that the old expectation is retired, "
            "per this project's rule of recording superseded behaviour "
            "rather than silently erasing it — see review_note.md's Update "
            "Log entries for the same principle applied to the document."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    # CATEGORY 3 — Scenario-based
    # ======================================================================
    {
        "id": "C3-2.3",
        "category": "3_scenario",
        "description": "Scenario 2.3 (roundabout, 20 km/h): raw Z_max below clamp threshold",
        "target": "z_max",
        "inputs": {"speed_kmh": 20.0},
        "expected_type": "value",
        "expected": 14.24,
        "tolerance": 0.05,
        "basis": "v=5.556 m/s; Z_max=5.556*2.1+5.556^2/12=11.67+2.57=14.24m.",
        "stage_dependency": None,
    },
    {
        "id": "C3-4.3",
        "category": "3_scenario",
        "description": "Scenario 4.3 (first frame after boot): first-ever call produces valid ROI",
        "target": "first_frame_ok",
        "inputs": {"speed_kmh": 50.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Design requirement: no prior state should cause a crash or invalid output on the first call.",
        "stage_dependency": None,
    },
    {
        "id": "C3-4.1",
        "category": "3_scenario",
        "description": "Scenario 4.1 (sudden braking): corridor widens quickly under ABS-reduced confidence",
        "target": "sudden_braking_widens_quickly",
        "inputs": {"speed_before_kmh": 100.0, "speed_after_kmh": 40.0, "n_frames": 3},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "IMPLEMENTED in Stage 4 (2026-08-06): the asymmetric fast-grow "
            "filter (ASYM_ALPHA_GROW_EDGE=0.30) ensures ABS-triggered "
            "corridor widening reaches at least 90% of its target within "
            "3 frames, rather than lagging under the old single-rate filter."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 4 — Asymmetric fast-grow/slow-shrink filter (added 2026-08-06)
    # ======================================================================
    {
        "id": "S4-001",
        "category": "4_stage4_smoothing",
        "description": "Growing edge reaches >95% of target within 3 frames",
        "target": "grow_reaches_target_fast",
        "inputs": {"speed_before_kmh": 20.0, "speed_after_kmh": 130.0, "n_frames": 3},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "ASYM_ALPHA_GROW_EDGE=0.30 means each frame keeps only 30% of the previous value -- convergence should be fast: after 3 frames, (0.3)^3=2.7% of the ORIGINAL gap should remain, i.e. >97% progress.",
        "stage_dependency": None,
    },
    {
        "id": "S4-002",
        "category": "4_stage4_smoothing",
        "description": "Shrinking edge retains noticeably more residual than growing loses, over the same number of frames",
        "target": "shrink_lags_grow",
        "inputs": {"speed_before_kmh": 130.0, "speed_after_kmh": 20.0, "n_frames": 3},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "ASYM_ALPHA_SHRINK_EDGE=0.85 means each frame keeps 85% of the previous value -- after 3 frames, (0.85)^3=61.4% of the original gap should remain, versus only (0.30)^3=2.7% for the growing case at the same elapsed time. Confirms the two alphas are actually applied asymmetrically, not equal or unused.",
        "stage_dependency": None,
    },
    {
        "id": "S4-003",
        "category": "4_stage4_smoothing",
        "description": "Vertical dimension (y_top/height) is now smoothed, unlike the old filter",
        "target": "vertical_now_smoothed",
        "inputs": {"speed_before_kmh": 20.0, "speed_after_kmh": 130.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Old _smooth_or_snap() only ever touched centre-x and width; height/y_top passed through unsmoothed. New _smooth_asymmetric() blends all four edges. A one-frame jump should NOT instantly reach the new target height.",
        "stage_dependency": None,
    },
    {
        "id": "S4-004",
        "category": "4_stage4_smoothing",
        "description": "No snap-threshold discontinuity remains across a wide range of jump sizes",
        "target": "no_snap_discontinuity",
        "inputs": {"target_speeds_kmh": [30, 50, 70, 90, 110, 130]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "The old filter's fixed jump threshold (0.06 norm) caused heavily-smoothed behaviour just below it and instant, unsmoothed snapping just above it. The new filter is a pure ratio (alpha) applied regardless of jump size, so the fraction-of-target reached in a fixed number of frames should stay roughly constant across very different jump magnitudes.",
        "stage_dependency": None,
    },
    {
        "id": "S4-005",
        "category": "4_stage4_finding",
        "description": "Single-frame detection dropout causes only a small (~1%) height reduction, confirming expansion-decay memory is not needed",
        "target": "dropout_minimal_impact",
        "inputs": {"n_frames": 8, "dropout_frame_index": 6},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Directly resolves the open question from the earlier design "
            "discussion (review_note.md Section 4.8): 'build the simple "
            "version first, add decay only if proven necessary.' Measured "
            "directly on 2026-08-06: a one-frame dropout during an active "
            "expansion produced roughly a 0.7% height reduction -- the "
            "slow-shrink asymmetry alone provides adequate flicker "
            "protection without a separate ExpansionMemory mechanism."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    {
        "id": "C3-5.1",
        "category": "3_scenario",
        "description": "Scenario 5.1 (no lane markings, curve): Level-2 fallback still shifts for curvature",
        "target": "level2_applies_curvature",
        "inputs": {"speed_kmh": 60.0, "steer_deg": 15.0, "lane_conf": 0.1},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Design requirement (review_note.md Section 2.4, Problem one): "
            "Level 2 should shift using CAN curvature, not return static_roi "
            "unmodified. IMPLEMENTED in Stage 2 (2026-08-05): "
            "_compute_base_roi() now always computes and applies the "
            "curvature-based shift regardless of lane confidence."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C3-2.2",
        "category": "3_scenario",
        "description": "Scenario 2.2 (highway fork): corridor widens on CAN-vision curvature disagreement",
        "target": "curvature_mismatch_widening",
        "inputs": {"speed_kmh": 90.0, "mismatch_inv_m": 0.015},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "IMPLEMENTED in Stage 5 (2026-08-07): LaneInfo now carries "
            "c2_curvature/c2_confidence, and _fuse_curvature() widens the "
            "corridor via _curvature_agreement_confidence() when CAN and "
            "vision disagree significantly."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 5 — CAN/vision curvature fusion (added 2026-08-07)
    # ======================================================================
    {
        "id": "S5-001",
        "category": "5_stage5_fusion",
        "description": "Vision unavailable falls back to CAN identically to pre-Stage-5 behaviour",
        "target": "vision_unavailable_fallback",
        "inputs": {"speed_kmh": 80.0, "steer_deg": 10.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Backward-compatibility requirement: default LaneInfo (c2_curvature=None) must produce fused_curvature == can_curvature and confidence == 1.0, matching Stages 1-4 exactly.",
        "stage_dependency": None,
    },
    {
        "id": "S5-002",
        "category": "5_stage5_fusion",
        "description": "Vision preferred over CAN when confident and available",
        "target": "vision_preferred_when_confident",
        "inputs": {"speed_kmh": 80.0, "steer_deg": 10.0, "vision_offset": 0.0002, "vision_conf": 0.8},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "VISION_CURVATURE_TRUST_THRESHOLD=0.5; with vision_conf=0.8 (above threshold), fused_curvature should equal the vision value, not the CAN value.",
        "stage_dependency": None,
    },
    {
        "id": "S5-003",
        "category": "5_stage5_fusion",
        "description": "Low vision confidence falls back to CAN despite a vision value being present",
        "target": "low_vision_confidence_fallback",
        "inputs": {"speed_kmh": 80.0, "steer_deg": 10.0, "vision_offset": 0.05, "vision_conf": 0.2},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "vision_conf=0.2 is below VISION_CURVATURE_TRUST_THRESHOLD=0.5, so fused_curvature must equal CAN's value even though a (untrusted) vision value exists.",
        "stage_dependency": None,
    },
    {
        "id": "S5-004",
        "category": "5_stage5_fusion",
        "description": "Mismatch-confidence tiers are genuinely graduated (mild > significant > severe distrust)",
        "target": "mismatch_tiers_graduated",
        "inputs": {"mismatches_inv_m": [0.0002, 0.002, 0.006, 0.015]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Measured directly on 2026-08-07: agreement confidence values "
            "1.0 -> 0.70 -> 0.40 -> 0.15 across the four mismatch levels, "
            "and end-to-end region width 0.4302 -> 0.6883 -> 0.9465 -> "
            "1.0000 (clamped) -- confirms the tiers discriminate by degree, "
            "not just presence/absence of disagreement."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S5-005",
        "category": "5_stage5_fusion",
        "description": "Corridor widens end-to-end on severe CAN/vision disagreement vs. agreement",
        "target": "corridor_widens_on_disagreement",
        "inputs": {"speed_kmh": 80.0, "agree_offset": 0.0002, "disagree_offset": 0.02},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Full-pipeline check: same speed/steering, only the vision curvature differs -- the disagreeing case must produce a strictly wider final region.",
        "stage_dependency": None,
    },

    # ======================================================================
    {
        "id": "C3-1.1",
        "category": "3_scenario",
        "description": "Scenario 1.1 (parked vehicle): does not inflate the region",
        "target": "corridor_gated_expansion",
        "inputs": {"vehicle_lateral_offset_norm": 0.02, "n_frames": 5},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "IMPLEMENTED in Stage 3 (2026-08-06): corridor-membership gating "
            "added to _apply_object_expansions(). A vehicle bbox centred "
            "well outside the corridor's lateral bounds must not cause the "
            "region to widen toward it, even once its track is CONFIRMED."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 3 — Corridor-membership gating (added 2026-08-06)
    # ======================================================================
    {
        "id": "S3-001",
        "category": "3_stage3_gating",
        "description": "In-corridor, confirmed, closing vehicle still expands the region",
        "target": "in_corridor_expansion_still_works",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "The flip side of the parked-vehicle fix: gating must not become "
            "so strict that legitimate FCW expansion stops working. Uses a "
            "down-and-growing motion pattern to obtain a valid TTC under the "
            "current TTC estimator (see S3-003/004 re: head-on limitation)."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S3-002",
        "category": "3_stage3_gating",
        "description": "A single-frame (TENTATIVE) detection does not expand the region",
        "target": "unconfirmed_track_no_expansion",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Track-confirmation gate (condition 2 of 3): N_INIT=3 hit_streak required before CONFIRMED; a first-frame detection is TENTATIVE and must not expand.",
        "stage_dependency": None,
    },
    {
        "id": "S3-003",
        "category": "3_stage3_kalman_fix",
        "description": "CRITICAL FIX: position-velocity covariance cross-term is now populated after predict()",
        "target": "cross_covariance_populated",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Found 2026-08-06 during Stage 3 verification: the previous "
            "predict() only updated P's diagonal, so P[0][4] (and P[1][5]) "
            "stayed at exactly 0.0 forever, making the Kalman gain for "
            "velocity states always zero regardless of motion. Confirmed by "
            "direct inspection: P[0][4]=0.0 before predict(), non-zero after, "
            "using the corrected F*P*F^T + Q propagation."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S3-004",
        "category": "3_stage3_kalman_fix",
        "description": "Velocity now converges toward true motion for lateral (cx) movement",
        "target": "velocity_converges_lateral",
        "inputs": {"true_velocity_per_frame": 0.02, "n_frames": 8},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Direct consequence of the S3-003 fix: with track continuity "
            "maintained (consecutive boxes overlapping enough for IoU "
            "matching to succeed), filtered vx should converge toward the "
            "true per-frame displacement, not stay at 0.0 as it did before "
            "the fix."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S3-005",
        "category": "3_stage3_known_limitation",
        "description": "TRACKED LIMITATION (not fixed): pure head-on approach always yields TTC=None",
        "target": "head_on_ttc_still_none",
        "inputs": {"n_frames": 8},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "DISTINCT from S3-003/004's fix. A vehicle approaching directly "
            "head-on has a bbox whose vertical CENTRE stays fixed (near the "
            "vanishing point) while only its HEIGHT grows. vy (centre "
            "velocity) is correctly ~0 for this motion — the real problem is "
            "that _estimate_ttc() has no height-velocity state to read "
            "instead. This entry documents the limitation is PRESENT, not "
            "that it is desired behaviour — if this ever scores FAIL, it "
            "means the limitation has been fixed and this row's expectation "
            "should be updated to False, not deleted."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 2 — Unified confidence and continuous blending (added 2026-08-05)
    # ======================================================================
    {
        "id": "S2-001",
        "category": "2_stage2_confidence",
        "description": "Dynamics confidence at zero mismatch, no ESC/ABS: full confidence",
        "target": "dynamics_confidence",
        "inputs": {"speed_kmh": 80.0, "steer_deg": 0.0, "yaw_dps": 0.0, "esc": False, "abs_flag": False},
        "expected_type": "value",
        "expected": 1.0,
        "tolerance": 1e-9,
        "basis": "Zero mismatch, no stability flags -> no penalty applied anywhere in _dynamics_confidence().",
        "stage_dependency": None,
    },
    {
        "id": "S2-002",
        "category": "2_stage2_confidence",
        "description": "Dynamics confidence with ESC active: capped at ESC_ACTIVE_CONF_CEILING",
        "target": "dynamics_confidence",
        "inputs": {"speed_kmh": 80.0, "steer_deg": 0.0, "yaw_dps": 0.0, "esc": True, "abs_flag": False},
        "expected_type": "value",
        "expected": 0.3,
        "tolerance": 1e-9,
        "basis": "ESC_ACTIVE_CONF_CEILING=0.3 constant; confidence=min(1.0, 0.3)=0.3.",
        "stage_dependency": None,
    },
    {
        "id": "S2-003",
        "category": "2_stage2_confidence",
        "description": "Dynamics confidence with ABS active: capped at ABS_ACTIVE_CONF_CEILING",
        "target": "dynamics_confidence",
        "inputs": {"speed_kmh": 80.0, "steer_deg": 0.0, "yaw_dps": 0.0, "esc": False, "abs_flag": True},
        "expected_type": "value",
        "expected": 0.5,
        "tolerance": 1e-9,
        "basis": "ABS_ACTIVE_CONF_CEILING=0.5 constant; confidence=min(1.0, 0.5)=0.5.",
        "stage_dependency": None,
    },
    {
        "id": "S2-004",
        "category": "2_stage2_confidence",
        "description": "Dynamics confidence with both ESC and ABS active: lower of the two ceilings applies",
        "target": "dynamics_confidence",
        "inputs": {"speed_kmh": 80.0, "steer_deg": 0.0, "yaw_dps": 0.0, "esc": True, "abs_flag": True},
        "expected_type": "value",
        "expected": 0.3,
        "tolerance": 1e-9,
        "basis": "min(1.0, ESC_CEILING=0.3, ABS_CEILING=0.5) = 0.3 (the more conservative of the two).",
        "stage_dependency": None,
    },
    {
        "id": "S2-005",
        "category": "2_stage2_confidence",
        "description": "Level 3 does NOT trigger from lane failure alone when CAN dynamics are healthy",
        "target": "level3_not_from_lane_alone",
        "inputs": {"speed_kmh": 80.0, "steer_deg": 5.0, "lane_conf": 0.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Deliberate design decision: Level 3 trigger uses dynamics_conf "
            "alone, not min(dynamics_conf, lane_conf), specifically so a lane "
            "dropout with healthy CAN does not force a full-frame fallback "
            "when a CAN-only corridor (Level 2) is a perfectly good response."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S2-006",
        "category": "2_stage2_confidence",
        "description": "Level 3 DOES trigger when dynamics confidence is severely degraded",
        "target": "level3_triggers_from_severe_mismatch",
        "inputs": {"speed_kmh": 80.0, "yaw_dps": 30.0, "steer_deg": 0.0, "lane_conf": 0.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "A 30 deg/s yaw rate against a 0 deg steering input is a huge "
            "mismatch, well past YAW_MISMATCH_SEVERE_DPS=15 -> "
            "dynamics_conf=DYNAMICS_CONF_SEVERE=0.1, below "
            "CONF_LEVEL3_THRESHOLD=0.15 -> Level 3."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S2-007",
        "category": "2_stage2_confidence",
        "description": "Confidence blend is continuous — no jump greater than 0.05 for a 0.02 confidence step",
        "target": "blend_smoothness",
        "inputs": {"speed_kmh": 60.0, "steer_deg": 15.0, "lane_center": 0.35, "lane_width": 0.3},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Direct test of the design goal replacing the old hard switch: "
            "verified by hand on 2026-08-05 across a 0.0-1.0 confidence "
            "sweep at 0.02 steps, max observed centre movement was 0.0184 "
            "per step, well under the 0.05 threshold used here."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    # CATEGORY 4 — Boundary values
    # ======================================================================
    {
        "id": "C4-001",
        "category": "4_boundary",
        "description": "Zero speed does not collapse floor to zero height",
        "target": "floor_height_positive",
        "inputs": {"speed_kmh": 0.0, "curvature": 0.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Z_MIN_FLOOR_DEPTH_M=15m guarantees z_far >= z_near+15 regardless of speed -> height > 0 always. This is the exact 2026-08-05 fix.",
        "stage_dependency": None,
    },
    {
        "id": "C4-002",
        "category": "4_boundary",
        "description": "Minimum-floor-depth clamp boundary speed, hand-solved",
        "target": "clamp_boundary_speed_kmh",
        "inputs": {},
        "expected_type": "value",
        "expected": 26.53,
        "tolerance": 0.05,
        "basis": (
            "Solve v*2.1 + v^2/12 = 20 for v (Z_near=5 + Z_MIN_FLOOR_DEPTH=15=20): "
            "a=1/12, b=2.1, c=-20; v=(-b+sqrt(b^2-4ac))/(2a) "
            "=(-2.1+sqrt(4.41+6.6667))/(0.16667) "
            "=(-2.1+3.3282)/0.16667 = 7.369 m/s = 26.53 km/h. "
            "CORRECTION LOG: an earlier version of this oracle entry (during "
            "conversation on 2026-08-05) stated 25.2 km/h from a mental "
            "arithmetic shortcut that was itself wrong. The independent "
            "runner calculation (run_oracle_analysis.py, computed with the "
            "same formula but typed separately) flagged the mismatch, and "
            "re-solving by hand with full precision confirmed 26.53 km/h is "
            "correct. This is recorded here deliberately, as an example of "
            "the oracle process catching an error in the oracle itself, not "
            "just in the code — which is exactly what having two "
            "independent derivations is for."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C4-003",
        "category": "4_boundary",
        "description": "TTC exactly at critical threshold (1.5s) maps to TTC_CRIT_SCALE",
        "target": "ttc_margin_scale",
        "inputs": {"ttc": 1.5},
        "expected_type": "value",
        "expected": 2.00,
        "tolerance": 1e-6,
        "basis": "Module constant TTC_CRIT_SCALE=2.00; _ttc_margin_scale() documented: ttc<=TTC_CRIT_S -> TTC_CRIT_SCALE.",
        "stage_dependency": None,
    },
    {
        "id": "C4-004",
        "category": "4_boundary",
        "description": "TTC exactly at warning threshold (3.5s) maps to TTC_WARN_SCALE",
        "target": "ttc_margin_scale",
        "inputs": {"ttc": 3.5},
        "expected_type": "value",
        "expected": 1.40,
        "tolerance": 1e-6,
        "basis": "At ttc=TTC_WARN_S, interpolation factor t=0 -> result=TTC_WARN_SCALE=1.40 exactly.",
        "stage_dependency": None,
    },
    {
        "id": "C4-005",
        "category": "4_boundary",
        "description": "TTC just above warning threshold maps to 1.0 (no threat)",
        "target": "ttc_margin_scale",
        "inputs": {"ttc": 3.500001},
        "expected_type": "value",
        "expected": 1.0,
        "tolerance": 1e-4,
        "basis": "ttc > TTC_WARN_S -> nominal multiplier 1.0.",
        "stage_dependency": None,
    },
    {
        "id": "C4-006",
        "category": "4_boundary",
        "description": "TTC=None (no valid track) maps to 1.0",
        "target": "ttc_margin_scale",
        "inputs": {"ttc": None},
        "expected_type": "value",
        "expected": 1.0,
        "tolerance": 1e-9,
        "basis": "_ttc_margin_scale(None) documented to return 1.0 (nominal, no adjustment).",
        "stage_dependency": None,
    },
    {
        "id": "C4-007",
        "category": "4_boundary",
        "description": "Bounding box touching image corner (0,0) does not raise and stays in-bounds after expansion",
        "target": "edge_bbox_expansion_ok",
        "inputs": {"bbox": [0.0, 0.0, 0.05, 0.05]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "_validate_inputs() is boundary-inclusive (0.0<=coord<=1.0); _clamp() bounds all outputs regardless.",
        "stage_dependency": None,
    },

    # ======================================================================
    # CATEGORY 5 — Invalid input
    # ======================================================================
    {
        "id": "C5-001",
        "category": "5_invalid",
        "description": "Negative speed raises ValueError mentioning speed_mps",
        "target": "validate_inputs_negative_speed",
        "inputs": {"speed_kmh": -10.0},
        "expected_type": "exception",
        "expected": "speed_mps",
        "tolerance": None,
        "basis": "_validate_inputs(): 'if signals.speed_mps < 0.0: raise ValueError(...)'.",
        "stage_dependency": None,
    },
    {
        "id": "C5-002",
        "category": "5_invalid",
        "description": "Confidence gate above 1.0 raises ValueError mentioning conf_gates",
        "target": "validate_inputs_bad_gate",
        "inputs": {"vehicle_gate": 1.5},
        "expected_type": "exception",
        "expected": "conf_gates",
        "tolerance": None,
        "basis": "_validate_inputs(): each gate checked 'if not (0.0<=gval<=1.0)'.",
        "stage_dependency": None,
    },
    {
        "id": "C5-003",
        "category": "5_invalid",
        "description": "Degenerate bbox (x2<x1) raises ValueError mentioning 'degenerate'",
        "target": "validate_inputs_degenerate_bbox",
        "inputs": {"bbox": [0.5, 0.5, 0.4, 0.6]},
        "expected_type": "exception",
        "expected": "degenerate",
        "tolerance": None,
        "basis": "_validate_inputs(): 'if x2<=x1 or y2<=y1: raise ValueError(f\"...degenerate...\")'.",
        "stage_dependency": None,
    },
    {
        "id": "C5-004",
        "category": "5_invalid",
        "description": "Bbox coordinate outside [0,1] raises ValueError",
        "target": "validate_inputs_oob_bbox",
        "inputs": {"bbox": [0.1, 0.1, 1.5, 0.5]},
        "expected_type": "exception",
        "expected": "out of [0, 1]",
        "tolerance": None,
        "basis": "_validate_inputs(): coordinate range check.",
        "stage_dependency": None,
    },
    {
        "id": "C5-005",
        "category": "5_invalid",
        "description": "Zero-width static_roi raises ValueError mentioning 'positive'",
        "target": "validate_inputs_zero_static_roi",
        "inputs": {"static_width": 0.0},
        "expected_type": "exception",
        "expected": "positive",
        "tolerance": None,
        "basis": "_validate_inputs(): 'if static_roi.width<=0.0 or ...: raise ValueError(...positive...)'.",
        "stage_dependency": None,
    },
    {
        "id": "C5-006",
        "category": "5_invalid",
        "description": "Negative focal length raises ValueError mentioning focal_px",
        "target": "validate_camera_bad_focal",
        "inputs": {"focal_px": -100.0},
        "expected_type": "exception",
        "expected": "focal_px",
        "tolerance": None,
        "basis": "_validate_camera_intrinsics() (Stage 1, new): 'if camera.focal_px<=0.0: raise ValueError(...)'.",
        "stage_dependency": None,
    },
    {
        "id": "C5-007",
        "category": "5_invalid",
        "description": "Zero mount height raises ValueError mentioning mount_height_m",
        "target": "validate_camera_bad_mount_height",
        "inputs": {"mount_height_m": 0.0},
        "expected_type": "exception",
        "expected": "mount_height_m",
        "tolerance": None,
        "basis": "_validate_camera_intrinsics(): mount height must be > 0.",
        "stage_dependency": None,
    },
    {
        "id": "C5-008",
        "category": "5_invalid",
        "description": "Zero image width raises ValueError mentioning image dimensions",
        "target": "validate_camera_bad_image_size",
        "inputs": {"image_width_px": 0.0},
        "expected_type": "exception",
        "expected": "image",
        "tolerance": None,
        "basis": "_validate_camera_intrinsics(): image dimensions must be > 0.",
        "stage_dependency": None,
    },

    # ======================================================================
    # CATEGORY 6 — Deliberate safety-guarantee attacks
    # ======================================================================
    {
        "id": "C6-001",
        "category": "6_safety_attack",
        "description": "Known object at Z=80m, X=0 stays within floor's lateral bound across speed x curvature grid",
        "target": "lateral_coverage_grid",
        "inputs": {"object_x_m": 0.0, "object_z_m": 80.0,
                    "speeds_kmh": [20, 60, 100, 120],
                    "curvatures": [0.0, 0.002, -0.002, 0.004, -0.004]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Object sits on the lane centreline (X=0); for any non-degenerate "
            "corridor half-width (>0), a centreline object must remain inside "
            "the corridor unless the corridor itself is broken. This is the "
            "core safety claim of the invariant floor."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C6-002",
        "category": "6_safety_attack",
        "description": "Ground-level object exactly at Z_max stays within the floor's vertical bound, all tested speeds",
        "target": "vertical_coverage_grid",
        "inputs": {"speeds_kmh": [30, 60, 100, 130]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "By construction, the floor's v_top is projected FROM the same "
            "z_far used for the object check, so this specific vector is "
            "PARTIALLY CIRCULAR (see test_vectors.py comment on this same "
            "limitation) — it verifies the clamp/z_far logic is internally "
            "consistent, not that the projection formula itself is correct "
            "against an independent ground truth."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C6-003",
        "category": "6_safety_attack",
        "description": "Extreme curvature (at MAX_CURVATURE_INV_M) never produces NaN/Inf or inverted bounds",
        "target": "extreme_curvature_robustness",
        "inputs": {"curvatures": ["MAX_CURVATURE_INV_M", "-MAX_CURVATURE_INV_M", "1e-7", "-1e-7"],
                    "speeds_kmh": [10, 100]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Robustness requirement: the critical-point search in "
            "_invariant_floor() involves a square root; near-degenerate "
            "curvature inputs must not produce non-finite results."
        ),
        "stage_dependency": None,
    },
    {
        "id": "C6-004",
        "category": "6_safety_attack",
        "description": "SABOTAGE CHECK: artificially narrowed corridor width IS detected as a violation",
        "target": "sabotage_detection_check",
        "inputs": {"object_x_m": 0.0, "object_z_m": 80.0, "sabotage_half_width_m": 0.01},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "This row is a check ON THE CHECKER, not on dynamic_roi.py "
            "itself: it confirms C6-001's methodology can actually detect a "
            "real violation when one is artificially introduced, rather "
            "than passing trivially regardless of input. Verified once by "
            "hand on 2026-08-05 (16/20 combinations correctly flagged as "
            "violations when the corridor was sabotaged to 0.01m half-width)."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 6 — Detection scheduling / gap survival (added 2026-08-07)
    # ======================================================================
    {
        "id": "S6-001",
        "category": "6_stage6_scheduling",
        "description": "Track survives a multi-frame detection gap and predicts forward consistently with its known velocity",
        "target": "gap_survival_prediction",
        "inputs": {"n_frames_before": 5, "gap_frames": 3, "step_per_frame": 0.02},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "TrackRegistry.update() calls predict() unconditionally every call; with a constant-velocity object, predicted position after a gap should equal last_measured + velocity*gap_frames, confirmed directly.",
        "stage_dependency": None,
    },
    {
        "id": "S6-002",
        "category": "6_stage6_scheduling",
        "description": "Re-association after a gap succeeds via the PREDICTED position and fails via the stale last-measured position (for a fast-moving object)",
        "target": "reassociation_needs_prediction",
        "inputs": {"n_frames_before": 5, "gap_frames": 4, "step_per_frame": 0.02, "iou_threshold": 0.30},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Measured directly 2026-08-07: IoU(fresh detection, stale box)=0.111 "
            "(below 0.30 threshold, match would fail), IoU(fresh detection, "
            "predicted box)=0.330 (above threshold, match succeeds). This is "
            "the concrete downstream case the Stage 3 Kalman covariance fix "
            "(Category 7) was needed for -- without that fix, vx would have "
            "stayed 0.0 and the predicted box would equal the stale box, "
            "causing this same re-association to fail."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S6-003",
        "category": "6_stage6_scheduling",
        "description": "MAX_AGE boundary is exact under the detection-scheduling-gap framing",
        "target": "max_age_boundary_scheduling",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "MAX_AGE-1 consecutive empty updates -> survives. MAX_AGE consecutive empty updates -> deleted. Confirmed exact, not off-by-one.",
        "stage_dependency": None,
    },
    {
        "id": "S6-004",
        "category": "6_stage6_hardware_dependency",
        "description": "PENDING CONFIRMATION: actual achievable detection rate on target hardware has not been measured",
        "target": None,
        "inputs": {},
        "expected_type": "bool",
        "expected": None,
        "tolerance": None,
        "basis": (
            "Cannot be scored -- requires real TDA4VM-class hardware, not "
            "available in this development environment. Recorded here "
            "explicitly so the gap is visible in the results file rather "
            "than silently absent. See review_note.md Section 3.3's "
            "existing 'Pending Confirmation' treatment of hardware timing."
        ),
        "stage_dependency": "Hardware (not a future software stage)",
    },

    # ======================================================================
    # STAGE 7 — Sign handling: readability + occlusion response (added 2026-08-07)
    # ======================================================================
    {
        "id": "S7-001",
        "category": "7_stage7_isa_readability",
        "description": "ISA readability check matches hand calculation (100->60 km/h, 0.9m sign)",
        "target": "isa_readability_hand_check",
        "inputs": {"speed_kmh": 100.0, "target_kmh": 60.0, "sign_diameter_m": 0.9},
        "expected_type": "value",
        "expected": 275.2,
        "tolerance": 1.0,
        "basis": (
            "v=27.78m/s, target=16.67m/s; speed_reduction=11.11m/s; "
            "t_required=2.5+11.11/1.5=9.907s; distance=27.78*9.907=275.2m. "
            "At this distance, a 0.9m sign with f=1000px gives ~3.27px, "
            "below the 20px readability threshold -- correctly flagged "
            "as NOT readable, matching the IRC 67 finding from earlier "
            "project analysis (signs at the advance-placement distance "
            "can be below native camera resolution)."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S7-002",
        "category": "7_stage7_isa_readability",
        "description": "IRC 67 sign diameter lookup table is correct across all five speed brackets",
        "target": "isa_diameter_lookup",
        "inputs": {"speeds_kmh": [40, 70, 90, 110, 140], "expected_diameters": [0.60, 0.75, 0.90, 1.20, 1.50]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Direct transcription of the IRC 67 regulatory sign sizing table already established earlier in this project.",
        "stage_dependency": None,
    },
    {
        "id": "S7-003",
        "category": "7_stage7_occlusion",
        "description": "Sign memory bridges a brief (within SIGN_MEMORY_MAX_AGE) occlusion",
        "target": "sign_memory_bridges_occlusion",
        "inputs": {"speed_kmh": 60.0, "sign_bbox": [0.85, 0.4, 0.90, 0.5], "frames_visible": 3, "frames_occluded": 2},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "SIGN_MEMORY_MAX_AGE=5; 2 occluded frames is well within that window, so the region should still cover the sign's last-known position via memory.",
        "stage_dependency": None,
    },
    {
        "id": "S7-004",
        "category": "7_stage7_occlusion",
        "description": "Sign memory is forgotten after exceeding SIGN_MEMORY_MAX_AGE",
        "target": "sign_memory_forgotten",
        "inputs": {"speed_kmh": 60.0, "sign_bbox": [0.85, 0.4, 0.90, 0.5]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "After SIGN_MEMORY_MAX_AGE+1 consecutive frames without the sign being seen, its entry must be removed from sign_memory.",
        "stage_dependency": None,
    },
    {
        "id": "S7-005",
        "category": "7_stage7_occlusion",
        "description": "Large confirmed in-corridor vehicle triggers lateral occlusion widening (despite having no valid closing TTC)",
        "target": "large_vehicle_triggers_widening",
        "inputs": {"speed_kmh": 60.0, "vehicle_bbox": [0.45, 0.5, 0.55, 0.66], "n_frames": 4},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "bbox height=0.16 exceeds LARGE_VEHICLE_HEIGHT_THRESHOLD_NORM=0.12. "
            "This vehicle is STATIONARY (no valid TTC), so it is correctly "
            "excluded from normal Stage 3 collision expansion -- the occlusion "
            "response is a SEPARATE mechanism that does not require closing TTC, "
            "since a stationary large vehicle can still occlude a sign."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S7-006",
        "category": "7_stage7_occlusion",
        "description": "Small (car-sized) vehicle does NOT trigger the occlusion response",
        "target": "small_vehicle_no_occlusion_response",
        "inputs": {"speed_kmh": 60.0, "vehicle_bbox": [0.48, 0.5, 0.52, 0.56], "n_frames": 4},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "bbox height=0.06, below LARGE_VEHICLE_HEIGHT_THRESHOLD_NORM=0.12 -- a car is not assumed to occlude signs the way a truck/bus does.",
        "stage_dependency": None,
    },
    {
        "id": "S7-007",
        "category": "7_stage7_occlusion",
        "description": "Vertical peek extends the region's top edge above a large confirmed vehicle",
        "target": "vertical_peek_extends_top",
        "inputs": {"speed_kmh": 60.0, "vehicle_bbox": [0.45, 0.5, 0.55, 0.66], "n_frames": 4},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "y_top should extend to <= the vehicle's own top edge (0.5), approximating the visible gap between a truck's height and an IRC 67 gantry's minimum clearance.",
        "stage_dependency": None,
    },
    {
        "id": "S7-008",
        "category": "7_stage7_occlusion",
        "description": "Occlusion response requires a CONFIRMED track, not a single-frame detection",
        "target": "occlusion_gated_on_confirmed",
        "inputs": {"speed_kmh": 60.0, "vehicle_bbox": [0.45, 0.5, 0.55, 0.66]},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Same ghost-track guard used throughout this module since Stage 3 -- a single-frame (TENTATIVE) large-vehicle detection must not trigger the occlusion response.",
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 8 — Area cap and canonical mapping (added 2026-08-07)
    # ======================================================================
    {
        "id": "S8-001",
        "category": "8_stage8_area_cap",
        "description": "Area cap never shrinks below the floor, even when the floor itself exceeds the cap (Level 3 full-frame)",
        "target": "area_cap_preserves_full_frame_floor",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "MAX_ROI_AREA_FRACTION=0.70; a Level 3 full-frame base (area=1.0) exceeds this, but the cap must have NO effect -- the floor always takes priority over the practical size limit.",
        "stage_dependency": None,
    },
    {
        "id": "S8-002",
        "category": "8_stage8_area_cap",
        "description": "Area cap measurably reduces an over-large combined expansion",
        "target": "area_cap_reduces_oversized",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "base area=0.09, expanded (pre-cap) area=1.0; capped result must be strictly less than 1.0.",
        "stage_dependency": None,
    },
    {
        "id": "S8-003",
        "category": "8_stage8_area_cap",
        "description": "Base ROI is fully retained on every edge after capping (structural floor-protection guarantee)",
        "target": "area_cap_retains_base_edges",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "By construction (margins can only shrink toward zero, never negative), capped_x_left<=base_x_left, capped_x_right>=base_x_right, and the equivalent for y -- verified directly, not merely asserted.",
        "stage_dependency": None,
    },
    {
        "id": "S8-004",
        "category": "8_stage8_canonical_mapping",
        "description": "Canonical mapping uses exactly one uniform scale factor, structurally preventing anisotropic stretching",
        "target": "canonical_single_scale_structural",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "CanonicalMapping dataclass has a single `scale` field, not separate scale_x/scale_y -- the data structure itself makes stretching impossible to accidentally introduce later.",
        "stage_dependency": None,
    },
    {
        "id": "S8-005",
        "category": "8_stage8_canonical_mapping",
        "description": "Round-trip conversion (full-frame -> canonical -> full-frame) recovers the correct position",
        "target": "canonical_roundtrip_accuracy",
        "inputs": {"roi": [0.2, 0.4, 0.6, 0.2], "canonical_w": 512, "canonical_h": 256},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "A synthetic detection at the exact centre of the canonical output must map back to within 0.01 (normalised) of the original crop's centre -- confirms map_roi_to_canonical and canonical_bbox_to_fullframe are true inverses of each other.",
        "stage_dependency": None,
    },
    {
        "id": "S8-006",
        "category": "8_stage8_canonical_mapping",
        "description": "Tall/narrow ROI produces horizontal padding, not vertical (padding lands on the correct axis)",
        "target": "canonical_tall_roi_padding_axis",
        "inputs": {"roi": [0.4, 0.1, 0.1, 0.8], "canonical_w": 512, "canonical_h": 256},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "A crop taller than the canonical aspect ratio should be scaled to exactly fill the canonical HEIGHT, leaving horizontal padding -- the opposite of the wide-crop case already verified, confirming the min() scale selection picks the correct constraining dimension in both directions.",
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 8B — c0 (off-centre) estimation (added 2026-08-11)
    # ======================================================================
    {
        "id": "S8B-001",
        "category": "8b_c0_estimation",
        "description": "Centred lane (at principal point) gives c0=0",
        "target": "c0_centred_is_zero",
        "inputs": {"lane_center": 0.5},
        "expected_type": "value",
        "expected": 0.0,
        "tolerance": 1e-9,
        "basis": "u_px = 0.5*1920 = 960 = principal_x_px -> difference is zero -> c0=0, matching the fixed assumption this replaces.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-002",
        "category": "8b_c0_estimation",
        "description": "c0 matches hand calculation for lane right of centre",
        "target": "c0_hand_calc",
        "inputs": {"lane_center": 0.6},
        "expected_type": "value",
        "expected": 0.96,
        "tolerance": 1e-9,
        "basis": "u_px=0.6*1920=1152; (1152-960)*5/1000 = 0.96m (z_near=Z_NEAR_CUTOFF_M=5.0, f=1000px).",
        "stage_dependency": None,
    },
    {
        "id": "S8B-003",
        "category": "8b_c0_estimation",
        "description": "c0 sign convention: lane left of centre gives negative c0",
        "target": "c0_hand_calc",
        "inputs": {"lane_center": 0.4},
        "expected_type": "value",
        "expected": -0.96,
        "tolerance": 1e-9,
        "basis": "u_px=0.4*1920=768; (768-960)*5/1000 = -0.96m -- opposite sign from S8B-002, confirming the convention matches _project_lateral_to_pixel's positive-X-projects-right rule.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-004",
        "category": "8b_c0_estimation",
        "description": "Missing lane centre falls back to c0=0 (backward compatible)",
        "target": "c0_missing_falls_back",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "No lane centre data available must give the same c0=0.0 as the fixed assumption previously in place -- no regression for callers without lane detection.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-005",
        "category": "8b_c0_estimation",
        "description": "Off-centre lane genuinely shifts the final region end-to-end",
        "target": "c0_shifts_region",
        "inputs": {"speed_kmh": 60.0, "lane_center_a": 0.5, "lane_center_b": 0.6},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "THE CORE FIX: prior to 2026-08-11, an off-centre lane detection had zero effect on the final region because c0 was hardcoded to 0.0 regardless of actual lane position.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-006",
        "category": "8b_c0_estimation",
        "description": "Floor coverage invariant holds across 216 combinations including off-centre lane positions",
        "target": "c0_floor_invariant",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Re-run of the Stage 3-era 96-combination safety sweep, extended to 216 by adding lane centre (0.4/0.5/0.6) as a new varying dimension -- confirms c0 does not break the core safety guarantee now that it genuinely varies.",
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 8B — FOV boundary clamp logging (added 2026-08-11)
    # ======================================================================
    {
        "id": "S8B-007",
        "category": "8b_fov_clamp_logging",
        "description": "No clamping recorded during normal driving",
        "target": "fov_no_clamp_normal",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "A typical speed/curvature combination should never push the floor's mathematical extent beyond the image -- any_clamped() must be False.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-008",
        "category": "8b_fov_clamp_logging",
        "description": "Extreme curvature triggers the clamp flag on the correct side, in both directions",
        "target": "fov_clamp_correct_side",
        "inputs": {"speed_kmh": 120.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Whichever side is flagged clamped must have its own raw (pre-clamp) value outside [0,1] -- checked for both positive and negative extreme curvature, so a left/right mix-up in the flag logic would be caught.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-009",
        "category": "8b_fov_clamp_logging",
        "description": "Passing no diagnostics object is fully backward compatible",
        "target": "fov_backward_compatible",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Every call site from Stages 1-8 calls _invariant_floor without a diagnostics argument -- the default None must produce identical, unaffected floor values.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-010",
        "category": "8b_fov_clamp_logging",
        "description": "ROIGenerator produces a fresh, inspectable diagnostics object every frame",
        "target": "fov_generator_populates_diagnostics",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "self.last_floor_diagnostics must be populated and of the correct type after every step() call when a camera is provided -- this is the actual usable hook for a calling system to inspect.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-011",
        "category": "8b_fov_clamp_logging",
        "description": "End-to-end clamp detection through the full ROIGenerator pipeline, using an anomalous vision-reported curvature",
        "target": "fov_e2e_extreme_vision_curvature",
        "inputs": {"speed_kmh": 120.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Measured directly 2026-08-11: a vision-reported curvature at "
            "MAX_CURVATURE_INV_M with high confidence produces raw_x_left_norm "
            "of approximately -16.6 (the corridor's true mathematical extent "
            "is dramatically beyond the image), correctly flagged as "
            "clamped_left=True. CAN-derived curvature alone could not reach "
            "this extreme, since it is naturally bounded by the lateral-"
            "acceleration limiter in _compute_curvature -- this is why the "
            "vision path was needed to construct a genuine trigger."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S8B-012",
        "category": "8b_fov_clamp_logging",
        "description": "No camera provided gives None diagnostics, not a falsely-reassuring default",
        "target": "fov_no_camera_gives_none",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Without a camera the floor never runs at all -- last_floor_diagnostics must be None, distinguishing 'the floor ran and found no clamping' from 'the floor did not run this frame'.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-013",
        "category": "8b_fov_clamp_logging",
        "description": "Level 3 (full-frame) gives default unclamped diagnostics, honestly reflecting that the floor did not run",
        "target": "fov_level3_default_diagnostics",
        "inputs": {"speed_kmh": 80.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "When dynamics confidence is catastrophically low, _compute_base_roi returns full-frame directly without ever calling _invariant_floor -- diagnostics correctly stays at its default (unclamped) state rather than showing a stale value from a previous frame.",
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 8B — Frame counter and warmed-up state (added 2026-08-11)
    # ======================================================================
    {
        "id": "S8B-014",
        "category": "8b_frame_counter",
        "description": "First-ever step() call reports frames_since_init=1, not 0, and is_warmed_up=False",
        "target": "first_frame_counter_value",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "frames_since_init counts 'how many frames processed', so the first call must report 1. A freshly constructed generator cannot yet be warmed up.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-015",
        "category": "8b_frame_counter",
        "description": "Warm-up boundary is exact: frame WARMUP_FRAMES_REQUIRED-1 is not warmed up, frame WARMUP_FRAMES_REQUIRED is",
        "target": "warmup_boundary_exact",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Direct boundary check -- exactly the kind of off-by-one that could easily hide in a >= vs > comparison. Verified both sides of the boundary explicitly.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-016",
        "category": "8b_frame_counter",
        "description": "Internal ROIParameters construction (not via step()) uses safe backward-compatible defaults",
        "target": "internal_construction_defaults",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "frames_since_init=0, is_warmed_up=True by default -- every pre-existing internal construction site throughout this module (dozens of call sites across Stages 1-8) continues to behave identically without modification.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-017",
        "category": "8b_frame_counter",
        "description": "Stateless generate_dynamic_roi() does not fabricate a meaningful warm-up status",
        "target": "stateless_api_defaults",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "The stateless API has no persistent frame count across calls -- its output correctly stays at the safe defaults rather than claiming a warm-up status it has no basis to report.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-018",
        "category": "8b_frame_counter",
        "description": "Frame counter keeps counting through a Level 3 event; is_warmed_up and roi_level answer different questions and are not conflated",
        "target": "counter_survives_level3",
        "inputs": {"speed_kmh": 60.0, "n_normal_frames": 15},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "A system with 15 frames of established history that then hits a "
            "genuinely bad-confidence frame (Level 3, dynamics catastrophically "
            "unreliable) must report is_warmed_up=True (history IS established) "
            "AND roi_level=3 (current conditions are bad) simultaneously -- the "
            "frame counter must not reset merely because current confidence "
            "dropped. Verified directly: frames_since_init=16 (not reset), "
            "is_warmed_up=True, roi_level=3, all at once."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 8B — Speed plausibility check (added 2026-08-11)
    # ======================================================================
    {
        "id": "S8B-019",
        "category": "8b_speed_plausibility",
        "description": "Normal moving speed is always plausible regardless of other signals",
        "target": "speed_normal_always_plausible",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "This check only ever questions a reading of approximately zero -- a speed of 20 m/s must be accepted at face value.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-020",
        "category": "8b_speed_plausibility",
        "description": "Genuinely stationary (zero speed, zero yaw, no ESC/ABS) is plausible",
        "target": "speed_stationary_plausible",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "An ordinary, unremarkable standstill must not be second-guessed -- both cross-checks correctly find nothing suspicious.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-021",
        "category": "8b_speed_plausibility",
        "description": "Zero speed with significant yaw rate is implausible",
        "target": "speed_zero_with_yaw_implausible",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "A vehicle rotating meaningfully cannot simultaneously be genuinely stationary -- cross-check 1.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-022",
        "category": "8b_speed_plausibility",
        "description": "Zero speed with ESC or ABS active is implausible",
        "target": "speed_zero_with_stability_systems_implausible",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "ESC and ABS are stability-control and anti-lock braking systems; neither engages at a genuine standstill -- cross-check 2, verified for both flags independently.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-023",
        "category": "8b_speed_plausibility",
        "description": "An invalid/unavailable yaw signal cannot be used as evidence of motion",
        "target": "speed_invalid_yaw_not_evidence",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "yaw_rate_valid=False must not let a numerically-present yaw_rate_dps value be treated as proof of anything -- absence of valid evidence is not evidence of motion.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-024",
        "category": "8b_speed_plausibility",
        "description": "REGRESSION: speed_was_implausible flag survives the full pipeline (expansion, smoothing, final clamp) to reach the final output",
        "target": "speed_flag_survives_pipeline",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Direct regression test for a real bug found and fixed during "
            "this same implementation session (2026-08-11): both "
            "_smooth_asymmetric() and the final safety-clamp reconstruction "
            "in ROIGenerator.step() build a brand new ROIParameters from "
            "scratch and were silently dropping speed_was_implausible before "
            "it ever reached the caller. Fixed by capturing the value in a "
            "local variable immediately after _compute_base_roi returns, "
            "and stamping it explicitly at the final clamp, plus fixing "
            "_smooth_asymmetric to carry it forward for any other caller."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S8B-025",
        "category": "8b_speed_plausibility",
        "description": "The substitution genuinely changes the computed region, not just the reported flag",
        "target": "speed_substitution_changes_region",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "The whole point of this feature is safety-relevant, not cosmetic -- the region height with an implausible-zero substitution must be measurably taller than the height a genuinely-trusted zero speed would produce.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-026",
        "category": "8b_speed_plausibility",
        "description": "The caller's original CanSignals object is never mutated",
        "target": "speed_no_mutation",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "_compute_base_roi uses dataclasses.replace() to build a corrected copy internally -- the caller's own CanSignals instance must be left completely untouched after the call returns.",
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 8B — Warm-restart state-restoration policy (added 2026-08-11)
    # ======================================================================
    {
        "id": "S8B-027",
        "category": "8b_warm_restart",
        "description": "reset_for_warm_restart() fully clears all time-dependent (unsafe) state",
        "target": "warm_restart_clears_unsafe_state",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Tracked objects, smoothing history, sign memory, frame count, and floor diagnostics must ALL reset -- each is time-dependent state that may be stale or wrong after an interruption of unknown length.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-028",
        "category": "8b_warm_restart",
        "description": "reset_for_warm_restart() fully preserves physically/configuration-fixed state, including a non-default iou_threshold",
        "target": "warm_restart_preserves_safe_config",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Camera calibration, confidence gates, and every construction-time configuration choice (including a deliberately non-default iou_threshold=0.45, to catch a silent fallback to the default) must survive untouched.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-029",
        "category": "8b_warm_restart",
        "description": "Generator behaves exactly like a fresh instance immediately after reset -- no smoothing lag against stale history",
        "target": "warm_restart_fresh_behaviour",
        "inputs": {"speed_kmh": 120.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "With prev_roi cleared, the first frame after reset must hit the true target height immediately (no lag) -- exactly correct for a vehicle still moving fast right through the restart.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-030",
        "category": "8b_warm_restart",
        "description": "Reset generator's tracker is genuinely fresh and functional, not a broken half-state",
        "target": "warm_restart_tracker_functional",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "A new detection introduced after reset must be able to complete the normal TENTATIVE -> CONFIRMED lifecycle exactly as it would on a brand-new generator.",
        "stage_dependency": None,
    },

    # ======================================================================
    # STAGE 8B — Input quantization (added 2026-08-11, final item)
    # ======================================================================
    {
        "id": "S8B-031",
        "category": "8b_quantization",
        "description": "Speed bucketing is monotonic across the full tested range (0-500 m/s)",
        "target": "quant_speed_monotonic",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "This is the property that makes simple upper-edge rounding safe for speed -- height must never decrease as speed increases.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-032",
        "category": "8b_quantization",
        "description": "Confidence bucketing is monotonic across the full range (0.0-1.0)",
        "target": "quant_confidence_monotonic",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "This is the property that makes simple lower-edge rounding safe for confidence -- width must never decrease as confidence decreases.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-033",
        "category": "8b_quantization",
        "description": "CRITICAL SAFETY FINDING: naive curvature rounding (round magnitude up, evaluate once) is demonstrably UNSAFE for the far/outside lateral edge",
        "target": "quant_naive_curvature_unsafe",
        "inputs": {"speed_kmh": 80.0, "true_kappa": 0.003, "rounded_kappa": 0.01},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "Measured directly 2026-08-11: at 80 km/h, true kappa=0.003 gives "
            "x_right=0.7073, but naively rounding up to the 0.01 band edge and "
            "evaluating once there gives x_right=0.6891 -- UNDER-covering by "
            "0.0182 (2.6%). This happens because a sharper curve correctly "
            "needs LESS coverage on the outside edge than a gentler one -- "
            "the underlying relationship is genuinely non-monotonic, not a "
            "bug in a single evaluation. This finding is why the envelope "
            "approach (S8B-034) exists instead of simple edge-rounding for "
            "curvature specifically."
        ),
        "stage_dependency": None,
    },
    {
        "id": "S8B-034",
        "category": "8b_quantization",
        "description": "The envelope approach correctly covers exactly the case naive rounding fails",
        "target": "quant_envelope_fixes_naive_failure",
        "inputs": {"speed_kmh": 80.0, "true_kappa": 0.003},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Sampling several points across the curvature band and taking the outer envelope is safe by construction regardless of which direction any individual edge moves -- verified directly against the exact case S8B-033 shows the naive approach failing.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-035",
        "category": "8b_quantization",
        "description": "Envelope contains the true floor across 500 random samples spanning the full speed/curvature/confidence range",
        "target": "quant_envelope_dense_random",
        "inputs": {"n_samples": 500, "seed": 123},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Broad randomized confirmation beyond the specific case already checked -- no violations found across 500 combinations independently sampled from the full 0-150km/h, +/-0.20/m, 0.0-1.0 operating range.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-036",
        "category": "8b_quantization",
        "description": "Default (quantize_inputs=False) behaviour is completely unchanged -- backward compatible with every existing hand-calculated value",
        "target": "quant_default_backward_compatible",
        "inputs": {"speed_kmh": 83.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "quantize_inputs is opt-in, defaulting to False, specifically so every existing hand-calculated test value throughout this entire project (which assumes the exact unbucketed formula) remains valid and unaffected.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-037",
        "category": "8b_quantization",
        "description": "Quantized result is equal-or-wider than the exact calculation for the same true inputs",
        "target": "quant_result_equal_or_wider",
        "inputs": {"speed_kmh": 83.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "The whole point of conservative bucketing -- the quantized region must never be narrower than what the exact formula would have produced for the same true inputs.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-038",
        "category": "8b_quantization",
        "description": "Same speed band gives a bit-identical result across genuinely different true speeds",
        "target": "quant_stability_property",
        "inputs": {"speed_kmh_a": 81.0, "speed_kmh_b": 99.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "THE ACTUAL PROMISED PROPERTY from the manager's original suggestion: two different true speeds (81 and 99 km/h) both falling in the 80-100km/h band must produce a bit-identical result -- this is what makes the input domain finite and exhaustively checkable.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-039",
        "category": "8b_quantization",
        "description": "FloorClampDiagnostics honestly reports None on the quantized path rather than a misleading single-sample status",
        "target": "quant_diagnostics_none",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "The envelope approach unions several evaluations -- there is no single 'the' raw clamp status left to report, so diagnostics must stay None rather than show one arbitrary sample's status.",
        "stage_dependency": None,
    },
    {
        "id": "S8B-040",
        "category": "8b_quantization",
        "description": "EXHAUSTIVE CHECK: every (speed band, curvature band, confidence band) combination in the full grid is verified safe",
        "target": "quant_exhaustive_grid_check",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "THE ACTUAL PAPER CLAIM, demonstrated rather than merely stated: "
            "verified directly on 2026-08-11 across the full grid of 7 speed "
            "bands x 7 curvature bands x 4 confidence bands x 3 sample points "
            "per dimension x 2 curvature signs = 10,584 total combinations "
            "checked, with zero violations. This is what 'a finite, "
            "exhaustively-checkable input domain' actually means in this "
            "project, not an assertion taken on faith."
        ),
        "stage_dependency": None,
    },

    # ======================================================================
    # Validation matrix gap closure (added 2026-08-12)
    # ======================================================================
    {
        "id": "VM-001",
        "category": "9_validation_matrix",
        "description": "Cut-in expansion is synchronized with corridor entry, not premature and not delayed",
        "target": "vm_cutin_timing",
        "inputs": {"speed_kmh": 60.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": (
            "A vehicle already confirmed and producing valid TTC while still "
            "outside the corridor must show zero expansion until it crosses "
            "in. Measured directly 2026-08-12: all three gating conditions "
            "(confirmed, in-corridor, valid TTC) become true simultaneously "
            "at the crossing frame in this construction, and a substantial "
            "expansion (delta > 0.25) appears in that exact frame -- no "
            "premature firing, no extra delay."
        ),
        "stage_dependency": None,
    },
    {
        "id": "VM-002",
        "category": "9_validation_matrix",
        "description": "Computed TTC matches a hand-derived ground-truth value within tolerance",
        "target": "vm_ttc_ground_truth",
        "inputs": {"true_vy_per_frame": 0.02, "n_frames": 8},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "For a sequence with a KNOWN injected true vertical velocity, the hand-evaluated TTC formula (h/vy)*dt_s using the TRUE vy is compared against _estimate_ttc()'s actual output (using Kalman-filtered vy) -- within 15% tolerance for filter convergence lag.",
        "stage_dependency": None,
    },
    {
        "id": "VM-003",
        "category": "9_validation_matrix",
        "description": "Two different sign categories in one frame coexist without collision",
        "target": "vm_signs_different_categories",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "SIGN_ROADSIDE and SIGN_OVERHEAD are different dict keys -- both must be independently remembered when detected simultaneously.",
        "stage_dependency": None,
    },
    {
        "id": "VM-004",
        "category": "9_validation_matrix",
        "description": "Two same-category signs collide into one memory slot with documented last-wins behaviour",
        "target": "vm_signs_same_category_collision",
        "inputs": {},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "sign_memory is keyed by category (documented scope limitation) -- two SIGN_ROADSIDE detections in one frame must deterministically keep the LAST-processed one, not crash or keep a stale value.",
        "stage_dependency": None,
    },
    {
        "id": "VM-005",
        "category": "9_validation_matrix",
        "description": "CAN dropout alone (both channels invalid), healthy lane, lands specifically at Level 1",
        "target": "vm_can_dropout_alone",
        "inputs": {"speed_kmh": 60.0, "lane_conf": 0.9},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Previously only tested with one CAN channel invalid at a time, always alongside other scenario framing -- this isolates the clean case: both steering_valid=False and yaw_rate_valid=False, confident lane, must give Level 1 with curvature=0.0 and floor still using speed alone.",
        "stage_dependency": None,
    },
    {
        "id": "VM-006",
        "category": "9_validation_matrix",
        "description": "FOV clamp frequency can be correctly derived by a caller from the provided per-frame flag",
        "target": "vm_fov_frequency_derivable",
        "inputs": {"total_frames": 20, "injection_period": 4},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Frequency aggregation is deliberately NOT built into this module (Section 22 design decision) -- this confirms the exposed per-frame flag is sufficient for a caller to correctly compute a known, injected clamp frequency.",
        "stage_dependency": None,
    },
    {
        "id": "VM-007",
        "category": "9_validation_matrix",
        "description": "Vertical margin under ABS-active braking isolated and directly verified (not just lateral, as in every prior test)",
        "target": "vm_vertical_margin_isolated",
        "inputs": {"speed_kmh": 40.0},
        "expected_type": "bool",
        "expected": True,
        "tolerance": None,
        "basis": "Every prior abs_active test checked lateral width only. Direct isolated comparison: y_top extends further (smaller), y_bottom extends further (larger), and total vertical extent is strictly larger with abs_active=True vs False, holding everything else fixed.",
        "stage_dependency": None,
    },
]
