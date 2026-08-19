"""
Dynamic ROI Generator  —  FCW + Lane + ISA (signals, signs, gantry)
====================================================================
STAGE 1 UPDATE (Foundation): adds camera intrinsics, the physics-based
stopping-distance calculation, and the invariant collision-coverage
floor. The vertical region is no longer a hand-tuned heuristic — it is
derived from ego speed, road curvature, and camera geometry.

Everything in this file marked "STAGE 2+" is deliberately unchanged
from the previous version and will be addressed in later stages, per
the consolidated roadmap in review_note.md.

Enhanced with:
  • TTC-adaptive ROI scaling for FCW urgency zones
  • Kalman filter per-track state estimation (BiTrack-compatible)
  • Track lifecycle management (tentative → confirmed → lost)
  • STAGE 1: Physics-based invariant collision-coverage floor

Coordinate space : fully normalised [0.0, 1.0]  (origin = top-left)
ROI output       : (x_left, y_top, width, height, roi_level) — normalised
Target platform  : L2- ADAS (truck / heavy vehicle), ~4 TOPS accelerator

Design
------
A single ROI is returned every frame covering:

  Zone D  Forward path      lane + curvature + speed            (always)
  Zone B  Traffic signals   vertical pull toward horizon         (conf-gated)
  Zone C  Roadside signs    full bbox lateral + vertical         (conf-gated)
  Zone A  Gantry / overhead y_top pulled toward 0.0             (conf-gated)

STAGE 1 — Invariant Collision-Coverage Floor (NEW)
---------------------------------------------------
  Before any lane-based or object-based logic runs, a minimum region
  is calculated directly from ego speed, road curvature, and camera
  geometry — no detections required. This floor guarantees that the
  road surface from a near cutoff out to the vehicle's stopping
  distance is always inside the final region.

  The final base region is the widest bound, edge by edge, of:
    (a) this physics-based floor, and
    (b) the existing lane-centred calculation.

  This means the floor can only be added to, never reduced below,
  by anything that follows it (lane centring, object expansion).

  Grounding for the constants used here:
    - Warning-time budget (T_TTC_MIN_S = 2.1 s) follows ISO 15623's
      minimum time-to-collision threshold for a stationary lead
      vehicle — the most demanding of that standard's three defined
      scenarios (2.1 s stopped, 2.4 s decelerating, 1.8 s slower-
      moving), and is broadly consistent with Euro NCAP's AEB test
      protocol (1.2 s assumed driver-reaction delay) once combined
      with typical braking response latency.
    - Base lane width (LANE_WIDTH_STD_M = 3.5 m) follows
      IRC:SP:73-2015 for Indian National Highways.
    - The lateral wander allowance and curvature-uncertainty
      coefficient below are engineering placeholders pending
      validation against recorded driving data (see review_note.md,
      Section 4.5 / Section 5.3 "Pending Confirmation" items).

TTC Integration
---------------
  TTC is estimated per tracked vehicle from Kalman-filtered relative
  depth velocity.  Three urgency zones map to ROI expansion multipliers:

    TTC > TTC_WARN_S           : nominal expansion (multiplier = 1.0)
    TTC_CRIT_S < TTC ≤ TTC_WARN_S : warning zone  (multiplier = TTC_WARN_SCALE)
    TTC ≤ TTC_CRIT_S           : critical zone    (multiplier = TTC_CRIT_SCALE)

  The multiplier inflates OBJECT_MARGIN for FCW vehicle expansions only.

  STAGE 3 (not yet implemented in this file): expansion should also be
  gated on corridor membership and a confirmed, valid TTC — currently
  any detected vehicle above the confidence threshold still expands
  the region. Tracked as a known, documented gap; see review_note.md.

Tracking Integration
--------------------
  Each DetectedObject may carry an optional track_id.  The tracker
  maintains a KalmanTrack per ID with a 4-state model:

    state = [cx, cy, w, h]   (normalised image coordinates)
    motion model             : constant velocity in cx, cy; constant w, h
    measurement              : [cx, cy, w, h] from detector

  Track lifecycle:
    TENTATIVE  : first N_INIT frames — detection confirmed, not yet used
                 for TTC (avoids ghost-track TTC triggers)
    CONFIRMED  : hit_streak >= N_INIT — full expansion + TTC applied
    LOST       : missed >= MAX_AGE    — track deleted

  BiTrack note:
    This module implements the Kalman state independently.  If BiTrack
    (or any external SORT/DeepSORT-compatible tracker) is used upstream,
    set use_external_tracker=True when constructing ROIGenerator.
    In that mode DetectedObject.track_id carries the external ID, and
    the POSITION returned downstream is always the raw external bbox
    (never overwritten by internal filtering). However, an internal
    Kalman filter STILL RUNS in the background even in external mode —
    it is fed the external bbox every frame purely so that TTC can
    still be estimated from it, since the external tracker interface
    has no field for supplying velocity/TTC directly. CORRECTED
    2026-08-06: earlier wording here claimed "the internal Kalman step
    is skipped," which was inaccurate — it is only the RETURNED
    POSITION that bypasses internal filtering, not the internal
    velocity/TTC machinery, which continues to run either way.

Vertical rules (STAGE 1 — UPDATED)
-----------------------------------
  y_top / height : derived from the invariant floor (Z_max projection),
                    combined with the previous lane-based calculation.
                    Replaces the old fixed SKY_CLIP_Y_TOP / _depth_ratio()
                    heuristic entirely — see "Deleted in Stage 1" below.
  y_bottom       : near-field cutoff from the floor's Z_min, clamped to
                    HOOD_Y_BOTTOM.

Deleted in Stage 1 (previously heuristic, now replaced by the floor)
----------------------------------------------------------------------
  SKY_CLIP_Y_TOP, DEPTH_RATIO_BASE, DEPTH_RATIO_MIN, _depth_ratio()
  FOCAL_NORM_APPROX, OBJECT_REAL_HEIGHT_M
    — confirmed dead code in the prior version (declared, never used
      anywhere in the TTC calculation or elsewhere).

Smoothing (STAGE 4 — not yet updated in this file)
----------------------------------------------------
  Still the original single-rate IIR with a snap threshold, applied
  only to (centre_x, width). Known limitations documented in
  review_note.md Section 2.7 / 4.x — fast-grow/slow-shrink asymmetric
  filtering is planned but not yet implemented here.

Object expansion (STAGE 3 — not yet updated in this file)
-------------------------------------------------------------
  Expansion fires only when confidence >= per-category threshold.
  Thresholds are injectable via ConfidenceGates.

  VEHICLE       lateral + vertical        (FCW; default threshold = 0.0)
                KNOWN GAP: not yet gated on corridor membership or
                valid TTC — see review_note.md Stage 3.
  SIGNAL        vertical only             (threshold = 0.55)
  SIGN_ROADSIDE lateral + vertical        (threshold = 0.60)
  SIGN_OVERHEAD y_top toward 0.0 only    (threshold = 0.60)

Degradation levels (surfaced via roi_level) — STAGE 2 IMPLEMENTED
---------------------------------------------------------------------------
  As of Stage 2, positioning is a CONTINUOUS confidence-weighted blend,
  not a hard level switch. `roi_level` is retained purely as a
  discretised diagnostic/logging label, not as branching logic:

  0 — high confidence : overall confidence >= CONF_BLEND_HIGH and
                          CAN steer/yaw signals valid; full lane-informed
                          centring, floor applied
  1 — degraded/blended: confidence between CONF_BLEND_LOW and
                          CONF_BLEND_HIGH (or CAN dynamics signals
                          themselves invalid); a continuous mix of
                          lane-informed and CAN-only centring, floor
                          applied with dynamics-confidence-scaled
                          corridor width
  2 — CAN-only fallback: overall confidence <= CONF_BLEND_LOW; lateral
                          centring uses CAN curvature only (no lane
                          trust at all), floor still applied — THIS IS
                          THE FIX for the previously-documented gap:
                          CAN curvature now always shifts this fallback,
                          it is never a frozen/unshifted rectangle
  3 — full-frame fallback: dynamics confidence itself (not lane
                          confidence — see _compute_base_roi docstring
                          for why these are kept separate) below
                          CONF_LEVEL3_THRESHOLD; returns the entire
                          image, no positioning attempted

  See _compute_base_roi()'s docstring for the full reasoning behind
  keeping three separate confidence questions distinct rather than
  collapsing them into one number.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ==========================================================================
# Constants  — all tunables in one flat block
# ==========================================================================

# --- Vehicle / road geometry ---
WHEELBASE_M                  = 5.5    # truck wheelbase [m]
LANE_WIDTH_M                 = 3.5    # nominal lane width, metric reference [m]

# --- Preview horizon (STAGE 2+: lane-based lateral shift, unchanged) ---
PREVIEW_TIME_S               = 0.60   # look-ahead time [s]
PREVIEW_DIST_MIN_M           = 15.0   # minimum preview distance — truck braking envelope [m]
MIN_SPEED_FOR_PREVIEW_MPS    = 3.0    # below this speed lateral offset is zeroed [m/s]

# --- Curvature / dynamics ---
MAX_LAT_ACC_MPS2             = 3.0    # comfortable lateral acceleration limit [m/s²]
MAX_SPEED_MPS                = 30.0   # speed normalisation ceiling [m/s]
MAX_CURVATURE_INV_M          = 0.20   # hard curvature saturation [1/m]
CURVATURE_NEAR_ZERO          = 1e-5   # threshold below which curvature is treated as zero [1/m]
SPEED_CURVATURE_FLOOR_MPS    = 0.5    # minimum speed used in curvature denominator [m/s]
SPEED_DYNAMICS_FLOOR_MPS     = 0.5    # below this speed dynamics are skipped entirely [m/s]
LAT_ACC_DENOM_FLOOR          = 0.5    # floor for speed² in lateral acc limit [m²/s²]

# ==========================================================================
# STAGE 8B — Cold/hot start: speed plausibility check (NEW, added 2026-08-11)
# ==========================================================================

SPEED_ZERO_THRESHOLD_MPS      = 0.5   # a reported speed at or below this is
    # treated as "reads approximately zero" for plausibility-checking
    # purposes — matches SPEED_DYNAMICS_FLOOR_MPS above, deliberately,
    # since both concern the same practical notion of "close enough to
    # stationary to matter."
YAW_SUGGESTS_MOVING_THRESHOLD_DPS = 2.0  # a yaw rate above this, while
    # speed reads approximately zero, is treated as evidence the
    # vehicle is very likely actually moving — a vehicle rotating
    # meaningfully cannot simultaneously be genuinely at a standstill.
    # Matches YAW_MISMATCH_MILD_DPS's own threshold (Stage 2), reusing
    # the same physically-motivated scale rather than inventing a new one.
DEFAULT_ASSUMED_SPEED_MPS_ON_IMPLAUSIBLE_ZERO = MAX_SPEED_MPS  # if a
    # near-zero speed reading is judged implausible (see
    # _is_speed_plausible), THIS value is used for every downstream
    # calculation instead. Deliberately reuses MAX_SPEED_MPS (30 m/s,
    # ~108 km/h) — an already-existing, already-referenced constant in
    # this module — rather than introducing a new arbitrary number.
    # This is a conservative, "assume the worst plausible case" choice:
    # addresses the specific hazard identified in review_note.md
    # Section 19.4 — a brief restart while genuinely moving, with CAN
    # not yet delivering valid speed data, must not be allowed to
    # produce a floor sized for a stationary vehicle when the vehicle
    # may actually be travelling at speed.

# --- Yaw reliability ---
YAW_MISMATCH_THRESHOLD_DPS   = 5.0    # max allowed yaw error before fallback to steering [deg/s]

# ==========================================================================
# STAGE 1 — Camera intrinsics and invariant floor constants (NEW)
# ==========================================================================

# --- Warning-time budget, grounded in ISO 15623 (see module docstring) ---
TTC_MIN_WARNING_S             = 2.1    # ISO 15623 stationary-lead-vehicle threshold [s]
MAX_BRAKING_DECEL_MPS2        = 6.0    # comfortable-to-firm braking deceleration, truck [m/s²]
Z_NEAR_CUTOFF_M               = 5.0    # near-field cutoff — below this, other sensors cover it [m]
Z_MIN_FLOOR_DEPTH_M           = 15.0   # minimum (Z_max - Z_near) even at zero speed — found
                                        # during Stage 1 verification: without this, a stationary
                                        # vehicle (Z_max formula naturally returns 0) collapses
                                        # the floor to zero height, which is degenerate/unsafe.
                                        # A stopped vehicle still needs forward visibility, e.g.
                                        # to see traffic ahead starting to move again.

# --- Corridor width, grounded in IRC 73 (see module docstring) ---
LANE_WIDTH_STD_M              = 3.5    # IRC:SP:73-2015 standard NH lane width [m]
LATERAL_WANDER_SIGMA_M        = 0.30   # PENDING CONFIRMATION — placeholder lane-centring
                                        # error; replace with measured LKA performance [m]
CURVATURE_ERROR_SIGMA_INV_M   = 0.0006 # PENDING CONFIRMATION — placeholder curvature
                                        # uncertainty; derived as ~15% of the worst-case
                                        # curvature at ISO 15623's 125 m minimum design
                                        # curve radius (c2 ≈ 1/(2*125) = 0.004 /m). [1/m]

# --- Vertical / pitch handling ---
CAMERA_PITCH_RAD_DEFAULT      = 0.0    # assumed level camera when no pitch estimate given [rad]
ABS_ACTIVE_VERTICAL_MARGIN_M  = 1.0    # STAGE 1 fallback: conservative extra vertical
                                        # margin applied when hard braking is flagged,
                                        # in lieu of precise pitch estimation (deferred
                                        # to Future Improvements per review_note.md) [m]

# --- Optional ISA / gantry vertical extension ---
GANTRY_HEIGHT_M                = 5.5   # IRC 67 minimum overhead gantry clearance [m]
GANTRY_MIN_READ_DISTANCE_M     = 30.0  # closest distance a gantry sign must be resolvable [m]

# ==========================================================================
# STAGE 7 — Sign handling: readability horizon and occlusion response (NEW)
# ==========================================================================

# --- ISA decision-time budget ---
# Deliberately DIFFERENT from FCW's ISO 15623-grounded TTC_MIN_WARNING_S
# (2.1s) — reading and reacting to a speed-limit sign is a routine
# driving task, not an emergency collision response, so the AASHTO
# perception-reaction reference (2.5s, already discussed and cited in
# Section 4.4 of review_note.md for a DIFFERENT purpose there) is the
# appropriate grounding here, not the FCW-specific standard.
ISA_REACTION_TIME_S           = 2.5    # AASHTO Green Book perception-reaction reference
ISA_COMFORTABLE_DECEL_MPS2    = 1.5    # gentle, non-emergency deceleration to a new lower limit
ISA_MIN_SIGN_PX               = 20.0   # minimum resolvable size for sign classification —
                                        # consistent with the pixel-budget reasoning discussed
                                        # earlier in this project (small fixed-class classifier,
                                        # not general OCR; ~20-30px established as the reliable
                                        # threshold for distinguishing similar sign values)

# --- IRC 67 reference sign diameters, by design-speed bracket ---
# Grounded values from IRC:67 (regulatory sign sizing table), kept here
# as named constants rather than embedded magic numbers so each can be
# cited independently.
IRC67_SIGN_DIAMETER_LE_65_M    = 0.60
IRC67_SIGN_DIAMETER_66_80_M    = 0.75
IRC67_SIGN_DIAMETER_81_100_M   = 0.90
IRC67_SIGN_DIAMETER_101_120_M  = 1.20
IRC67_SIGN_DIAMETER_121_150_M  = 1.50

# --- Occlusion response ---
LARGE_VEHICLE_HEIGHT_THRESHOLD_NORM = 0.12  # bbox height above which a
    # tracked vehicle is treated as "large" (truck/bus-scale, tall
    # enough to plausibly occlude a roadside or overhead sign) rather
    # than a car. PENDING CONFIRMATION — a reasonable starting estimate
    # consistent with the scale used throughout this module's test
    # fixtures; not yet validated against real large-vehicle imagery.
SIGN_MEMORY_MAX_AGE           = 5      # frames a remembered sign's
    # position is still trusted after last being directly seen, before
    # the occlusion response stops reaching for it. Deliberately reuses
    # the SAME numeric value as MAX_AGE (vehicle track lifecycle) for
    # consistency, not because the two concepts must always match.
OCCLUSION_LATERAL_WIDEN_NORM  = 0.15  # additional lateral half-width
    # margin applied when a large vehicle is judged sign-occluding —
    # approximates reaching toward an opposite-side redundant sign
    # (IRC 67 multi-lane placement requirement) without a precise
    # distance-based projection, given known limitations in this
    # module's monocular distance estimation (see Section 12.5's
    # documented head-on-approach limitation). PENDING CONFIRMATION.
OCCLUSION_VERTICAL_PEEK_NORM  = 0.06  # additional upward extension
    # (toward smaller y) applied above a large tracked vehicle's bbox,
    # approximating the gap between a typical truck's height and IRC
    # 67's minimum gantry clearance (GANTRY_HEIGHT_M) so an overhead
    # sign partially visible above the vehicle stays inside the region.
    # PENDING CONFIRMATION — a fixed normalised margin rather than a
    # precise per-frame projection, for the same reason as above.

# ==========================================================================
# STAGE 8 — Region size cap and canonical (accelerator) mapping (NEW)
# ==========================================================================

MAX_ROI_AREA_FRACTION = 0.70   # cap on the FINAL region's area, as a
    # fraction of the full frame — prevents multiple simultaneous
    # expansions (vehicle + sign + occlusion + peek) from silently
    # combining toward full-frame when no single one looks excessive
    # on its own. This cap can NEVER intrude into the invariant floor
    # — see _apply_area_cap()'s docstring for exactly how that is
    # guaranteed by construction, not merely by a runtime check.

# --- Bounded resize (letterbox padding, not stretching) ---
# Grounded in the design decision recorded in review_note.md Section
# 4.3: uniform scaling preserves object proportions; anisotropic
# stretching would distort them relative to what a detector trained on
# undistorted images expects. No new tunable constants needed here —
# the canonical (accelerator) input size is supplied by the caller,
# since it is a property of the target hardware/model, not of this
# module.

# --- Vertical ROI (STAGE 2+ level-2 static fallback only; dynamic path now uses the floor) ---
HOOD_Y_BOTTOM                 = 0.97   # fixed bottom edge — ego hood [norm]
IMAGE_Y_MIN                   = 0.0    # hard upper floor — gantry expansion limit [norm]
IMAGE_Y_MAX                   = 1.0    # hard lower floor [norm]

# --- Lateral ROI ---
LATERAL_HALF_SCALE_L0        = 0.75   # half-lane multiplier at level 0 (full dynamic)
LATERAL_HALF_SCALE_L1        = 0.85   # half-lane multiplier at level 1 (lane only)
MIN_WIDTH_SCALE_AT_MAX_SPEED = 0.85   # lateral squeeze factor at MAX_SPEED_MPS
MAX_LATERAL_SHIFT_FRACTION   = 0.40   # max shift as fraction of lane width [norm]
LANE_HW_CLAMP_MIN            = 0.05   # minimum half-lane-width clamp [norm]
LANE_HW_CLAMP_MAX            = 0.45   # maximum half-lane-width clamp [norm]

# --- Lane / level thresholds ---
LANE_CONF_MIN                = 0.5    # STAGE 2: retained only as a reference
                                        # value inside _estimate_confidence()'s
                                        # lane-confidence contribution; no
                                        # longer used as a hard switch anywhere
                                        # in _compute_base_roi().

# ==========================================================================
# STAGE 2 — Unified confidence and degradation-blending constants (NEW)
# ==========================================================================

# --- Yaw-steering mismatch -> dynamics confidence mapping ---
# Grounded in the mismatch-magnitude tiers discussed during design
# (review_note.md Section 4.1 / the slip-handling conversation):
#   <2 deg/s   : normal, full confidence
#   2-5 deg/s  : mild instability
#   5-15 deg/s : significant slip
#   >15 deg/s  : loss of control
YAW_MISMATCH_MILD_DPS         = 2.0    # below this: no dynamics confidence penalty
YAW_MISMATCH_SIGNIFICANT_DPS  = 5.0
YAW_MISMATCH_SEVERE_DPS       = 15.0
DYNAMICS_CONF_MILD            = 0.7    # confidence ceiling once mild mismatch begins
DYNAMICS_CONF_SIGNIFICANT     = 0.3
DYNAMICS_CONF_SEVERE          = 0.1

ESC_ACTIVE_CONF_CEILING       = 0.3    # confidence ceiling when ESC is intervening
ABS_ACTIVE_CONF_CEILING       = 0.5    # confidence ceiling when ABS is active

# --- Overall confidence -> region blending ---
# Below CONF_BLEND_LOW: pure fallback (CAN-only centring, widest corridor)
# Above CONF_BLEND_HIGH: pure dynamic (full lane-based centring)
# Between: linear interpolation — this is what replaces the old hard
# switch at LANE_CONF_MIN, eliminating the single-frame jump documented
# as a known defect in review_note.md Section 2.4/2.7.
CONF_BLEND_LOW                = 0.40
CONF_BLEND_HIGH               = 0.65

# Below this, abandon even the CAN-only fallback and return the full
# image — Stage 2's new Level 3, addressing the previously-missing
# "nothing reliable" floor described in review_note.md Section 5.3.
CONF_LEVEL3_THRESHOLD         = 0.15

# --- Confidence-scaled corridor widening ---
# expansion = 1 + (MAX_EXPANSION-1)*(1-confidence); at confidence=1.0
# the corridor is at its base width, at confidence=0.0 it is
# MAX_EXPANSION times wider. Matches the design derivation in
# review_note.md Section 4.1 / Section 8.5 of the earlier technical
# notes (ported here as this project's authoritative design record).
CORRIDOR_MAX_EXPANSION_FACTOR = 3.0

# ==========================================================================
# STAGE 5 — Curvature source fusion constants (NEW)
# ==========================================================================

# --- Vision trust threshold ---
VISION_CURVATURE_TRUST_THRESHOLD = 0.5   # c2_confidence must be at least
    # this for vision curvature to be used as the SOURCE VALUE at all.
    # Below this, fall back to the CAN bicycle-model curvature — vision
    # is preferred over CAN whenever it is reasonably confident, because
    # it provides genuine look-ahead (the road shape ahead), whereas CAN
    # only reflects instantaneous curvature at the vehicle's own position
    # (see review_note.md Section 7.2 for the full reasoning).

# --- CAN/vision mismatch -> curvature-agreement confidence tiers ---
# Grounded the same way as CURVATURE_ERROR_SIGMA_INV_M (Section 6.4/23.2):
# using ISO 15623's 125m minimum design curve radius as the reference
# scale (c2 ~= 1/(2*125) = 0.004 /m). Mismatch tiers are expressed as
# fractions/multiples of that reference value, not arbitrary numbers.
CURVATURE_MISMATCH_MILD_INV_M        = 0.001   # ~25% of the ISO 15623 reference curvature
CURVATURE_MISMATCH_SIGNIFICANT_INV_M = 0.004   # == the ISO 15623 reference curvature itself
CURVATURE_MISMATCH_SEVERE_INV_M      = 0.010   # 2.5x the reference — sources fundamentally disagree
CURVATURE_AGREEMENT_CONF_MILD        = 0.70
CURVATURE_AGREEMENT_CONF_SIGNIFICANT = 0.40
CURVATURE_AGREEMENT_CONF_SEVERE      = 0.15

# --- Object expansion ---
OBJECT_MARGIN                = 0.015  # base padding added around expanded bboxes [norm]

# --- Default confidence gates ---
DEFAULT_CONF_VEHICLE         = 0.0    # confidence threshold is intentionally
                                       # permissive (FCW-critical) — STAGE 3:
                                       # the actual gating that prevents
                                       # over-eager expansion is no longer
                                       # this threshold, but the separate
                                       # corridor-membership + confirmed-track
                                       # + valid-TTC checks in
                                       # _apply_object_expansions(). See
                                       # _is_in_corridor() and that function's
                                       # docstring.
DEFAULT_CONF_SIGNAL          = 0.55   # traffic light minimum confidence
DEFAULT_CONF_SIGN_ROADSIDE   = 0.60   # roadside sign minimum confidence
DEFAULT_CONF_SIGN_OVERHEAD   = 0.60   # gantry sign minimum confidence

# --- Smoothing: STAGE 4 IMPLEMENTED. Old single-rate IIR (SMOOTHING_ALPHA)
# and snap-threshold constants (ROI_SNAP_CX_THRESHOLD, ROI_SNAP_W_THRESHOLD)
# REMOVED 2026-08-06 — replaced by the asymmetric per-edge filter
# (ASYM_ALPHA_GROW_EDGE, ASYM_ALPHA_SHRINK_EDGE, defined alongside
# _smooth_asymmetric() further down this file, where the filter itself
# lives) rather than kept as unused dead constants alongside it.

# --- Safety clamp bounds ---
ROI_WIDTH_MIN                = 0.05   # minimum allowed ROI width [norm]
ROI_WIDTH_MAX                = 1.0    # maximum allowed ROI width [norm]
ROI_HEIGHT_MIN               = 0.05   # minimum allowed ROI height [norm]
ROI_HEIGHT_MAX               = HOOD_Y_BOTTOM  # maximum allowed ROI height [norm]

# ==========================================================================
# TTC constants
# ==========================================================================

TTC_WARN_S                   = 3.5    # TTC warning threshold [s]
TTC_CRIT_S                   = 1.5    # TTC critical threshold [s]
TTC_WARN_SCALE               = 1.40   # margin multiplier in warning zone
TTC_CRIT_SCALE               = 2.00   # margin multiplier in critical zone
TTC_MAX_VALID_S              = 10.0   # TTC values beyond this treated as no-threat [s]

# NOTE: FOCAL_NORM_APPROX and OBJECT_REAL_HEIGHT_M deleted in Stage 1 —
# confirmed dead code in the prior version. The TTC estimate below uses a
# normalised-height growth-rate approximation that never referenced them.

# ==========================================================================
# Kalman tracker constants
# ==========================================================================

N_INIT                       = 3      # hit_streak required to confirm a track
MAX_AGE                      = 5      # frames before a lost track is deleted
    # STAGE 6 INTEGRATION NOTE (2026-08-07): this is FRAMES, not seconds.
    # If the calling system runs the detector at a reduced rate (per the
    # Stage 6 detection-scheduling design — see review_note.md) but still
    # calls ROIGenerator.step() every camera frame, MAX_AGE frames no
    # longer represents the real-world time window it may have been
    # tuned against. Example: at 30fps with the detector running every
    # frame, MAX_AGE=5 means ~167ms before a track is dropped. If the
    # detector is instead scheduled at half-rate (every 2nd camera
    # frame), the SAME MAX_AGE=5 raw-frame-count either represents
    # ~333ms of real time (if step() is still called every camera
    # frame and skipped-detector frames pass empty detection lists) or
    # a different value again depending on exactly how the calling
    # system wires the schedule. This constant should be RE-TUNED in
    # terms of real elapsed time once the actual detector schedule is
    # fixed by real hardware timing (see Section 3.3's "Pending
    # Confirmation" on hardware timing) — it is not automatically
    # correct just because the underlying predict/re-associate
    # mechanism has been confirmed to work across gaps (verified
    # 2026-08-07, see test_vectors.py Category 11).

# ==========================================================================
# STAGE 8B — Cold/hot start: warm-up threshold (NEW, added 2026-08-11)
# ==========================================================================

WARMUP_FRAMES_REQUIRED = 10   # frames a ROIGenerator instance must have
    # processed before it reports is_warmed_up=True on its output.
    # PENDING CONFIRMATION — an engineering estimate, not a precisely
    # derived figure. Reasoning: set to more than 3x N_INIT (the
    # confirmed-track threshold, =3), giving margin for at least one
    # full track-confirmation cycle plus some settling time for the
    # asymmetric smoothing filter (which separately converges to >95%
    # of target within roughly 3 frames on a step change — see the
    # Stage 4 fast-grow verification) and for camera-based curvature
    # to become available, if it is going to be. At 30fps this is
    # ~333ms — the same order of magnitude as the MAX_AGE=5 track
    # lifecycle window above, deliberately, since both concern how
    # long this module needs to trust its own freshly-started state.
    # Like MAX_AGE, this is a FRAME count, not a time — see that
    # constant's note on why frame counts and real time can diverge
    # under detection scheduling.

# Kalman process noise (state = [cx, cy, w, h, vx, vy])
KF_Q_POS                     = 1e-2   # position process noise variance
KF_Q_VEL                     = 1e-3   # velocity process noise variance
KF_Q_SIZE                    = 1e-3   # width/height process noise variance

# Kalman measurement noise
KF_R_POS                     = 5e-3   # position measurement noise variance
KF_R_SIZE                    = 1e-2   # size measurement noise variance


# ==========================================================================
# Enumerations
# ==========================================================================

class ObjectCategory(Enum):
    VEHICLE       = auto()
    SIGNAL        = auto()
    SIGN_ROADSIDE = auto()
    SIGN_OVERHEAD = auto()


class TrackState(Enum):
    TENTATIVE = auto()   # not yet used for TTC
    CONFIRMED = auto()   # full TTC + expansion
    LOST      = auto()   # pending deletion


# ==========================================================================
# Data structures
# ==========================================================================

@dataclass
class ConfidenceGates:
    vehicle:       float = DEFAULT_CONF_VEHICLE
    signal:        float = DEFAULT_CONF_SIGNAL
    sign_roadside: float = DEFAULT_CONF_SIGN_ROADSIDE
    sign_overhead: float = DEFAULT_CONF_SIGN_OVERHEAD


@dataclass
class CanSignals:
    speed_mps:          float
    steering_angle_deg: Optional[float]
    yaw_rate_dps:       Optional[float]
    steering_valid:     bool
    yaw_rate_valid:     bool
    esc_active:         bool = False   # STAGE 2 (NEW): Electronic Stability
                                        # Control intervening — strong slip
                                        # indicator, used by _estimate_confidence()
    abs_active:         bool = False   # STAGE 2 (NEW): Anti-lock Braking active
                                        # — used both by _estimate_confidence()
                                        # and by the Stage 1 vertical-margin
                                        # fallback in _invariant_floor(), which
                                        # previously required this to be passed
                                        # in separately rather than read from
                                        # CanSignals directly.


@dataclass
class LaneInfo:
    center_norm:  Optional[float]
    width_norm:   Optional[float]
    confidence:   float
    c2_curvature:  Optional[float] = None  # STAGE 5 (NEW): vision-estimated
        # road curvature (1/m), from the PREVIOUS frame's lane detection
        # output — see review_note.md Section 7 for why previous-frame
        # vision output (not current-frame) avoids a circular dependency
        # between the region and the model that runs inside it.
    c2_confidence: float = 0.0             # STAGE 5 (NEW): confidence in
        # c2_curvature specifically, separate from the overall lane
        # position confidence above — a lane detector can be confident
        # about WHERE the lane centre is nearby while being much less
        # certain about the FAR-FIELD CURVATURE shape, and vice versa.


@dataclass
class CameraIntrinsics:
    """
    STAGE 1 (NEW): camera geometry required for all physics-based
    ROI calculations. Nothing in this module could compute a real-world
    projection without these values — they were entirely absent before
    Stage 1.

    focal_px       : focal length in pixels (horizontal and vertical
                      assumed equal — a common simplification for
                      square-pixel sensors; revisit if the sensor has
                      non-square pixels).
    principal_x_px,
    principal_y_px : principal point (usually close to image centre).
    image_width_px,
    image_height_px: sensor / frame resolution actually used for the
                      curvature and floor projection (this should match
                      whatever resolution CanSignals-derived geometry
                      assumes — typically the full, undistorted frame
                      resolution after lens distortion correction).
    mount_height_m : camera height above the road surface.
    pitch_rad      : camera pitch angle, positive = tilted downward
                      (nose-down). Defaults to 0.0 (level camera).
                      STAGE 1 does not estimate this precisely — see
                      ABS_ACTIVE_VERTICAL_MARGIN_M for the interim
                      conservative-widening fallback used instead.
    """
    focal_px:        float
    principal_x_px:  float
    principal_y_px:  float
    image_width_px:  float
    image_height_px: float
    mount_height_m:  float
    pitch_rad:       float = CAMERA_PITCH_RAD_DEFAULT


@dataclass
class DetectedObject:
    """
    Single detection.
    bbox = (x1, y1, x2, y2) normalised [0, 1].
    track_id : None  → untracked detection (internal Kalman assigned)
               int   → pre-assigned by external tracker (BiTrack / SORT)
    """
    category:   ObjectCategory
    bbox:       Tuple[float, float, float, float]
    confidence: float
    track_id:   Optional[int] = None


@dataclass
class ROIParameters:
    x_left:    float
    y_top:     float
    width:     float
    height:    float
    roi_level: int = 0
    frames_since_init: int = 0   # STAGE 8B (NEW, 2026-08-11): how many
        # step() calls this generator instance has processed, including
        # this one. Default 0 preserves prior behaviour for any
        # ROIParameters constructed WITHOUT going through
        # ROIGenerator.step() — internal intermediate values throughout
        # this module, and the stateless generate_dynamic_roi() API,
        # which has no persistent frame count to report and does not
        # touch this field at all (warm-up tracking is inherently a
        # STATEFUL concept). Only the FINAL output of ROIGenerator.step()
        # has this field stamped with a real, meaningful value.
    is_warmed_up: bool = True    # STAGE 8B (NEW, 2026-08-11): whether
        # enough frames have passed for the system's own internal state
        # (confirmed tracks, settled smoothing, any available vision
        # curvature) to be considered established, as opposed to a
        # value produced during the first handful of frames after
        # start-up. Default True is the SAFE choice for anything that
        # does not explicitly set it: internal intermediate values and
        # the stateless API were never subject to a warm-up concept
        # before this field existed, so leaving them at the permissive
        # default changes nothing for existing behaviour. Only
        # ROIGenerator.step()'s actual final output computes this
        # honestly, from a real frame count — see WARMUP_FRAMES_REQUIRED.
        #
        # IMPORTANT: this says nothing about CURRENT confidence — a
        # well-warmed-up system can still report roi_level==3 (Level 3,
        # nothing currently trustworthy) if conditions genuinely
        # deteriorate later in a drive. is_warmed_up and roi_level
        # answer two different questions and are meant to be read
        # together, not as substitutes for each other: "has enough time
        # passed for my internal state to be established" versus "is my
        # current confidence high right now."
    speed_was_implausible: bool = False  # STAGE 8B (NEW, 2026-08-11):
        # True if the reported speed read approximately zero AND was
        # judged physically implausible (see _is_speed_plausible) —
        # e.g. a brief restart while the vehicle is genuinely still
        # moving, with CAN not yet delivering valid data. When True,
        # DEFAULT_ASSUMED_SPEED_MPS_ON_IMPLAUSIBLE_ZERO was substituted
        # for every calculation this frame, INCLUDING the one that
        # produced this very ROIParameters. Default False preserves
        # prior behaviour for anything not explicitly set by
        # _compute_base_roi (which stamps this honestly on every
        # ROIParameters it constructs, at all of its return points).


# ==========================================================================
# STAGE 1 — Invariant collision-coverage floor
# ==========================================================================

def _z_max(speed_mps: float,
           ttc_min_s: float = TTC_MIN_WARNING_S,
           max_decel_mps2: float = MAX_BRAKING_DECEL_MPS2) -> float:
    """
    Maximum relevant look-ahead distance: the distance within which any
    FCW-relevant object must be visible, given current speed.

    Grounded in ISO 15623's minimum warning time-to-collision (stationary
    lead vehicle, 2.1 s — the most demanding of the standard's three
    defined scenarios) plus the physical distance needed to actually stop
    once braking begins. See module docstring for full citation reasoning.

    Z_max(v) = v * ttc_min_s + v^2 / (2 * max_decel_mps2)
    """
    return speed_mps * ttc_min_s + (speed_mps ** 2) / (2.0 * max_decel_mps2)


# ==========================================================================
# STAGE 7 — ISA sign readability horizon (NEW)
# ==========================================================================

def isa_sign_diameter_for_design_speed_kmh(design_speed_kmh: float) -> float:
    """
    STAGE 7 (NEW): looks up the IRC 67 regulatory sign diameter for a
    given road design speed. A convenience function — callers may also
    supply a sign diameter directly to _isa_readability_check() without
    going through this lookup, e.g. if the actual road class is known
    from a source other than a raw speed value.
    """
    if design_speed_kmh <= 65.0:
        return IRC67_SIGN_DIAMETER_LE_65_M
    elif design_speed_kmh <= 80.0:
        return IRC67_SIGN_DIAMETER_66_80_M
    elif design_speed_kmh <= 100.0:
        return IRC67_SIGN_DIAMETER_81_100_M
    elif design_speed_kmh <= 120.0:
        return IRC67_SIGN_DIAMETER_101_120_M
    else:
        return IRC67_SIGN_DIAMETER_121_150_M


def _isa_required_decision_time_s(speed_mps: float, target_speed_mps: float,
                                   reaction_time_s: float = ISA_REACTION_TIME_S,
                                   decel_mps2: float = ISA_COMFORTABLE_DECEL_MPS2) -> float:
    """
    STAGE 7 (NEW): total time the driver needs between seeing a speed
    limit sign and having comfortably adjusted to it — perception/
    reaction time plus the time to gently decelerate to the new limit
    (zero if the new limit is not lower than current speed).
    """
    speed_reduction = max(0.0, speed_mps - target_speed_mps)
    return reaction_time_s + speed_reduction / decel_mps2


def _isa_min_detection_distance_m(speed_mps: float, target_speed_mps: float,
                                   reaction_time_s: float = ISA_REACTION_TIME_S,
                                   decel_mps2: float = ISA_COMFORTABLE_DECEL_MPS2) -> float:
    """
    STAGE 7 (NEW): the minimum distance at which a sign must be
    detected AND read for the driver to have the full decision-time
    budget available. Working backward from
    _isa_required_decision_time_s: distance = speed * time_required.
    """
    t_required = _isa_required_decision_time_s(speed_mps, target_speed_mps, reaction_time_s, decel_mps2)
    return speed_mps * t_required


def _isa_readability_check(speed_mps: float, target_speed_mps: float,
                            sign_diameter_m: float, camera: CameraIntrinsics,
                            min_sign_px: float = ISA_MIN_SIGN_PX) -> Tuple[bool, float, float]:
    """
    STAGE 7 (NEW): the combined ISA readability check. Answers: "if
    this sign is detected right at the minimum distance the driver
    needs for adequate decision time, will it actually have enough
    pixels to be classified?"

    Returns (is_adequately_readable, required_distance_m, pixel_size_at_that_distance).

    A False result is an important, honest finding in its own right —
    it means the physical decision-time requirement and the camera's
    resolving power are in tension at this speed, exactly the situation
    identified during this project's IRC 67 analysis (a sign at the IRC
    67 advance-placement distance can be below reliable classification
    threshold at native camera resolution — the core argument for why
    the adaptive high-resolution crop matters for ISA specifically).
    """
    required_distance = _isa_min_detection_distance_m(speed_mps, target_speed_mps)
    if required_distance <= 1e-6:
        # Already at or below target speed — no meaningful distance
        # requirement; treat as trivially satisfied.
        return True, required_distance, float('inf')
    px_at_required = camera.focal_px * sign_diameter_m / required_distance
    return (px_at_required >= min_sign_px), required_distance, px_at_required


def _corridor_half_width_m(z_m: float,
                            lane_width_m: float = LANE_WIDTH_STD_M,
                            wander_sigma_m: float = LATERAL_WANDER_SIGMA_M,
                            curvature_sigma_inv_m: float = CURVATURE_ERROR_SIGMA_INV_M,
                            confidence: float = 1.0,
                            ) -> float:
    """
    Half-width of the safety corridor at a given forward distance z_m.

    w(Z) = lane_width + 2*wander_sigma + 2*Z^2*curvature_sigma
    Half-width returned is w(Z) / 2, then scaled by a STAGE 2 (NEW)
    confidence-driven expansion factor:

        expansion = 1 + (CORRIDOR_MAX_EXPANSION_FACTOR - 1) * (1 - confidence)

    At confidence=1.0, expansion=1.0 (no change — base width only).
    At confidence=0.0, expansion=CORRIDOR_MAX_EXPANSION_FACTOR (widest).
    This directly implements the "when uncertain, widen rather than
    guess precisely" principle used throughout this module (see the
    ABS-active vertical margin in _invariant_floor for the same
    philosophy applied to the vertical bound).

    The curvature-uncertainty term grows with the SQUARE of distance —
    this mirrors the c2*Z divergence already present in the lateral
    bound projection itself (see _invariant_floor below): a small error
    in curvature estimate translates into a lateral position error that
    compounds with range, exactly as the curvature term itself does.

    NOTE: wander_sigma_m and curvature_sigma_inv_m are PENDING
    CONFIRMATION placeholders (see constants block). Replace with
    measured values once the curvature fusion module (Stage 5) has
    validation data.
    """
    confidence = _clamp(confidence)
    full_width = lane_width_m + 2.0 * wander_sigma_m + 2.0 * (z_m ** 2) * curvature_sigma_inv_m
    expansion = 1.0 + (CORRIDOR_MAX_EXPANSION_FACTOR - 1.0) * (1.0 - confidence)
    return (full_width * expansion) / 2.0


def _lane_lateral_position_m(z_m: float, c0_m: float, c1_rad: float, c2_inv_m: float) -> float:
    """
    Lane centre lateral offset at forward distance z_m, from the
    standard second-order lane polynomial:
        x_lane(Z) = c0 + c1*Z + c2*Z^2

    SIGN CONVENTION NOTE: c2_inv_m is expected to already be negated
    relative to the raw _compute_curvature() output, to match the sign
    convention used by the pre-existing _lateral_offset_norm() function
    elsewhere in this module (which applies
    math.copysign(magnitude, -curvature) — i.e. positive raw curvature
    shifts the lane-based region toward negative image-x). Callers of
    this function (see _invariant_floor) must pass -curvature, not
    curvature, to keep the floor and the lane-based region agreeing on
    which direction the corridor sweeps for a given curvature sign. This
    was found and fixed during Stage 1 verification testing — the two
    subsystems disagreed on sign before this fix.

    c0_m   : lateral offset of the lane centre at the vehicle (m)
    c1_rad : heading angle of the lane relative to vehicle heading (rad,
             small-angle assumption consistent with the rest of this
             derivation)
    c2_inv_m: curvature term, sign-adjusted per the note above
    """
    return c0_m + c1_rad * z_m + c2_inv_m * (z_m ** 2)


def _project_lateral_to_pixel(x_m: float, z_m: float, camera: CameraIntrinsics) -> float:
    """Pinhole projection of a lateral world position to a horizontal pixel column."""
    if z_m <= 1e-6:
        z_m = 1e-6
    return camera.principal_x_px + camera.focal_px * (x_m / z_m)


def _project_vertical_to_pixel(height_m: float, z_m: float, camera: CameraIntrinsics) -> float:
    """
    Pinhole projection of a point at world height `height_m` above the
    road, at forward distance z_m, to a vertical pixel row. Includes the
    camera pitch correction term (Stage 1: pitch defaults to 0.0 unless
    explicitly supplied on the CameraIntrinsics instance).

    Convention: image v increases DOWNWARD (row 0 = top of image,
    consistent with the rest of this module's normalised [0,1] top-left
    origin). A point below the camera (H_cam > H, the usual case for the
    road surface) must therefore project to v > cy (below image centre).
    Near-field road points project close to the bottom of the image;
    far-field points converge toward v ≈ cy (the horizon).

    v(Z, H) = cy + f * ( (H_cam - H) / Z - pitch_rad )

    NOTE ON PITCH SIGN: the sign of the pitch_rad term above has NOT
    been verified against a real recorded hard-braking or uphill/
    downhill sequence — it is a reasoned first-principles derivation
    (camera axis rotating toward the ground reduces the apparent
    downward angle to a ground point), not an empirically confirmed
    convention. Treat as PENDING CONFIRMATION. This is precisely why
    the ABS-active fallback below is implemented symmetrically
    (widening both edges) rather than assuming a specific direction —
    see _invariant_floor.
    """
    if z_m <= 1e-6:
        z_m = 1e-6
    return camera.principal_y_px + camera.focal_px * (
        (camera.mount_height_m - height_m) / z_m - camera.pitch_rad
    )


@dataclass
class FloorClampDiagnostics:
    """
    STAGE 8B (NEW, added 2026-08-11): optional diagnostic record,
    populated by _invariant_floor() when a diagnostics object is
    passed in. Records whether the floor's RAW (pre-clamp)
    mathematical extent would have gone beyond the actual image on
    each of the four sides, before the existing safety clamp forces it
    back within [0, 1].

    Found as a gap during the 2026-08-10 code review: the clamp itself
    already existed and already worked correctly (the floor can never
    report a position outside the real image) — but nothing recorded
    WHEN this happened. Without that record, there was no way to find
    out, after the fact, whether a sharp curve at a given speed had
    pushed the required corridor past what the camera's field of view
    can physically show — a case where the floor's true, safety-
    required extent and what the camera can actually deliver have
    diverged, which is exactly the kind of condition a review would
    want visibility into.

    This is deliberately a plain, inspectable record rather than a
    printed warning fired every frame it occurs — on a sustained sharp
    curve this condition could legitimately persist for many
    consecutive frames, and a warning on every one of those frames
    would be operational noise, not a useful signal. Whether and how
    to log, count, or alert on this is left to the calling system,
    which is better placed to decide what counts as noteworthy for its
    own purposes (e.g. counting the fraction of frames affected over a
    drive, rather than each individual frame).
    """
    clamped_left:   bool = False
    clamped_right:  bool = False
    clamped_top:    bool = False
    clamped_bottom: bool = False
    raw_x_left_norm:   float = 0.0  # pre-clamp value; may be < 0 or > 1
    raw_x_right_norm:  float = 0.0
    raw_y_top_norm:    float = 0.0
    raw_y_bottom_norm: float = 0.0

    def any_clamped(self) -> bool:
        """True if the camera's field of view was the limiting factor on any side this frame."""
        return self.clamped_left or self.clamped_right or self.clamped_top or self.clamped_bottom


def _invariant_floor(
    speed_mps: float,
    curvature_inv_m: float,
    camera: CameraIntrinsics,
    lane_c0_m: float = 0.0,
    lane_c1_rad: float = 0.0,
    abs_active: bool = False,
    isa_enabled: bool = False,
    z_near_m: float = Z_NEAR_CUTOFF_M,
    confidence: float = 1.0,
    diagnostics: Optional[FloorClampDiagnostics] = None,
) -> Tuple[float, float, float, float]:
    """
    STAGE 1 core function, extended in STAGE 2 with a `confidence`
    parameter (default 1.0 preserves exact Stage 1 behaviour), and in
    STAGE 8B with an optional `diagnostics` parameter (default None
    preserves exact prior behaviour for every existing caller).
    Computes the physics-based minimum ROI — the invariant
    collision-coverage floor — from ego speed, road curvature, and
    camera geometry alone. No object detections are required or used.

    Returns (x_left_norm, x_right_norm, y_top_norm, y_bottom_norm),
    all normalised to [0, 1] and already clamped to valid image bounds.

    --- Range bound ---
    Z_max from _z_max(speed) [ISO 15623-grounded].
    Z_min is the fixed near-field cutoff.

    --- Lateral bound ---
    The corridor edges are x_lane(Z) ± half_width(Z), projected to
    pixels. Because both the curvature term (c2*Z) and the corridor
    half-width itself (Stage 1: ~Z^2) vary with Z, the true extremum
    is evaluated by sampling Z_near, Z_max, and an interior critical
    point derived from the DOMINANT curvature term (holding width's own
    Z-dependence fixed at its value at that candidate point). This is a
    deliberate, documented engineering approximation — see review_note.md
    Section 4.5 — rather than a full closed-form solution of the
    combined quartic, which was judged unnecessary complexity for the
    accuracy this floor requires. All three candidate points are
    evaluated and the true min/max taken, so the approximation can only
    ever make the floor slightly more conservative (wider), never
    unsafe.

    STAGE 2 (NEW): `confidence` (0.0-1.0) scales the corridor half-width
    via _corridor_half_width_m's expansion factor — lower confidence
    (from CAN/lane disagreement, ESC/ABS activity, or low lane
    detection confidence; see _estimate_confidence) widens the corridor
    up to CORRIDOR_MAX_EXPANSION_FACTOR times its base width.

    --- Vertical bound ---
    Upper bound: road surface (H=0) at Z_max.
    If isa_enabled: also checks the gantry height at its minimum read
    distance, and the floor extends further up if that requirement is
    more demanding (this is the direct, quantifiable "ISA cost" to the
    shared ROI — see review_note.md Section 6.4 discussion).
    Lower bound: road surface at Z_near.
    If abs_active: adds a fixed conservative vertical margin in place
    of precise pitch estimation (see ABS_ACTIVE_VERTICAL_MARGIN_M).
    """
    z_far = max(_z_max(speed_mps), z_near_m + Z_MIN_FLOOR_DEPTH_M)
    z_near = z_near_m

    # --- Lateral bound: sample near, far, and one interior critical point ---
    # Sign convention: negate curvature_inv_m here to match the existing
    # _lateral_offset_norm() convention used by the lane-based region
    # elsewhere in this module (see _lane_lateral_position_m docstring).
    # Found and fixed during Stage 1 verification testing.
    c2_signed = -curvature_inv_m
    candidate_zs = [z_near, z_far]

    # Interior critical point of the DOMINANT (c2*Z) term, ignoring the
    # secondary Z-dependence of the width term for this candidate-point
    # search only (see docstring note on the deliberate approximation).
    if abs(c2_signed) > 1e-6:
        # d/dZ [c2*Z - (half_width_component)/Z] = 0 at this approx
        # uses the corridor half-width evaluated at z_far as a
        # representative scale for the critical-point search.
        approx_half_w = _corridor_half_width_m(z_far, confidence=confidence)
        z_crit_sq = approx_half_w / abs(c2_signed)
        if z_crit_sq > 0:
            z_crit = math.sqrt(z_crit_sq)
            if z_near < z_crit < z_far:
                candidate_zs.append(z_crit)

    u_min_px = math.inf
    u_max_px = -math.inf
    for z in candidate_zs:
        half_w = _corridor_half_width_m(z, confidence=confidence)
        x_lane = _lane_lateral_position_m(z, lane_c0_m, lane_c1_rad, c2_signed)
        u_left  = _project_lateral_to_pixel(x_lane - half_w, z, camera)
        u_right = _project_lateral_to_pixel(x_lane + half_w, z, camera)
        u_min_px = min(u_min_px, u_left, u_right)
        u_max_px = max(u_max_px, u_left, u_right)

    # --- Vertical bound ---
    # With the corrected projection convention: near field (small Z)
    # projects to LARGE v (near image bottom); far field (large Z)
    # projects to v close to cy (near the horizon). So v_top (smaller
    # value, higher in image) comes from Z_FAR, and v_bottom (larger
    # value, lower in image) comes from Z_NEAR.
    v_top_px = _project_vertical_to_pixel(0.0, z_far, camera)   # horizon-ward bound, at Z_max

    if isa_enabled:
        v_gantry_px = _project_vertical_to_pixel(
            GANTRY_HEIGHT_M, GANTRY_MIN_READ_DISTANCE_M, camera
        )
        v_top_px = min(v_top_px, v_gantry_px)  # smaller v = higher in image = more coverage

    v_bottom_px = _project_vertical_to_pixel(0.0, z_near, camera)  # near-field bound, at Z_near

    if abs_active:
        # STAGE 1 fallback for unmodelled pitch during hard braking.
        # The true direction of the pitch-induced shift is PENDING
        # CONFIRMATION (see _project_vertical_to_pixel docstring), so
        # this margin is applied SYMMETRICALLY — widening both the
        # upper and lower bound — rather than guessing a direction and
        # potentially widening the wrong way. This is deliberately more
        # conservative (costs a little extra frame area) in exchange
        # for not depending on an unverified sign convention.
        pitch_margin_px = camera.focal_px * ABS_ACTIVE_VERTICAL_MARGIN_M / max(z_near, 1.0)
        v_top_px -= pitch_margin_px     # extend further toward/past the horizon
        v_bottom_px += pitch_margin_px  # extend further toward/past the hood line

    # --- Normalise and clamp ---
    x_left_raw   = u_min_px / camera.image_width_px
    x_right_raw  = u_max_px / camera.image_width_px
    y_top_raw    = v_top_px / camera.image_height_px
    y_bottom_raw = v_bottom_px / camera.image_height_px

    x_left_norm   = _clamp(x_left_raw)
    x_right_norm  = _clamp(x_right_raw)
    y_top_norm    = _clamp(y_top_raw)
    y_bottom_norm = _clamp(y_bottom_raw)

    if diagnostics is not None:
        # STAGE 8B (NEW): record whether the camera's field of view
        # was the limiting factor on each side this frame — i.e.
        # whether the RAW (pre-clamp) value actually fell outside
        # [0, 1], meaning the true, physics-required extent of the
        # floor could not be fully delivered by the sensor's field of
        # view at the current speed/curvature combination.
        diagnostics.clamped_left   = x_left_raw   < 0.0
        diagnostics.clamped_right  = x_right_raw  > 1.0
        diagnostics.clamped_top    = y_top_raw    < 0.0
        diagnostics.clamped_bottom = y_bottom_raw > 1.0
        diagnostics.raw_x_left_norm   = x_left_raw
        diagnostics.raw_x_right_norm  = x_right_raw
        diagnostics.raw_y_top_norm    = y_top_raw
        diagnostics.raw_y_bottom_norm = y_bottom_raw

    return x_left_norm, x_right_norm, y_top_norm, y_bottom_norm


# ==========================================================================
# STAGE 8B — Input quantization (NEW, added 2026-08-11)
# ==========================================================================
#
# Per the manager's suggestion (review_note.md Section 19) and the
# follow-up decision recorded in Section 20: group speed, curvature,
# and confidence into a small, fixed set of bands BEFORE running the
# existing floor calculation, rather than storing a precomputed table.
# This makes the input domain finite (so it can be checked exhaustively
# rather than trusted by analytical argument alone) and guarantees the
# result cannot change unless the vehicle has genuinely moved into a
# different band — WITHOUT the two-sources-of-truth risk or the
# per-vehicle regeneration burden a stored table would carry.
#
# IMPORTANT SAFETY FINDING (2026-08-11, during implementation): a naive
# "round curvature magnitude up to the next band edge" approach was
# tested directly and found to be UNSAFE for one of the two lateral
# edges. As curvature magnitude grows, the corridor's NEAR edge
# (towards the inside of the turn) correctly saturates toward the image
# boundary — but the FAR edge (towards the outside of the turn)
# correctly SHRINKS, because a sharper curve genuinely does not extend
# as far to the outside as a gentler one does. Rounding curvature up
# would therefore shrink the far edge below what the true, gentler
# curvature actually requires — an under-coverage, in the exact
# direction this module exists to prevent. Confirmed directly: at
# 80 km/h, x_right shrinks from 0.7151 (kappa=0) to 0.1943
# (kappa=0.20), a clear, non-monotonic, safety-relevant reversal.
#
# The fix: curvature is NOT bucketed by rounding to one edge. Instead,
# the floor is evaluated at several sample points spanning the width of
# the curvature band, and the ENVELOPE (widest edges) across all of
# them is used — this is safe by construction, regardless of which
# direction the true relationship moves, since it takes the outer
# bound of whatever was actually sampled rather than assuming a
# monotonic trend. Speed and confidence, by contrast, were verified
# directly to be cleanly monotonic across their full ranges (see the
# same 2026-08-11 verification), so a single representative edge is
# sufficient and safe for those two dimensions.

SPEED_BUCKET_UPPER_EDGES_KMH = [26.5, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0]
    # Below 26.5 km/h: Z_MIN_FLOOR_DEPTH_M's minimum-depth clamp already
    # makes every speed in this range produce an IDENTICAL floor — this
    # first band is not an approximation at all, just recognising an
    # exact equivalence that already existed. Each subsequent band uses
    # its UPPER edge as the effective speed (the fastest, most
    # demanding case within that band) — matching the manager's own
    # proposed structure, verified in Section 20 to cost only
    # 2.8%-13.5% extra margin, shrinking as speed increases. A raw
    # speed ABOVE the last edge (150 km/h) is NOT bucketed at all — the
    # true speed is used directly, since bucketing an unexpectedly
    # extreme value into a possibly-wrong band is a worse risk than
    # simply computing it fresh.

CONFIDENCE_BUCKET_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]
    # 4 coarse bands. Verified directly (2026-08-11) to be cleanly
    # monotonic: lower confidence never produces a narrower corridor
    # across the full range tested. Each value maps to the LOWER edge
    # of its band — the more cautious (wider-corridor) value within
    # that band — consistent with "when uncertain, widen" used
    # throughout this module.

CURVATURE_BUCKET_EDGES_INV_M = [0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20]
    # Deliberately finer near zero, where Section 20's own analysis
    # found the greatest sensitivity to curvature error. Unlike the
    # other two dimensions, curvature is NOT reduced to a single edge
    # value — see the envelope approach below and the safety finding
    # documented above.
CURVATURE_ENVELOPE_SAMPLES = 5
    # Number of evenly-spaced sample points evaluated across a
    # curvature band's magnitude range when building its envelope.
    # PENDING CONFIRMATION as a specific number — chosen to give the
    # exhaustive verification (test_vectors.py, oracle.py) a reasonable
    # chance of catching a local non-monotonic wiggle INSIDE a band,
    # not just the global trend reversal already found across bands.
    # Increasing this number only ever makes the envelope MORE
    # conservative (wider or equal), never less — see
    # _floor_envelope_for_curvature_band's docstring.


def _bucket_speed_mps(speed_mps: float) -> float:
    """
    STAGE 8B (NEW): maps a raw speed to the upper edge of the band it
    falls into — verified safe because Z_max(v), and therefore the
    floor's vertical bound, is monotonically non-decreasing in speed
    across the full range tested (0 to 500 m/s), with no reversal.
    """
    speed_kmh = speed_mps * 3.6
    for edge_kmh in SPEED_BUCKET_UPPER_EDGES_KMH:
        if speed_kmh <= edge_kmh:
            return edge_kmh / 3.6
    return speed_mps  # beyond the known operating envelope — do not bucket


def _bucket_confidence(confidence: float) -> float:
    """
    STAGE 8B (NEW): maps a raw confidence value down to the lower edge
    of its band — verified safe because the corridor-width expansion
    factor is monotonically non-increasing in confidence (i.e. never
    narrower at lower confidence) across the full range tested.
    """
    confidence = _clamp(confidence)
    edges = CONFIDENCE_BUCKET_EDGES
    for i in range(len(edges) - 1):
        if confidence < edges[i + 1] or i == len(edges) - 2:
            return edges[i]
    return edges[0]


def _curvature_bucket_range(curvature_inv_m: float) -> Tuple[float, float, float]:
    """
    STAGE 8B (NEW): returns (lower_magnitude, upper_magnitude, sign)
    for the curvature band the input falls into. The CALLER is
    responsible for sampling across this range and taking the
    envelope — see _floor_envelope_for_curvature_band — rather than
    picking one edge, per the safety finding documented above.
    """
    if curvature_inv_m == 0.0:
        return 0.0, 0.0, 1.0
    magnitude = abs(curvature_inv_m)
    sign = math.copysign(1.0, curvature_inv_m)
    edges = CURVATURE_BUCKET_EDGES_INV_M
    for i in range(len(edges) - 1):
        if magnitude <= edges[i + 1]:
            return edges[i], edges[i + 1], sign
    return edges[-2], edges[-1], sign  # beyond the last edge — use the outermost band


def _floor_envelope_for_curvature_band(
    speed_mps: float,
    curvature_inv_m: float,
    camera: CameraIntrinsics,
    lane_c0_m: float = 0.0,
    lane_c1_rad: float = 0.0,
    abs_active: bool = False,
    isa_enabled: bool = False,
    z_near_m: float = Z_NEAR_CUTOFF_M,
    confidence: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    STAGE 8B (NEW): the safe replacement for naive curvature bucketing.

    Rather than evaluating _invariant_floor once at a single
    representative curvature value, this evaluates it at
    CURVATURE_ENVELOPE_SAMPLES points spread evenly across the
    magnitude range of the curvature band the true value falls into
    (preserving sign), and takes the ENVELOPE — the widest x_left, the
    widest x_right, the highest y_top, and the lowest y_bottom — across
    all of them.

    This is safe by construction regardless of which direction any
    individual edge moves as curvature changes: taking the outer bound
    of several actual evaluations can only ever produce an equal-or-
    wider region than any single one of those evaluations, including
    whatever the true, unbucketed curvature would have produced (as
    long as the true value falls within the sampled band, which it
    does by construction — see _curvature_bucket_range).

    Speed and confidence are NOT re-sampled here — they are bucketed
    once, safely, via the simple edge-rounding functions above, since
    both were verified to be cleanly monotonic.
    """
    bucketed_speed = _bucket_speed_mps(speed_mps)
    bucketed_confidence = _bucket_confidence(confidence)
    lower_mag, upper_mag, sign = _curvature_bucket_range(curvature_inv_m)

    if upper_mag == lower_mag:
        sample_kappas = [0.0]
    else:
        n = CURVATURE_ENVELOPE_SAMPLES
        sample_kappas = [
            sign * (lower_mag + (upper_mag - lower_mag) * i / (n - 1))
            for i in range(n)
        ]

    x_left_env, x_right_env = math.inf, -math.inf
    y_top_env, y_bottom_env = math.inf, -math.inf

    for kappa_sample in sample_kappas:
        xl, xr, yt, yb = _invariant_floor(
            bucketed_speed, kappa_sample, camera,
            lane_c0_m=lane_c0_m, lane_c1_rad=lane_c1_rad,
            abs_active=abs_active, isa_enabled=isa_enabled,
            z_near_m=z_near_m, confidence=bucketed_confidence,
        )
        x_left_env  = min(x_left_env, xl)
        x_right_env = max(x_right_env, xr)
        y_top_env    = min(y_top_env, yt)
        y_bottom_env = max(y_bottom_env, yb)

    return x_left_env, x_right_env, y_top_env, y_bottom_env


# ==========================================================================
# Minimal 6-state Kalman filter  [cx, cy, w, h, vx, vy]
# ==========================================================================

class KalmanTrack:
    """
    Constant-velocity Kalman filter for a single bounding-box track.

    State vector  : x = [cx, cy, w, h, vx, vy]
    Measurement   : z = [cx, cy, w, h]
    Motion model  : cx' = cx + vx*dt,  cy' = cy + vy*dt,  w'=w, h'=h
    dt            : assumed 1 frame (normalised).  Scale vx/vy by actual
                    frame interval for physical TTC estimates.

    Notation follows Welch & Bishop "An Introduction to the Kalman Filter".
    """

    _id_counter: int = 0

    def __init__(self, bbox: Tuple[float, float, float, float]):
        KalmanTrack._id_counter += 1
        self.track_id   : int        = KalmanTrack._id_counter
        self.state      : TrackState = TrackState.TENTATIVE
        self.hit_streak : int        = 1
        self.age        : int        = 1
        self.missed     : int        = 0
        self.ttc_s      : Optional[float] = None

        cx, cy, w, h = _bbox_to_cxcywh(bbox)

        # State: [cx, cy, w, h, vx, vy]
        self.x = [cx, cy, w, h, 0.0, 0.0]

        # Covariance P (diagonal init — high uncertainty on velocity)
        self.P = [
            [1e-1, 0,    0,    0,    0,    0   ],
            [0,    1e-1, 0,    0,    0,    0   ],
            [0,    0,    1e-1, 0,    0,    0   ],
            [0,    0,    0,    1e-1, 0,    0   ],
            [0,    0,    0,    0,    1.0,  0   ],
            [0,    0,    0,    0,    0,    1.0 ],
        ]

    # ------------------------------------------------------------------
    # Predict step  (F = constant velocity)
    # ------------------------------------------------------------------
    def predict(self) -> None:
        """
        CRITICAL FIX (found 2026-08-06, during Stage 3 verification):
        the previous version of this method only updated the DIAGONAL
        of the covariance matrix P, adding P[4][4] into P[0][0] as a
        variance contribution but never creating the actual off-diagonal
        position-velocity cross-covariance term P[0][4] (and P[1][5])
        that a correct F*P*F^T propagation produces for this motion
        model. Without that cross-covariance, the Kalman gain for the
        velocity states (K[4][*], K[5][*]) is always exactly zero,
        because K = P*H^T*S^-1 and H only measures [cx,cy,w,h] — the
        gain that lets a POSITION measurement update the VELOCITY
        estimate depends entirely on that cross-covariance existing.

        The practical consequence, confirmed directly by testing: vx
        and vy stayed at exactly 0.0 forever, for every track, regardless
        of how fast or how obviously an object was moving. TTC
        estimation (_estimate_ttc, which reads vy) has therefore never
        worked for ANY motion pattern, not just the head-on-approach
        case this bug was first noticed through — this affects lateral
        motion (cut-ins, crossing traffic) exactly as much as it
        affects vertical motion.

        This fix replaces the ad-hoc diagonal-only update with an
        explicit F*P*F^T + Q computation using the actual constant-
        velocity transition matrix F (identity, plus F[0][4]=1 and
        F[1][5]=1 for the cx+=vx, cy+=vy motion model). Cost is
        negligible (one 6x6 matrix multiply per frame per track).
        """
        x = self.x
        F = [
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

        # State predict: x = F x
        self.x = [sum(F[i][k] * x[k] for k in range(6)) for i in range(6)]

        # Covariance predict: P = F P F^T + Q
        P = self.P
        FP = [[sum(F[i][k] * P[k][j] for k in range(6)) for j in range(6)] for i in range(6)]
        FPFT = [[sum(FP[i][k] * F[j][k] for k in range(6)) for j in range(6)] for i in range(6)]

        q_diag = [KF_Q_POS, KF_Q_POS, KF_Q_SIZE, KF_Q_SIZE, KF_Q_VEL, KF_Q_VEL]
        new_P = [[FPFT[i][j] + (q_diag[i] if i == j else 0.0) for j in range(6)] for i in range(6)]

        # Cap velocity variance to prevent unbounded growth on long tracks
        # (unchanged behaviour from the previous version, just applied
        # after the now-correct propagation).
        new_P[4][4] = min(new_P[4][4], 10.0)
        new_P[5][5] = min(new_P[5][5], 10.0)

        self.P = new_P
        self.age += 1
        self.missed += 1

    # ------------------------------------------------------------------
    # Update step  (measurement = [cx, cy, w, h])
    # ------------------------------------------------------------------
    def update(self, bbox: Tuple[float, float, float, float]) -> None:
        z = list(_bbox_to_cxcywh(bbox))  # [cx, cy, w, h]

        # H maps state → measurement (first 4 components)
        # Innovation: y = z - H*x
        y = [z[i] - self.x[i] for i in range(4)]

        # S = H P Hᵀ + R
        r_pos  = KF_R_POS
        r_size = KF_R_SIZE
        S = [
            self.P[0][0] + r_pos,
            self.P[1][1] + r_pos,
            self.P[2][2] + r_size,
            self.P[3][3] + r_size,
        ]

        # Kalman gain K = P Hᵀ S⁻¹  (diagonal approx)
        K = [[self.P[i][j] / S[j] if j < 4 else 0.0
              for j in range(6)] for i in range(6)]

        # Update state
        for i in range(6):
            for j in range(4):
                self.x[i] += K[i][j] * y[j]

        # Update covariance P = (I - KH) P  (simplified diagonal)
        for i in range(4):
            self.P[i][i] *= (1.0 - K[i][i])

        self.hit_streak += 1
        self.missed = 0
        if self.hit_streak >= N_INIT:
            self.state = TrackState.CONFIRMED

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Return predicted/filtered bbox as (x1, y1, x2, y2)."""
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)

    def get_velocity(self) -> Tuple[float, float]:
        """Return filtered (vx, vy) in normalised coords per frame."""
        return self.x[4], self.x[5]


# ==========================================================================
# Track registry
# ==========================================================================

class TrackRegistry:
    """
    Maintains all active KalmanTracks.

    Matching strategy: greedy IoU matching (threshold-based).
    Compatible with external tracker IDs — if DetectedObject.track_id
    is set, internal matching is bypassed and the external ID is used
    directly (BiTrack / SORT / DeepSORT drop-in).

    For production, replace greedy IoU with Hungarian assignment.

    STAGE 6 note: this class already supports being called with an
    empty detection list on frames where the detector is skipped for
    compute-budget reasons — predict() runs unconditionally every call.
    No structural change needed here for the reduced-rate detection
    scheduling planned in Stage 6; only the calling convention changes.
    """

    def __init__(self, iou_threshold: float = 0.30):
        self.tracks     : Dict[int, KalmanTrack] = {}
        self.iou_thresh : float = iou_threshold

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[DetectedObject],
        use_external_ids: bool = False,
    ) -> List[DetectedObject]:
        """
        Run predict → match → update lifecycle.

        Returns a new list of DetectedObject with:
          - track_id  filled in
          - bbox      replaced with Kalman-filtered bbox (if internal mode)
          - ttc_s     attached as a side-channel via the registry dict
        """

        # --- Step 1: predict all tracks ---
        for trk in self.tracks.values():
            trk.predict()

        # --- Step 2: match / create ---
        updated_dets: List[DetectedObject] = []

        if use_external_ids:
            updated_dets = self._update_external(detections)
        else:
            updated_dets = self._update_internal(detections)

        # --- Step 3: remove dead tracks ---
        dead = [tid for tid, trk in self.tracks.items()
                if trk.state == TrackState.LOST or trk.missed >= MAX_AGE]
        for tid in dead:
            del self.tracks[tid]

        return updated_dets

    # ------------------------------------------------------------------
    def _update_internal(
        self, detections: List[DetectedObject]
    ) -> List[DetectedObject]:
        vehicle_dets = [d for d in detections
                        if d.category == ObjectCategory.VEHICLE]
        other_dets   = [d for d in detections
                        if d.category != ObjectCategory.VEHICLE]

        active_tracks = [trk for trk in self.tracks.values()
                         if trk.state != TrackState.LOST]

        matched, unmatched_dets, unmatched_trks = _greedy_iou_match(
            vehicle_dets,
            active_tracks,
            self.iou_thresh,
        )

        result: List[DetectedObject] = []

        # Matched — update existing tracks
        for det, trk in matched:
            trk.update(det.bbox)
            filtered_bbox = trk.get_bbox()
            result.append(DetectedObject(
                category=det.category,
                bbox=filtered_bbox,
                confidence=det.confidence,
                track_id=trk.track_id,
            ))

        # Unmatched detections — create new tracks
        for det in unmatched_dets:
            trk = KalmanTrack(det.bbox)
            self.tracks[trk.track_id] = trk
            result.append(DetectedObject(
                category=det.category,
                bbox=det.bbox,  # raw; track tentative
                confidence=det.confidence,
                track_id=trk.track_id,
            ))

        # Mark unmatched tracks as missed (already done in predict)
        for trk in unmatched_trks:
            if trk.missed >= MAX_AGE:
                trk.state = TrackState.LOST

        # Pass through non-vehicle detections unchanged
        result.extend(other_dets)
        return result

    # ------------------------------------------------------------------
    def _update_external(
        self, detections: List[DetectedObject]
    ) -> List[DetectedObject]:
        """
        BiTrack / SORT compatible path: the bbox RETURNED downstream is
        always the raw external bbox, never overwritten by internal
        filtering. CORRECTED 2026-08-06: this docstring previously
        claimed "we only maintain lifecycle state... here," which
        implied no Kalman filtering happens in this mode — that was
        inaccurate. self.tracks[tid].update(det.bbox) below DOES run
        the full internal Kalman update, feeding it the external bbox
        purely so TTC can still be estimated from the resulting
        internal velocity state, since the external tracker interface
        has no field for supplying velocity/TTC directly. Only the
        RETURNED DetectedObject's position bypasses this — it is
        always `det` (the original, externally-supplied detection),
        never the internally-filtered bbox.
        """
        result: List[DetectedObject] = []
        seen_ids = set()

        for det in detections:
            if det.track_id is None or det.category != ObjectCategory.VEHICLE:
                result.append(det)
                continue

            tid = det.track_id
            seen_ids.add(tid)

            if tid not in self.tracks:
                trk = KalmanTrack(det.bbox)
                trk.track_id = tid
                self.tracks[tid] = trk
            else:
                self.tracks[tid].update(det.bbox)

            result.append(det)

        # Tracks not seen this frame are aged out by predict() + step-3 removal.
        # No additional state mutation needed here.

        return result

    # ------------------------------------------------------------------
    def get_track(self, track_id: int) -> Optional[KalmanTrack]:
        return self.tracks.get(track_id)


# ==========================================================================
# TTC estimation
# ==========================================================================

def _estimate_ttc(
    trk: KalmanTrack,
    ego_speed_mps: float,
    dt_s: float = 1.0 / 30.0,
) -> Optional[float]:
    """
    Estimate TTC from the Kalman-filtered bbox height velocity.

    Derivation:
      Apparent height h ≈ f * H_real / depth_m
      → depth_m ≈ f * H_real / h      (f in normalised units)
      → d(depth)/dt ≈ -f * H_real * dh/dt / h²

    vy (positive = moving down = growing taller = approaching)
    TTC = -depth / (d(depth)/dt)  =  h / (vy)   [simplified normalised]

    We use the height component of velocity only.  For improved accuracy
    in production, fuse with radar range-rate or stereo depth.

    NOTE: this function does not use camera focal length or object
    real-world height — it is a simplified normalised-coordinate
    approximation. FOCAL_NORM_APPROX and OBJECT_REAL_HEIGHT_M from the
    prior version were unused dead code and have been removed in Stage 1.
    """
    if trk.state != TrackState.CONFIRMED:
        return None

    _, vy = trk.get_velocity()
    x1, y1, x2, y2 = trk.get_bbox()
    h = max(y2 - y1, 1e-4)

    # vy > 0 means object growing → approaching
    if vy <= 0.0:
        return None  # receding or stationary

    # TTC from apparent size growth rate
    ttc = (h / vy) * dt_s   # convert per-frame rate to seconds
    return ttc if ttc <= TTC_MAX_VALID_S else None


def _ttc_margin_scale(ttc: Optional[float]) -> float:
    """Map TTC to an OBJECT_MARGIN multiplier."""
    if ttc is None:
        return 1.0
    if ttc <= TTC_CRIT_S:
        return TTC_CRIT_SCALE
    if ttc <= TTC_WARN_S:
        # Linear interpolation between warn and crit scales
        t = (TTC_WARN_S - ttc) / (TTC_WARN_S - TTC_CRIT_S)
        return TTC_WARN_SCALE + t * (TTC_CRIT_SCALE - TTC_WARN_SCALE)
    return 1.0


# ==========================================================================
# IoU matching utilities
# ==========================================================================

def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _greedy_iou_match(
    detections: List[DetectedObject],
    tracks:     List[KalmanTrack],
    threshold:  float,
) -> Tuple[List[Tuple], List[DetectedObject], List[KalmanTrack]]:
    """Greedy IoU matching — O(N*M).  Replace with Hungarian for N>50."""
    matched_dets : set  = set()
    matched_trks : set  = set()
    pairs        : list = []

    for di, det in enumerate(detections):
        best_iou = threshold
        best_ti  = -1
        for ti, trk in enumerate(tracks):
            if ti in matched_trks:
                continue
            score = _iou(det.bbox, trk.get_bbox())
            if score > best_iou:
                best_iou = score
                best_ti  = ti
        if best_ti >= 0:
            pairs.append((det, tracks[best_ti]))
            matched_dets.add(di)
            matched_trks.add(best_ti)

    unmatched_dets = [detections[i] for i in range(len(detections)) if i not in matched_dets]
    unmatched_trks = [tracks[i]     for i in range(len(tracks))     if i not in matched_trks]
    return pairs, unmatched_dets, unmatched_trks


def _bbox_to_cxcywh(
    bbox: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1


# ==========================================================================
# Input validation
# ==========================================================================

def _validate_inputs(
    signals:    CanSignals,
    static_roi: ROIParameters,
    objects:    Optional[List[DetectedObject]],
    prev_roi:   Optional[ROIParameters],
    conf_gates: ConfidenceGates,
) -> None:
    if signals.speed_mps < 0.0:
        raise ValueError(f"speed_mps must be >= 0.0, got {signals.speed_mps}")

    for fname, val in [
        ("x_left", static_roi.x_left),
        ("y_top",  static_roi.y_top),
        ("width",  static_roi.width),
        ("height", static_roi.height),
    ]:
        if not math.isfinite(val):
            raise ValueError(f"static_roi.{fname} is non-finite: {val}")

    if static_roi.width <= 0.0 or static_roi.height <= 0.0:
        raise ValueError(
            f"static_roi dimensions must be positive, "
            f"got width={static_roi.width}, height={static_roi.height}"
        )

    for gname, gval in [
        ("vehicle",       conf_gates.vehicle),
        ("signal",        conf_gates.signal),
        ("sign_roadside", conf_gates.sign_roadside),
        ("sign_overhead", conf_gates.sign_overhead),
    ]:
        if not (0.0 <= gval <= 1.0):
            raise ValueError(f"conf_gates.{gname} must be in [0.0, 1.0], got {gval}")

    if objects:
        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj.bbox
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"objects[{i}] bbox is degenerate: {obj.bbox}")
            for coord in (x1, y1, x2, y2):
                if not (0.0 <= coord <= 1.0):
                    raise ValueError(
                        f"objects[{i}] bbox coordinate out of [0, 1]: {coord}"
                    )


def _validate_camera_intrinsics(camera: CameraIntrinsics) -> None:
    """STAGE 1 (NEW): sanity checks on camera geometry before it is used
    in any projection — a bad intrinsics value would otherwise silently
    produce a wrong (and potentially unsafe) floor."""
    if camera.focal_px <= 0.0:
        raise ValueError(f"camera.focal_px must be > 0, got {camera.focal_px}")
    if camera.image_width_px <= 0.0 or camera.image_height_px <= 0.0:
        raise ValueError(
            f"camera image dimensions must be > 0, got "
            f"{camera.image_width_px}x{camera.image_height_px}"
        )
    if camera.mount_height_m <= 0.0:
        raise ValueError(f"camera.mount_height_m must be > 0, got {camera.mount_height_m}")
    if not math.isfinite(camera.pitch_rad):
        raise ValueError(f"camera.pitch_rad is non-finite: {camera.pitch_rad}")


def _prev_roi_is_usable(prev_roi: ROIParameters) -> bool:
    return all(
        math.isfinite(v)
        for v in (prev_roi.x_left, prev_roi.y_top,
                  prev_roi.width,  prev_roi.height)
    )


# ==========================================================================
# Helpers
# ==========================================================================

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# NOTE: _roi_center() was REMOVED 2026-08-06 — it was used only by the
# old _smooth_or_snap() (centre-x/width based smoothing), which Stage 4
# replaced with _smooth_asymmetric() (per-edge based smoothing, which
# has no need for a centre calculation). Confirmed unused elsewhere
# before removal, following the same dead-code cleanup practice as the
# Stage 1 removal of FOCAL_NORM_APPROX/OBJECT_REAL_HEIGHT_M.


def _roi_from_center(
    cx: float, half_w: float,
    y_top: float, height: float,
    level: int,
) -> ROIParameters:
    x1 = _clamp(cx - half_w)
    x2 = _clamp(cx + half_w)
    return ROIParameters(
        x_left=x1,
        y_top=_clamp(y_top),
        width=_clamp(x2 - x1),
        height=_clamp(height),
        roi_level=level,
    )


def _roi_from_edges(x_left: float, x_right: float,
                     y_top: float, y_bottom: float, level: int) -> ROIParameters:
    """STAGE 1 (NEW): build an ROIParameters directly from edge coordinates,
    used to convert the floor's (x_left, x_right, y_top, y_bottom) output
    into the module's (x_left, y_top, width, height) representation."""
    x_left  = _clamp(min(x_left, x_right))
    x_right = _clamp(max(x_left, x_right))
    y_top    = _clamp(min(y_top, y_bottom))
    y_bottom = _clamp(max(y_top, y_bottom))
    return ROIParameters(
        x_left=x_left,
        y_top=y_top,
        width=_clamp(x_right - x_left),
        height=_clamp(y_bottom - y_top),
        roi_level=level,
    )


def _union_roi(a: ROIParameters, b: ROIParameters, level: int) -> ROIParameters:
    """
    STAGE 1 (NEW): combine two ROIParameters by taking the widest bound
    on each edge — i.e. the smallest enclosing rectangle covering both.

    This is the mechanism that enforces "the floor can only be grown,
    never shrunk below" — used in _compute_base_roi to combine the
    invariant floor with the lane-based region, and available for reuse
    wherever a similar guarantee is needed later (e.g. Stage 3 object
    expansion, which already achieves the same effect via inline
    min/max operations).
    """
    x_left  = min(a.x_left, b.x_left)
    x_right = max(a.x_left + a.width, b.x_left + b.width)
    y_top   = min(a.y_top, b.y_top)
    y_bottom = max(a.y_top + a.height, b.y_top + b.height)
    return _roi_from_edges(x_left, x_right, y_top, y_bottom, level)


# ==========================================================================
# STAGE 8B — Speed plausibility check (NEW, added 2026-08-11)
# ==========================================================================

def _is_speed_plausible(sig: CanSignals,
                         zero_threshold_mps: float = SPEED_ZERO_THRESHOLD_MPS,
                         yaw_threshold_dps: float = YAW_SUGGESTS_MOVING_THRESHOLD_DPS) -> bool:
    """
    STAGE 8B (NEW): cross-checks a near-zero reported speed against
    other signals already available on CanSignals, for physical
    plausibility.

    Addresses the specific hazard identified in review_note.md Section
    19.4: a brief restart while the vehicle is genuinely moving could
    leave speed_mps momentarily reading zero — because CAN has not yet
    resumed delivering valid data — while the vehicle is actually
    travelling at speed. A floor sized for a stationary vehicle, on a
    vehicle actually moving fast, would be dangerously short.

    Two independent physical cross-checks, using nothing beyond what
    this module already reads from CanSignals:

      1. A vehicle rotating meaningfully (yaw rate above a small
         threshold) cannot simultaneously be genuinely stationary.
      2. ESC and ABS are stability-control and anti-lock braking
         systems; neither engages at a genuine standstill. Either
         being active while speed reads ~0 is itself evidence the
         zero reading is not trustworthy.

    DELIBERATELY LIMITED SCOPE: this function only ever questions a
    reported speed that is approximately zero. It does not attempt to
    validate any other speed value — that would be a much broader
    plausibility problem this module has no independent way to check,
    and is not the specific hazard being addressed here. A speed
    reading of, say, 80 km/h is accepted at face value regardless of
    what the other signals say.

    Returns True (plausible, use as reported) in every case except:
    speed reads at or below zero_threshold_mps AND at least one of the
    two cross-checks suggests the vehicle is very likely actually moving.
    """
    if sig.speed_mps > zero_threshold_mps:
        return True  # not reporting near-zero at all — nothing to question

    yaw_suggests_moving = (
        sig.yaw_rate_valid and sig.yaw_rate_dps is not None
        and abs(sig.yaw_rate_dps) > yaw_threshold_dps
    )
    stability_systems_suggest_moving = sig.esc_active or sig.abs_active

    return not (yaw_suggests_moving or stability_systems_suggest_moving)


def _effective_speed_mps(sig: CanSignals) -> Tuple[float, bool]:
    """
    STAGE 8B (NEW): returns (effective_speed_mps, was_implausible) —
    the speed value every downstream calculation in this module should
    actually use.

    If the reported speed passes _is_speed_plausible(), it is returned
    unchanged and was_implausible=False. Otherwise,
    DEFAULT_ASSUMED_SPEED_MPS_ON_IMPLAUSIBLE_ZERO is returned instead,
    and was_implausible=True — allowing the caller to both use a safe
    speed value AND know that the substitution happened, for the same
    "inspectable, not silent" reasons as FloorClampDiagnostics
    (Section 22).
    """
    if _is_speed_plausible(sig):
        return sig.speed_mps, False
    return DEFAULT_ASSUMED_SPEED_MPS_ON_IMPLAUSIBLE_ZERO, True


# ==========================================================================
# Yaw / steering reliability
# ==========================================================================

def _yaw_steering_mismatch_dps(sig: CanSignals) -> float:
    """
    STAGE 2 (partially added here as a foundation): returns the signed
    mismatch in degrees/sec between measured yaw rate and the yaw rate
    the bicycle model predicts from steering angle. Previously this
    information was discarded — _yaw_is_reliable() returned only a
    boolean. The magnitude is needed for Stage 2's confidence scoring;
    exposing it now avoids a breaking change to this function later.

    Returns 0.0 if either signal is unavailable (treated as "no evidence
    of mismatch" rather than "mismatch confirmed absent" — callers
    should also check signal validity independently where that
    distinction matters).
    """
    if not (sig.yaw_rate_valid and sig.steering_valid):
        return 0.0
    if sig.yaw_rate_dps is None or sig.steering_angle_deg is None:
        return 0.0
    steer_clamped = _clamp(sig.steering_angle_deg, -45.0, 45.0)
    steer_rad     = math.radians(steer_clamped)
    expected_dps  = math.degrees(
        sig.speed_mps * math.tan(steer_rad) / WHEELBASE_M
    )
    return sig.yaw_rate_dps - expected_dps


def _yaw_is_reliable(sig: CanSignals) -> bool:
    if not (sig.yaw_rate_valid and sig.steering_valid):
        return False
    if sig.yaw_rate_dps is None or sig.steering_angle_deg is None:
        return False
    return abs(_yaw_steering_mismatch_dps(sig)) < YAW_MISMATCH_THRESHOLD_DPS


# ==========================================================================
# STAGE 2 — Unified confidence score (NEW)
# ==========================================================================

def _dynamics_confidence(sig: CanSignals) -> float:
    """
    Confidence in the CURRENT VEHICLE DYNAMICS reading (i.e. whether
    the CAN-derived curvature can be trusted as reflecting the road,
    as opposed to a slipping/unstable vehicle whose signals reflect
    confused vehicle motion instead — see review_note.md Section 4.1).

    Combines two independent pieces of evidence, taking the more
    conservative (lower) of the two wherever both apply:
      1. Magnitude of yaw-steering mismatch (see constants block for
         the tier boundaries).
      2. ESC/ABS intervention flags, which are direct evidence of
         reduced tyre grip regardless of what the mismatch calculation
         shows (a vehicle can be slipping in a way the simple bicycle-
         model mismatch check does not fully capture).

    Returns 1.0 (full confidence) when yaw/steering signals are
    unavailable — absence of evidence is treated as no reason to
    distrust the dynamics, not as evidence of instability. This
    matches the module's existing convention in
    _yaw_steering_mismatch_dps(), which returns 0.0 mismatch (not an
    error) when inputs are missing.
    """
    confidence = 1.0

    mismatch = abs(_yaw_steering_mismatch_dps(sig))
    if sig.yaw_rate_valid and sig.steering_valid and mismatch > 0.0:
        if mismatch > YAW_MISMATCH_SEVERE_DPS:
            confidence = min(confidence, DYNAMICS_CONF_SEVERE)
        elif mismatch > YAW_MISMATCH_SIGNIFICANT_DPS:
            confidence = min(confidence, DYNAMICS_CONF_SIGNIFICANT)
        elif mismatch > YAW_MISMATCH_MILD_DPS:
            confidence = min(confidence, DYNAMICS_CONF_MILD)

    if sig.esc_active:
        confidence = min(confidence, ESC_ACTIVE_CONF_CEILING)
    if sig.abs_active:
        confidence = min(confidence, ABS_ACTIVE_CONF_CEILING)

    return _clamp(confidence)


def _estimate_confidence(sig: CanSignals, lane: LaneInfo) -> float:
    """
    STAGE 2 core function. Produces the single, unified 0.0-1.0
    confidence score that drives BOTH the corridor-width widening (via
    _corridor_half_width_m) AND the region-blending logic in
    _compute_base_roi (replacing the old hard LANE_CONF_MIN switch).

    This deliberately replaces what earlier design discussion had
    treated as two separate systems — a "dynamics confidence" system
    and a "degradation-tier confidence" system — with one combined
    score, per the consolidation decision recorded in review_note.md
    Section 4.8 ("two independent mechanisms both answered the
    question 'how uncertain am I right now'").

    Combines, conservatively (minimum, not average — the weakest
    signal governs, consistent with this module's "widen rather than
    guess" philosophy throughout):
      - dynamics confidence (see _dynamics_confidence)
      - lane detection confidence (lane.confidence, already 0.0-1.0
        by construction from the perception system)
    """
    dynamics_conf = _dynamics_confidence(sig)
    lane_conf = _clamp(lane.confidence) if lane is not None else 0.0
    return min(dynamics_conf, lane_conf)


# ==========================================================================
# STAGE 5 — Curvature source fusion (NEW)
# ==========================================================================

def _curvature_agreement_confidence(can_curvature: float,
                                     vision_curvature: Optional[float]) -> float:
    """
    STAGE 5 (NEW): a THIRD, distinct confidence question, separate from
    both _dynamics_confidence (question 1: can I trust CAN's curvature
    reading?) and _estimate_confidence (question 2: should I trust the
    lane-detected centre position?). This one asks: "do my two
    independent curvature sources (CAN and vision) actually agree with
    each other?" A large disagreement is itself informative — it
    usually means the road geometry is changing in a way neither
    source alone captures well (an S-curve transition, a highway fork,
    wet-road phantom lane markings — see review_note.md's earlier
    scenario analysis, Scenarios 2.1/2.2/5.3).

    Returns 1.0 (no reason to distrust) when vision curvature is
    unavailable — absence of a second opinion is not evidence of
    disagreement, consistent with this module's existing convention
    for missing-signal cases elsewhere (e.g. _yaw_steering_mismatch_dps).
    """
    if vision_curvature is None:
        return 1.0
    mismatch = abs(can_curvature - vision_curvature)
    if mismatch > CURVATURE_MISMATCH_SEVERE_INV_M:
        return CURVATURE_AGREEMENT_CONF_SEVERE
    elif mismatch > CURVATURE_MISMATCH_SIGNIFICANT_INV_M:
        return CURVATURE_AGREEMENT_CONF_SIGNIFICANT
    elif mismatch > CURVATURE_MISMATCH_MILD_INV_M:
        return CURVATURE_AGREEMENT_CONF_MILD
    return 1.0


def _fuse_curvature(can_curvature: float, lane: LaneInfo,
                     dynamics_conf: float) -> Tuple[float, float]:
    """
    STAGE 5 core function. Combines the CAN bicycle-model curvature
    (available before inference, but only reflects curvature AT the
    vehicle's current position — see review_note.md Section 7.2) with
    the previous frame's vision-based curvature estimate (genuine
    look-ahead, but one frame stale and dependent on lane-detection
    quality).

    Returns (fused_curvature, curvature_confidence):
      fused_curvature   — the single curvature value to use everywhere
                            curvature is needed (the floor's lateral
                            bound AND the lane-based lateral shift).
      curvature_confidence — combines dynamics_conf (question 1) with
                            the CAN/vision agreement confidence (the
                            new question above) via minimum, consistent
                            with this module's established pattern for
                            combining independent uncertainty sources.
                            This is what actually gets passed to
                            _invariant_floor's corridor-width scaling.

    SOURCE SELECTION RULE (deliberately simple — see design discussion
    in review_note.md Section 14 planning notes): prefer vision
    whenever it is reasonably confident, REGARDLESS of vehicle
    stability. This is simpler than an explicit "invert the hierarchy
    during slip" rule, and produces the same practical outcome: vision
    reflects the actual road regardless of how the vehicle is moving,
    so there is no scenario where CAN should be preferred over a
    confident vision reading. The slip-specific danger (CAN curvature
    reflecting confused vehicle motion, not the road) is handled by
    dynamics_conf separately widening the corridor and, in the most
    severe cases, triggering Level 3 — not by this function picking a
    different source during slip.
    """
    vision_curvature = lane.c2_curvature if lane is not None else None
    vision_confidence = lane.c2_confidence if lane is not None else 0.0

    if vision_curvature is not None and vision_confidence >= VISION_CURVATURE_TRUST_THRESHOLD:
        fused_curvature = vision_curvature
    else:
        fused_curvature = can_curvature

    agreement_conf = _curvature_agreement_confidence(can_curvature, vision_curvature)
    curvature_confidence = min(dynamics_conf, agreement_conf)

    return fused_curvature, curvature_confidence


# ==========================================================================
# Speed-dependent scaling
# ==========================================================================

def _speed_norm(speed_mps: float) -> float:
    return _clamp(speed_mps / MAX_SPEED_MPS)


def _lateral_speed_scale(speed_mps: float) -> float:
    return 1.0 - (1.0 - MIN_WIDTH_SCALE_AT_MAX_SPEED) * _speed_norm(speed_mps)


# NOTE: _depth_ratio(), DEPTH_RATIO_BASE, DEPTH_RATIO_MIN, and
# SKY_CLIP_Y_TOP have been REMOVED in Stage 1. The vertical extent is
# now derived by _invariant_floor() from actual speed and camera
# geometry, replacing this heuristic entirely (see module docstring,
# "Deleted in Stage 1").


# ==========================================================================
# STAGE 8B — Off-centre (c0) estimation (NEW, added 2026-08-11)
# ==========================================================================

def _estimate_c0_m(lane: LaneInfo, camera: CameraIntrinsics,
                    z_near_m: float = Z_NEAR_CUTOFF_M) -> float:
    """
    STAGE 8B (NEW): estimates c0 — the vehicle's lateral offset from the
    lane centre, in the standard lane polynomial
    x_lane(Z) = c0 + c1*Z + c2*Z^2 — using the lane detector's reported
    near-field centre position and the inverse pinhole projection.

    Found as a gap during the 2026-08-10 code review: the floor
    computation already has a place for this term (lane_c0_m, passed
    into _invariant_floor), but it was always fed a fixed 0.0, meaning
    the floor assumed the vehicle sits exactly in the centre of its
    lane. This function replaces that fixed assumption with an actual
    estimate.

    APPROXIMATION, stated plainly: `lane.center_norm` is treated as the
    lane centre position at z_near_m specifically, and the c1*Z and
    c2*Z^2 contributions at that short range are ignored — i.e. this
    computes x_lane(z_near_m) and uses it directly as c0, rather than
    solving for the true c0 term in isolation. This is reasonable
    because z_near_m is small (a few metres) and because this module
    does not yet separately estimate c1 (heading) either — see the
    lane_c1_rad=0.0 note in _compute_base_roi. This approximation is
    no worse than the assumption it sits alongside, and is a strict
    improvement over the fixed 0.0 it replaces.

    Sign convention check: if the lane appears to the RIGHT of the
    image's principal point (u_px > principal_x_px), the lane is
    physically to the right of the vehicle's current position, so c0
    should be positive — matching the same positive-X-projects-right
    convention already used throughout this module (e.g.
    _project_lateral_to_pixel).

    Returns 0.0 (ego assumed centred — the prior, unchanged behaviour)
    if the lane centre is unavailable.
    """
    if lane is None or lane.center_norm is None:
        return 0.0
    u_px = lane.center_norm * camera.image_width_px
    return (u_px - camera.principal_x_px) * z_near_m / camera.focal_px


# ==========================================================================
# Curvature-aware lateral centre
# ==========================================================================

def _compute_curvature(sig: CanSignals) -> float:
    speed = sig.speed_mps
    if speed > SPEED_DYNAMICS_FLOOR_MPS:
        lat_limit = MAX_LAT_ACC_MPS2 / max(speed * speed, LAT_ACC_DENOM_FLOOR)
    else:
        lat_limit = MAX_CURVATURE_INV_M
    limit = min(lat_limit, MAX_CURVATURE_INV_M)

    curvature = 0.0
    if (sig.yaw_rate_valid
            and sig.yaw_rate_dps is not None
            and _yaw_is_reliable(sig)):
        curvature = math.radians(sig.yaw_rate_dps) / max(speed, SPEED_CURVATURE_FLOOR_MPS)
    elif sig.steering_valid and sig.steering_angle_deg is not None:
        steer_rad = math.radians(sig.steering_angle_deg)
        curvature = math.tan(steer_rad) / WHEELBASE_M

    return _clamp(curvature, -limit, limit)


def _lateral_offset_norm(sig: CanSignals, lane_width_norm: float,
                          curvature_override: Optional[float] = None) -> float:
    """
    STAGE 5 CHANGE: accepts an optional `curvature_override`. When
    provided (by _compute_base_roi, using _fuse_curvature's result),
    this is used INSTEAD of recomputing curvature from CAN signals
    alone — ensuring the lane-based lateral shift and the invariant
    floor's lateral bound both use the SAME fused curvature value,
    rather than the floor using vision-aware fusion while this
    function silently kept using CAN-only curvature underneath it.
    When omitted (default), behaviour is IDENTICAL to Stages 1-4 —
    existing callers and tests that call this function directly are
    unaffected.
    """
    speed = sig.speed_mps
    if speed < MIN_SPEED_FOR_PREVIEW_MPS:
        return 0.0

    curvature    = _compute_curvature(sig) if curvature_override is None else curvature_override
    preview_dist = max(speed * PREVIEW_TIME_S, PREVIEW_DIST_MIN_M)

    if abs(curvature) > CURVATURE_NEAR_ZERO:
        R         = abs(1.0 / curvature)
        theta     = preview_dist * abs(curvature)
        magnitude = R * (1.0 - math.cos(theta))
        lateral_m = math.copysign(magnitude, -curvature)
    else:
        lateral_m = 0.0

    norm_per_m = lane_width_norm / LANE_WIDTH_M
    shift      = lateral_m * norm_per_m
    max_shift  = lane_width_norm * MAX_LATERAL_SHIFT_FRACTION
    return _clamp(shift, -max_shift, max_shift)


# ==========================================================================
# Base ROI  (STAGE 1 — REWIRED to combine the invariant floor with the
#            existing lane-based calculation)
# ==========================================================================

DEFAULT_LANE_WIDTH_NORM_FALLBACK = 0.3  # STAGE 2 (NEW): used only as a
    # metres-to-normalised-units scale factor inside _lateral_offset_norm()
    # when lane.width_norm is unavailable — NOT a trust signal. This lets
    # the CAN-only curvature shift still be computed in normalised image
    # units even when the lane detector has provided nothing at all.


def _compute_base_roi(
    lane:       LaneInfo,
    sig:        CanSignals,
    static_roi: ROIParameters,
    camera:     Optional[CameraIntrinsics] = None,
    abs_active: bool = False,
    isa_enabled: bool = False,
    diagnostics: Optional[FloorClampDiagnostics] = None,
    quantize_inputs: bool = False,
) -> Tuple[ROIParameters, int]:
    """
    STAGE 2 REWRITE. Replaces the old hard switch
    (`if lane.confidence < LANE_CONF_MIN: return static_roi, 2`) with
    continuous confidence-weighted blending between a lane-informed
    region and a CAN-only fallback region, plus a new Level 3
    (full-frame) fallback for genuinely catastrophic input loss.

    STAGE 8B (NEW): accepts an optional `diagnostics` parameter,
    passed straight through to _invariant_floor() when a camera is
    provided. Default None preserves exact prior behaviour for every
    existing caller. See FloorClampDiagnostics's docstring for what
    this records and why.

    STAGE 8B (NEW): accepts an optional `quantize_inputs` parameter
    (default False, preserving EXACT prior behaviour — including every
    existing hand-calculated test value throughout this project, which
    assumes the unquantized formula). When True, the floor is computed
    via _floor_envelope_for_curvature_band() instead of calling
    _invariant_floor() directly — grouping speed, curvature, and
    confidence into a small, fixed set of bands first, per the
    manager's suggestion (review_note.md Sections 19-20). This is
    deliberately opt-in rather than the default: it trades a small
    amount of extra conservatism (see Section 20's measured costs) for
    a finite, exhaustively-checkable input domain and a guarantee that
    the result cannot change unless the vehicle moves into a different
    band — properties a caller must deliberately choose, not one
    silently imposed on every existing user of this function.

    This fixes the specific defect documented in review_note.md
    Section 2.4 ("Problem one"): CAN curvature is now ALWAYS computed
    and ALWAYS contributes to lateral positioning, regardless of lane
    detection confidence — previously, low lane confidence caused an
    early return that discarded perfectly good curvature information.

    --- Three separate confidence questions, deliberately kept distinct ---
    A single blended "how uncertain am I" number was considered and
    rejected during design (see review_note.md Section 4.8's
    consolidation discussion) BECAUSE collapsing these into one number
    creates a new bug: a total lane-detection dropout (lane.confidence
    -> 0) with perfectly healthy CAN signals would otherwise force the
    same drastic response as a genuine total system failure, when in
    fact a CAN-only corridor is a perfectly good fallback in that case.

    So three distinct questions are asked, each answered by the
    appropriate signal:

      1. "Can I trust the CURVATURE estimate itself?"
         -> dynamics_conf = _dynamics_confidence(sig) [CAN-only]
         -> drives corridor WIDTH scaling in _invariant_floor (a bad
            curvature estimate means the corridor must be wider,
            regardless of whether lane detection is working).

      2. "Should I trust the LANE-DETECTED centre position over a
          CAN-only centred guess?"
         -> overall_conf = _estimate_confidence(sig, lane)
            = min(dynamics_conf, lane.confidence)
         -> drives the BLEND WEIGHT between lane-informed and
            CAN-only lateral centring. Using min() here is correct:
            if EITHER source is compromised, trust the lane position
            less — a confident-looking lane detection paired with an
            untrustworthy curvature estimate for the shift term is
            not fully trustworthy either.

      3. "Is NOTHING reliable — should I give up on any positioning
          guess and just process the whole frame?"
         -> triggered by dynamics_conf ALONE, not overall_conf. A
            lane dropout with healthy CAN signals must NOT trigger
            this — the CAN-only fallback (question 2's low end) is
            exactly the correct, graceful response to that case, and
            using overall_conf here would trigger full-frame far too
            often (every time lane detection has a bad frame).

    `static_roi` is retained in the signature for backward API
    compatibility but is no longer used substantively — per
    review_note.md Section 5.3 ("the static ROI is caller-defined, not
    system-defined... the system should enforce a minimum safe static
    ROI internally"), both the CAN-only fallback (Level 2) and the
    full-frame fallback (Level 3) are now computed internally rather
    than relying on a caller-supplied rectangle.

    STAGE 8B (NEW): before anything else runs, the reported speed is
    passed through _effective_speed_mps() — a near-zero reading judged
    physically implausible (see _is_speed_plausible) is replaced with
    a conservative default for the REST OF THIS FUNCTION's purposes.
    This is done by constructing a corrected copy of `sig` via
    dataclasses.replace() rather than mutating the caller's object —
    the caller's original CanSignals instance is left untouched.
    """
    effective_speed, speed_was_implausible = _effective_speed_mps(sig)
    if speed_was_implausible:
        sig = replace(sig, speed_mps=effective_speed)

    steer_valid = sig.steering_valid and sig.steering_angle_deg is not None
    yaw_valid   = sig.yaw_rate_valid  and sig.yaw_rate_dps    is not None

    dynamics_conf = _dynamics_confidence(sig)
    overall_conf  = _estimate_confidence(sig, lane)
    effective_abs_active = abs_active or sig.abs_active

    # --- Level 3: nothing reliable — give up on positioning, use full frame ---
    if dynamics_conf < CONF_LEVEL3_THRESHOLD:
        return ROIParameters(x_left=0.0, y_top=0.0, width=1.0, height=1.0, roi_level=3,
                              speed_was_implausible=speed_was_implausible), 3

    curvature_can = _compute_curvature(sig)
    curvature, curvature_conf = _fuse_curvature(curvature_can, lane, dynamics_conf)
    sp_scale  = _lateral_speed_scale(sig.speed_mps)

    # --- Lane-informed lateral centring (used when trusted) ---
    lane_width_for_shift = lane.width_norm if lane.width_norm is not None else DEFAULT_LANE_WIDTH_NORM_FALLBACK
    shift = _lateral_offset_norm(sig, lane_width_for_shift, curvature_override=curvature)  # STAGE 5: uses fused curvature

    if lane.center_norm is not None and lane.width_norm is not None:
        lane_cx = lane.center_norm
        lane_hw = _clamp(lane.width_norm * 0.5, LANE_HW_CLAMP_MIN, LANE_HW_CLAMP_MAX)
        cx_dynamic     = _clamp(lane_cx + shift)
        half_w_dynamic = lane_hw * LATERAL_HALF_SCALE_L0 * sp_scale
    else:
        # No lane centre/width at all — "dynamic" degenerates to the
        # same thing as the fallback below.
        cx_dynamic     = _clamp(0.5 + shift)
        half_w_dynamic = LANE_HW_CLAMP_MAX * sp_scale

    # --- CAN-only fallback lateral centring (used when lane is not trusted) ---
    cx_fallback     = _clamp(0.5 + shift)
    half_w_fallback = LANE_HW_CLAMP_MAX * sp_scale  # widest allowed half-width

    # --- Blend weight from overall confidence (question 2 above) ---
    if overall_conf >= CONF_BLEND_HIGH:
        blend_weight = 1.0
    elif overall_conf <= CONF_BLEND_LOW:
        blend_weight = 0.0
    else:
        blend_weight = (overall_conf - CONF_BLEND_LOW) / (CONF_BLEND_HIGH - CONF_BLEND_LOW)

    cx     = blend_weight * cx_dynamic + (1.0 - blend_weight) * cx_fallback
    half_w = blend_weight * half_w_dynamic + (1.0 - blend_weight) * half_w_fallback

    # --- Discretised level, for diagnostics/logging only — the actual
    # positioning above is continuous, not level-switched. ---
    if blend_weight <= 0.0:
        level = 2
    elif blend_weight >= 1.0 and (steer_valid or yaw_valid):
        level = 0
    else:
        level = 1

    if camera is None:
        raise ValueError(
            "_compute_base_roi() called without camera intrinsics. "
            "The invariant collision-coverage floor cannot be computed. "
            "Pass a CameraIntrinsics instance."
        )

    _validate_camera_intrinsics(camera)

    c0_estimate = _estimate_c0_m(lane, camera)  # STAGE 8B: replaces the
        # previous fixed 0.0 — see _estimate_c0_m's docstring for the
        # approximation this relies on.

    if quantize_inputs:
        # STAGE 8B (NEW): grouped-input path. NOTE: `diagnostics` is
        # not populated on this path — FloorClampDiagnostics records
        # the clamp status of a SINGLE evaluation, and the envelope
        # approach deliberately combines several; there is no single
        # "the" raw value left to report once they have been unioned.
        # A caller needing FOV clamp visibility should use the default
        # (quantize_inputs=False) path.
        x_left, x_right, y_top, y_bottom = _floor_envelope_for_curvature_band(
            speed_mps=sig.speed_mps,
            curvature_inv_m=curvature,
            camera=camera,
            lane_c0_m=c0_estimate,
            lane_c1_rad=0.0,
            abs_active=effective_abs_active,
            isa_enabled=isa_enabled,
            confidence=curvature_conf,
        )
    else:
        x_left, x_right, y_top, y_bottom = _invariant_floor(
            speed_mps=sig.speed_mps,
            curvature_inv_m=curvature,
            camera=camera,
            lane_c0_m=c0_estimate,
            lane_c1_rad=0.0,
            abs_active=effective_abs_active,
            isa_enabled=isa_enabled,
            confidence=curvature_conf,  # STAGE 5: now incorporates BOTH dynamics
                                         # confidence (question 1) AND CAN/vision
                                         # agreement confidence (question 3) via
                                         # _fuse_curvature's min() combination —
                                         # a curvature-source disagreement widens
                                         # the corridor exactly like a dynamics
                                         # reliability problem does.
            diagnostics=diagnostics,  # STAGE 8B: passthrough, None by default
        )
    floor_roi = _roi_from_edges(x_left, x_right, y_top, y_bottom, level)

    # Vertical extent comes exclusively from the floor (unchanged Stage 1
    # rule — see Stage 1 verification notes on why this must not be
    # unioned with a wide placeholder). Only the lateral bound is unioned
    # with the blended lane/fallback centring above.
    lane_x_left  = _clamp(cx - half_w)
    lane_x_right = _clamp(cx + half_w)
    combined_x_left  = min(floor_roi.x_left, lane_x_left)
    combined_x_right = max(floor_roi.x_left + floor_roi.width, lane_x_right)

    combined = ROIParameters(
        x_left=combined_x_left,
        y_top=floor_roi.y_top,
        width=_clamp(combined_x_right - combined_x_left),
        height=floor_roi.height,
        roi_level=level,
        speed_was_implausible=speed_was_implausible,
    )
    return combined, level



# ==========================================================================
# Per-category expansion functions  (TTC-aware for VEHICLE)
# STAGE 3 IMPLEMENTED: corridor-membership + confirmed-track + valid-TTC
# gating added for vehicle expansion — see _is_in_corridor() and
# _apply_object_expansions() below.
# ==========================================================================

def _is_in_corridor(bbox: Tuple[float, float, float, float],
                     corridor_x_left: float,
                     corridor_x_right: float) -> bool:
    """
    STAGE 3 (NEW): is the detection's horizontal centre laterally
    inside the driving corridor?

    This is the core fix for the documented parked-vehicle bug
    (review_note.md Section 2.5 / 5.2): a car parked on the roadside,
    well outside the corridor, should not cause the region to expand
    toward it. Only vehicles whose centre falls within the corridor
    bounds are even considered for expansion.

    IMPORTANT: `corridor_x_left`/`corridor_x_right` must be captured
    ONCE per frame, BEFORE any expansion is applied that frame — not
    read from a `roi` that is being progressively widened inside the
    same expansion loop. If the corridor bounds were re-read after an
    earlier expansion, a genuinely parked vehicle could wrongly "pass"
    the membership test simply because an earlier, unrelated expansion
    had already widened the region. See _apply_object_expansions()
    for where this capture happens.
    """
    bx1, _, bx2, _ = bbox
    obj_cx = (bx1 + bx2) * 0.5
    return corridor_x_left <= obj_cx <= corridor_x_right


def _expand_vehicle(
    roi:          ROIParameters,
    bbox:         Tuple[float, float, float, float],
    margin_scale: float = 1.0,
) -> ROIParameters:
    """
    FCW vehicle — expand laterally and vertically.
    margin_scale: TTC-derived multiplier (1.0 nominal, up to TTC_CRIT_SCALE).

    NOTE: this function itself performs NO gating — it unconditionally
    expands for whatever bbox it is given. The three-condition gate
    (corridor membership, confirmed track, valid TTC) is applied by the
    CALLER (_apply_object_expansions), which decides whether to call
    this function at all for a given vehicle. Keeping the gating in the
    caller, not here, means this function stays a pure "expand to
    include this box" primitive that could in principle be reused
    for other gating rules later without modification.
    """
    bx1, by1, bx2, by2 = bbox
    m = OBJECT_MARGIN * margin_scale

    new_x1 = _clamp(min(roi.x_left,              bx1 - m))
    new_x2 = _clamp(max(roi.x_left + roi.width,  bx2 + m))
    new_y1 = _clamp(min(roi.y_top,               by1 - m))
    new_y2 = _clamp(max(roi.y_top + roi.height,  by2 + m))

    return ROIParameters(
        x_left=new_x1, y_top=new_y1,
        width=_clamp(new_x2 - new_x1),
        height=_clamp(new_y2 - new_y1),
        roi_level=roi.roi_level,
    )


def _expand_signal(
    roi: ROIParameters,
    bbox: Tuple[float, float, float, float],
) -> ROIParameters:
    _, by1, _, by2 = bbox
    m = OBJECT_MARGIN
    new_y1 = _clamp(min(roi.y_top,              by1 - m))
    new_y2 = _clamp(max(roi.y_top + roi.height, by2 + m))
    return ROIParameters(
        x_left=roi.x_left, y_top=new_y1,
        width=roi.width,
        height=_clamp(new_y2 - new_y1),
        roi_level=roi.roi_level,
    )


def _expand_sign_roadside(
    roi: ROIParameters,
    bbox: Tuple[float, float, float, float],
) -> ROIParameters:
    bx1, by1, bx2, by2 = bbox
    m = OBJECT_MARGIN
    new_x1 = _clamp(min(roi.x_left,              bx1 - m))
    new_x2 = _clamp(max(roi.x_left + roi.width,  bx2 + m))
    new_y1 = _clamp(min(roi.y_top,               by1 - m))
    new_y2 = _clamp(max(roi.y_top + roi.height,  by2 + m))
    return ROIParameters(
        x_left=new_x1, y_top=new_y1,
        width=_clamp(new_x2 - new_x1),
        height=_clamp(new_y2 - new_y1),
        roi_level=roi.roi_level,
    )


def _expand_sign_overhead(
    roi: ROIParameters,
    bbox: Tuple[float, float, float, float],
) -> ROIParameters:
    _, by1, _, by2 = bbox
    m = OBJECT_MARGIN
    new_y1 = _clamp(min(roi.y_top, by1 - m), IMAGE_Y_MIN, IMAGE_Y_MAX)
    new_y2 = _clamp(max(roi.y_top + roi.height, by2 + m))
    return ROIParameters(
        x_left=roi.x_left, y_top=new_y1,
        width=roi.width,
        height=_clamp(new_y2 - new_y1),
        roi_level=roi.roi_level,
    )


# ==========================================================================
# Expansion dispatch  (TTC-aware, STAGE 3: corridor-gated for vehicles)
# ==========================================================================

_CATEGORY_ORDER = (
    ObjectCategory.VEHICLE,
    ObjectCategory.SIGNAL,
    ObjectCategory.SIGN_ROADSIDE,
    ObjectCategory.SIGN_OVERHEAD,
)


def _apply_object_expansions(
    roi:          ROIParameters,
    objects:      List[DetectedObject],
    gates:        ConfidenceGates,
    registry:     Optional[TrackRegistry],
    ego_speed_mps: float,
    frame_dt_s:   float = 1.0 / 30.0,
    corridor_bounds: Optional[Tuple[float, float]] = None,
) -> ROIParameters:
    """
    Expand ROI for each detection that clears its confidence gate.

    STAGE 3 CHANGE: vehicle expansion now additionally requires ALL
    THREE of the following (previously only the confidence gate
    applied, which is why a stationary, roadside-parked vehicle with
    confidence >= DEFAULT_CONF_VEHICLE=0.0 — i.e. any detected vehicle
    at all — used to expand the region regardless of position or
    motion):

      1. The vehicle's centre is laterally inside the driving corridor
         (_is_in_corridor) — a parked car on the shoulder fails this.
      2. Its track has reached CONFIRMED state — a single-frame,
         possibly-spurious detection fails this (already existing
         track-lifecycle logic, just now also gating expansion itself
         rather than only gating TTC estimation).
      3. A valid (non-None) closing time-to-collision exists — a
         vehicle that is not actually closing (parked, or moving away)
         fails this, since _estimate_ttc() already returns None for
         vy<=0 (non-approaching objects).

    Signal and sign categories are UNCHANGED — they remain
    confidence-gated only, with no corridor or TTC requirement, since
    those categories do not have an analogous "is this actually a
    collision threat" question the way an arbitrary detected vehicle
    does.

    The corridor bounds used for condition 1 are captured ONCE, from
    the `roi` as it is BEFORE this function applies any expansion —
    see _is_in_corridor()'s docstring for why re-reading a
    progressively-widening `roi` inside the loop would be a bug.

    STAGE 7 (NEW): accepts an optional `corridor_bounds` override so
    that ROIGenerator.step() can capture ONE frozen pre-expansion
    corridor and share it with both this function AND the new Stage 7
    occlusion-response/vertical-peek functions, rather than each
    deriving its own independently. When omitted, behaviour is
    identical to Stages 1-6 (derives it from `roi` directly).
    """
    if corridor_bounds is not None:
        corridor_x_left, corridor_x_right = corridor_bounds
    else:
        corridor_x_left  = roi.x_left
        corridor_x_right = roi.x_left + roi.width

    conf_map = {
        ObjectCategory.VEHICLE:       gates.vehicle,
        ObjectCategory.SIGNAL:        gates.signal,
        ObjectCategory.SIGN_ROADSIDE: gates.sign_roadside,
        ObjectCategory.SIGN_OVERHEAD: gates.sign_overhead,
    }

    for cat in _CATEGORY_ORDER:
        threshold = conf_map[cat]
        for obj in objects:
            if obj.category != cat or obj.confidence < threshold:
                continue

            if cat == ObjectCategory.VEHICLE:
                # --- Condition 1: corridor membership ---
                if not _is_in_corridor(obj.bbox, corridor_x_left, corridor_x_right):
                    continue  # e.g. a car parked on the shoulder — skip

                # --- Condition 2: confirmed track ---
                if registry is None or obj.track_id is None:
                    continue
                trk = registry.get_track(obj.track_id)
                if trk is None or trk.state != TrackState.CONFIRMED:
                    continue  # tentative/unconfirmed — skip (ghost-track guard)

                # --- Condition 3: valid closing TTC ---
                ttc = _estimate_ttc(trk, ego_speed_mps, dt_s=frame_dt_s)
                if ttc is None:
                    continue  # not actually closing — e.g. parked, or receding

                margin_scale = _ttc_margin_scale(ttc)
                roi = _expand_vehicle(roi, obj.bbox, margin_scale)

            elif cat == ObjectCategory.SIGNAL:
                roi = _expand_signal(roi, obj.bbox)
            elif cat == ObjectCategory.SIGN_ROADSIDE:
                roi = _expand_sign_roadside(roi, obj.bbox)
            elif cat == ObjectCategory.SIGN_OVERHEAD:
                roi = _expand_sign_overhead(roi, obj.bbox)

    return roi


# ==========================================================================
# STAGE 7 — Sign occlusion response (NEW)
# ==========================================================================

@dataclass
class _RememberedSign:
    """
    STAGE 7 (NEW): a single remembered sign detection, keyed by
    category in ROIGenerator.sign_memory. Deliberately keyed by
    CATEGORY, not by a per-instance identity — this module does not
    track individual signs the way it tracks individual vehicles
    (signs have no Kalman filter, no track_id assignment); one
    remembered position per sign category is the level of detail this
    project's occlusion response operates at. Multiple simultaneous
    roadside signs are outside this scope.

    NOTE: unlike vehicle positions, no ego-motion compensation is
    applied to the remembered bbox while it ages — the remembered
    position is used as-is for up to SIGN_MEMORY_MAX_AGE frames. At
    typical highway speeds and this frame count, the resulting
    positional error is small relative to the expansion margin already
    applied, but this is a simplification, not a precise projection —
    see review_note.md for the fuller design discussion of why a
    precise ego-motion-compensated version was not built at this
    stage (the same "prove it's needed first" philosophy used
    elsewhere in this module).
    """
    bbox: Tuple[float, float, float, float]
    frames_since_seen: int = 0


def _update_sign_memory(
    sign_memory: Dict[ObjectCategory, _RememberedSign],
    tracked_objects: List[DetectedObject],
    gates: ConfidenceGates,
) -> None:
    """
    STAGE 7 (NEW): mutates `sign_memory` in place — refreshes entries
    for sign categories seen this frame (above their confidence gate),
    ages entries not seen this frame, and forgets entries that have
    exceeded SIGN_MEMORY_MAX_AGE.
    """
    conf_map = {
        ObjectCategory.SIGN_ROADSIDE: gates.sign_roadside,
        ObjectCategory.SIGN_OVERHEAD: gates.sign_overhead,
    }
    seen_this_frame = set()
    for obj in tracked_objects:
        if obj.category not in conf_map:
            continue
        if obj.confidence < conf_map[obj.category]:
            continue
        sign_memory[obj.category] = _RememberedSign(bbox=obj.bbox, frames_since_seen=0)
        seen_this_frame.add(obj.category)

    to_forget = []
    for category, remembered in sign_memory.items():
        if category in seen_this_frame:
            continue
        remembered.frames_since_seen += 1
        if remembered.frames_since_seen > SIGN_MEMORY_MAX_AGE:
            to_forget.append(category)
    for category in to_forget:
        del sign_memory[category]


def _is_large_confirmed_in_corridor_vehicle(
    obj: DetectedObject,
    registry: Optional[TrackRegistry],
    corridor_x_left: float,
    corridor_x_right: float,
) -> bool:
    """
    STAGE 7 (NEW): shared predicate for both the occlusion response and
    the vertical peek — a vehicle is treated as a plausible sign
    occluder only if it is a VEHICLE detection, large enough
    (LARGE_VEHICLE_HEIGHT_THRESHOLD_NORM), on a CONFIRMED track (same
    ghost-track guard used throughout this module since Stage 3), and
    laterally inside the driving corridor.
    """
    if obj.category != ObjectCategory.VEHICLE:
        return False
    bx1, by1, bx2, by2 = obj.bbox
    if (by2 - by1) < LARGE_VEHICLE_HEIGHT_THRESHOLD_NORM:
        return False
    if not _is_in_corridor(obj.bbox, corridor_x_left, corridor_x_right):
        return False
    if registry is None or obj.track_id is None:
        return False
    trk = registry.get_track(obj.track_id)
    return trk is not None and trk.state == TrackState.CONFIRMED


def _apply_occlusion_response(
    roi: ROIParameters,
    tracked_objects: List[DetectedObject],
    sign_memory: Dict[ObjectCategory, _RememberedSign],
    registry: Optional[TrackRegistry],
    corridor_x_left: float,
    corridor_x_right: float,
) -> ROIParameters:
    """
    STAGE 7 (NEW): the unified occlusion response — combining temporal
    sign memory and large-vehicle-triggered lateral widening into ONE
    mechanism, per the consolidation decision recorded in
    review_note.md (five originally-separate mitigations discussed
    during design reduced to two: this function, and the separate
    _apply_vertical_peek below, kept apart because the geometry is
    genuinely different — lateral reach for redundant opposite-side
    signs vs. vertical reach for overhead gantries).

    Two things happen, both purely additive (union-style growth, same
    "can only grow, never shrink below what came before" discipline
    used throughout this module):

      1. Any sign category still within SIGN_MEMORY_MAX_AGE of last
         being directly seen has the region expanded to still cover
         its last-known position — bridging a brief occlusion rather
         than immediately forgetting the sign the instant it is
         blocked for even one frame.

      2. If a large, confirmed, in-corridor vehicle is present (a
         plausible occluder for anything behind it), the region is
         proactively widened laterally by OCCLUSION_LATERAL_WIDEN_NORM
         on both sides — approximating reach toward an IRC 67-mandated
         opposite-side redundant sign that the large vehicle might be
         blocking on ONE side, without needing to know which side.
    """
    for category, remembered in sign_memory.items():
        if remembered.frames_since_seen <= SIGN_MEMORY_MAX_AGE:
            if category == ObjectCategory.SIGN_OVERHEAD:
                roi = _expand_sign_overhead(roi, remembered.bbox)
            elif category == ObjectCategory.SIGN_ROADSIDE:
                roi = _expand_sign_roadside(roi, remembered.bbox)

    for obj in tracked_objects:
        if _is_large_confirmed_in_corridor_vehicle(obj, registry, corridor_x_left, corridor_x_right):
            new_x1 = _clamp(roi.x_left - OCCLUSION_LATERAL_WIDEN_NORM)
            new_x2 = _clamp(roi.x_left + roi.width + OCCLUSION_LATERAL_WIDEN_NORM)
            roi = ROIParameters(
                x_left=new_x1, y_top=roi.y_top,
                width=_clamp(new_x2 - new_x1), height=roi.height,
                roi_level=roi.roi_level,
            )
            break  # one qualifying large vehicle is enough to trigger the response

    return roi


def _apply_vertical_peek(
    roi: ROIParameters,
    tracked_objects: List[DetectedObject],
    registry: Optional[TrackRegistry],
    corridor_x_left: float,
    corridor_x_right: float,
) -> ROIParameters:
    """
    STAGE 7 (NEW): extends the region's top edge upward above a large,
    confirmed, in-corridor tracked vehicle, so an overhead gantry sign
    partially visible above the vehicle (per IRC 67's minimum gantry
    clearance vs. typical truck height — see OCCLUSION_VERTICAL_PEEK_NORM)
    stays inside the region. Kept SEPARATE from _apply_occlusion_response
    per the consolidation decision — genuinely different geometry
    (vertical extension vs. lateral reach), not an arbitrary split.

    The region's BOTTOM edge is deliberately held fixed while extending
    the top — this is an extension of the existing region upward, not a
    shift of the whole region.
    """
    for obj in tracked_objects:
        if not _is_large_confirmed_in_corridor_vehicle(obj, registry, corridor_x_left, corridor_x_right):
            continue
        bx1, by1, bx2, by2 = obj.bbox
        old_y_bottom = roi.y_top + roi.height
        new_y_top = _clamp(min(roi.y_top, by1 - OCCLUSION_VERTICAL_PEEK_NORM), IMAGE_Y_MIN, IMAGE_Y_MAX)
        new_height = _clamp(old_y_bottom - new_y_top)
        roi = ROIParameters(
            x_left=roi.x_left, y_top=new_y_top,
            width=roi.width, height=new_height,
            roi_level=roi.roi_level,
        )
        break  # one qualifying large vehicle is enough to trigger the peek

    return roi


# ==========================================================================
# STAGE 8 — Region size cap (NEW)
# ==========================================================================

def _apply_area_cap(
    roi: ROIParameters,
    base_roi: ROIParameters,
    max_area_fraction: float = MAX_ROI_AREA_FRACTION,
) -> ROIParameters:
    """
    STAGE 8 (NEW): caps the FINAL region's area (after all object,
    occlusion, and peek expansions) to at most `max_area_fraction` of
    the full frame — preventing several simultaneous, individually
    reasonable-looking expansions from silently combining toward
    full-frame.

    CRITICAL SAFETY PROPERTY: this can NEVER shrink below `base_roi`
    (the floor+lane baseline, captured before any expansion this
    frame) on any edge. This is guaranteed BY CONSTRUCTION, not by a
    runtime check that could have an edge case: the function only ever
    scales the four MARGIN amounts (how far `roi` extends beyond
    `base_roi` on each side) toward zero, and a margin can never become
    negative. If `base_roi` itself already exceeds `max_area_fraction`
    (e.g. a very low-confidence corridor, or Level 3's full-frame
    fallback), the cap has no effect at all — the floor always wins,
    exactly as intended.

    APPROXIMATION NOTE: the scaling factor is computed as
    sqrt(target_area / current_area), which exactly holds only when
    width and height are scaled independently without the base
    offsets — the true relationship involving the base dimensions has
    a cross term this does not solve for exactly. This is intentional:
    Stage 8 is scoped as "a maximum region size limit... region size
    stays predictable" (review_note.md Section 6), a practical
    predictability measure, not a precision requirement. The
    approximation always shrinks margins somewhat when the cap is
    exceeded; it does not guarantee hitting the cap exactly.
    """
    base_x_left, base_x_right = base_roi.x_left, base_roi.x_left + base_roi.width
    base_y_top, base_y_bottom = base_roi.y_top, base_roi.y_top + base_roi.height
    roi_x_left, roi_x_right = roi.x_left, roi.x_left + roi.width
    roi_y_top, roi_y_bottom = roi.y_top, roi.y_top + roi.height

    current_area = roi.width * roi.height
    if current_area <= max_area_fraction or current_area <= 1e-9:
        return roi  # already within budget, or degenerate — nothing to do

    scale = math.sqrt(max_area_fraction / current_area)
    scale = _clamp(scale)  # guard against any pathological input

    delta_left   = max(0.0, base_x_left - roi_x_left)     # margin grown to the LEFT of base
    delta_right  = max(0.0, roi_x_right - base_x_right)   # margin grown to the RIGHT of base
    delta_top    = max(0.0, base_y_top - roi_y_top)       # margin grown ABOVE base
    delta_bottom = max(0.0, roi_y_bottom - base_y_bottom) # margin grown BELOW base

    new_x_left   = base_x_left   - delta_left   * scale
    new_x_right  = base_x_right  + delta_right  * scale
    new_y_top    = base_y_top    - delta_top    * scale
    new_y_bottom = base_y_bottom + delta_bottom * scale

    # By construction, new_x_left <= base_x_left <= base_x_right <= new_x_right
    # (and the equivalent for y) — the base is always fully retained.
    return ROIParameters(
        x_left=_clamp(new_x_left),
        y_top=_clamp(new_y_top),
        width=_clamp(new_x_right - new_x_left),
        height=_clamp(new_y_bottom - new_y_top),
        roi_level=roi.roi_level,
    )


# ==========================================================================
# STAGE 8 — Bounded resize: mapping a variable ROI to a fixed canonical
# (accelerator) input size via uniform scaling and letterbox padding (NEW)
# ==========================================================================

@dataclass
class CanonicalMapping:
    """
    STAGE 8 (NEW): describes how a variable-shaped ROI (in full-frame
    pixel coordinates) maps onto a fixed canonical (accelerator) input
    size. Produced by map_roi_to_canonical(); consumed by
    canonical_bbox_to_fullframe() to convert a detection produced by
    the model (in canonical pixel coordinates) back into full-frame
    normalised coordinates for the tracker.
    """
    crop_x_px:            float
    crop_y_px:            float
    crop_width_px:        float
    crop_height_px:       float
    scale:                float  # single uniform scale factor (never two different ones)
    pad_x_px:             float  # letterbox padding on EACH side horizontally
    pad_y_px:             float  # letterbox padding on EACH side vertically
    canonical_width_px:   float
    canonical_height_px:  float


def map_roi_to_canonical(
    roi: ROIParameters,
    camera: CameraIntrinsics,
    canonical_width_px: float,
    canonical_height_px: float,
) -> CanonicalMapping:
    """
    STAGE 8 (NEW): converts a normalised ROI into pixel-space crop
    coordinates, then computes the SINGLE uniform scale factor (never
    two different scale factors for width/height) that fits the crop
    within the canonical input size, with any leftover space handled
    as letterbox padding rather than stretching.

    This is the design decision recorded in review_note.md Section 4.3:
    stretching would distort object proportions relative to what a
    detector trained on undistorted images expects; uniform scaling
    only ever changes SIZE, which detectors handle far better than a
    SHAPE distortion.
    """
    crop_x = roi.x_left * camera.image_width_px
    crop_y = roi.y_top * camera.image_height_px
    crop_w = roi.width * camera.image_width_px
    crop_h = roi.height * camera.image_height_px

    if crop_w <= 1e-6 or crop_h <= 1e-6:
        scale = 1.0
    else:
        scale = min(canonical_width_px / crop_w, canonical_height_px / crop_h)

    scaled_w = crop_w * scale
    scaled_h = crop_h * scale
    pad_x = max(0.0, (canonical_width_px - scaled_w) / 2.0)
    pad_y = max(0.0, (canonical_height_px - scaled_h) / 2.0)

    return CanonicalMapping(
        crop_x_px=crop_x, crop_y_px=crop_y,
        crop_width_px=crop_w, crop_height_px=crop_h,
        scale=scale, pad_x_px=pad_x, pad_y_px=pad_y,
        canonical_width_px=canonical_width_px, canonical_height_px=canonical_height_px,
    )


def canonical_bbox_to_fullframe(
    bbox_canonical_px: Tuple[float, float, float, float],
    mapping: CanonicalMapping,
    camera: CameraIntrinsics,
) -> Tuple[float, float, float, float]:
    """
    STAGE 8 (NEW): inverse of map_roi_to_canonical's geometry — converts
    a detection bbox produced by the model (in canonical pixel
    coordinates, e.g. 512x256) back into full-frame NORMALISED
    coordinates, suitable for handing to TrackRegistry/the tracker.

    This is the "track in full-frame coordinates, not crop-relative
    coordinates" fix already established in this module's design
    discussion — a detection must be converted back to a common frame
    before the tracker ever sees it, or crop changes between frames
    would look like sudden object motion to the Kalman filter.
    """
    x1, y1, x2, y2 = bbox_canonical_px
    # Undo padding, undo scale, add back the crop's own offset.
    full_x1 = (x1 - mapping.pad_x_px) / mapping.scale + mapping.crop_x_px
    full_y1 = (y1 - mapping.pad_y_px) / mapping.scale + mapping.crop_y_px
    full_x2 = (x2 - mapping.pad_x_px) / mapping.scale + mapping.crop_x_px
    full_y2 = (y2 - mapping.pad_y_px) / mapping.scale + mapping.crop_y_px

    return (
        _clamp(full_x1 / camera.image_width_px),
        _clamp(full_y1 / camera.image_height_px),
        _clamp(full_x2 / camera.image_width_px),
        _clamp(full_y2 / camera.image_height_px),
    )


# ==========================================================================
# Stability smoothing  (STAGE 4 IMPLEMENTED: asymmetric per-edge filter)
# ==========================================================================

def _union_with_floor(roi: ROIParameters, floor: ROIParameters) -> ROIParameters:
    """Return the union of roi and floor edge by edge. Called after IIR
    smoothing and the safety clamp to guarantee the invariant floor is
    never violated — smoothing can hold edges inside the floor on the
    first frame of a sudden event, and the height clamp (HOOD_Y_BOTTOM)
    must not override the ABS-active floor that can legitimately reach 1.0."""
    x_left  = min(roi.x_left,  floor.x_left)
    y_top   = min(roi.y_top,   floor.y_top)
    x_right = max(roi.x_left  + roi.width,  floor.x_left + floor.width)
    y_bot   = max(roi.y_top   + roi.height, floor.y_top  + floor.height)
    return replace(roi,
        x_left=x_left,
        y_top=y_top,
        width=_clamp(x_right - x_left),
        height=_clamp(y_bot  - y_top),
    )


ASYM_ALPHA_GROW_EDGE   = 0.30   # weight on PREVIOUS value when an edge is
                                  # GROWING (expanding outward) — low weight
                                  # on the past means the new, larger value
                                  # dominates quickly. Growing is the safety
                                  # direction (covering a new or closer
                                  # threat), so it should never lag.
ASYM_ALPHA_SHRINK_EDGE = 0.85   # weight on PREVIOUS value when an edge is
                                  # SHRINKING (contracting inward) — high
                                  # weight on the past means the old, larger
                                  # value persists for a while. Shrinking is
                                  # the conservative direction to be slow
                                  # about: prematurely narrowing the region
                                  # risks losing coverage the instant a
                                  # detector has one noisy or missed frame.


def _blend_edge(prev_val: float, new_val: float, is_growing: bool) -> float:
    """
    STAGE 4 (NEW): one edge's asymmetric blend. `is_growing` must be
    determined by the CALLER based on which direction growth means for
    that specific edge (see _smooth_asymmetric below) — this function
    itself has no notion of "which way is outward" for a given edge.
    """
    alpha = ASYM_ALPHA_GROW_EDGE if is_growing else ASYM_ALPHA_SHRINK_EDGE
    return prev_val * alpha + new_val * (1.0 - alpha)


def _smooth_asymmetric(prev: ROIParameters, new: ROIParameters) -> ROIParameters:
    """
    STAGE 4 REPLACEMENT for the old _smooth_or_snap(). Applies the
    fast-grow/slow-shrink filter independently to all FOUR edges
    (x_left, x_right, y_top, y_bottom) — not just centre-x and width as
    the old filter did, and NOT the old single alpha + snap-threshold
    mechanism at all.

    Why this replaces the old approach entirely, per the consolidation
    decision recorded in review_note.md Section 4.8 / the original
    design discussion:
      - The old single alpha (0.70) resisted ALL changes equally,
        including genuinely urgent ones (e.g. the region needing to
        widen fast during hard braking) — see review_note.md Section
        2.7's documented limitation.
      - The old snap threshold created a discontinuity: a change just
        below the threshold was heavily smoothed and sluggish, a change
        just above it snapped instantly with no smoothing at all. There
        was no continuous middle ground.
      - Vertical extent (y_top/height) was never smoothed at all in the
        old version — tolerable only because it used to be a nearly-
        fixed heuristic (Stage 1 removed that heuristic; the floor's
        vertical bound is now genuinely speed-dependent, so leaving it
        unsmoothed would reintroduce jitter Stage 1 didn't have a
        reason to smooth away yet).

    Direction-of-growth convention for each edge, given this module's
    coordinate system (origin top-left, y increasing downward):
      x_left:    growing = moving LEFT  (new < prev)
      x_right:   growing = moving RIGHT (new > prev)
      y_top:     growing = moving UP, toward the horizon (new < prev) —
                 see _project_vertical_to_pixel's convention note
      y_bottom:  growing = moving DOWN, toward the hood (new > prev)

    After blending, edges are re-sorted (min/max) before constructing
    the result, purely as a defensive measure against an extreme edge
    case where independent per-edge blending could theoretically leave
    x_left slightly past x_right (or y_top past y_bottom) — this has
    not been observed in testing, but costs nothing to guard against.
    """
    prev_x_left, prev_x_right = prev.x_left, prev.x_left + prev.width
    prev_y_top, prev_y_bottom = prev.y_top, prev.y_top + prev.height
    new_x_left, new_x_right   = new.x_left, new.x_left + new.width
    new_y_top, new_y_bottom   = new.y_top, new.y_top + new.height

    x_left   = _blend_edge(prev_x_left,   new_x_left,   is_growing=(new_x_left   < prev_x_left))
    x_right  = _blend_edge(prev_x_right,  new_x_right,  is_growing=(new_x_right  > prev_x_right))
    y_top    = _blend_edge(prev_y_top,    new_y_top,    is_growing=(new_y_top    < prev_y_top))
    y_bottom = _blend_edge(prev_y_bottom, new_y_bottom, is_growing=(new_y_bottom > prev_y_bottom))

    x_left_f, x_right_f = min(x_left, x_right), max(x_left, x_right)
    y_top_f, y_bottom_f = min(y_top, y_bottom), max(y_top, y_bottom)

    return ROIParameters(
        x_left=_clamp(x_left_f),
        y_top=_clamp(y_top_f),
        width=_clamp(x_right_f - x_left_f),
        height=_clamp(y_bottom_f - y_top_f),
        roi_level=new.roi_level,
        speed_was_implausible=new.speed_was_implausible,  # STAGE 8B (NEW):
            # previously dropped — found and fixed during the same
            # implementation session as the field's introduction,
            # 2026-08-11. ROIGenerator.step() also captures this value
            # independently before calling this function, as defence in
            # depth, but this function should not silently discard it
            # for any OTHER caller either.
    )


# ==========================================================================
# Public API  — stateful ROIGenerator class
# ==========================================================================

class ROIGenerator:
    """
    Stateful wrapper around generate_dynamic_roi.

    Maintains:
      - TrackRegistry  (Kalman tracks per vehicle detection)
      - prev_roi       (for IIR smoothing)

    STAGE 1 CHANGE: accepts an optional `camera` (CameraIntrinsics).
    When provided, the invariant floor is computed and combined into
    the base ROI every frame. When omitted, the module falls back to
    lane-based-only positioning with no physics-based safety guarantee
    (see _compute_base_roi docstring) — intended as a temporary
    compatibility path during rollout, not a supported long-term mode.

    Usage
    -----
    gen = ROIGenerator(camera=my_camera_intrinsics)

    Per-frame loop:
        roi = gen.step(lane, signals, static_roi, detections)

    BiTrack / external tracker:
        gen = ROIGenerator(camera=my_camera_intrinsics, use_external_tracker=True)
        # Pass DetectedObject with track_id already assigned by BiTrack.
        # The RETURNED POSITION always trusts the external bbox as-is —
        # internal filtering never overwrites it. An internal Kalman
        # filter still runs in the background regardless, purely to
        # support TTC estimation (see TrackRegistry._update_external's
        # docstring / the BiTrack note earlier in this file for the
        # 2026-08-06 correction to this previously-inaccurate claim).
    """

    def __init__(
        self,
        camera:               Optional[CameraIntrinsics] = None,
        conf_gates:           Optional[ConfidenceGates] = None,
        iou_threshold:        float = 0.30,
        use_external_tracker: bool  = False,
        frame_dt_s:           float = 1.0 / 30.0,
        abs_active_default:   bool  = False,
        isa_enabled:          bool  = False,
        quantize_inputs:      bool  = False,
    ):
        self.camera                = camera
        self.conf_gates            = conf_gates or ConfidenceGates()
        self.registry              = TrackRegistry(iou_threshold)
        self.prev_roi: Optional[ROIParameters] = None
        self.use_external_tracker  = use_external_tracker
        self.frame_dt_s            = frame_dt_s
        self.abs_active_default    = abs_active_default
        self.isa_enabled           = isa_enabled
        self.quantize_inputs       = quantize_inputs  # STAGE 8B (NEW, 2026-08-11):
            # default False preserves exact prior behaviour. When True,
            # step() computes the floor via the grouped-input envelope
            # approach instead of the direct per-frame calculation —
            # see _compute_base_roi's docstring for the full reasoning.
        self.sign_memory: Dict[ObjectCategory, _RememberedSign] = {}  # STAGE 7 (NEW)
        self.last_floor_diagnostics: Optional[FloorClampDiagnostics] = None  # STAGE 8B (NEW)
            # Populated fresh every step() call. Inspect this after
            # calling step() to find out whether the camera's field of
            # view was the limiting factor on any side this frame —
            # see FloorClampDiagnostics's docstring. Remains None if no
            # camera was provided to this generator (the no-camera
            # compatibility path does not use the floor at all).
        self.frames_since_init: int = 0  # STAGE 8B (NEW, 2026-08-11):
            # incremented at the start of every step() call. Frame 1
            # (the first call this instance ever processes) is
            # frames_since_init=1, not 0 — see step() for exactly where
            # this increments and how it drives is_warmed_up on the
            # returned ROIParameters.

        if camera is not None:
            _validate_camera_intrinsics(camera)
        else:
            raise ValueError(
                "ROIGenerator requires camera intrinsics — "
                "the invariant collision-coverage floor cannot be computed without them. "
                "Pass camera=CameraIntrinsics(...) to the constructor."
            )

    def reset_for_warm_restart(self) -> None:
        """
        STAGE 8B (NEW, 2026-08-11): implements the state-restoration
        policy for a brief WARM restart — a software or watchdog reset
        WITHOUT power loss, where some memory may have survived. See
        review_note.md Section 19.4 for the full reasoning behind this
        policy; this method is the enforced, testable version of that
        written rule, rather than leaving it as a comment someone
        integrating this code might follow incorrectly by hand.

        THE RULE: information that is PHYSICALLY FIXED is safe to keep
        across a restart, because it cannot have changed in the
        meantime. Anything TIME-DEPENDENT is not safe to keep, because
        it may now be stale or actively wrong, and must be reset to a
        fresh, cautious starting state rather than trusted.

        Call this method after resuming a ROIGenerator instance
        following a brief warm restart, INSTEAD OF constructing a
        brand new instance — using this method rather than a fresh
        instance is only worthwhile because it preserves the
        configuration below without the caller needing to remember
        and re-supply it.

        SAFE TO KEEP — left completely untouched by this method:
          - self.camera               (camera calibration; physically fixed)
          - self.conf_gates           (confidence-gate configuration)
          - self.use_external_tracker (construction-time configuration)
          - self.frame_dt_s           (construction-time configuration)
          - self.abs_active_default   (construction-time configuration)
          - self.isa_enabled          (construction-time configuration)

        NOT SAFE TO KEEP — this method resets ALL of these to a fresh
        starting state:
          - self.registry             (tracked objects may have moved
                                        arbitrarily during the gap; a
                                        stale track could produce a
                                        wrong TTC estimate)
          - self.prev_roi             (smoothing history would otherwise
                                        blend the next frame against a
                                        region computed under conditions
                                        that may no longer apply at all)
          - self.sign_memory          (remembered sign positions are
                                        stale for the same reason as
                                        tracked objects)
          - self.frames_since_init    (a warm restart IS a fresh start
                                        for warm-up purposes — see
                                        Section 23's is_warmed_up
                                        reasoning; established history
                                        from before the restart cannot
                                        be assumed to still apply)
          - self.last_floor_diagnostics (stale by definition once
                                        frames_since_init resets)

        DELIBERATELY NOT ADDRESSED HERE: previous-frame vision
        curvature sits in a genuine grey area (Section 19.4) — safe to
        keep ONLY if the interruption was extremely brief, since the
        vehicle has then moved a negligible distance. This module has
        no way to measure how long the interruption actually lasted,
        so it cannot safely make that judgement on the caller's
        behalf. Critically, this value is not held as state INSIDE
        ROIGenerator at all — it lives on the LaneInfo object the
        caller constructs fresh and passes into step() every single
        frame. There is nothing here for this method to reset; the
        calling system itself must decide, based on its own knowledge
        of how long the interruption lasted, whether to pass a
        genuinely remembered curvature value or None on the first
        call after a warm restart.

        The IoU matching threshold originally supplied at construction
        is preserved when the tracker is reset, since that is
        configuration, not time-dependent state.
        """
        self.registry = TrackRegistry(self.registry.iou_thresh)
        self.prev_roi = None
        self.sign_memory = {}
        self.frames_since_init = 0
        self.last_floor_diagnostics = None

    def step(
        self,
        lane:       LaneInfo,
        signals:    CanSignals,
        static_roi: ROIParameters,
        objects:    Optional[List[DetectedObject]] = None,
        abs_active: Optional[bool] = None,
    ) -> ROIParameters:
        """
        Process one frame.  Returns ROIParameters with roi_level.

        abs_active: if provided, overrides the constructor's
        abs_active_default for this frame AND the value of
        signals.abs_active (STAGE 2: CanSignals now carries its own
        abs_active field — this parameter is retained for callers that
        want to override it, e.g. for testing, but sig.abs_active is
        the primary source as of Stage 2).

        STAGE 2 CHANGE: the previous hard branch for level==2 (which
        returned static_roi verbatim and reset self.prev_roi to force
        an un-smoothed jump on recovery) has been removed. Level 2 is
        now just one point on a continuous confidence-driven blend
        computed inside _compute_base_roi, so it flows through the
        same expansion/smoothing/clamp path as every other level —
        eliminating the single-frame jump documented as a known defect
        in review_note.md Section 2.4/2.7.
        """
        gates = self.conf_gates
        _validate_inputs(signals, static_roi, objects, self.prev_roi, gates)

        self.frames_since_init += 1  # STAGE 8B (NEW): frame 1 = first ever call

        effective_abs_active = self.abs_active_default if abs_active is None else abs_active

        # --- Track update (always run, even with empty detections) ---
        tracked_objects = self.registry.update(
            objects or [], use_external_ids=self.use_external_tracker
        )

        # --- Base ROI (Stage 2: continuous confidence blending + Level 3) ---
        frame_diagnostics = FloorClampDiagnostics()  # STAGE 8B: fresh each frame
        roi, level = _compute_base_roi(
            lane, signals, static_roi,
            camera=self.camera,
            abs_active=effective_abs_active,
            isa_enabled=self.isa_enabled,
            diagnostics=frame_diagnostics,
            quantize_inputs=self.quantize_inputs,
        )
        # STAGE 8B: frame_diagnostics is not populated on the quantized
        # path (see _compute_base_roi's docstring) — reflect that
        # honestly rather than exposing a diagnostics object that looks
        # populated but was never actually filled in for this frame.
        self.last_floor_diagnostics = (
            frame_diagnostics if (self.camera is not None and not self.quantize_inputs) else None
        )
        speed_was_implausible_this_frame = roi.speed_was_implausible  # STAGE 8B (NEW):
            # captured HERE, immediately, because both _smooth_asymmetric()
            # and the final safety-clamp reconstruction below build a NEW
            # ROIParameters from scratch and would otherwise silently drop
            # this field (they already did not carry it forward — found
            # and fixed during this same implementation session, 2026-08-11).

        # STAGE 8: retained as the hard floor the area cap (below) can
        # never intrude into — this is the SAME rectangle frozen_corridor
        # is derived from, captured here under its own name for clarity
        # at the point it is actually used.
        base_roi_for_cap = roi

        # STAGE 7: freeze the corridor ONCE here, before ANY expansion
        # this frame, and share it across object expansion, occlusion
        # response, and vertical peek — consistent with the Stage 3
        # principle that re-deriving the corridor from a
        # progressively-widening `roi` mid-frame would let one
        # expansion wrongly make a later, unrelated check more
        # permissive than it should be.
        frozen_corridor = (roi.x_left, roi.x_left + roi.width)

        # --- Object expansions (TTC-aware) — now applied uniformly at
        # every level, including Level 3 (full-frame), where expansion
        # is a no-op in practice since the frame is already maximal. ---
        if tracked_objects:
            roi = _apply_object_expansions(
                roi, tracked_objects, gates,
                self.registry, signals.speed_mps,
                frame_dt_s=self.frame_dt_s,
                corridor_bounds=frozen_corridor,
            )

        # --- STAGE 7: sign memory + occlusion response + vertical peek ---
        _update_sign_memory(self.sign_memory, tracked_objects, gates)
        roi = _apply_occlusion_response(
            roi, tracked_objects, self.sign_memory, self.registry,
            frozen_corridor[0], frozen_corridor[1],
        )
        roi = _apply_vertical_peek(
            roi, tracked_objects, self.registry,
            frozen_corridor[0], frozen_corridor[1],
        )

        # --- STAGE 8: cap total area, never intruding into the floor ---
        roi = _apply_area_cap(roi, base_roi_for_cap)

        # --- IIR smoothing ---
        if self.prev_roi is not None and _prev_roi_is_usable(self.prev_roi):
            roi = _smooth_asymmetric(self.prev_roi, roi)

        # --- Safety clamp ---
        roi = ROIParameters(
            x_left=_clamp(roi.x_left),
            y_top=_clamp(roi.y_top, IMAGE_Y_MIN, IMAGE_Y_MAX),
            width=_clamp(roi.width,  ROI_WIDTH_MIN,  ROI_WIDTH_MAX),
            height=_clamp(roi.height, ROI_HEIGHT_MIN, ROI_HEIGHT_MAX),
            roi_level=roi.roi_level,
            frames_since_init=self.frames_since_init,  # STAGE 8B (NEW)
            is_warmed_up=(self.frames_since_init >= WARMUP_FRAMES_REQUIRED),  # STAGE 8B (NEW)
            speed_was_implausible=speed_was_implausible_this_frame,  # STAGE 8B (NEW)
        )

        # Re-enforce floor after clamp: the height clamp (HOOD_Y_BOTTOM=0.97)
        # must not override the ABS-active floor that can legitimately reach 1.0.
        roi = _union_with_floor(roi, base_roi_for_cap)

        self.prev_roi = roi
        return roi


# ==========================================================================
# Functional API (stateless — original interface preserved)
# ==========================================================================

def generate_dynamic_roi(
    lane:       LaneInfo,
    signals:    CanSignals,
    static_roi: ROIParameters,
    objects:    Optional[List[DetectedObject]] = None,
    prev_roi:   Optional[ROIParameters]        = None,
    conf_gates: Optional[ConfidenceGates]      = None,
    camera:     Optional[CameraIntrinsics]     = None,
    abs_active: bool = False,
    isa_enabled: bool = False,
) -> ROIParameters:
    """
    Stateless functional interface.  Backward-compatible with v1.

    STAGE 1 CHANGE: accepts an optional `camera` argument, same
    semantics as ROIGenerator — see that class's docstring.

    Note: TTC and tracking require the stateful ROIGenerator class.
    When objects contain track_id (pre-assigned externally), basic
    TTC estimation is attempted via a one-shot track lookup —
    registry is not maintained between calls.
    """
    gates = conf_gates if conf_gates is not None else ConfidenceGates()
    _validate_inputs(signals, static_roi, objects, prev_roi, gates)

    if camera is None:
        raise ValueError(
            "generate_dynamic_roi() requires camera intrinsics — "
            "the invariant collision-coverage floor cannot be computed without them. "
            "Pass camera=CameraIntrinsics(...) to enable it."
        )

    roi, level = _compute_base_roi(
        lane, signals, static_roi,
        camera=camera, abs_active=abs_active, isa_enabled=isa_enabled,
    )
    speed_was_implausible_this_call = roi.speed_was_implausible  # STAGE 8B (NEW):
        # captured here for the same reason as in ROIGenerator.step() —
        # _apply_object_expansions/_smooth_asymmetric/the final clamp
        # below all build a new ROIParameters and would otherwise lose it.

    # STAGE 2 CHANGE: the old hard level==2 branch (returning static_roi
    # verbatim) has been removed — level 2 is now one point on the
    # continuous confidence blend computed inside _compute_base_roi, and
    # flows through the same expansion path as every other level.

    if objects:
        tracked = [o for o in objects if o.track_id is not None]
        if tracked:
            warnings.warn(
                f"{len(tracked)} DetectedObject(s) carry track_id but the stateless "
                "generate_dynamic_roi() API does not maintain a TrackRegistry — "
                "TTC estimation is disabled. Use ROIGenerator for full TTC support.",
                UserWarning,
                stacklevel=2,
            )
        roi = _apply_object_expansions(
            roi, objects, gates, None, signals.speed_mps
        )

    if prev_roi is not None and _prev_roi_is_usable(prev_roi):
        roi = _smooth_asymmetric(prev_roi, roi)

    roi = ROIParameters(
        x_left=_clamp(roi.x_left),
        y_top=_clamp(roi.y_top, IMAGE_Y_MIN, IMAGE_Y_MAX),
        width=_clamp(roi.width,  ROI_WIDTH_MIN,  ROI_WIDTH_MAX),
        height=_clamp(roi.height, ROI_HEIGHT_MIN, ROI_HEIGHT_MAX),
        roi_level=roi.roi_level,
        speed_was_implausible=speed_was_implausible_this_call,  # STAGE 8B (NEW)
    )

    return roi
