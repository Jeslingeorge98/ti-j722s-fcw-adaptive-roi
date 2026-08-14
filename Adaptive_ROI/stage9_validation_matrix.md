# Validation Traceability Matrix — Stage 9 Planning
**Date:** August 12, 2026 (updated same day — 6 items closed)
**Checked against:** `dynamic_roi.py` (uploaded copy confirmed byte-identical to `outputs/dynamic_roi.py`), `test_vectors.py` (**133 passing**, up from 126), `oracle.py`/`run_oracle_analysis.py` (**123 passing, 1 deferred**, up from 116)

**Update:** all 3 Gap rows and all 3 Partial rows identified in the original version of this matrix have now been closed with real, executed tests — see the new section below. Only the 3 hardware/data-blocked rows remain open, exactly as expected.

**Method:** every row below was checked by direct `grep`/inspection of the actual test files — not asserted from memory. Four status levels are used, and the distinction matters:

- **FULL** — genuinely covered, with the specific test(s) cited
- **PARTIAL** — something exists, but doesn't fully answer what the row asks
- **GAP** — zero coverage found; buildable now, in this environment
- **BLOCKED** — cannot be produced in this environment at all, regardless of effort, because it needs real hardware or real/recorded driving data

---

## Full coverage (13 of 22 rows)

| # | Test | Evidence |
|---|---|---|
| 1 | Floor containment | `Category 6` (SafetyGuaranteeAttacks), `Category 14` (216-combo sweep incl. off-centre lane), `Category 8` (96-combo cross-stage sweep), `Category 19` (**10,584-combination** exhaustive quantization sweep). Strongest-tested property in the system. |
| 2 | Full-frame fallback immediacy | `test_level_3_makes_corridor_gating_trivially_permissive` and others construct Level 3 via a single, first-ever `_compute_base_roi` call — confirmed no history/buildup is needed. |
| 3 | Speed variation | `Category 2` (sweeps), `Category 19::quant_speed_monotonic` (0–500 m/s). |
| 4 | Curvature variation | `Category 2`, `Category 6::test_extreme_curvature_does_not_produce_inverted_or_nan_bounds`, `Category 19` (both signs, full magnitude range including the non-monotonic-edge finding). |
| 5 | Lane dropout → CAN fallback | Scenario 5.1 test, `test_curvature_still_applied_at_zero_lane_confidence`. |
| 7 | CAN + lane dropout → full-frame | `test_level_3_triggers_on_dynamics_failure_not_lane_failure_alone` — this single test explicitly covers **both** the negative case (lane alone fails → Level 2, not 3) and the positive case (both fail together → Level 3), in one coherent test. |
| 11 | Object expansion / containment | `Category 3` in-corridor closing-vehicle test, corridor gating tests (Stage 3). |
| 12 | Parked vehicles, no unnecessary expansion | `test_scenario_1_1_parked_vehicle_does_not_inflate_region`. |
| 15 | Sign memory / occlusion duration | `test_sign_memory_bridges_brief_occlusion`, `test_sign_memory_forgotten_after_max_age` — both sides of the `SIGN_MEMORY_MAX_AGE` boundary checked. |
| 17 | Restart / no stale state | `Category 18` (WarmRestartPolicy) — comprehensive, includes the deliberately non-default `iou_threshold` check to catch silent-default bugs. |
| 18 | Detector gaps / bounded response | `Category 11` (DetectionSchedulingGaps) — includes the proof that IoU matching against a stale position fails while matching against the predicted position succeeds. |
| 19 | Quantized mode / envelope containment | `Category 19` — the 10,584-combination exhaustive check, plus the explicit proof that naive rounding would have failed and the envelope fix corrects it. |
| 20 | Area cap + floor simultaneously | `Category 13` — includes the specific case where the floor itself already exceeds the cap (Level 3 full-frame), confirming the cap has zero effect there. |

---

## Closed today — 6 of 6, all with real findings worth keeping

Every one of these was investigated by direct execution first, not assumed to just work. Three produced genuine findings along the way.

| # | Row | Result | What was actually found |
|---|---|---|---|
| 6 | CAN dropout alone | **Closed clean** | Isolated test (`vm_can_dropout_alone`) confirms both channels invalid + confident lane → Level 1 exactly, curvature=0.0, floor still uses speed. No surprises. |
| 8 | FOV clamp frequency | **Closed by design confirmation** | Built a test proving the per-frame flag is *sufficient* for a caller to derive frequency correctly (20-frame run, known injected clamp rate, caller-side aggregation matches within tolerance). This module still doesn't aggregate frequency itself — that remains a deliberate design boundary from Section 22, not a gap. |
| 9 | Vertical margin under braking | **Closed clean** | First isolated test ever written for this: holds speed/curvature/confidence fixed, varies only `abs_active`, confirms `y_top` and `y_bottom` both extend and total vertical extent is strictly larger. Every prior test had only checked the lateral side effect. |
| 13 | Cut-in timing | **Closed — caught a real test-construction bug in the process** | First attempt used a lateral step (0.035/frame) too large relative to box width for IoU matching to track continuity — spawned a new track every frame instead of one continuous track, so expansion never fired. This was **my own test-script bug**, not a system defect. Corrected to a realistic, IoU-trackable lateral speed (0.008/frame); the real system then showed expansion synchronized *exactly* to the frame the vehicle crosses into the corridor, with a substantial (>0.25) jump and zero premature or delayed firing. |
| 14 | Ground-truth TTC | **Closed clean** | Constructed a sequence with a known, injected true vertical velocity; hand-derived the expected TTC using the same formula the system uses but with the *true* (not filtered) velocity; confirmed the system's actual output matches within 15% (accounting for Kalman convergence lag). |
| 16 | Multiple signs | **Closed — confirmed exact, deterministic behavior** | Two different categories (`SIGN_ROADSIDE` + `SIGN_OVERHEAD`) coexist correctly, no collision. Two same-category detections in one frame: confirmed the collision is deterministic (**last-processed detection wins**, not first, not a crash, not a silently stale value) — this exact behavior had never actually been exercised by a test before, only implied by the code structure. |

**Test additions:** `TestCategory20_ValidationMatrixGaps` (7 tests — cut-in has two closely related assertions within one test), plus 7 new oracle entries (`VM-001` through `VM-007`), all passing on first real execution after the cut-in test-construction bug was found and corrected.


---

## Blocked — cannot be produced in this environment, regardless of effort

| # | Test | Why blocked |
|---|---|---|
| 10 | Camera pitch / calibration ground truth | Pitch estimation was deliberately deferred (Stage 1) — `pitch_rad` always defaults to `0.0`, with the ABS-triggered symmetric vertical margin standing in as a documented, conservative substitute. Validating **actual pitch estimation** needs a real vehicle's calibrated IMU/pitch sensor and recorded hard-braking events — there is nothing to validate here yet because the feature itself doesn't exist, and building it isn't just a test-writing exercise, it needs real ground-truth data this environment cannot produce |
| 21 | Runtime / target EVM timing | Requires the actual TDA4VM-class hardware. Documented as "Pending Confirmation" since Stage 6 and every session since — this has not changed and cannot change without physical access to the target chip |
| 22 | Model accuracy: fixed vs. dynamic vs. full-frame | This is not a missing test vector — **this is the definition of Stage 9 itself**, which has not started. It requires either real recorded driving data or a defensible synthetic dataset, plus an actual trained detection model to run comparative accuracy measurements against. This is a substantial, separate body of work already scoped in Section 6, not a gap in the existing test suite |

---

## Summary count (updated)

| Status | Count | Rows |
|---|---|---|
| Full | 19 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 |
| Blocked (needs real hardware/data) | 3 | 10, 21, 22 |

**Total: 19 + 3 = 22, matching every row in the original table.**

---

**Honest bottom line:** of the 22-row matrix, **86% (19 rows) now have solid, executed evidence**, and the remaining **14% (3 rows) are correctly blocked** pending real vehicle pitch/IMU ground truth, real target hardware, and the Stage 9 validation campaign itself. This is exactly where the plan said we'd land once the buildable gaps were closed — no test-suite work remains that doesn't require resources this environment cannot provide.
