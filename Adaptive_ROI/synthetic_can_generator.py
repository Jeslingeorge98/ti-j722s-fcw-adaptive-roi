"""
Synthetic CAN signal generator for testing the main pipeline without a
real vehicle -- solves the self-consistency problem: video and CAN
generated independently will disagree with each other. This derives
CAN signals directly from a chosen trajectory (speed + curvature over
time), respecting BOTH physical limits the pipeline itself enforces,
so whatever drives your video generator's road geometry should use
the SAME (possibly-clamped) effective curvature this returns -- not
the raw value you originally requested.
"""
import math
import dynamic_roi as m


def generate_synthetic_can_sequence(speed_profile_kmh, curvature_profile_inv_m,
                                       wheelbase_m=m.WHEELBASE_M):
    """
    Returns (sequence, effective_curvatures, warnings) where:
      - sequence: list of CanSignals, self-consistent with each other
      - effective_curvatures: what the pipeline will ACTUALLY compute
        internally for each frame, after both clamps -- feed THIS to
        your video generator's road-curvature parameter, not the raw
        requested value, or video and pipeline state will disagree
      - warnings: human-readable notes on which frames were clamped and why
    """
    sequence = []
    effective_curvatures = []
    warnings_list = []

    for i, (speed_kmh, kappa) in enumerate(zip(speed_profile_kmh, curvature_profile_inv_m)):
        speed_mps = speed_kmh / 3.6

        # --- Clamp 1: mechanical steering limit (+/-45 deg) ---
        steer_rad = math.atan(kappa * wheelbase_m)
        steer_deg = math.degrees(steer_rad)
        if abs(steer_deg) > 45.0:
            warnings_list.append(
                f"frame {i}: requested kappa={kappa:.4f}/m needs {steer_deg:.1f}deg steering "
                f"(exceeds +/-45deg mechanical clamp) -- clamped at the steering-angle stage"
            )
            steer_deg = math.copysign(45.0, steer_deg)
            steer_rad = math.radians(steer_deg)

        yaw_rate_dps = math.degrees(math.tan(steer_rad) / wheelbase_m * speed_mps)

        sig = m.CanSignals(
            speed_mps=speed_mps, steering_angle_deg=steer_deg, yaw_rate_dps=yaw_rate_dps,
            steering_valid=True, yaw_rate_valid=True,
        )
        sequence.append(sig)

        # --- Clamp 2: the pipeline's OWN lateral-acceleration comfort limit ---
        # This is often the TIGHTER limit at realistic highway speeds --
        # found directly while building this generator (2026-08-12).
        effective_kappa = m._compute_curvature(sig)
        effective_curvatures.append(effective_kappa)
        if abs(effective_kappa - kappa) > 1e-6 and abs(steer_deg) <= 45.0:
            lat_limit = m.MAX_LAT_ACC_MPS2 / max(speed_mps**2, m.LAT_ACC_DENOM_FLOOR)
            warnings_list.append(
                f"frame {i}: requested kappa={kappa:.4f}/m exceeds the lateral-acceleration "
                f"comfort limit at {speed_kmh}km/h ({lat_limit:.4f}/m) -- the pipeline will "
                f"actually use {effective_kappa:.4f}/m. Feed THIS value to your video "
                f"generator's road curvature, not the originally requested one."
            )

    return sequence, effective_curvatures, warnings_list


if __name__ == "__main__":
    print("=== Demo: a deliberately aggressive trajectory (sharp curve, high speed) ===")
    speed_profile = [100, 100, 100, 100, 100]
    curvature_profile = [0.0, 0.005, 0.02, 0.05, 0.02]
    seq, effective, warns = generate_synthetic_can_sequence(speed_profile, curvature_profile)

    for w in warns:
        print(f"  {w}")
    print()
    for i, (sig, eff) in enumerate(zip(seq, effective)):
        print(f"frame {i}: requested={curvature_profile[i]:.4f}  effective={eff:.4f}  "
              f"steer={sig.steering_angle_deg:.2f}deg  yaw={sig.yaw_rate_dps:.2f}dps")
