"""
CAN signal reader for the Adaptive ROI pipeline.

Mock mode  — replays a CSV file row-by-row, looping at end-of-file.
             Required column : speed_kmh
             Optional columns: steering_angle_deg, yaw_rate_dps,
                               steering_valid, yaw_rate_valid,
                               abs_active, esc_active
             Missing optional columns get safe defaults (0 / True / False).

Real mode  — background thread listens on `channel` (default 'can0') via
             python-can and decodes signals using the arbitration IDs and
             byte layouts from INTEGRATION_PLAN.md §Step 2.
             get_latest() is thread-safe and returns the most recent
             CanSignals (stopped-vehicle defaults until first message).

Signal mapping (real mode):
    0x100  bytes [0:2]  uint16 big-endian  × 0.01  → speed_kmh
    0x200  bytes [0:2]  int16  big-endian  × 0.1   → steering_angle_deg
    0x300  bytes [0:2]  int16  big-endian  × 0.01  → yaw_rate rad/s → dps
    0x400  byte 0 bit0                             → abs_active
    0x400  byte 0 bit1                             → esc_active
"""

import csv
import math
import struct
import threading
import time
from typing import List, Optional

from roi.dynamic_roi import CanSignals

# ---------------------------------------------------------------------------
# Real-mode arbitration IDs and scaling factors
# ---------------------------------------------------------------------------
_AID_SPEED   = 0x100
_AID_STEER   = 0x200
_AID_YAW     = 0x300
_AID_ABS_ESC = 0x400

_SPEED_SCALE = 0.01   # raw uint16 → km/h
_STEER_SCALE = 0.1    # raw int16  → degrees
_YAW_SCALE   = 0.01   # raw int16  → rad/s (converted to dps after)

# Real mode: if no CAN message has been received within this window,
# steering_valid and yaw_rate_valid are forced False so the ROI stops
# using stale curvature estimates. abs_active/esc_active are kept at
# their last value (conservative: ABS floor stays expanded if it was on).
_STALE_TIMEOUT_S = 0.5

_STOPPED = CanSignals(
    speed_mps=0.0,
    steering_angle_deg=0.0,
    yaw_rate_dps=0.0,
    steering_valid=False,
    yaw_rate_valid=False,
)


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ('1', 'true', 'yes')


class CANSignalReader:
    """
    Parameters
    ----------
    mode     : 'mock' or 'real'
    csv_path : path to CSV file — required when mode='mock'
    channel  : CAN interface name (mode='real' only), e.g. 'can0'
    bitrate  : bus bitrate in bps (mode='real' only, informational)
    """

    def __init__(self, mode: str = 'mock', csv_path: Optional[str] = None,
                 channel: str = 'can0', bitrate: int = 500_000):
        self._mode = mode

        if mode == 'mock':
            if csv_path is None:
                raise ValueError("csv_path is required for mock mode")
            self._frames: List[CanSignals] = _load_csv(csv_path)
            if not self._frames:
                raise ValueError(f"CSV file is empty or has no valid rows: {csv_path}")
            self._idx = 0
            self._lock = threading.Lock()

        elif mode == 'real':
            import can  # python-can; lazy import keeps mock installs clean
            self._latest = _STOPPED
            self._last_rx_time: float = 0.0  # monotonic timestamp of last received message
            self._lock = threading.Lock()
            self._bus = can.interface.Bus(channel=channel, bustype='socketcan')
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'mock' or 'real'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest(self) -> CanSignals:
        """Return current CanSignals. Thread-safe. Advances replay index in mock mode.

        Real mode: if no message has arrived within _STALE_TIMEOUT_S, returns
        the last signals with steering_valid=False and yaw_rate_valid=False so
        the ROI stops using stale curvature estimates. abs_active/esc_active
        are kept at their last value (conservative: ABS floor stays expanded).
        """
        if self._mode == 'mock':
            with self._lock:
                sig = self._frames[self._idx]
                self._idx = (self._idx + 1) % len(self._frames)
            return sig
        with self._lock:
            sig = self._latest
            age = time.monotonic() - self._last_rx_time
        if age > _STALE_TIMEOUT_S:
            from dataclasses import replace
            sig = replace(sig, steering_valid=False, yaw_rate_valid=False)
        return sig

    def close(self) -> None:
        """Shut down the CAN bus (real mode only). No-op in mock mode."""
        if self._mode == 'real':
            self._bus.shutdown()

    # ------------------------------------------------------------------
    # Real-mode background reader
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        speed_mps   = 0.0
        steer_deg   = None
        yaw_dps     = None
        steer_valid = False
        yaw_valid   = False
        abs_active  = False
        esc_active  = False

        while True:
            try:
                msg = self._bus.recv(timeout=1.0)
                if msg is None:
                    continue

                aid = msg.arbitration_id

                if aid == _AID_SPEED and len(msg.data) >= 2:
                    raw = struct.unpack_from('>H', msg.data, 0)[0]
                    speed_mps = (raw * _SPEED_SCALE) / 3.6

                elif aid == _AID_STEER and len(msg.data) >= 2:
                    raw = struct.unpack_from('>h', msg.data, 0)[0]
                    steer_deg   = raw * _STEER_SCALE
                    steer_valid = True

                elif aid == _AID_YAW and len(msg.data) >= 2:
                    raw = struct.unpack_from('>h', msg.data, 0)[0]
                    yaw_dps   = math.degrees(raw * _YAW_SCALE)
                    yaw_valid = True

                elif aid == _AID_ABS_ESC and len(msg.data) >= 1:
                    b          = msg.data[0]
                    abs_active = bool(b & 0x01)
                    esc_active = bool(b & 0x02)

                else:
                    continue

                with self._lock:
                    self._latest = CanSignals(
                        speed_mps=speed_mps,
                        steering_angle_deg=steer_deg,
                        yaw_rate_dps=yaw_dps,
                        steering_valid=steer_valid,
                        yaw_rate_valid=yaw_valid,
                        abs_active=abs_active,
                        esc_active=esc_active,
                    )
                    self._last_rx_time = time.monotonic()

            except Exception:
                pass  # bus errors (cable pulled, etc.) — keep thread alive


# ---------------------------------------------------------------------------
# CSV loader (module-level, shared by all mock instances)
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> List[CanSignals]:
    frames: List[CanSignals] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            speed_kmh = float(row['speed_kmh'])
            frames.append(CanSignals(
                speed_mps=speed_kmh / 3.6,
                steering_angle_deg=float(row.get('steering_angle_deg') or 0.0),
                yaw_rate_dps=float(row.get('yaw_rate_dps') or 0.0),
                steering_valid=_to_bool(row.get('steering_valid', True)),
                yaw_rate_valid=_to_bool(row.get('yaw_rate_valid', True)),
                abs_active=_to_bool(row.get('abs_active', False)),
                esc_active=_to_bool(row.get('esc_active', False)),
            ))
    return frames
