#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import numpy as np
import copy
import debug
import os
import csv
import json
import yaml
import datetime
import time
import threading
import queue
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def _env_flag(name, default=False):
    """Read a boolean from the environment ("1"/"true"/"yes"/"on" are true)."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def decode_detections(results, model):
    """
    Turn a detection model's raw outputs into an [N, 6] array of
    [x1, y1, x2, y2, class_idx, score], with the box coordinates normalised to
    0..1 of the model input.

    This is the reshape / head-reorder / formatter / normalisation sequence that
    detection post-processing needs before anything can be drawn. It lives here
    so PostProcessDetection and the lane overlay share one implementation.
    """
    results = list(results)
    for i, r in enumerate(results):
        r = np.squeeze(r)
        if r.ndim == 1:
            r = np.expand_dims(r, 1)
        results[i] = r

    # Reorder heads if needed
    if model.shuffle_indices:
        results = [results[i] for i in model.shuffle_indices]

    # Some models stash a scalar; drop it
    if results[-1].ndim < 2:
        results = results[:-1]

    bbox = np.concatenate(results, axis=-1)

    # Optional field re-mapping
    if model.formatter:
        if model.ignore_index is None:
            bbox_copy = copy.deepcopy(bbox)
        else:
            bbox_copy = copy.deepcopy(np.delete(bbox, model.ignore_index, 1))
        bbox[..., model.formatter["dst_indices"]] = bbox_copy[
            ..., model.formatter["src_indices"]
        ]

    # If detections are in pixels, convert to normalized [0..1]
    if not model.normalized_detections:
        bbox[..., (0, 2)] /= model.resize[0]  # width
        bbox[..., (1, 3)] /= model.resize[1]  # height

    return bbox


def resolve_class(model, class_idx):
    """
    Map a model's raw class index to (display name, rgb_color) via its
    label_offset and dataset.yaml.
    """
    if isinstance(model.label_offset, dict):
        dataset_idx = model.label_offset.get(class_idx, class_idx)
    else:
        dataset_idx = model.label_offset + class_idx

    if dataset_idx in model.dataset_info:
        info = model.dataset_info[dataset_idx]
        class_name = info.name or "UNDEFINED"
        if info.supercategory:
            class_name = info.supercategory + "/" + class_name
        return class_name, info.rgb_color

    return "UNDEFINED", (20, 220, 20)


# =============================================================================
# UFLDv2 Lane Detection Utilities (ported from lanedetection_gst.py)
# =============================================================================
# Tusimple annotations (row_anchor_start/end) are defined against the
# dataset's native 720px-tall images, independent of the actual camera
# frame size passed into pred2coords.
_TUSIMPLE_ROW_ANCHOR_REF_HEIGHT = 720.0

# Fallback palette used when the model dir has no dataset.yaml. Indexed by lane
# id (channel index in the output tensors), so it stays in sync with the
# color_map in dataset.yaml. Values are B,G,R as cv2 consumes them.
LANE_COLORS = [
    (255, 180, 0),  # id 0  left_adjacent   — light blue
    (0, 255, 0),    # id 1  ego_left        — green
    (0, 0, 255),    # id 2  ego_right       — red
    (0, 255, 255),  # id 3  right_adjacent  — yellow
]


def _softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _expected_grid_positions(loc, lane_id, ks, local_width):
    """
    Vectorised form of UFLDv2's per-anchor refinement loop.

    For every valid anchor in `ks`, take the (2*local_width+1) window of grid
    logits centred on that anchor's argmax, softmax it, and return the
    probability-weighted grid position (+0.5, as in the reference).

    Windows are *truncated* at the grid edges rather than clamped onto
    duplicate indices, matching the reference's arange(lo, hi+1): positions
    that fall outside the grid get a -inf logit so they contribute no
    probability and drop out of the softmax denominator.
    """
    num_grid = loc.shape[0]

    centers = loc[:, ks, lane_id].argmax(axis=0)                # (K,)
    offsets = np.arange(-local_width, local_width + 1)
    window = centers[:, None] + offsets[None, :]                # (K, W)

    in_grid = (window >= 0) & (window < num_grid)
    idx = np.clip(window, 0, num_grid - 1)

    logits = loc[idx, ks[:, None], lane_id]                     # (K, W)
    logits = np.where(in_grid, logits, -np.inf)
    probs = _softmax(logits, axis=1)

    return (probs * idx).sum(axis=1) + 0.5


def pred2coords(loc_row, loc_col, exist_row, exist_col, row_anchor, col_anchor,
                 row_lane_idx=(1, 2), col_lane_idx=(0, 3), local_width=1,
                 original_image_width=640, original_image_height=256):
    """
    Numpy re-implementation of UFLDv2's demo.pred2coords (batch dim already
    squeezed out of every tensor).

    loc_row/exist_row : (num_grid_row, num_cls_row, num_lane_row) / (2, num_cls_row, num_lane_row)
    loc_col/exist_col : (num_grid_col, num_cls_col, num_lane_col) / (2, num_cls_col, num_lane_col)

    Returns a list of (lane_id, points) pairs. Lanes failing the existence
    check are omitted entirely, so the lane id is carried explicitly rather
    than inferred from list position — position would shift frame to frame and
    make per-lane colours/labels unstable.
    """
    num_grid_row, num_cls_row, _ = loc_row.shape
    num_grid_col, num_cls_col, _ = loc_col.shape

    valid_row = exist_row.argmax(axis=0)
    valid_col = exist_col.argmax(axis=0)

    coords = []

    for i in row_lane_idx:
        if valid_row[:, i].sum() <= num_cls_row / 2:
            continue
        ks = np.flatnonzero(valid_row[:, i])
        pos = _expected_grid_positions(loc_row, i, ks, local_width)
        xs = (pos / (num_grid_row - 1) * original_image_width).astype(np.int32)
        ys = (row_anchor[ks] * original_image_height).astype(np.int32)
        coords.append((i, list(zip(xs.tolist(), ys.tolist()))))

    for i in col_lane_idx:
        if valid_col[:, i].sum() <= num_cls_col / 4:
            continue
        ks = np.flatnonzero(valid_col[:, i])
        pos = _expected_grid_positions(loc_col, i, ks, local_width)
        ys = (pos / (num_grid_col - 1) * original_image_height).astype(np.int32)
        xs = (col_anchor[ks] * original_image_width).astype(np.int32)
        coords.append((i, list(zip(xs.tolist(), ys.tolist()))))

    return coords


def filter_lane(lane, img_h, img_w):
    y_min = int(img_h * 0.65)
    lane = [c for c in lane if c[1] >= y_min]
    if len(lane) < 6:
        return []

    # Build the point array once so the line fit below can be evaluated as a
    # single vector op. The original called np.polyval() once per point inside a
    # list comprehension, which cost ~9.9 ms/frame on the AM67A's ARM cores.
    arr = np.asarray(lane, dtype=np.float64)
    xs_a = arr[:, 0]
    ys_a = arr[:, 1]

    x_span = xs_a.max() - xs_a.min()
    y_span = ys_a.max() - ys_a.min()
    if y_span < img_h * 0.05 or x_span > y_span * 2.0:
        return []

    pts = sorted(lane, key=lambda p: -p[1])
    n_ref = max(3, len(pts) // 2)
    ref = np.asarray(pts[:n_ref], dtype=np.float32)
    coeffs = np.polyfit(ref[:, 1], ref[:, 0], 1)

    if abs(coeffs[0]) > 1.8:
        return []

    # np.polyval() on two coefficients is exactly c0*x + c1, so evaluating it
    # vectorised here is bit-identical to the per-point calls it replaces.
    thr = img_w * 0.05
    keep = np.abs(xs_a - (coeffs[0] * ys_a + coeffs[1])) < thr
    kept = [lane[i] for i in np.flatnonzero(keep)]
    if len(kept) < 2:
        return lane

    kept_sorted = sorted(kept, key=lambda p: p[1])
    clean = [kept_sorted[0]]
    for i in range(1, len(kept_sorted)):
        dy = kept_sorted[i][1] - clean[-1][1]
        dx = abs(kept_sorted[i][0] - clean[-1][0])
        if dy > 0 and dx / dy > 2.5:
            continue
        clean.append(kept_sorted[i])
    kept = clean if len(clean) >= 2 else kept

    k = np.asarray(kept, dtype=np.float64)
    residuals = np.abs(k[:, 0] - (coeffs[0] * k[:, 1] + coeffs[1]))
    if residuals.mean() > img_w * 0.025:
        return []

    return kept


def _smooth_lane(lane, img_h, n_pts=80):
    """
    Fit a quadratic through the lane points and resample it.

    Returns the polyline as an (n_pts, 1, 2) int32 array, i.e. already in the
    shape cv2.polylines() wants -- building a list of n_pts Python tuples only
    for the caller to convert it straight back to an array was pure overhead.
    """
    pts = np.asarray(sorted(lane, key=lambda p: p[1]), dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]
    deg = min(2, len(pts) - 1)
    coeffs = np.polyfit(ys, xs, deg)
    y_dense = np.linspace(ys[0], ys[-1], n_pts)
    x_dense = np.polyval(coeffs, y_dense)
    x_dense = np.clip(x_dense, 0, 1e6)
    return np.stack((x_dense, y_dense), axis=1).astype(np.int32).reshape(-1, 1, 2)


# =============================================================================
# Utility Functions with Improved Error Handling
# =============================================================================
def create_title_frame(title, width, height):
    """Create title frame with Texas Instruments branding."""
    frame = np.zeros((height, width, 3), np.uint8)
    if title is not None:
        frame = cv2.putText(
            frame,
            "Texas Instruments - Edge Analytics",
            (40, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            2,
        )
        frame = cv2.putText(
            frame, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    return frame


def overlay_model_name(frame, model_name, start_x, start_y, width, height):
    """Overlay model name on frame."""
    row_size = 40 * width // 1280
    font_size = width / 1280
    cv2.putText(
        frame,
        "Model : " + model_name,
        (start_x + 5, start_y - row_size // 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        2,
    )
    return frame


def load_camera_params(json_path="camera_calibration_c270.json", camera_name=None):
    """
    Robust camera parameter loading with proper error handling and defaults.
    """
    default_params = {
        "fx_px": 1406.7165978907144,
        "fy_px": 1408.8291227778363,
        "cx_px": 636.0321092582369,
        "cy_px": 350.38274035639483,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "image_width": 1280,
        "image_height": 720
    }
    
    try:
        if not os.path.exists(json_path):
            print(f"Warning: Camera parameters file '{json_path}' not found. Using defaults.")
            raise FileNotFoundError(f"Camera params file not found: {json_path}")
        
        with open(json_path, "r") as f:
            all_params = json.load(f)
        
        if camera_name and camera_name in all_params:
            params = all_params[camera_name]
            print(f"Loaded parameters for camera: {camera_name}")
        elif "fx_px" in all_params:
            params = all_params
            print("Loaded single camera parameters")
        elif len(all_params) == 1:
            camera_name = list(all_params.keys())[0]
            params = all_params[camera_name]
            print(f"Loaded parameters for camera: {camera_name}")
        else:
            print(f"Warning: Could not find camera parameters structure. Using defaults.")
            params = default_params
        
        required_keys = ["fx_px", "fy_px", "cx_px", "cy_px", "distortion"]
        missing_keys = [key for key in required_keys if key not in params]
        
        if missing_keys:
            print(f"Warning: Missing keys {missing_keys} in camera params, using defaults")
            for key in missing_keys:
                params[key] = default_params[key]
        
        K = np.array([
            [float(params["fx_px"]), 0, float(params["cx_px"])],
            [0, float(params["fy_px"]), float(params["cy_px"])],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = params["distortion"]
        if len(dist_coeffs) == 4:
            dist = np.array([dist_coeffs[0], dist_coeffs[1], 
                           dist_coeffs[2], dist_coeffs[3], 0.0], dtype=np.float32)
        elif len(dist_coeffs) == 5:
            dist = np.array(dist_coeffs, dtype=np.float32)
        else:
            print(f"Warning: Unexpected distortion coefficients length {len(dist_coeffs)}. Using zeros.")
            dist = np.zeros((5,), dtype=np.float32)
        
        return {
            "fx": float(params["fx_px"]),
            "fy": float(params["fy_px"]),
            "cx": float(params["cx_px"]),
            "cy": float(params["cy_px"]),
            "dist": dist,
            "K": K,
            "camera_name": camera_name or "default_camera",
            "image_width": params.get("image_width", default_params["image_width"]),
            "image_height": params.get("image_height", default_params["image_height"]),
            "original_json": params
        }
        
    except FileNotFoundError:
        print(f"Warning: Camera parameters file '{json_path}' not found. Using defaults.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in camera parameters file: {e}")
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
    
    # Return robust defaults
    K = np.array([
        [default_params["fx_px"], 0, default_params["cx_px"]],
        [0, default_params["fy_px"], default_params["cy_px"]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return {
        "fx": default_params["fx_px"],
        "fy": default_params["fy_px"],
        "cx": default_params["cx_px"],
        "cy": default_params["cy_px"],
        "dist": np.array(default_params["distortion"], dtype=np.float32),
        "K": K,
        "camera_name": "default_camera",
        "image_width": default_params["image_width"],
        "image_height": default_params["image_height"],
        "original_json": default_params
    }


def get_risk_color(ttc):
    """Get color based on TTC for FCW visualization (NCAP compliant)."""
    if ttc < 2.0:
        return (0, 0, 255)      # Red - Critical
    elif ttc < 4.0:
        return (0, 165, 255)    # Orange - Warning
    elif ttc < 6.0:
        return (0, 255, 255)    # Yellow - Caution
    else:
        return (0, 255, 0)      # Green - Safe


def get_track_state_color(state):
    """Get color based on track freshness state."""
    if state == "FRESH":
        return (0, 255, 0)      # Green - Fresh measurement
    elif state == "COAST_1":
        return (0, 255, 255)    # Yellow - One frame coasted
    elif state == "COAST_2":
        return (0, 165, 255)    # Orange - Two frames coasted
    elif state == "STALE":
        return (0, 0, 255)      # Red - Stale (should not be published)
    else:
        return (255, 255, 255)  # White - Unknown


def ema_filter(prev_value, curr_value, alpha=0.4):
    """Exponential Moving Average filter with initialization handling."""
    if prev_value is None:
        return curr_value
    return alpha * curr_value + (1 - alpha) * prev_value


def apply_nms_with_indices(boxes, scores, class_indices, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression and return indices to maintain alignment.
    Returns: (kept_boxes, kept_scores, kept_class_indices, kept_indices)
    """
    if len(boxes) == 0:
        return [], [], [], []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_indices = np.array(class_indices)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    keep_original_indices = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        keep_original_indices.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return (boxes[keep], scores[keep], class_indices[keep], keep_original_indices)


# =============================================================================
# Data Structures for Temporal Consistency
# =============================================================================
@dataclass
class FrameMetadata:
    """Metadata for frame timing and synchronization."""
    frame_id: int
    capture_timestamp: float  # Monotonic timestamp when frame was captured
    processing_timestamp: float  # Monotonic timestamp when processing started
    publish_timestamp: float = 0.0  # Monotonic timestamp when results published
    latency_ms: float = 0.0  # Processing latency
    
    def __post_init__(self):
        """Calculate latency after initialization."""
        if self.publish_timestamp > 0:
            self.latency_ms = (self.publish_timestamp - self.capture_timestamp) * 1000


# =============================================================================
# Async Logger to Remove Disk I/O from Real-time Path
# =============================================================================
class AsyncLogger:
    """Asynchronous logger to prevent disk I/O from blocking real-time processing."""
    
    def __init__(self, output_dir: Path, max_queue_size: int = 100):
        self.output_dir = output_dir
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.logger_thread = None
        self.running = False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file
        self.csv_path = self.output_dir / "results.csv"
        self.csv_file = None
        self.csv_writer = None
        
        print(f"Async logger initialized. Output directory: {self.output_dir}")
    
    def start(self):
        """Start the logger thread."""
        self.running = True
        self.logger_thread = threading.Thread(target=self._logger_worker, daemon=True)
        self.logger_thread.start()
        print("Async logger started.")
    
    def stop(self):
        """Stop the logger thread."""
        self.running = False
        if self.logger_thread:
            self.logger_thread.join(timeout=5.0)
        
        if self.csv_file:
            self.csv_file.close()
        
        print("Async logger stopped.")
    
    def log_frame(self, frame_id: int, timestamp: float, image: np.ndarray, 
                  detections: List, metadata: Optional[Dict] = None):
        """
        Queue a frame for logging. Non-blocking - drops frames if queue is full.
        """
        try:
            # Use put_nowait to avoid blocking
            self.frame_queue.put_nowait({
                'frame_id': frame_id,
                'timestamp': timestamp,
                'image': image.copy() if image is not None else None,
                'detections': copy.deepcopy(detections),
                'metadata': metadata or {}
            })
            return True
        except queue.Full:
            # Drop frame if queue is full (best-effort logging)
            return False
    
    def _logger_worker(self):
        """Worker thread that handles disk I/O."""
        # Open CSV file in worker thread
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write CSV header
        self.csv_writer.writerow([
            "frame_id", "timestamp", "track_id", "class_name", "distance_m",
            "ttc_s", "risk_level", "relative_speed_mps", "fcw_trigger",
            "x1", "y1", "x2", "y2", "confidence", "track_state",
            "capture_latency_ms", "image_path"
        ])
        
        while self.running or not self.frame_queue.empty():
            try:
                # Get frame with timeout to allow checking running flag
                frame_data = self.frame_queue.get(timeout=1.0)
                
                try:
                    self._process_frame_data(frame_data)
                except Exception as e:
                    print(f"Warning: Error processing frame data: {e}")
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in logger worker: {e}")
                continue
    
    def _process_frame_data(self, frame_data: Dict):
        """Process and save frame data."""
        frame_id = frame_data['frame_id']
        timestamp = frame_data['timestamp']
        image = frame_data['image']
        detections = frame_data['detections']
        metadata = frame_data['metadata']
        
        # Only save every 10th frame to avoid disk overload
        if frame_id % 10 != 0:
            return
        
        # Save image
        image_filename = f"frame_{frame_id:06d}_{timestamp}.jpg"
        image_path = self.output_dir / image_filename
        
        if image is not None:
            try:
                cv2.imwrite(str(image_path), image)
            except Exception as e:
                print(f"Warning: Could not save image {image_path}: {e}")
                return
        
        # Save detection data to CSV
        for detection in detections:
            self.csv_writer.writerow([
                frame_id,
                timestamp,
                detection.get('track_id', -1),
                detection.get('class_name', 'UNKNOWN'),
                detection.get('distance', -1.0),
                detection.get('ttc', 999.0),
                detection.get('risk_level', 'UNKNOWN'),
                detection.get('relative_speed', 0.0),
                detection.get('fcw_trigger', 0),
                detection.get('x1', 0),
                detection.get('y1', 0),
                detection.get('x2', 0),
                detection.get('y2', 0),
                detection.get('confidence', 0.0),
                detection.get('track_state', 'UNKNOWN'),
                metadata.get('latency_ms', 0.0),
                str(image_path) if image is not None else ''
            ])
        
        # Flush periodically
        if frame_id % 50 == 0:
            self.csv_file.flush()


# =============================================================================
# Enhanced Kalman Filter with Variable Time Step
# =============================================================================
class AdaptiveKalmanFilter:
    """
    Kalman filter with adaptive time step and noise covariance.
    Uses capture timestamps for accurate prediction.
    """
    
    def __init__(self, initial_dt: float = 1/30):
        self.dt = initial_dt
        self.kf = cv2.KalmanFilter(7, 4)  # 7 states, 4 measurements
        
        # State: [x, y, w, h, vx, vy, vw]
        # Measurement: [x, y, w, h]
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], np.float32)
        
        # Store last prediction timestamp
        self.last_prediction_time = None
        self.last_update_time = None
        
    def init(self, bbox: List[float], timestamp: float):
        """Initialize filter with bounding box and timestamp."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0], dtype=np.float32
        )
        
        # Initialize error covariance
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 0.1
        
        self.last_update_time = timestamp
        self.last_prediction_time = timestamp
        
        # Adaptive noise based on initial size
        self._update_noise_covariance(w, h)
    
    def _update_noise_covariance(self, w: float, h: float):
        """Update process noise based on object size and velocity."""
        # Larger objects have more stable motion
        size_factor = np.sqrt(w * h) / 100.0
        
        # Process noise covariance
        pos_noise = 0.01 * size_factor
        size_noise = 0.001 * size_factor
        vel_noise = 0.1 * size_factor
        
        self.kf.processNoiseCov = np.diag([
            pos_noise, pos_noise, size_noise, size_noise,
            vel_noise, vel_noise, vel_noise * 0.5
        ]).astype(np.float32)
        
        # Measurement noise covariance (higher for smaller objects)
        meas_noise = 0.5 / size_factor if size_factor > 0 else 0.5
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * meas_noise
    
    def predict(self, current_time: float) -> np.ndarray:
        """Predict state at current time using actual time difference."""
        if self.last_prediction_time is None:
            self.last_prediction_time = current_time
            return self.kf.statePost
        
        # Calculate actual time difference
        dt = max(0.001, current_time - self.last_prediction_time)  # Avoid zero dt
        self.dt = dt
        
        # Update transition matrix with actual dt
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        self.last_prediction_time = current_time
        return self.kf.predict()
    
    def update(self, bbox: List[float], timestamp: float):
        """Update filter with new measurement and timestamp."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # Update noise based on new size
        self._update_noise_covariance(w, h)
        
        self.kf.correct(np.array([cx, cy, w, h], dtype=np.float32))
        self.last_update_time = timestamp
        self.last_prediction_time = timestamp
    
    def get_bbox(self) -> List[int]:
        """Get bounding box from current state."""
        cx, cy, w, h = self.kf.statePost[:4]
        return [
            int(cx - w / 2),
            int(cy - h / 2),
            int(cx + w / 2),
            int(cy + h / 2),
        ]
    
    def get_velocity_pixels(self) -> Tuple[float, float]:
        """Get velocity in pixels per second."""
        vx, vy = self.kf.statePost[4:6]
        return vx / self.dt if self.dt > 0 else 0, vy / self.dt if self.dt > 0 else 0
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty from error covariance."""
        return np.sqrt(self.kf.errorCovPost[0, 0] + self.kf.errorCovPost[1, 1])


# =============================================================================
# Track Object with Freshness Management
# =============================================================================
class ADAS_Track:
    """
    Track object with proper freshness management and state machine.
    Implements tier-1 ADAS track lifecycle.
    """
    
    def __init__(self, track_id: int, bbox: List[float], class_name: str, 
                 confidence: float, frame_metadata: FrameMetadata):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        
        # Temporal metadata
        self.creation_metadata = frame_metadata
        self.last_update_metadata = frame_metadata
        self.last_prediction_time = frame_metadata.capture_timestamp
        
        # Track state machine
        self.state = "FRESH"  # FRESH, COAST_1, COAST_2, STALE
        self.hits = 1
        self.time_since_update = 0
        self.consecutive_coasts = 0
        self.quality_score = 1.0
        
        # Kalman filter with proper time handling
        self.kalman = AdaptiveKalmanFilter()
        self.kalman.init(bbox, frame_metadata.capture_timestamp)
        self.bbox = self.kalman.get_bbox()
        
        # Distance measurements (for closing speed calculation)
        self.distance_history = deque(maxlen=5)  # (distance, timestamp)
        self.filtered_distance = None
        self.closing_speed_mps = 0.0  # Closing speed in m/s
        
        # Position history for velocity validation
        self.position_history = deque(maxlen=3)  # (cx, cy, timestamp)
        self.add_position(bbox, frame_metadata.capture_timestamp)
        
        # Coast verification
        self.last_valid_roi = None  # For template matching during coasting
        self.coast_verification_failures = 0
    
    def update(self, bbox: List[float], class_name: str, confidence: float, 
               frame_metadata: FrameMetadata):
        """Update track with fresh measurement."""
        self.class_name = class_name
        self.confidence = confidence
        self.last_update_metadata = frame_metadata
        self.time_since_update = 0
        self.consecutive_coasts = 0
        self.coast_verification_failures = 0
        self.state = "FRESH"
        self.hits += 1
        
        # Update Kalman filter with capture timestamp
        self.kalman.update(bbox, frame_metadata.capture_timestamp)
        self.bbox = self.kalman.get_bbox()
        
        # Store last valid ROI for coast verification
        self._store_roi_for_verification()
        
        # Add to position history
        self.add_position(bbox, frame_metadata.capture_timestamp)
        
        # Update quality score
        self._update_quality_score()
    
    def predict(self, frame_metadata: FrameMetadata):
        """Predict track state at current frame time."""
        # Predict using actual capture timestamp
        self.kalman.predict(frame_metadata.capture_timestamp)
        self.bbox = self.kalman.get_bbox()
        self.last_prediction_time = frame_metadata.capture_timestamp
        
        # Update track state
        self.time_since_update += 1
        self.consecutive_coasts += 1
        
        # State machine transition
        if self.time_since_update == 1:
            self.state = "COAST_1"
        elif self.time_since_update == 2:
            self.state = "COAST_2"
        elif self.time_since_update >= 3:
            self.state = "STALE"
        
        # Decay quality during coasting
        if self.state != "FRESH":
            decay_factor = 0.8 if self.state == "COAST_1" else 0.6 if self.state == "COAST_2" else 0.3
            self.quality_score *= decay_factor
    
    def add_distance_measurement(self, distance: float, timestamp: float):
        """Add distance measurement for closing speed calculation."""
        self.distance_history.append((distance, timestamp))
        self.filtered_distance = ema_filter(self.filtered_distance, distance, alpha=0.4)
        
        # Calculate closing speed from distance history
        if len(self.distance_history) >= 2:
            dist1, t1 = self.distance_history[0]
            dist2, t2 = self.distance_history[-1]
            
            if t2 > t1 and abs(dist2 - dist1) > 0.1:
                dt = t2 - t1
                closing_speed = (dist2 - dist1) / dt  # Positive = getting closer
                self.closing_speed_mps = ema_filter(self.closing_speed_mps, closing_speed, alpha=0.3)
    
    def add_position(self, bbox: List[float], timestamp: float):
        """Add position to history for motion validation."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self.position_history.append((cx, cy, timestamp))
    
    def _store_roi_for_verification(self):
        """Store ROI for coast verification (placeholder for actual implementation)."""
        # In real implementation, store image patch for template matching
        self.last_valid_roi = self.bbox
    
    def _update_quality_score(self):
        """Update track quality based on various factors."""
        # Base quality from hits
        hits_quality = min(self.hits / 10.0, 1.0)
        
        # Freshness penalty
        freshness_penalty = 1.0 / (1.0 + self.time_since_update * 0.5)
        
        # Combine factors
        self.quality_score = 0.4 * hits_quality + 0.4 * freshness_penalty + 0.2 * self.confidence
    
    def should_be_published(self) -> bool:
        """Determine if track should be published based on freshness."""
        # Tier-1 rule: only publish FRESH or very recent COAST_1 tracks
        return self.state in ["FRESH", "COAST_1"] and self.time_since_update <= 2
    
    def should_trigger_fcw(self) -> bool:
        """Determine if track can trigger FCW (stricter than publishing)."""
        # FCW requires fresh measurements only
        return self.state == "FRESH" and self.time_since_update == 0
    
    def get_age_ms(self, current_time: float) -> float:
        """Get track age in milliseconds."""
        return (current_time - self.last_update_metadata.capture_timestamp) * 1000
    
    def is_valid_for_display(self) -> bool:
        """Check if track is valid for display."""
        # Display only fresh or coasted tracks with good quality
        return (self.state in ["FRESH", "COAST_1", "COAST_2"] and 
                self.quality_score > 0.3 and
                self.time_since_update <= 3)


# =============================================================================
# ADAS Tracker with Freshness Gating
# =============================================================================
class ADAS_Tracker:
    """
    ADAS-compliant tracker with proper freshness management and publish gating.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Track management parameters
        self.max_age_frames = self.config.get('max_age_frames', 30)  # Internal track retention
        self.min_hits = self.config.get('min_hits', 3)  # Track confirmation threshold
        self.iou_threshold = self.config.get('iou_threshold', 0.3)  # Association threshold
        self.publish_max_age = self.config.get('publish_max_age', 2)  # Max frames stale for publishing
        
        # Track storage
        self.tracks: List[ADAS_Track] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Performance monitoring
        self.track_stats = {
            'created': 0,
            'updated': 0,
            'deleted': 0,
            'published': 0
        }
        
        print(f"ADAS Tracker initialized with publish_max_age={self.publish_max_age}")
    
    def update(self, detections: List, frame_metadata: FrameMetadata) -> List[ADAS_Track]:
        """
        Update tracker with new detections.
        Returns only tracks that should be published.
        """
        self.frame_count += 1
        
        # Step 1: Predict all existing tracks to current frame time
        for track in self.tracks:
            track.predict(frame_metadata)
        
        # Step 2: Associate detections with tracks using IoU
        matches, unmatched_tracks, unmatched_dets = self._associate_detections(detections)
        
        # Step 3: Update matched tracks
        for track_idx, det_idx in matches:
            det = detections[det_idx]
            bbox = det[:4]
            class_name = det[5]
            confidence = det[4]
            
            self.tracks[track_idx].update(bbox, class_name, confidence, frame_metadata)
            self.track_stats['updated'] += 1
        
        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            bbox = det[:4]
            class_name = det[5]
            confidence = det[4]
            
            new_track = ADAS_Track(
                self.next_id, bbox, class_name, confidence, frame_metadata
            )
            self.tracks.append(new_track)
            self.next_id += 1
            self.track_stats['created'] += 1
        
        # Step 5: Remove stale tracks (internal cleanup)
        initial_count = len(self.tracks)
        self.tracks = [
            t for t in self.tracks 
            if t.time_since_update <= self.max_age_frames and t.quality_score > 0.1
        ]
        self.track_stats['deleted'] += initial_count - len(self.tracks)
        
        # Step 6: Return only publishable tracks
        publishable_tracks = [
            t for t in self.tracks 
            if t.should_be_published() and t.hits >= self.min_hits
        ]
        self.track_stats['published'] = len(publishable_tracks)
        
        return publishable_tracks
    
    def _associate_detections(self, detections: List) -> Tuple[List, List, List]:
        """Associate detections with tracks using IoU."""
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det[:4])
        
        # Threshold IoU matrix
        cost_matrix = 1.0 - iou_matrix
        cost_matrix[iou_matrix < self.iou_threshold] = np.inf
        
        # Greedy assignment (simple and effective)
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))
        
        # Sort by best IoU
        valid_pairs = np.where(cost_matrix < np.inf)
        if len(valid_pairs[0]) > 0:
            # Sort by cost (ascending)
            sorted_indices = np.argsort(cost_matrix[valid_pairs])
            
            for idx in sorted_indices:
                i, j = valid_pairs[0][idx], valid_pairs[1][idx]
                if i in unmatched_tracks and j in unmatched_dets:
                    matches.append((i, j))
                    unmatched_tracks.remove(i)
                    unmatched_dets.remove(j)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter_area = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return self.track_stats.copy()


# =============================================================================
# CAN Fusion Module (Enhanced)
# =============================================================================
class CANFusion:
    """
    CAN data fusion with time alignment.
    """
    
    def __init__(self):
        self.ego_speed_mps = 0.0
        self.ego_yaw_rate = 0.0
        self.ego_acceleration = 0.0
        self.brake_status = False
        self.turn_signal = "NONE"
        
        # CAN data history for time alignment
        self.can_history = deque(maxlen=100)  # (timestamp, can_data)
        
    def update_from_can(self, can_data: Dict, timestamp: float):
        """Update with timestamped CAN data."""
        self.can_history.append((timestamp, can_data.copy()))
        
        # Update current state
        self.ego_speed_mps = can_data.get('speed', self.ego_speed_mps)
        self.ego_yaw_rate = can_data.get('yaw_rate', self.ego_yaw_rate)
        self.ego_acceleration = can_data.get('acceleration', self.ego_acceleration)
        self.brake_status = can_data.get('brake', self.brake_status)
        self.turn_signal = can_data.get('turn_signal', self.turn_signal)
    
    def get_aligned_data(self, target_timestamp: float) -> Dict:
        """Get CAN data aligned to target timestamp (simple nearest neighbor)."""
        if not self.can_history:
            return {
                'speed': self.ego_speed_mps,
                'yaw_rate': self.ego_yaw_rate,
                'acceleration': self.ego_acceleration,
                'brake': self.brake_status,
                'turn_signal': self.turn_signal
            }
        
        # Find nearest CAN data
        closest_idx = 0
        min_diff = float('inf')
        
        for i, (ts, _) in enumerate(self.can_history):
            diff = abs(ts - target_timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        _, aligned_data = self.can_history[closest_idx]
        return aligned_data
    
    def compute_safe_distance(self, relative_speed: float, reaction_time: float = 1.5) -> float:
        """Compute safe following distance based on NCAP standards."""
        if self.brake_status:
            braking_deceleration = 8.0  # m/s² (emergency braking)
        else:
            braking_deceleration = 4.0  # m/s² (comfortable braking)
        
        # Safe distance formula according to UNECE R152
        closing_speed = max(relative_speed, 0.0)
        safe_distance = (self.ego_speed_mps * reaction_time) + \
                       ((closing_speed ** 2) / (2 * braking_deceleration))
        
        return max(safe_distance, 2.0)  # Minimum 2 meters


# =============================================================================
# PostProcess Base Class (Enhanced)
# =============================================================================
class PostProcess:
    """
    Base post process class with async logging and proper timing.
    """
    
    def __init__(self, flow):
        self.flow = flow
        self.model = flow.model
        self.debug = None
        self.debug_str = ""
        self.frame_count = 0
        self.last_capture_time = time.monotonic()
        
        # Result logging is opt-in because it creates CSV/image output and adds
        # disk I/O. Set ENABLE_ASYNC_LOGGER=1 to enable it.
        self.async_logger = None
        if _env_flag("ENABLE_ASYNC_LOGGER", default=False):
            self._init_async_logger()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.frame_latencies = deque(maxlen=100)
        
        if flow.debug_config and flow.debug_config.post_proc:
            self.debug = debug.Debug(flow.debug_config, "post")
        
        if self.async_logger:
            print("PostProcess initialized with async logging")
        else:
            print("PostProcess initialized (async logging disabled)")
    
    def _init_async_logger(self):
        """Initialize async logger for non-blocking disk I/O."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (
            Path(self.model.model_path).name if self.model.model_path else "unknown_model"
        )
        
        # Create output directory
        self.output_base_dir = (
            Path.cwd()
            / "Outputs"
            / "Test_results"
            / f"{model_name}_{timestamp}"
        )
        
        self.annotation_dir = self.output_base_dir / "annotation" / "frames"
        
        # Initialize async logger
        self.async_logger = AsyncLogger(self.output_base_dir, max_queue_size=50)
        self.async_logger.start()
        
        print(f"Results will be saved to: {self.output_base_dir}")
    
    def _log_frame_async(self, frame_id: int, image: np.ndarray, 
                        detections: List, metadata: Dict):
        """Log frame data asynchronously."""
        if self.async_logger is None:
            return False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.async_logger.log_frame(
            frame_id=frame_id,
            timestamp=timestamp,
            image=image,
            detections=detections,
            metadata=metadata
        )
    
    def __del__(self):
        """Cleanup - stop async logger."""
        if getattr(self, 'async_logger', None) is not None:
            self.async_logger.stop()
            print(f"Results saved to: {self.output_base_dir}")
    
    def get(flow):
        """
        Create an object of a subclass based on the task type.
        """
        if flow.model.task_type == "classification":
            return PostProcessClassification(flow)
        elif flow.model.task_type == "detection":
            return PostProcessDetection(flow)
        elif flow.model.task_type == "segmentation":
            return PostProcessSegmentation(flow)
        elif flow.model.task_type == "keypoint_detection":
            return PostProcessKeypointDetection(flow)
        elif flow.model.task_type == "lane_detection":
            return PostProcessLaneDetection(flow)


# =============================================================================
# PostProcess Classification (Unchanged)
# =============================================================================
class PostProcessClassification(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        results = np.squeeze(results)
        img = self.overlay_topN_classnames(img, results)

        classification_data = self._get_classification_data(results)
        
        # Log asynchronously
        metadata = {
            'latency_ms': 0.0,
            'frame_id': self.frame_count
        }
        self._log_frame_async(self.frame_count, img, classification_data, metadata)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        self.frame_count += 1
        return img

    def _get_classification_data(self, results):
        """Extract classification data for CSV."""
        N = self.model.topN
        topN_classes = np.argsort(results)[: (-1 * N) - 1 : -1]
        classification_data = []

        for idx in topN_classes:
            class_idx = idx + self.model.label_offset
            confidence = results[idx]

            if class_idx in self.model.dataset_info:
                class_name = self.model.dataset_info[class_idx].name
                if not class_name:
                    class_name = "UNDEFINED"
                if self.model.dataset_info[class_idx].supercategory:
                    class_name = (
                        self.model.dataset_info[class_idx].supercategory + "/" + class_name
                    )
            else:
                class_name = "UNDEFINED"

            classification_data.append({
                'class_name': class_name,
                'confidence': float(confidence)
            })

        return classification_data

    def overlay_topN_classnames(self, frame, results):
        orig_width = frame.shape[1]
        row_size = 40 * orig_width // 1280
        font_size = orig_width / 1280
        N = self.model.topN
        topN_classes = np.argsort(results)[: (-1 * N) - 1 : -1]
        title_text = f"Recognized Classes (Top {N}):"
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size, _ = cv2.getTextSize(title_text, font, font_size, 2)

        bg_top_left = (0, (2 * row_size) - text_size[1] - 5)
        bg_bottom_right = (text_size[0] + 10, (2 * row_size) + 3 + 5)
        font_coord = (5, 2 * row_size)

        cv2.rectangle(frame, bg_top_left, bg_bottom_right, (5, 11, 120), -1)

        cv2.putText(
            frame,
            title_text,
            font_coord,
            font,
            font_size,
            (0, 255, 0),
            2,
        )
        row = 3
        for idx in topN_classes:
            idx = idx + self.model.label_offset
            if idx in self.model.dataset_info:
                class_name = self.model.dataset_info[idx].name
                if not class_name:
                    class_name = "UNDEFINED"
                if self.model.dataset_info[idx].supercategory:
                    class_name = (
                        self.model.dataset_info[idx].supercategory + "/" + class_name
                    )
            else:
                class_name = "UNDEFINED"

            text_size, _ = cv2.getTextSize(class_name, font, font_size, 2)

            bg_top_left = (0, (row_size * row) - text_size[1] - 5)
            bg_bottom_right = (text_size[0] + 10, (row_size * row) + 3 + 5)
            font_coord = (5, row_size * row)

            cv2.rectangle(frame, bg_top_left, bg_bottom_right, (5, 11, 120), -1)
            cv2.putText(
                frame,
                class_name,
                font_coord,
                font,
                font_size,
                (255, 255, 0),
                2,
            )
            row += 1
            if self.debug:
                self.debug_str += class_name + "\n"

        return frame


# =============================================================================
# Enhanced PostProcess Detection with ADAS Compliance
# =============================================================================
class PostProcessDetection(PostProcess):
    """
    ADAS-compliant detection post-processing with:
    - stable track publishing
    - bbox smoothing
    - distance / relative-speed / TTC / FCW computation
    - ADAS-style overlay panels
    """

    def __init__(self, flow):
        super().__init__(flow)

        # Load camera parameters
        cam_params = load_camera_params("camera_calibration_c270.json")
        self.fx = cam_params["fx"]
        self.fy = cam_params["fy"]
        self.cx = cam_params["cx"]
        self.cy = cam_params["cy"]
        self.dist = cam_params["dist"]
        self.K = cam_params["K"]
        self.camera_name = cam_params.get("camera_name", "default_camera")

        print(f"Loaded camera parameters: fx={self.fx:.2f}, fy={self.fy:.2f}")

        # Mount parameters
        self.camera_height_m = 1.5
        self.camera_pitch_deg = 6.0
        self.pitch_rad = float(np.deg2rad(self.camera_pitch_deg))

        # Initialize tracker (kept intact)
        tracker_config = {
            'max_age_frames': 30,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'publish_max_age': 2
        }
        self.tracker = ADAS_Tracker(tracker_config)

        # CAN fusion
        self.can_fusion = CANFusion()

        # Lane parameters
        self.ego_lane_center = None
        self.lane_width_px = None
        self.lane_width_m = 3.6

        # Object dimensions for fallback
        self.object_dimensions = {
            "person": {"width": 0.45, "height": 1.75},
            "pedestrian": {"width": 0.45, "height": 1.75},
            "bicycle": {"width": 0.60, "height": 1.40},
            "motorcycle": {"width": 0.80, "height": 1.40},
            "car": {"width": 1.69, "height": 1.60},
            "truck": {"width": 2.50, "height": 3.00},
            "bus": {"width": 2.60, "height": 3.20},
            "van": {"width": 2.00, "height": 1.90},
        }

        self.alias_map = {
            "motorbike": "motorcycle",
            "bike": "bicycle",
            "cycle": "bicycle",
            "cyclist": "bicycle",
            "rider": "person",
            "pedestrian": "person",
        }

        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.frame_latencies = deque(maxlen=100)

        # Post-process-only track memory (does not change tracker internals)
        self.track_memory = {}
        self.min_stable_hits = 3
        self.max_track_missing_frames = 15
        self.bbox_smooth_alpha = 0.35
        self.distance_smooth_alpha = 0.35
        self.speed_smooth_alpha = 0.25
        self.min_relative_speed_mps = 0.5
        self.fcw_distance_min_m = 3.0
        self.fcw_distance_max_m = 40.0
        self.fcw_ttc_warning_s = 4.0
        self.fcw_ttc_critical_s = 2.0
        self.object_panel_max_rows = 8

        print("ADAS-compliant detection post-process initialized")

    def undistort_points(self, pts):
        """Undistort points using camera calibration."""
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return undistorted.reshape(-1, 2)

    def estimate_distance_groundplane(self, x1: float, y1: float, x2: float, y2: float,
                                    img_h: int, img_w: int) -> float:
        """
        Single-camera metric distance using ground-plane intersection.
        Uses bbox bottom-center (footpoint).
        """
        u = 0.5 * (x1 + x2)
        v = float(y2)

        u = max(0.0, min(float(img_w - 1), float(u)))
        v = max(0.0, min(float(img_h - 1), float(v)))

        try:
            uu, vv = self.undistort_points([[u, v]])[0]
        except Exception:
            uu, vv = u, v

        y = (float(vv) - self.cy) / self.fy
        phi = float(np.arctan(y))
        denom = self.pitch_rad + phi

        if denom <= 1e-4:
            return -1.0

        z_m = float(self.camera_height_m / np.tan(denom))
        if not np.isfinite(z_m):
            return -1.0

        z_m = float(np.clip(z_m, 0.1, 60.0))
        return round(z_m, 2)

    def compute_ttc(self, distance: float, relative_speed: float) -> float:
        """Compute TTC using positive closing speed (m/s)."""
        if distance is None or distance <= 0:
            return float("inf")
        if relative_speed is None or relative_speed < self.min_relative_speed_mps:
            return float("inf")

        ttc = float(distance) / max(float(relative_speed), 1e-3)
        if not np.isfinite(ttc) or ttc > 99.0:
            return float("inf")
        return round(ttc, 2)

    def compute_risk_level(self, ttc: float, distance: float) -> str:
        """Risk levels requested by user."""
        if not np.isfinite(ttc):
            return "SAFE"
        if ttc < 2.0:
            return "CRITICAL"
        if ttc < 4.0:
            return "WARNING"
        if ttc < 6.0:
            return "CAUTION"
        return "SAFE"

    def get_risk_color_from_level(self, risk_level: str):
        color_map = {
            "CRITICAL": (0, 0, 255),
            "WARNING": (0, 165, 255),
            "CAUTION": (0, 255, 255),
            "SAFE": (0, 255, 0),
        }
        return color_map.get(risk_level, (255, 255, 255))

    def is_in_ego_lane(self, bbox: List[int], frame_width: int) -> bool:
        """Check if object is in ego vehicle's lane."""
        if self.ego_lane_center is None:
            self.ego_lane_center = frame_width // 2

        if self.lane_width_px is None:
            self.lane_width_px = int(frame_width * 0.25)

        x1, y1, x2, y2 = map(int, bbox)
        obj_center_x = (x1 + x2) // 2
        return abs(obj_center_x - self.ego_lane_center) <= self.lane_width_px

    def _get_track_slot(self, track_id: int) -> Dict:
        if track_id not in self.track_memory:
            self.track_memory[track_id] = {
                "bbox": None,
                "distance": None,
                "relative_speed": 0.0,
                "distance_history": deque(maxlen=6),
                "last_seen_frame": self.frame_count,
            }
        self.track_memory[track_id]["last_seen_frame"] = self.frame_count
        return self.track_memory[track_id]

    def _cleanup_track_memory(self, live_track_ids: List[int]):
        live_ids = set(live_track_ids)
        stale_ids = []
        for track_id, slot in self.track_memory.items():
            age = self.frame_count - slot.get("last_seen_frame", self.frame_count)
            if track_id not in live_ids and age > self.max_track_missing_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self.track_memory.pop(track_id, None)

    def _smooth_bbox(self, track_id: int, bbox: List[int]) -> List[int]:
        slot = self._get_track_slot(track_id)
        curr = np.array(bbox, dtype=np.float32)
        prev = slot.get("bbox")

        if prev is None:
            slot["bbox"] = curr
        else:
            slot["bbox"] = (
                self.bbox_smooth_alpha * curr +
                (1.0 - self.bbox_smooth_alpha) * prev
            )

        smoothed = slot["bbox"]
        return [int(round(v)) for v in smoothed.tolist()]

    def _update_distance_and_speed(self, track_id: int, distance: float, timestamp: float) -> Tuple[float, float]:
        slot = self._get_track_slot(track_id)

        if distance is None or distance <= 0:
            return slot.get("distance", -1.0) or -1.0, float(slot.get("relative_speed", 0.0))

        prev_distance = slot.get("distance")
        if prev_distance is None:
            filtered_distance = float(distance)
        else:
            filtered_distance = ema_filter(prev_distance, float(distance), alpha=self.distance_smooth_alpha)

        slot["distance"] = float(filtered_distance)
        slot["distance_history"].append((float(timestamp), float(filtered_distance)))

        relative_speed = float(slot.get("relative_speed", 0.0))
        hist = slot["distance_history"]
        if len(hist) >= 2:
            t0, d0 = hist[0]
            t1, d1 = hist[-1]
            dt = max(t1 - t0, 1e-3)

            # Positive value means the object is getting closer.
            raw_closing_speed = max(0.0, (d0 - d1) / dt)
            filtered_speed = ema_filter(relative_speed, raw_closing_speed, alpha=self.speed_smooth_alpha)
            relative_speed = float(max(0.0, filtered_speed if filtered_speed is not None else 0.0))
            slot["relative_speed"] = relative_speed

        return round(slot["distance"], 2), round(relative_speed, 2)

    def _is_track_stable(self, track) -> bool:
        return track.hits >= self.min_stable_hits and track.state in ("FRESH", "COAST_1")

    def _evaluate_fcw(self, track, distance: float, relative_speed: float, ttc: float, in_ego_lane: bool) -> int:
        if not in_ego_lane:
            return 0
        if track.state != "FRESH":
            return 0
        if track.hits < self.min_stable_hits:
            return 0
        if distance <= 0 or distance < self.fcw_distance_min_m or distance > self.fcw_distance_max_m:
            return 0
        if not np.isfinite(ttc) or relative_speed < self.min_relative_speed_mps:
            return 0

        safe_distance = self.can_fusion.compute_safe_distance(relative_speed)
        if ttc < self.fcw_ttc_warning_s and distance <= max(safe_distance, self.fcw_distance_min_m):
            return 1
        return 0

    def _get_inference_time_ms(self, fallback_ms: float) -> float:
        for obj in (getattr(self, "flow", None), getattr(self, "model", None)):
            if obj is None:
                continue
            for attr in ("inference_time_ms", "last_inference_time_ms", "inference_ms", "last_inference_ms"):
                if hasattr(obj, attr):
                    try:
                        val = float(getattr(obj, attr))
                        if np.isfinite(val):
                            return val
                    except Exception:
                        pass
        return float(fallback_ms)

    def __call__(self, img: np.ndarray, results) -> np.ndarray:
        """Main processing function."""
        processing_start = time.monotonic()
        capture_time = processing_start

        frame_metadata = FrameMetadata(
            frame_id=self.frame_count,
            capture_timestamp=capture_time,
            processing_timestamp=processing_start
        )

        if self.ego_lane_center is None:
            self.ego_lane_center = img.shape[1] // 2
            self.lane_width_px = int(img.shape[1] * 0.25)

        img = self.draw_reference_road_overlay(img)

        can_data = {
            'speed': 10.0,
            'yaw_rate': 0.0,
            'acceleration': 0.0,
            'brake': False,
            'turn_signal': 'NONE'
        }
        self.can_fusion.update_from_can(can_data, capture_time)

        for i, r in enumerate(results):
            r = np.squeeze(r)
            if r.ndim == 1:
                r = np.expand_dims(r, 1)
            results[i] = r

        if self.model.shuffle_indices:
            results_reordered = []
            for i in self.model.shuffle_indices:
                results_reordered.append(results[i])
            results = results_reordered

        if results[-1].ndim < 2:
            results = results[:-1]

        bbox = np.concatenate(results, axis=-1)

        if self.model.formatter:
            if self.model.ignore_index is None:
                bbox_copy = copy.deepcopy(bbox)
            else:
                bbox_copy = copy.deepcopy(np.delete(bbox, self.model.ignore_index, 1))
            bbox[..., self.model.formatter["dst_indices"]] = bbox_copy[
                ..., self.model.formatter["src_indices"]
            ]

        if not self.model.normalized_detections:
            bbox[..., (0, 2)] /= self.model.resize[0]
            bbox[..., (1, 3)] /= self.model.resize[1]

        boxes = []
        scores = []
        class_indices = []

        for b in bbox:
            if b[5] > self.model.viz_threshold:
                x1 = int(b[0] * img.shape[1])
                y1 = int(b[1] * img.shape[0])
                x2 = int(b[2] * img.shape[1])
                y2 = int(b[3] * img.shape[0])
                boxes.append([x1, y1, x2, y2])
                scores.append(float(b[5]))
                class_indices.append(int(b[4]))

        if len(boxes) > 0:
            nms_boxes, nms_scores, nms_class_indices, kept_indices = apply_nms_with_indices(
                boxes, scores, class_indices, iou_threshold=0.5
            )
        else:
            nms_boxes, nms_scores, nms_class_indices = [], [], []

        detections_for_tracking = []
        for i, (box, score) in enumerate(zip(nms_boxes, nms_scores)):
            x1, y1, x2, y2 = box

            class_idx = nms_class_indices[i] if i < len(nms_class_indices) else 0
            if isinstance(self.model.label_offset, dict):
                dataset_idx = self.model.label_offset[class_idx]
            else:
                dataset_idx = self.model.label_offset + class_idx

            if dataset_idx in self.model.dataset_info:
                class_name = self.model.dataset_info[dataset_idx].name
                if not class_name:
                    class_name = "UNDEFINED"
                if self.model.dataset_info[dataset_idx].supercategory:
                    class_name = (
                        self.model.dataset_info[dataset_idx].supercategory
                        + "/" + class_name
                    )
            else:
                class_name = "UNDEFINED"

            if class_name.lower() in self.alias_map:
                class_name = self.alias_map[class_name.lower()]

            detections_for_tracking.append([x1, y1, x2, y2, float(score), class_name])

        tracks = self.tracker.update(detections_for_tracking, frame_metadata)
        self._cleanup_track_memory([t.track_id for t in tracks])

        detection_data = []
        objects_for_panel = []
        fcw_warnings = []

        for track in tracks:
            if not self._is_track_stable(track):
                continue

            raw_box = list(map(int, track.bbox))
            x1, y1, x2, y2 = self._smooth_bbox(track.track_id, raw_box)
            track_id = track.track_id

            if x2 <= x1 or y2 <= y1:
                continue

            raw_distance = self.estimate_distance_groundplane(
                x1, y1, x2, y2, img_h=img.shape[0], img_w=img.shape[1]
            )
            distance_m, relative_speed_mps = self._update_distance_and_speed(
                track_id, raw_distance, capture_time
            )

            ttc_s = self.compute_ttc(distance_m, relative_speed_mps)
            risk_level = self.compute_risk_level(ttc_s, distance_m)
            in_ego_lane = self.is_in_ego_lane([x1, y1, x2, y2], img.shape[1])
            fcw_trigger = self._evaluate_fcw(track, distance_m, relative_speed_mps, ttc_s, in_ego_lane)
            risk_color = self.get_risk_color_from_level(risk_level)

            if fcw_trigger:
                fcw_warnings.append({
                    'track_id': track_id,
                    'class_name': track.class_name,
                    'ttc': ttc_s,
                    'distance': distance_m,
                    'risk_level': risk_level,
                })

            img = self.overlay_bounding_box_with_state(
                img,
                [x1, y1, x2, y2],
                track.class_name,
                track_id,
                risk_color,
                distance_m,
                ttc_s,
                risk_level,
                track.confidence,
                relative_speed_mps,
                in_ego_lane,
                track.state,
                track.hits,
                fcw_trigger,
            )

            detection_data.append({
                'track_id': track_id,
                'class_name': track.class_name,
                'distance': float(distance_m) if distance_m > 0 else -1.0,
                'ttc': float(ttc_s) if np.isfinite(ttc_s) else 999.0,
                'risk_level': risk_level,
                'relative_speed': float(relative_speed_mps),
                'fcw_trigger': int(fcw_trigger),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': float(track.confidence),
                'track_state': track.state
            })

            objects_for_panel.append({
                'track_id': track_id,
                'class_name': track.class_name,
                'distance': distance_m,
                'ttc': ttc_s,
                'risk_level': risk_level,
                'relative_speed': relative_speed_mps,
                'color': risk_color,
            })

        frame_metadata.publish_timestamp = time.monotonic()
        frame_metadata.latency_ms = (frame_metadata.publish_timestamp - frame_metadata.capture_timestamp) * 1000

        metadata = {
            'latency_ms': frame_metadata.latency_ms,
            'frame_id': self.frame_count,
            'track_count': len(detection_data),
            'fcw_warnings': len(fcw_warnings)
        }
        self._log_frame_async(self.frame_count, img, detection_data, metadata)

        processing_time_ms = (time.monotonic() - processing_start) * 1000
        inference_time_ms = self._get_inference_time_ms(processing_time_ms)
        self.processing_times.append(processing_time_ms)
        self.frame_latencies.append(frame_metadata.latency_ms)

        objects_for_panel.sort(key=lambda x: (9999 if x['distance'] <= 0 else x['distance']))
        fcw_status = "ACTIVE" if fcw_warnings else "CLEAR"

        img = self.draw_system_info(
            img,
            inference_time_ms=inference_time_ms,
            postproc_time_ms=processing_time_ms,
            track_count=len(detection_data),
            latency_ms=frame_metadata.latency_ms,
            fcw_status=fcw_status,
            objects_for_panel=objects_for_panel,
            fcw_warnings=fcw_warnings,
        )
        img = self.draw_ego_info(img)
        img = self.draw_detected_objects_panel(img, objects_for_panel)
        img = self.draw_fcw_warnings(img, fcw_warnings)

        self.frame_count += 1

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        return img

    def overlay_bounding_box_with_state(self, frame, box, class_name, track_id, color,
                                       distance, ttc, risk_level, confidence,
                                       relative_speed, in_lane, state, hits, fcw_trigger=0):
        """Draw bounding boxes and labels in a reference-demo style."""
        x1, y1, x2, y2 = box
        if state == "STALE":
            return frame

        thickness = 3 if risk_level in ("CRITICAL", "WARNING") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        main_label = class_name.upper()
        sub_label = f"{distance:.1f} m" if distance > 0 else "N/A"
        if np.isfinite(ttc):
            sub_label += f" | TTC {ttc:.1f}s"
        if fcw_trigger:
            sub_label += " | FCW"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_main = 0.72
        scale_sub = 0.45

        (w1, h1), _ = cv2.getTextSize(main_label, font, scale_main, 2)
        (w2, h2), _ = cv2.getTextSize(sub_label, font, scale_sub, 1)
        label_w = max(w1, w2) + 12
        label_h = h1 + h2 + 16
        by1 = max(4, y1 - label_h - 4)
        if by1 <= 4:
            by1 = min(frame.shape[0] - label_h - 4, y2 + 4)
        bx2 = min(frame.shape[1] - 4, x1 + label_w)

        cv2.rectangle(frame, (x1, by1), (bx2, by1 + label_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, by1), (bx2, by1 + label_h), color, 1)
        cv2.putText(frame, main_label, (x1 + 6, by1 + h1 + 2), font, scale_main, color, 2, cv2.LINE_AA)
        cv2.putText(frame, sub_label, (x1 + 6, by1 + h1 + h2 + 8), font, scale_sub, (255, 255, 255), 1, cv2.LINE_AA)

        if not in_lane:
            cv2.putText(frame, "OFF-LANE", (x1, y2 + 18), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def _draw_dotted_line(self, frame, pt1, pt2, color, thickness=2, gap=16):
        x1, y1 = pt1
        x2, y2 = pt2
        dist = max(int(np.hypot(x2 - x1, y2 - y1)), 1)
        steps = max(dist // gap, 1)
        for i in range(steps):
            if i % 2 != 0:
                continue
            t0 = i / steps
            t1 = min((i + 1) / steps, 1.0)
            sx = int(round(x1 + (x2 - x1) * t0))
            sy = int(round(y1 + (y2 - y1) * t0))
            ex = int(round(x1 + (x2 - x1) * t1))
            ey = int(round(y1 + (y2 - y1) * t1))
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

    def draw_reference_road_overlay(self, frame):
        """Draw ego-lane corridor similar to the provided reference frame."""
        h, w = frame.shape[:2]
        cx = int(self.ego_lane_center if self.ego_lane_center is not None else w // 2)
        lane_half_bottom = int(max(w * 0.12, self.lane_width_px * 0.55 if self.lane_width_px else w * 0.12))
        lane_half_top = int(max(w * 0.04, lane_half_bottom * 0.35))
        horizon_y = int(h * 0.58)
        bottom_y = h - 1

        left_bottom = (cx - lane_half_bottom, bottom_y)
        right_bottom = (cx + lane_half_bottom, bottom_y)
        left_top = (cx - lane_half_top, horizon_y)
        right_top = (cx + lane_half_top, horizon_y)

        overlay = frame.copy()
        poly = np.array([left_bottom, left_top, right_top, right_bottom], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (30, 110, 45))
        frame[:] = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0.0)

        self._draw_dotted_line(frame, left_bottom, left_top, (0, 255, 0), thickness=2, gap=18)
        self._draw_dotted_line(frame, right_bottom, right_top, (0, 255, 0), thickness=2, gap=18)

        outer_left_bottom = (max(0, cx - int(lane_half_bottom * 1.9)), bottom_y - 30)
        outer_left_top = (max(0, cx - int(lane_half_top * 2.0)), horizon_y + 10)
        outer_right_bottom = (min(w - 1, cx + int(lane_half_bottom * 1.9)), bottom_y - 30)
        outer_right_top = (min(w - 1, cx + int(lane_half_top * 2.0)), horizon_y + 10)
        self._draw_dotted_line(frame, outer_left_bottom, outer_left_top, (255, 0, 0), thickness=2, gap=16)
        self._draw_dotted_line(frame, outer_right_bottom, outer_right_top, (0, 215, 255), thickness=2, gap=16)

        mid_y = int(h * 0.78)
        arrow_top = (cx, int(h * 0.68))
        arrow_bot = (cx, mid_y)
        cv2.line(frame, arrow_bot, arrow_top, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.line(frame, arrow_top, (cx - 10, arrow_top[1] + 10), (220, 220, 220), 2, cv2.LINE_AA)
        cv2.line(frame, arrow_top, (cx + 10, arrow_top[1] + 10), (220, 220, 220), 2, cv2.LINE_AA)

        return frame

    def draw_system_info(self, frame, inference_time_ms, postproc_time_ms, track_count, latency_ms, fcw_status, objects_for_panel=None, fcw_warnings=None):
        """Top-right panel styled after the reference image."""
        h, w = frame.shape[:2]
        objects_for_panel = objects_for_panel or []
        fcw_warnings = fcw_warnings or []

        if len(self.processing_times) > 0:
            avg_time = np.mean(list(self.processing_times)[-10:])
            fps = 1000.0 / avg_time if avg_time > 0 else 0.0
        else:
            fps = 0.0

        panel_w = min(250, int(w * 0.34))
        panel_h = min(210, int(h * 0.44))
        x1 = w - panel_w - 10
        y1 = 10
        x2 = w - 10
        y2 = y1 + panel_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        closest = objects_for_panel[0] if objects_for_panel else None
        top_lines = []
        if closest and closest.get("distance", -1) > 0:
            top_lines.append((f"Obj {closest['distance']:.2f} m", (0, 0, 255)))
            if np.isfinite(closest.get('ttc', float('inf'))):
                top_lines.append((f"TTC {closest['ttc']:.2f} s", (0, 0, 255)))
        else:
            top_lines.append(("Obj --.-- m", (0, 0, 255)))
            top_lines.append(("TTC --.-- s", (0, 0, 255)))

        for i, (text_line, clr) in enumerate(top_lines):
            cv2.putText(frame, text_line, (x1 + 10, y1 + 18 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, clr, 1, cv2.LINE_AA)

        mini_x1 = x1 + 16
        mini_y1 = y1 + 38
        mini_x2 = x2 - 16
        mini_y2 = mini_y1 + 80
        cv2.rectangle(frame, (mini_x1, mini_y1), (mini_x2, mini_y2), (40, 40, 40), 1)
        cx = (mini_x1 + mini_x2) // 2
        cv2.line(frame, (cx - 28, mini_y2), (cx - 10, mini_y1 + 6), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (cx + 28, mini_y2), (cx + 10, mini_y1 + 6), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (cx, mini_y2), (cx, mini_y1 + 14), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx - 10, mini_y2 - 20), (cx + 10, mini_y2 - 2), (180, 180, 180), 1)

        status_color = (0, 0, 255) if fcw_warnings else (0, 255, 0)
        status_text = "FCWS : Warning" if fcw_warnings else "FCWS : Normal Risk"
        cv2.putText(frame, status_text, (x1 + 10, mini_y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"object-infer : {inference_time_ms/1000.0:.02f} s", (x1 + 10, mini_y2 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"lane-infer   : {max(postproc_time_ms - inference_time_ms, 0.0)/1000.0:.02f} s", (x1 + 10, mini_y2 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS {fps:.2f} | Tracks {track_count}", (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_ego_info(self, frame):
        """Top-left warning/status panel styled after the reference image."""
        h, w = frame.shape[:2]
        panel_w = min(210, int(w * 0.32))
        panel_h = min(260, int(h * 0.58))
        x1, y1 = 10, 10
        x2, y2 = x1 + panel_w, y1 + panel_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        tri = np.array([
            [x1 + panel_w // 2, y1 + 18],
            [x1 + 38, y1 + 110],
            [x2 - 38, y1 + 110],
        ], dtype=np.int32)
        cv2.fillConvexPoly(frame, tri, (0, 215, 255))
        cv2.polylines(frame, [tri], True, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.line(frame, (x1 + panel_w // 2, y1 + 42), (x1 + panel_w // 2, y1 + 82), (0, 0, 0), 7, cv2.LINE_AA)
        cv2.circle(frame, (x1 + panel_w // 2, y1 + 96), 5, (0, 0, 0), -1)

        fps = 0.0
        if len(self.processing_times) > 0:
            avg_time = np.mean(list(self.processing_times)[-10:])
            fps = 1000.0 / avg_time if avg_time > 0 else 0.0

        lines = [
            "LDWS : To Be Determined ...",
            "LKAS : To Be Determined ...",
            f"FPS  : {fps:.2f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x1 + 10, y1 + 142 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)

        return frame

    def draw_detected_objects_panel(self, frame, objects_for_panel):
        """Compact object list near the top center so the UI still exposes tracked targets."""
        if not objects_for_panel:
            return frame

        x1 = max(220, frame.shape[1] // 2 - 120)
        y1 = 14
        rows = min(len(objects_for_panel), 3)
        panel_w = 250
        panel_h = 18 + rows * 18
        cv2.rectangle(frame, (x1, y1), (x1 + panel_w, y1 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x1 + panel_w, y1 + panel_h), (80, 80, 80), 1)
        for i, obj in enumerate(objects_for_panel[:rows]):
            line = f"{obj['class_name'][:10]}  {obj['distance']:.1f}m"
            cv2.putText(frame, line, (x1 + 8, y1 + 14 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, obj['color'], 1, cv2.LINE_AA)
        return frame

    def draw_fcw_warnings(self, frame, fcw_warnings):
        """Bottom footer with timestamp, camera, and FCW summary."""
        h, w = frame.shape[:2]
        timestamp_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        speed_kmh = self.can_fusion.ego_speed_mps * 3.6
        footer_y = h - 10

        left_text = f"{timestamp_str}   {self.camera_name}"
        cv2.putText(frame, left_text, (12, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        if fcw_warnings:
            strongest = sorted(fcw_warnings, key=lambda x: (0 if x['risk_level'] == 'CRITICAL' else 1, x['ttc']))[0]
            right_text = f"FCWS ALERT  |  {strongest['class_name']} {strongest['distance']:.1f}m  TTC {strongest['ttc']:.1f}s"
            right_color = (0, 0, 255)
        else:
            right_text = f"FCWS NORMAL  |  {speed_kmh:.0f} km/h"
            right_color = (255, 255, 255)

        (tw, th), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(frame, right_text, (w - tw - 12, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, right_color, 1, cv2.LINE_AA)
        return frame


# =============================================================================
# PostProcess Lane Detection (UFLDv2)
# =============================================================================
class PostProcessLaneDetection(PostProcess):
    """
    UFLDv2 lane-detection post-processing.

    Decodes the model's loc_row/loc_col/exist_row/exist_col tensors into lane
    points (pred2coords), rejects unstable candidates (filter_lane), fits a
    smooth polyline per lane (_smooth_lane) and overlays them on the frame.
    Ported from lanedetection_gst.py; model-specific parameters (row anchors,
    grid sizes, lane index groups) come from the model's param.yaml
    `postprocess:` section, exposed here as `self.model.<key>`.
    """

    def __init__(self, flow):
        super().__init__(flow)
        self.current_roi = None      # set by InferPipe before each __call__
        self._cached_lane_dict = None  # {lane_id: filtered_pts}, populated each __call__
        self._cached_od_bbox   = []    # raw bbox rows from last _draw_detections

        # ModelConfig (edgeai_dl_inferer) only promotes a fixed, hardcoded set
        # of postprocess/preprocess keys to attributes and knows nothing about
        # lane_detection, so the UFLDv2-specific params aren't on self.model.
        # Read param.yaml directly, the same way ModelConfig itself reads
        # dataset.yaml off self.model.path.
        with open(os.path.join(self.model.path, "param.yaml"), "r") as f:
            params = yaml.safe_load(f)
        pp = params["postprocess"]

        # Tusimple's row_anchor_start/end are raw pixel rows of the dataset's
        # 720-tall frames, so they need normalising. CULane's are already
        # fractions of image height -- utils/common.py builds them as
        # np.linspace(0.42, 1, num_row) -- and dividing those by 720 a second
        # time collapses every lane point onto row 0, where filter_lane's
        # horizon test then discards the lot and nothing is ever drawn.
        # param.yaml declares which form it ships; packages predating the key
        # are Tusimple-style, so that stays the default.
        self.row_anchor = np.linspace(
            pp["row_anchor_start"], pp["row_anchor_end"], pp["num_row"]
        )
        if not pp.get("row_anchor_normalized", False):
            self.row_anchor = self.row_anchor / _TUSIMPLE_ROW_ANCHOR_REF_HEIGHT
        self.col_anchor = np.linspace(0, 1, pp["num_col"])
        self.row_lane_idx = pp["row_lane_idx"]
        self.col_lane_idx = pp["col_lane_idx"]
        self.local_width = pp["local_width"]

        # Optional second model drawn onto the same frame, so lanes and object
        # boxes share one view instead of needing a second flow (which the
        # framework would give its own mosaic window and display sink).
        #
        # Its tensor is prepared here in numpy rather than by tiovxdlpreproc.
        # That costs ~2-3 ms of ARM per frame for a 416x416 uint8 input, but a
        # second flow would cost more: another scaler branch pair, colorconvert,
        # preproc, appsink, appsrc and mosaic pad.
        self.od_model = None
        od_path = getattr(self.model, "overlay_detection_model", None)
        if od_path:
            self._init_overlay_detection(od_path)

    def _init_overlay_detection(self, od_path):
        """Load the secondary detection model that gets drawn over the lanes."""
        try:
            from edgeai_dl_inferer import ModelConfig
        except ImportError as e:
            print(f"[post_process] overlay detection unavailable: {e}")
            return

        try:
            core = int(getattr(self.model, "overlay_detection_core", 2))
            od = ModelConfig(od_path, self.model.enable_tidl, core)
            od.viz_threshold = float(
                getattr(self.model, "overlay_detection_threshold", 0.4)
            )
            od.create_runtime()
        except Exception as e:
            print(f"[post_process] ERROR loading overlay detection model: {e}")
            return

        self.od_model = od
        # crop is (width, height) after ModelConfig reverses the param.yaml list
        self.od_w, self.od_h = int(od.crop[0]), int(od.crop[1])
        self.od_dtype = od.input_tensor_types[0]
        print(
            f"Overlay detection: {od.model_name} on C7x_{core} "
            f"({self.od_w}x{self.od_h}, thr={od.viz_threshold})"
        )

    def _draw_detections(self, img):
        """Run the secondary detector on the current ROI crop and draw its boxes.

        When current_roi is set, YOLOX runs on the same cropped region used for
        lane inference (same FOV as UFLDv2). Bounding boxes are translated back
        to full-frame normalised coordinates before caching and drawing so that
        downstream ROI-expansion logic and display coordinates are always
        expressed in full-frame space.
        """
        h, w = img.shape[0], img.shape[1]

        # Determine crop region: ROI crop when available, full frame otherwise.
        roi = self.current_roi
        ox_norm, oy_norm, crop_w_norm, crop_h_norm = 0.0, 0.0, 1.0, 1.0
        source = img
        if roi is not None:
            cx0 = max(0, int(roi.x_left * w))
            cy0 = max(0, int(roi.y_top * h))
            cx1 = min(w, int((roi.x_left + roi.width) * w))
            cy1 = min(h, int((roi.y_top + roi.height) * h))
            if cx1 > cx0 and cy1 > cy0:
                source = img[cy0:cy1, cx0:cx1]
                ox_norm      = cx0 / w
                oy_norm      = cy0 / h
                crop_w_norm  = (cx1 - cx0) / w
                crop_h_norm  = (cy1 - cy0) / h

        # Plain resize, matching what tiovxdlpreproc does for this model
        # (caps are a bare 416x416 with no videobox/letterbox padding).
        resized = cv2.resize(
            source, (self.od_w, self.od_h), interpolation=cv2.INTER_LINEAR
        )
        tensor = np.ascontiguousarray(
            resized.transpose(2, 0, 1)[None].astype(self.od_dtype)
        )

        bbox_raw = decode_detections(self.od_model.run_time(tensor), self.od_model)
        bbox = []
        for b in bbox_raw:
            translated = list(b)
            translated[0] = ox_norm + b[0] * crop_w_norm  # x1
            translated[1] = oy_norm + b[1] * crop_h_norm  # y1
            translated[2] = ox_norm + b[2] * crop_w_norm  # x2
            translated[3] = oy_norm + b[3] * crop_h_norm  # y2
            bbox.append(translated)

        self._cached_od_bbox = bbox  # full-frame coords; used by _extract_detections

        font = cv2.FONT_HERSHEY_SIMPLEX
        for b in bbox:
            if b[5] <= self.od_model.viz_threshold:
                continue
            class_name, color = resolve_class(self.od_model, int(b[4]))
            x1 = max(0, min(w - 1, int(b[0] * w)))
            y1 = max(0, min(h - 1, int(b[1] * h)))
            x2 = max(0, min(w - 1, int(b[2] * w)))
            y2 = max(0, min(h - 1, int(b[3] * h)))
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = "%s %.0f%%" % (class_name, b[5] * 100)
            (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
            ty = y1 - 4 if y1 > th + 6 else y2 + th + 4
            cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 6, ty + 3), color, -1)
            cv2.putText(img, label, (x1 + 3, ty), font, 0.45,
                        (0, 0, 0), 1, cv2.LINE_8)

        return img

    def get_lane_style(self, lane_id):
        """
        Resolve the colour and label for a lane id, preferring the model's
        dataset.yaml and falling back to the builtin palette when the model
        directory has no dataset.yaml (dataset_info is then empty).
        """
        info = self.model.dataset_info.get(lane_id)
        if info is not None:
            # rgb_color goes straight into cv2, which reads it as B,G,R
            color = tuple(int(c) for c in info.rgb_color)
            return color, (info.name or "lane_%d" % lane_id)

        return LANE_COLORS[lane_id % len(LANE_COLORS)], "lane_%d" % lane_id

    def __call__(self, img, results):
        """
        Post process function for lane detection
        Args:
            img: Input frame
            results: output of inference (loc_row, loc_col, exist_row, exist_col)
        """
        loc_row, loc_col, exist_row, exist_col = (np.squeeze(r) for r in results[:4])

        img_h, img_w = img.shape[0], img.shape[1]

        lanes = pred2coords(
            loc_row, loc_col, exist_row, exist_col,
            self.row_anchor, self.col_anchor,
            row_lane_idx=self.row_lane_idx, col_lane_idx=self.col_lane_idx,
            local_width=self.local_width,
            original_image_width=img_w, original_image_height=img_h,
        )

        lane_dict = {}
        for lane_id, points in lanes:
            lane = filter_lane(points, img_h, img_w)
            lane_dict[lane_id] = lane  # cache all lanes (including short ones) for ROI feedback
            if len(lane) < 2:
                continue

            color, lane_name = self.get_lane_style(lane_id)
            pts = _smooth_lane(lane, img_h)
            # LINE_8 rather than LINE_AA: antialiasing cost 3.1 ms/frame vs
            # 1.0 ms on the AM67A's ARM cores, and it is not discernible on a
            # 4 px line. Matches the line type used elsewhere in this app.
            cv2.polylines(img, [pts], isClosed=False, color=color,
                          thickness=4, lineType=cv2.LINE_8)

            if self.debug:
                self.debug_str += "%s: %s\n" % (lane_name, lane)

        self._cached_lane_dict = lane_dict

        # Boxes go on top of the lane lines
        if self.od_model is not None:
            img = self._draw_detections(img)

        # ROI rectangle on top of everything
        if self.current_roi is not None:
            roi = self.current_roi
            h, w = img.shape[0], img.shape[1]
            x1 = int(roi.x_left * w)
            y1 = int(roi.y_top  * h)
            x2 = int((roi.x_left + roi.width)  * w)
            y2 = int((roi.y_top  + roi.height) * h)
            colours = {0: (0,255,0), 1: (0,255,255), 2: (0,165,255), 3: (0,0,255)}
            colour  = colours.get(roi.roi_level, (255,255,255))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(img,
                f"ROI L{roi.roi_level}  {roi.width*roi.height*100:.0f}%",
                (x1 + 4, max(y1 + 20, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        return img


# =============================================================================
# PostProcess Segmentation (Unchanged)
# =============================================================================
class PostProcessSegmentation(PostProcess):
    def __call__(self, img, results):
        img = self.blend_segmentation_mask(img, results[0])
        
        # Log asynchronously
        metadata = {
            'latency_ms': 0.0,
            'frame_id': self.frame_count
        }
        self._log_frame_async(self.frame_count, img, [], metadata)
        
        return img

    def blend_segmentation_mask(self, frame, results):
        mask = np.squeeze(results)

        if len(mask.shape) > 2:
            mask = mask[0]

        if self.debug:
            self.debug_str += str(mask.flatten()) + "\n"
            self.debug.log(self.debug_str)
            self.debug_str = ""

        org_width = frame.shape[1]
        org_height = frame.shape[0]

        mask_image_rgb = self.gen_segment_mask(mask)
        mask_image_rgb = cv2.resize(
            mask_image_rgb, (org_width, org_height), interpolation=cv2.INTER_LINEAR
        )

        blend_image = cv2.addWeighted(
            mask_image_rgb, 1 - self.model.alpha, frame, self.model.alpha, 0
        )

        return blend_image

    def gen_segment_mask(self, inp):
        r_map = (inp * 10).astype(np.uint8)
        g_map = (inp * 20).astype(np.uint8)
        b_map = (inp * 30).astype(np.uint8)

        return cv2.merge((r_map, g_map, b_map))


# =============================================================================
# PostProcess Keypoint Detection (Unchanged)
# =============================================================================
class PostProcessKeypointDetection(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        output = np.squeeze(results[0])

        scale_x = img.shape[1] / self.model.resize[0]
        scale_y = img.shape[0] / self.model.resize[1]

        det_bboxes, det_scores, det_labels, kpts = (
            np.array(output[:, 0:4]),
            np.array(output[:, 4]),
            np.array(output[:, 5]),
            np.array(output[:, 6:]),
        )

        keypoint_data = []
        for idx in range(len(det_bboxes)):
            det_bbox = det_bboxes[idx]
            kpt = kpts[idx]
            if det_scores[idx] > self.model.viz_threshold:
                det_bbox[..., (0, 2)] *= scale_x
                det_bbox[..., (1, 3)] *= scale_y

                img = cv2.rectangle(
                    img,
                    (int(det_bbox[0]), int(det_bbox[1])),
                    (int(det_bbox[2]), int(det_bbox[3])),
                    (0, 255, 0),
                    2,
                )

                dataset_idx = int(det_labels[idx])
                if isinstance(self.model.label_offset, dict):
                    dataset_idx = self.model.label_offset[dataset_idx]
                else:
                    dataset_idx = self.model.label_offset + dataset_idx

                if dataset_idx in self.model.dataset_info:
                    class_name = self.model.dataset_info[dataset_idx].name
                    if not class_name:
                        class_name = "UNDEFINED"
                    if self.model.dataset_info[dataset_idx].supercategory:
                        class_name = (
                            self.model.dataset_info[dataset_idx].supercategory
                            + "/" + class_name
                        )
                    skeleton = self.model.dataset_info[dataset_idx].skeleton
                    if not skeleton:
                        skeleton = []
                else:
                    class_name = "UNDEFINED"
                    skeleton = []

                cv2.putText(
                    img,
                    class_name,
                    (int(det_bbox[0]), int(det_bbox[1]) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                keypoints = []
                num_kpts = len(kpt) // 3
                for kidx in range(num_kpts):
                    kx, ky, conf = (
                        kpt[3 * kidx],
                        kpt[3 * kidx + 1],
                        kpt[3 * kidx + 2],
                    )
                    kx = int(kx * scale_x)
                    ky = int(ky * scale_y)
                    keypoints.append(f"{kx},{ky},{conf}")

                    if conf > 0.5:
                        cv2.circle(img, (kx, ky), 3, (255, 0, 0), -1)

                for sk in skeleton:
                    pos1 = (
                        kpt[(sk[0] - 1) * 3],
                        kpt[(sk[0] - 1) * 3 + 1],
                    )
                    pos1 = (int(pos1[0] * scale_x), int(pos1[1] * scale_y))

                    pos2 = (
                        kpt[(sk[1] - 1) * 3],
                        kpt[(sk[1] - 1) * 3 + 1],
                    )
                    pos2 = (int(pos2[0] * scale_x), int(pos2[1] * scale_y))

                    conf1 = kpt[(sk[0] - 1) * 3 + 2]
                    conf2 = kpt[(sk[1] - 1) * 3 + 2]
                    if conf1 > 0.5 and conf2 > 0.5:
                        cv2.line(img, pos1, pos2, (255, 0, 0), 1)

                keypoint_data.append({
                    'class_name': class_name,
                    'confidence': float(det_scores[idx]),
                    'keypoints': ";".join(keypoints)
                })

        # Log asynchronously
        metadata = {
            'latency_ms': 0.0,
            'frame_id': self.frame_count
        }
        self._log_frame_async(self.frame_count, img, keypoint_data, metadata)

        return img
