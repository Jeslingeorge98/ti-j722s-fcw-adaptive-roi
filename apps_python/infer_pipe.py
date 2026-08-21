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
#    its contributors may be used to endorse or promote products derived
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

import numpy as np
from time import time
import threading
import queue
import utils
import debug
import gst_wrapper
from post_process import PostProcess
from roi.dynamic_roi import ROIParameters, LaneInfo, DetectedObject, ObjectCategory

class InferPipe:
    """
    Class to abstract the threading of multiple inference pipelines
    """

    def __init__(self, sub_flow, gst_pipe, roi_generator=None, can_reader=None,
                 shared_roi_state=None, shared_feedback=None):
        """
        Constructor to create an InferPipe object.
        Args:
            sub_flow: sub_flow configuration
            gst_pipe: gstreamer pipe object
            roi_generator: ROIGenerator instance — only the ROI master pipe gets
                one. Followers receive None and read the current ROI from
                shared_roi_state instead of stepping the generator themselves.
            can_reader: CANSignalReader — only the ROI master pipe gets one.
            shared_roi_state: SharedROIState — the single ROI value published by
                the master and read by all followers. None when ROI is disabled.
            shared_feedback: SharedFeedbackState — cross-subflow feedback so the
                master's ROIGenerator.step() can see lane info (from itself)
                and object detections (from the follower).
        """
        self.sub_flow = sub_flow
        self.gst_pipe = gst_pipe
        self.roi_generator = roi_generator
        self.can_reader    = can_reader
        self.shared_roi_state = shared_roi_state
        self.shared_feedback  = shared_feedback
        # A pipe is the "ROI master" if it owns the generator + CAN reader.
        # Every pipe still applies the current ROI to its own scaler pad.
        self._is_roi_master = (roi_generator is not None and can_reader is not None)
        self._hw_roi_direct = (
            gst_wrapper.HARDWARE_ROI and gst_wrapper.ROI_RESIZE_MODE == "direct"
        )
        self.fallback_roi  = ROIParameters(x_left=0.0, y_top=0.0, width=1.0, height=1.0)
        self._current_roi  = self.fallback_roi
        self._roi_lock     = threading.Lock()
        # Sequence of the last ROI this follower consumed from shared state.
        # Used by wait_next() so followers block until the master publishes a
        # fresh ROI for the next frame (rough per-frame synchronisation).
        self._last_roi_seq = 0
        self._roi_lag_frames = 0  # diagnostic: how many frames follower missed
        self._latest_lane_info = LaneInfo(center_norm=None, width_norm=None, confidence=0.0)
        self._latest_objects   = []
        self._feedback_lock    = threading.Lock()
        self.gst_pre_inp = gst_pipe.get_src(sub_flow.gst_pre_src_name, sub_flow.flow.id)
        self.gst_sen_inp = gst_pipe.get_src(sub_flow.gst_sen_src_name, sub_flow.flow.id)
        self.run_time = sub_flow.model.run_time
        self.post_proc = PostProcess.get(sub_flow)

        self.gst_post_out = gst_pipe.get_sink(
            sub_flow.gst_post_sink_name,
            sub_flow.sensor_width,
            sub_flow.sensor_height,
            sub_flow.input.fps,
        )
        self.param = sub_flow.model
        self.pre_proc_debug = None
        self.infer_debug = None

        if sub_flow.debug_config:
            if sub_flow.debug_config.pre_proc:
                self.pre_proc_debug = debug.Debug(sub_flow.debug_config, "pre")
            if sub_flow.debug_config.inference:
                self.infer_debug = debug.Debug(sub_flow.debug_config, "infer")

        self._roi_latency_ms = []
        self.pipeline_thread = threading.Thread(target=self.pipeline)
        # Post-processing runs in its own thread so the ARM work for frame N-1
        # overlaps the C7x inference for frame N. Depth 2 bounds the added
        # latency and gives stage 1 backpressure when stage 2 falls behind.
        self.result_queue = queue.Queue(maxsize=2)
        self.post_thread = threading.Thread(target=self.post_pipeline)
        self._post_frame_idx = 0
        self.stop_thread = False

    def start(self):
        """
        Start the pipeline
        """
        self.post_thread.start()
        self.pipeline_thread.start()

    def stop(self):
        """
        Stop the pipeline
        """
        self.stop_thread = True
        if self._roi_latency_ms:
            lats = self._roi_latency_ms
            print(
                f"[ROI latency] n={len(lats)} frames | "
                f"mean={sum(lats)/len(lats):.3f} ms | "
                f"max={max(lats):.3f} ms | "
                f"p99={sorted(lats)[int(len(lats)*0.99)]:.3f} ms"
            )

    def pipeline(self):
        """
        Callback function for pipeline thread.

        Stage 1 of two: capture the input tensor and run inference. Everything
        after inference happens in post_pipeline() on a second thread, so the
        per-frame cost is max(stage1, stage2) rather than their sum.
        """
        try:
            while self.stop_thread == False:
                # ROI: master computes once and publishes; followers just read.
                # Doing it any other way would step() twice per frame and
                # advance the mock CAN reader twice.
                roi = None
                if self._is_roi_master:
                    can_sig = self.can_reader.get_latest()
                    # Prefer cross-subflow feedback (the object subflow writes
                    # its detections there). Fall back to self-cached feedback
                    # when no shared state is wired.
                    if self.shared_feedback is not None:
                        lane_info = self.shared_feedback.get_lane_info()
                        objects   = self.shared_feedback.get_objects()
                    else:
                        lane_info = self._get_latest_lane_info()
                        objects   = self._get_latest_objects()
                    _t0 = time()
                    roi = self.roi_generator.step(
                        lane_info, can_sig, self.fallback_roi,
                        objects=objects,
                    )
                    _roi_ms = (time() - _t0) * 1000.0
                    self._roi_latency_ms.append(_roi_ms)
                    if self.shared_roi_state is not None:
                        self._last_roi_seq = self.shared_roi_state.set(roi)
                    if self.sub_flow.debug_config:
                        print(
                            f"ROI L{roi.roi_level} | "
                            f"area={roi.width * roi.height:.4f} | "
                            f"roi_ms={_roi_ms:.3f} | "
                            f"warmed={roi.is_warmed_up} | "
                            f"implausible_spd={roi.speed_was_implausible}"
                        )
                elif self.shared_roi_state is not None:
                    # Follower: block until the master publishes a ROI newer
                    # than the one we consumed last iteration. Timeout keeps
                    # this responsive on shutdown; if it fires we just reuse
                    # the previous ROI (one-frame lag) and log the miss.
                    roi, seq = self.shared_roi_state.wait_next(
                        self._last_roi_seq, timeout=0.2
                    )
                    if seq == self._last_roi_seq:
                        self._roi_lag_frames += 1
                    self._last_roi_seq = seq

                if roi is not None:
                    with self._roi_lock:
                        self._current_roi = roi
                    # Apply the ROI to this branch's scaler src pad *before*
                    # pulling the tensor — the very next appsink buffer will
                    # then be produced from the cropped region.
                    if self._hw_roi_direct:
                        self.gst_pipe.set_hw_roi(self.sub_flow, roi)

                input_img = self.gst_pipe.pull_tensor(
                    self.gst_pre_inp,
                    self.sub_flow.input.loop,
                    self.sub_flow.model.crop[0],
                    self.sub_flow.model.crop[1],
                    self.sub_flow.model.data_layout,
                    self.sub_flow.model.input_tensor_types[0],
                )
                if input_img is None:
                    break

                if self.pre_proc_debug:
                    self.pre_proc_debug.log(str(input_img.flatten()))

                # Inference
                start = time()
                result = self.run_time(input_img)
                end = time()
                self.sub_flow.report.report_proctime("dl-inference", (end - start))

                if self.infer_debug:
                    self.infer_debug.log(str(result))

                # Copy before handing off: this thread starts the next inference
                # immediately and must not overwrite tensors stage 2 is reading.
                # Ship the ROI alongside so stage 2 can remap ROI-local coords
                # (bboxes, lane points) back to full frame.
                envelope = ([r.copy() for r in result], roi)
                if not self._enqueue(envelope):
                    break
        finally:
            # Sentinel tells stage 2 no more frames are coming. Stage 2 owns the
            # EOS so queued frames are not dropped on shutdown.
            try:
                self.result_queue.put(None, timeout=0.5)
            except queue.Full:
                pass

    def _enqueue(self, item):
        """
        Hand an item to the post-process thread, staying responsive to stop().
        Returns False if the pipeline was asked to stop while waiting.
        """
        while self.stop_thread == False:
            try:
                self.result_queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def post_pipeline(self):
        """
        Stage 2 of two: pull the display frame, post-process it and push it out.
        Runs concurrently with stage 1 so ARM post-processing overlaps inference.
        """
        try:
            while True:
                try:
                    envelope = self.result_queue.get(timeout=0.2)
                except queue.Empty:
                    if self.stop_thread:
                        break
                    continue

                if envelope is None:
                    break

                result, roi = envelope

                frame, src_pts, src_dur = self.gst_pipe.pull_frame(
                    self.gst_sen_inp, self.sub_flow.input.loop
                )
                if frame is None:
                    break

                # Pass current ROI to post_proc so it can (a) draw the
                # rectangle on the full-frame sensor image and (b) unmap
                # ROI-local model outputs back to full-frame coords when
                # running in hardware_roi direct mode.
                if roi is not None:
                    self.post_proc.current_roi = roi
                    self.post_proc.hw_roi_direct = self._hw_roi_direct

                out_frame = self.post_proc(frame, result)

                # Feed previous-frame lane/detection results back for next ROI
                img_h, img_w = frame.shape[0], frame.shape[1]
                lane_info = _extract_lane_info(result, self.post_proc, img_h, img_w)
                # Remap ROI-local detections back to full-frame normalised
                # coords BEFORE feeding them into the ROI generator — the
                # generator's corridor / expansion math assumes full-frame
                # coordinates.
                objects   = _extract_detections(
                    result, self.post_proc,
                    roi=roi if self._hw_roi_direct else None,
                )
                with self._feedback_lock:
                    self._latest_lane_info = lane_info
                    self._latest_objects   = objects
                # Publish to the cross-subflow feedback so the ROI master
                # (lane subflow) sees the object subflow's detections next
                # iteration. Each subflow writes what its own model produces;
                # None-writes are ignored inside SharedFeedbackState.
                if self.shared_feedback is not None:
                    self.shared_feedback.set_lane_info(lane_info)
                    self.shared_feedback.set_objects(objects)

                # Prefer the source buffer's PTS so the mkv timeline matches
                # the source video's timeline (dropped source frames just
                # leave PTS gaps instead of stretching wall-clock). Fall back
                # to frame_id×1/fps when the source didn't stamp PTS.
                if src_pts is not None:
                    self.gst_pipe.push_frame(
                        out_frame, self.gst_post_out,
                        pts=src_pts, duration=src_dur,
                    )
                else:
                    self.gst_pipe.push_frame(
                        out_frame, self.gst_post_out,
                        frame_id=self._post_frame_idx,
                        fps=self.sub_flow.input.fps,
                    )
                self._post_frame_idx += 1
                # Increment frame count
                self.sub_flow.report.report_frame()
        finally:
            self.stop_thread = True
            self.gst_pipe.send_eos(self.gst_post_out)

    def _get_latest_lane_info(self):
        with self._feedback_lock:
            return self._latest_lane_info

    def _get_latest_objects(self):
        with self._feedback_lock:
            return list(self._latest_objects)


# =============================================================================
# Module-level extractors (Step 5)
# =============================================================================

def _extract_lane_info(result, post_proc, img_h, img_w):
    """
    Build LaneInfo for the next frame's ROI from the lane dict cached by
    PostProcessLaneDetection.__call__() during the draw pass. Falls back to a
    full pred2coords decode on the first frame (before the cache is populated).
    Returns None if post_proc is not a lane detector or decoding fails.
    """
    from post_process import PostProcessLaneDetection, pred2coords, filter_lane
    if not isinstance(post_proc, PostProcessLaneDetection):
        return None
    try:
        lane_dict = getattr(post_proc, '_cached_lane_dict', None)
        if lane_dict is None:
            # First frame: cache not yet populated, do a full decode.
            loc_row, loc_col, exist_row, exist_col = (np.squeeze(r) for r in result[:4])
            lanes = pred2coords(
                loc_row, loc_col, exist_row, exist_col,
                post_proc.row_anchor, post_proc.col_anchor,
                row_lane_idx=post_proc.row_lane_idx,
                col_lane_idx=post_proc.col_lane_idx,
                local_width=post_proc.local_width,
                original_image_width=img_w,
                original_image_height=img_h,
            )
            lane_dict = {lid: filter_lane(pts, img_h, img_w) for lid, pts in lanes}

        ego_left_id  = post_proc.row_lane_idx[0] if len(post_proc.row_lane_idx) > 0 else 1
        ego_right_id = post_proc.row_lane_idx[1] if len(post_proc.row_lane_idx) > 1 else 2
        left_pts  = lane_dict.get(ego_left_id,  [])
        right_pts = lane_dict.get(ego_right_id, [])

        if not left_pts and not right_pts:
            return LaneInfo(center_norm=None, width_norm=None, confidence=0.0)

        n_row = max(len(post_proc.row_anchor), 1)
        left_conf  = len(left_pts)  / n_row if left_pts  else 0.0
        right_conf = len(right_pts) / n_row if right_pts else 0.0
        confidence = (left_conf + right_conf) / 2.0

        def x_at_y(pts, target_y):
            arr = sorted(pts, key=lambda p: p[1])
            for i in range(len(arr) - 1):
                y0, y1 = arr[i][1], arr[i + 1][1]
                if y0 <= target_y <= y1:
                    t = (target_y - y0) / max(y1 - y0, 1)
                    return arr[i][0] + t * (arr[i + 1][0] - arr[i][0])
            return arr[-1][0] if target_y >= arr[-1][1] else arr[0][0]

        ref_y  = int(img_h * 0.80)
        left_x = x_at_y(left_pts,  ref_y) if left_pts  else None
        right_x = x_at_y(right_pts, ref_y) if right_pts else None

        if left_x is not None and right_x is not None:
            center_norm = ((left_x + right_x) / 2.0) / img_w
            width_norm  = abs(right_x - left_x) / img_w
        else:
            center_norm = None
            width_norm  = None

        # Curvature: fit 2nd-order polynomial to lane midline in normalised coords.
        # c2_confidence is capped at 0.4 so CAN curvature takes precedence until
        # the polynomial estimate is validated against calibrated ground truth.
        c2_curvature  = None
        c2_confidence = 0.0
        if left_pts and right_pts and confidence > 0.3:
            mid_pts = []
            for lx, ly in sorted(left_pts, key=lambda p: p[1]):
                rx = x_at_y(right_pts, ly)
                if rx is not None:
                    mid_pts.append(((lx + rx) / 2.0 / img_w, ly / img_h))
            if len(mid_pts) >= 5:
                mid = np.array(mid_pts, dtype=np.float64)
                coeffs = np.polyfit(mid[:, 1], mid[:, 0], 2)
                c2_curvature  = float(coeffs[0] * 0.01)
                c2_confidence = min(confidence, 0.4)

        return LaneInfo(
            center_norm=center_norm,
            width_norm=width_norm,
            confidence=confidence,
            c2_curvature=c2_curvature,
            c2_confidence=c2_confidence,
        )
    except Exception:
        return None


# COCO class-name → ObjectCategory mapping (lowercase, last segment after '/')
_VEHICLE_NAMES    = {'car', 'truck', 'bus', 'motorcycle', 'motorbike',
                     'bicycle', 'train', 'van'}
_SIGNAL_NAMES     = {'traffic light'}
_SIGN_ROAD_NAMES  = {'stop sign'}


def _extract_detections(result, post_proc, roi=None):
    """
    Build List[DetectedObject] for ROI expansion, in full-frame coords.

    For a primary-detector pipeline (PostProcessDetection): decodes the main
    model's output tensors. When roi is given (hardware ROI direct mode),
    the decoded bboxes are ROI-local; they get remapped to full-frame
    normalised coords before being wrapped in DetectedObject.
    For a lane-detection pipeline (PostProcessLaneDetection): reads the YOLOX
    overlay bbox results already decoded by _draw_detections() — those are
    already in full-frame coords, so no remap is needed here.
    Returns [] if no usable detection results are available.
    """
    from post_process import PostProcessDetection, PostProcessLaneDetection, \
        decode_detections, resolve_class

    remap = False
    if isinstance(post_proc, PostProcessDetection):
        try:
            bbox  = decode_detections(result, post_proc.model)
            model = post_proc.model
            # decode_detections returns ROI-local normalised coords whenever
            # the DL branch was cropped by the hardware scaler.
            remap = roi is not None
        except Exception:
            return []
    elif isinstance(post_proc, PostProcessLaneDetection) and post_proc.od_model is not None:
        bbox  = getattr(post_proc, '_cached_od_bbox', [])
        model = post_proc.od_model
    else:
        return []

    try:
        objects = []
        for b in bbox:
            score = float(b[5])
            if score < model.viz_threshold:
                continue
            class_name, _ = resolve_class(model, int(b[4]))
            name = class_name.split('/')[-1].lower().strip()
            if name in _VEHICLE_NAMES:
                category = ObjectCategory.VEHICLE
            elif name in _SIGNAL_NAMES:
                category = ObjectCategory.SIGNAL
            elif name in _SIGN_ROAD_NAMES:
                category = ObjectCategory.SIGN_ROADSIDE
            else:
                continue
            x1, y1, x2, y2 = (float(b[i]) for i in range(4))
            if remap:
                x1 = roi.x_left + x1 * roi.width
                y1 = roi.y_top  + y1 * roi.height
                x2 = roi.x_left + x2 * roi.width
                y2 = roi.y_top  + y2 * roi.height
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            objects.append(DetectedObject(
                category=category,
                bbox=(x1, y1, x2, y2),
                confidence=score,
            ))
        return objects
    except Exception:
        return []
