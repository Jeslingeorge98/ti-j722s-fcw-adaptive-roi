#!/usr/bin/env python3
"""
FCW with Adaptive ROI — Runtime Pipeline Diagram
Run:  python3 tools/gen_pipeline_pdf.py
Out:  tools/pipeline_diagram.pdf
"""
import os
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas

os.makedirs("tools", exist_ok=True)

PW, PH = landscape(A3)   # 1190.55 × 841.89 pt

# ── palette ───────────────────────────────────────────────────────────────────
NAVY  = colors.HexColor("#1c2e4a")
WHT   = colors.white
DARK  = colors.HexColor("#1a2329")
MED   = colors.HexColor("#5d6d7e")
FAINT = colors.HexColor("#95a5a6")

VPU_F = colors.HexColor("#fde0dc"); VPU_S = colors.HexColor("#a93226")
MSC_F = colors.HexColor("#fef0e0"); MSC_S = colors.HexColor("#ca6f1e")
C7X_F = colors.HexColor("#d4efdf"); C7X_S = colors.HexColor("#1e8449")
ARM_F = colors.HexColor("#d6eaf8"); ARM_S = colors.HexColor("#1a5276")
PY_F  = colors.HexColor("#e8daef"); PY_S  = colors.HexColor("#6c3483")
ROI_F = colors.HexColor("#fef9e7"); ROI_S = colors.HexColor("#9a7d0a")
SEC_F = colors.HexColor("#f4f6f7"); SEC_S = colors.HexColor("#abb2b9")
BYP_F = colors.HexColor("#fdfefe"); BYP_S = colors.HexColor("#bdc3c7")

GANTT_LD  = colors.HexColor("#a9dfbf")
GANTT_OD  = colors.HexColor("#f9e79f")
GANTT_ARM = colors.HexColor("#d6eaf8")
GANTT_BG  = colors.HexColor("#f2f3f4")


# ── drawing primitives ────────────────────────────────────────────────────────
def rbox(cv, x, y, w, h, fill, stroke, r=5, lw=1.0, dash=None):
    cv.saveState()
    cv.setFillColor(fill)
    cv.setStrokeColor(stroke)
    cv.setLineWidth(lw)
    if dash:
        cv.setDash(dash[0], dash[1])
    cv.roundRect(x, y, w, h, r, stroke=1, fill=1)
    cv.restoreState()


def txt(cv, x, y, w, h, rows, sizes, bolds=None, fcolors=None):
    """Centre-draw multi-line label inside box (x,y,w,h)."""
    n = len(rows)
    total = sum(s + 1.5 for s in sizes)
    cy = y + (h + total) / 2
    for i, (row, sz) in enumerate(zip(rows, sizes)):
        cy -= sz
        bold = bolds[i] if bolds else (i == 0)
        cv.setFont("Helvetica-Bold" if bold else "Helvetica", sz)
        cv.setFillColor(fcolors[i] if fcolors else (DARK if i == 0 else MED))
        cv.drawCentredString(x + w / 2, cy, row)
        cy -= 1.5


def harr(cv, x1, x2, y, col=DARK, lbl=None, dash=None):
    """Horizontal arrow from x1 to x2 at height y."""
    cv.saveState()
    cv.setStrokeColor(col)
    cv.setLineWidth(0.9)
    if dash:
        cv.setDash(dash[0], dash[1])
    cv.line(x1, y, x2 - 6, y)
    cv.setDash()
    cv.setFillColor(col)
    cv.setStrokeColor(col)
    p = cv.beginPath()
    p.moveTo(x2, y)
    p.lineTo(x2 - 7, y + 3.5)
    p.lineTo(x2 - 7, y - 3.5)
    p.close()
    cv.drawPath(p, fill=1, stroke=0)
    if lbl:
        cv.setFont("Helvetica-Oblique", 5.5)
        cv.setFillColor(MED)
        cv.drawCentredString((x1 + x2) / 2, y + 4, lbl)
    cv.restoreState()


def varr(cv, x, y_from, y_to, col=DARK, lbl=None, lbl_side="right"):
    """Vertical arrow."""
    cv.saveState()
    cv.setStrokeColor(col)
    cv.setLineWidth(0.9)
    tip = y_to
    going_up = y_to > y_from
    shaft_end = tip + (-7 if going_up else 7)
    cv.line(x, y_from, x, shaft_end)
    cv.setFillColor(col)
    p = cv.beginPath()
    p.moveTo(x, tip)
    p.lineTo(x - 3.5, tip + (-7 if going_up else 7))
    p.lineTo(x + 3.5, tip + (-7 if going_up else 7))
    p.close()
    cv.drawPath(p, fill=1, stroke=0)
    if lbl:
        cv.setFont("Helvetica-Oblique", 5.5)
        cv.setFillColor(MED)
        mid = (y_from + y_to) / 2
        if lbl_side == "right":
            cv.drawString(x + 4, mid, lbl)
        else:
            cv.drawRightString(x - 4, mid, lbl)
    cv.restoreState()


def section(cv, x, y, w, h, title):
    rbox(cv, x, y, w, h, SEC_F, SEC_S, r=8, lw=1.3)
    cv.setFont("Helvetica-Bold", 7.5)
    cv.setFillColor(MED)
    cv.drawString(x + 10, y + h - 14, title)


def lbox(cv, x, y, lw_box, lh, fill, stroke, title, sub, tsz=7.5, ssz=5.5):
    """Convenience: draw box + 2-line label."""
    rbox(cv, x, y, lw_box, lh, fill, stroke)
    txt(cv, x, y, lw_box, lh,
        [title, sub], [tsz, ssz],
        bolds=[True, False])


# ── page layout ───────────────────────────────────────────────────────────────
M  = 16          # outer margin
SW = PW - 2 * M  # section width ≈ 1158 pt

# Section Y-ranges (bottom to top):
LY, LH = 16, 40           # legend
DY, DH = 61, 210          # DSP timing + metrics
OY, OH = 276, 56          # output pipeline
BY, BH = 337, 290         # python bridge
SY, SH = 632, 156         # source pipeline
TY, TH = 793, 49          # title bar  (793+49=842 ≈ PH)

cv = Canvas("tools/pipeline_diagram.pdf", pagesize=landscape(A3))
cv.setTitle("FCW with Adaptive ROI — Runtime Pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# TITLE BAR
# ══════════════════════════════════════════════════════════════════════════════
cv.setFillColor(NAVY)
cv.rect(0, TY, PW, TH, fill=1, stroke=0)
cv.setFillColor(WHT)
cv.setFont("Helvetica-Bold", 13.5)
cv.drawCentredString(PW / 2, TY + 29,
    "FCW with Adaptive ROI — Runtime Pipeline  "
    "(TI J722S EVM · 172.16.76.106 · 2026-08-20)")
cv.setFont("Helvetica", 7.5)
cv.setFillColor(colors.HexColor("#aed6f1"))
cv.drawCentredString(PW / 2, TY + 13,
    "1 GStreamer src pipe + Python bridge (2 threads) + 1 GStreamer sink pipe  ·  "
    "UFLDv2 640×256 on C7x-2  ·  YOLOX-nano-lite 416×416 on C7x-1  ·  "
    "1918×1136@30fps → 1280×720 mkv  ·  colour = executing hardware")

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
section(cv, M, SY, SW, SH,
        "SOURCE PIPELINE (GStreamer) — decode + split + pre-proc  "
        "[1918×1136@30fps mp4 in, NV12 dmabuf internal]")

EH  = 44    # element height
GAP = 10    # horizontal gap between elements

# Vertical centres of the two branches inside source section
# Upper branch (pad0 → sen_0): full 1280×720 RGB
# Lower branch (pad1 → pre_0): 640×256 pre-processed tensor
BR1_Y = SY + SH - 20 - EH   # upper branch box top
BR2_Y = SY + 20              # lower branch box top
# Decode chain runs at mid height between the two branches
DC_Y  = SY + (SH - EH) / 2 - 2   # decode chain box top

sx = M + 14   # start x

# ── decode chain ──────────────────────────────────────────────────────────────
W_fs = 82
rbox(cv, sx, DC_Y, W_fs, EH, ARM_F, ARM_S)
txt(cv, sx, DC_Y, W_fs, EH,
    ["filesrc", "indian_road_curve", "_30fps.mp4"], [7.5, 5.5, 5])
x1 = sx + W_fs

W_qd = 58
harr(cv, x1, x1 + GAP + W_qd, DC_Y + EH / 2)
rbox(cv, x1 + GAP, DC_Y, W_qd, EH, ARM_F, ARM_S)
txt(cv, x1 + GAP, DC_Y, W_qd, EH, ["qtdemux", "H.264 ES"], [7.5, 5.5])
x1 = x1 + GAP + W_qd

W_vd = 72
harr(cv, x1, x1 + GAP + W_vd, DC_Y + EH / 2)
rbox(cv, x1 + GAP, DC_Y, W_vd, EH, VPU_F, VPU_S)
txt(cv, x1 + GAP, DC_Y, W_vd, EH,
    ["v4l2h264dec", "Wave5 VPU → NV12", "dmabuf"], [7.5, 5.5, 5])
x1 = x1 + GAP + W_vd

W_cc0 = 70
harr(cv, x1, x1 + GAP + W_cc0, DC_Y + EH / 2)
rbox(cv, x1 + GAP, DC_Y, W_cc0, EH, C7X_F, C7X_S)
txt(cv, x1 + GAP, DC_Y, W_cc0, EH,
    ["tiovxdl", "colorconvert", "NV12→NV12"], [7.5, 5.5, 5])
x1 = x1 + GAP + W_cc0

# ── tiovxmultiscaler (tall box spanning both branches) ────────────────────────
W_msc = 78
harr(cv, x1, x1 + GAP + W_msc, DC_Y + EH / 2)
MSC_X = x1 + GAP
MSC_BOT = SY + 16
MSC_TOP = SY + SH - 16
MSC_H   = MSC_TOP - MSC_BOT
rbox(cv, MSC_X, MSC_BOT, W_msc, MSC_H, MSC_F, MSC_S, r=6, lw=1.2)
txt(cv, MSC_X, MSC_BOT, W_msc, MSC_H,
    ["tiovxmulti", "scaler", "VPAC MSC1", "1 sink / 2 src"],
    [8, 8, 6, 5.5])
x_post_msc = MSC_X + W_msc   # x where branches start

# ── upper branch: pad0 → tiovxdlcolorconvert → appsink sen_0 ─────────────────
cv.setFont("Helvetica-Oblique", 5.5)
cv.setFillColor(MSC_S)
cv.drawString(x_post_msc + 3, BR1_Y + EH + 3, "pad0: 1280×720")

W_cc1 = 70
harr(cv, x_post_msc, x_post_msc + GAP + W_cc1, BR1_Y + EH / 2)
rbox(cv, x_post_msc + GAP, BR1_Y, W_cc1, EH, C7X_F, C7X_S)
txt(cv, x_post_msc + GAP, BR1_Y, W_cc1, EH,
    ["tiovxdl", "colorconvert", "NV12→RGB"], [7.5, 5.5, 5])
x2 = x_post_msc + GAP + W_cc1

W_as0 = 86
harr(cv, x2, x2 + GAP + W_as0, BR1_Y + EH / 2)
rbox(cv, x2 + GAP, BR1_Y, W_as0, EH, ARM_F, ARM_S)
txt(cv, x2 + GAP, BR1_Y, W_as0, EH,
    ["appsink  sen_0", "1280×720 RGB", "drop=true  max-buf=2"], [7.5, 5.5, 5])
SEN0_CX = x2 + GAP + W_as0 / 2   # centre x, for arrow down to python bridge

# GST thread note (upper right of source section)
cv.setFont("Helvetica-Oblique", 5.5)
cv.setFillColor(FAINT)
cv.drawRightString(M + SW - 8, BR1_Y + EH + 4,
    "GStreamer streaming threads: demux · decoder output · one thread per queue/branch")
cv.drawRightString(M + SW - 8, BR1_Y + EH - 3,
    "pull_sample  (tensor + frame, blocking, from stage-1 and stage-2 threads)")

# ── lower branch: pad1 → colorconvert → videobox → tiovxdlpreproc → appsink pre_0 ──
cv.setFont("Helvetica-Oblique", 5.5)
cv.setFillColor(MSC_S)
cv.drawString(x_post_msc + 3, BR2_Y - 9, "pad1: 640×426")

W_cc2 = 70
harr(cv, x_post_msc, x_post_msc + GAP + W_cc2, BR2_Y + EH / 2)
rbox(cv, x_post_msc + GAP, BR2_Y, W_cc2, EH, C7X_F, C7X_S)
txt(cv, x_post_msc + GAP, BR2_Y, W_cc2, EH,
    ["tiovxdl", "colorconvert", "NV12→RGB 640×426"], [7.5, 5.5, 5])
x3 = x_post_msc + GAP + W_cc2

W_vb = 58
harr(cv, x3, x3 + GAP + W_vb, BR2_Y + EH / 2)
rbox(cv, x3 + GAP, BR2_Y, W_vb, EH, ARM_F, ARM_S)
txt(cv, x3 + GAP, BR2_Y, W_vb, EH,
    ["videobox", "top=170", "→ 640×256 crop"], [7.5, 5.5, 5])
x3 = x3 + GAP + W_vb

W_pp = 80
harr(cv, x3, x3 + GAP + W_pp, BR2_Y + EH / 2)
rbox(cv, x3 + GAP, BR2_Y, W_pp, EH, C7X_F, C7X_S)
txt(cv, x3 + GAP, BR2_Y, W_pp, EH,
    ["tiovxdlpreproc", "mean/scale → float32", "NCHW  640×256"], [7.5, 5.5, 5])
x3 = x3 + GAP + W_pp

W_as1 = 86
harr(cv, x3, x3 + GAP + W_as1, BR2_Y + EH / 2)
rbox(cv, x3 + GAP, BR2_Y, W_as1, EH, ARM_F, ARM_S)
txt(cv, x3 + GAP, BR2_Y, W_as1, EH,
    ["appsink  pre_0", "640×256 tensor", "drop=true  max-buf=2"], [7.5, 5.5, 5])
PRE0_CX = x3 + GAP + W_as1 / 2   # centre x, for arrow down to python bridge

# ══════════════════════════════════════════════════════════════════════════════
# PYTHON BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
section(cv, M, BY, SW, BH,
        "PYTHON BRIDGE — app_edgeai process  "
        "(infer_pipe.py · post_process.py · 2 threads per InferPipe)")

PE_H = 46   # python element height
PGAP = 10

# Stage 1 (pipeline thread) — top row inside bridge section
S1_Y = BY + BH - 74   # top of stage-1 boxes

# Stage 2 (post_pipeline thread) — bottom row
S2_Y = BY + 22         # top of stage-2 boxes

# Thread banners
cv.setFont("Helvetica-Bold", 6.5)
cv.setFillColor(PY_S)
cv.drawString(M + 14, S1_Y + PE_H + 8,
    "Stage-1 thread — pipeline()   [ROI · pull tensor · UFLDv2 inference · enqueue]")
cv.drawString(M + 14, S2_Y + PE_H + 8,
    "Stage-2 thread — post_pipeline()   [dequeue · pull frame · PostProcess · YOLOX · push frame]")

px = M + 14   # running x for stage-1

# ① ROI generator
W_roi = 94
rbox(cv, px, S1_Y, W_roi, PE_H, ROI_F, ROI_S)
txt(cv, px, S1_Y, W_roi, PE_H,
    ["① ROIGenerator.step()", "lane_info + CAN signals", "→ ROIParameters"],
    [7.5, 5.5, 5.5])
ROI_BOX_X = px; ROI_BOX_W = W_roi
px = px + W_roi

# ② pull_tensor (pre_0)
W_pt = 82
harr(cv, px, px + PGAP + W_pt, S1_Y + PE_H / 2, lbl="roi")
rbox(cv, px + PGAP, S1_Y, W_pt, PE_H, ARM_F, ARM_S)
txt(cv, px + PGAP, S1_Y, W_pt, PE_H,
    ["② pull_tensor", "appsink pre_0", "(blocking)"],
    [7.5, 5.5, 5.5])
px = px + PGAP + W_pt

# ③ UFLDv2 on C7x-2
W_ld = 86
harr(cv, px, px + PGAP + W_ld, S1_Y + PE_H / 2)
rbox(cv, px + PGAP, S1_Y, W_ld, PE_H, C7X_F, C7X_S)
txt(cv, px + PGAP, S1_Y, W_ld, PE_H,
    ["③ UFLDv2  TIDL", "C7x-2 · ~55 ms", "640×256 float32 NCHW"],
    [7.5, 6, 5.5])
UFDL_CX = px + PGAP + W_ld / 2
px = px + PGAP + W_ld

# result_queue (connects stage 1 → stage 2, centred vertically)
W_rq = 78
RQ_GAP = PGAP
harr(cv, px, px + RQ_GAP + W_rq, S1_Y + PE_H / 2)
RQ_X = px + RQ_GAP
RQ_BOT = S2_Y - 4
RQ_TOP = S1_Y + PE_H + 4
RQ_H   = RQ_TOP - RQ_BOT
rbox(cv, RQ_X, RQ_BOT, W_rq, RQ_H, PY_F, PY_S, r=5, lw=1.2)
txt(cv, RQ_X, RQ_BOT, W_rq, RQ_H,
    ["result_queue", "Queue(maxsize=2)", "stage-1 → stage-2"],
    [7.5, 5.5, 5.5],
    bolds=[True, False, False])
RQ_MID_X = RQ_X + W_rq / 2
px2 = RQ_X + W_rq   # stage-2 continues from right edge of queue

# ── Stage 2 elements (continue rightward from queue) ─────────────────────────
# ④ pull_frame (sen_0)
W_pf = 82
harr(cv, px2, px2 + PGAP + W_pf, S2_Y + PE_H / 2)
rbox(cv, px2 + PGAP, S2_Y, W_pf, PE_H, ARM_F, ARM_S)
txt(cv, px2 + PGAP, S2_Y, W_pf, PE_H,
    ["④ pull_frame", "appsink sen_0", "1280×720 RGB"],
    [7.5, 5.5, 5.5])
px2 = px2 + PGAP + W_pf

# ⑤ PostProcess: pred2coords + draw lanes
W_pp2 = 100
harr(cv, px2, px2 + PGAP + W_pp2, S2_Y + PE_H / 2)
rbox(cv, px2 + PGAP, S2_Y, W_pp2, PE_H, ARM_F, ARM_S)
txt(cv, px2 + PGAP, S2_Y, W_pp2, PE_H,
    ["⑤ PostProcess (Arm A55)", "pred2coords → lane pts", "draw lanes (OpenCV)"],
    [7.5, 5.5, 5.5])
PP_CX = px2 + PGAP + W_pp2 / 2
px2 = px2 + PGAP + W_pp2

# ⑥ YOLOX on C7x-1
W_od = 86
harr(cv, px2, px2 + PGAP + W_od, S2_Y + PE_H / 2)
rbox(cv, px2 + PGAP, S2_Y, W_od, PE_H, C7X_F, C7X_S)
txt(cv, px2 + PGAP, S2_Y, W_od, PE_H,
    ["⑥ YOLOX  TIDL", "C7x-1 · ~40 ms", "416×416 uint8 NCHW"],
    [7.5, 6, 5.5])
YOLOX_CX = px2 + PGAP + W_od / 2
px2 = px2 + PGAP + W_od

# ⑦ push_frame (appsrc post_0)
W_ph = 82
harr(cv, px2, px2 + PGAP + W_ph, S2_Y + PE_H / 2)
rbox(cv, px2 + PGAP, S2_Y, W_ph, PE_H, ARM_F, ARM_S)
txt(cv, px2 + PGAP, S2_Y, W_ph, PE_H,
    ["⑦ push_frame", "appsrc post_0", "(blocking)"],
    [7.5, 5.5, 5.5])
PUSH_CX = px2 + PGAP + W_ph / 2
px2 = px2 + PGAP + W_ph

# ── Vertical arrows: source → stage-1 and source → stage-2 ───────────────────
# pre_0 → stage-1 pull_tensor (arrow down from appsink pre_0 to pipeline thread)
PT_CX = M + 14 + W_roi + PGAP + W_pt / 2   # centre of pull_tensor box
varr(cv, PRE0_CX, SY, S1_Y + PE_H, col=ARM_S, lbl="tensor")

# sen_0 → stage-2 pull_frame
PF_CX = RQ_X + W_rq + PGAP + W_pf / 2
varr(cv, SEN0_CX, SY, S2_Y + PE_H, col=ARM_S, lbl="frame")

# ── Feedback arrow: lane_info + objects → ROI generator ──────────────────────
# From PostProcess (PP_CX, S2_Y) bottom down and then left to ROI generator
FB_Y  = BY + 10   # horizontal feedback bus y
FB_LX = ROI_BOX_X + ROI_BOX_W / 2   # left end at ROI box centre
FB_RX = PP_CX                         # right end at PostProcess centre

cv.saveState()
cv.setStrokeColor(ROI_S)
cv.setLineWidth(0.9)
cv.setDash(4, 3)
# Vertical down from PP centre to feedback bus
cv.line(FB_RX, S2_Y, FB_RX, FB_Y)
# Horizontal left along feedback bus
cv.line(FB_RX, FB_Y, FB_LX, FB_Y)
cv.setDash()
# Vertical up from bus to ROI generator bottom
cv.line(FB_LX, FB_Y, FB_LX, BY + 22)
# Arrowhead up into ROI box bottom
cv.setFillColor(ROI_S)
p = cv.beginPath()
p.moveTo(FB_LX, BY + 22)
p.lineTo(FB_LX - 3.5, BY + 29)
p.lineTo(FB_LX + 3.5, BY + 29)
p.close()
cv.drawPath(p, fill=1, stroke=0)
cv.restoreState()
cv.setFont("Helvetica-Oblique", 5.5)
cv.setFillColor(ROI_S)
cv.drawCentredString((FB_LX + FB_RX) / 2, FB_Y + 3,
    "lane_info + detected_objects  (prev-frame feedback)")

# ── Thread note ───────────────────────────────────────────────────────────────
cv.setFont("Helvetica-Oblique", 5.5)
cv.setFillColor(FAINT)
cv.drawString(M + 14, BY + BH - 13,
    "threads in process: T_main (config · pipeline build · lifecycle · stats)  ·  "
    "T_pipeline — runs ①–③ once per frame  ·  "
    "T_post — runs ④–⑦ once per frame  ·  "
    "T_post runs concurrently with T_pipeline: UFLDv2(N) ∥ YOLOX(N-1)")

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
section(cv, M, OY, SW, OH,
        "OUTPUT PIPELINE (GStreamer) — encode + mux + record")

OE_H = 36
ox = M + 14
OEY = OY + (OH - OE_H) / 2

W_oap = 80
rbox(cv, ox, OEY, W_oap, OE_H, ARM_F, ARM_S)
txt(cv, ox, OEY, W_oap, OE_H,
    ["appsrc post_0", "RGB 1280×720"], [7.5, 5.5])
ox = ox + W_oap

W_occ = 72
harr(cv, ox, ox + 8 + W_occ, OEY + OE_H / 2)
rbox(cv, ox + 8, OEY, W_occ, OE_H, C7X_F, C7X_S)
txt(cv, ox + 8, OEY, W_occ, OE_H,
    ["tiovxdl", "colorconvert", "RGB→NV12"], [7.5, 5.5, 5])
ox = ox + 8 + W_occ

W_enc = 74
harr(cv, ox, ox + 8 + W_enc, OEY + OE_H / 2)
rbox(cv, ox + 8, OEY, W_enc, OE_H, VPU_F, VPU_S)
txt(cv, ox + 8, OEY, W_enc, OE_H,
    ["v4l2h264enc", "Wave5 · 10 Mbps"], [7.5, 5.5])
ox = ox + 8 + W_enc

W_hp = 60
harr(cv, ox, ox + 8 + W_hp, OEY + OE_H / 2)
rbox(cv, ox + 8, OEY, W_hp, OE_H, ARM_F, ARM_S)
txt(cv, ox + 8, OEY, W_hp, OE_H,
    ["h264parse", "AU align"], [7.5, 5.5])
ox = ox + 8 + W_hp

W_mx = 68
harr(cv, ox, ox + 8 + W_mx, OEY + OE_H / 2)
rbox(cv, ox + 8, OEY, W_mx, OE_H, ARM_F, ARM_S)
txt(cv, ox + 8, OEY, W_mx, OE_H,
    ["matroskamux", "GStreamer Matroska"], [7.5, 5.5])
ox = ox + 8 + W_mx

W_fs2 = 130
harr(cv, ox, ox + 8 + W_fs2, OEY + OE_H / 2)
rbox(cv, ox + 8, OEY, W_fs2, OE_H, ARM_F, ARM_S)
txt(cv, ox + 8, OEY, W_fs2, OE_H,
    ["filesink", "output_fcw_roi_curve.mkv"], [7.5, 5.5])

# Vertical: push_frame → appsrc post_0
OSTART_CX = M + 14 + W_oap / 2
varr(cv, PUSH_CX, OY + OH, OY + OH + 5, col=ARM_S)
# Connect with a line from PUSH_CX down to output pipeline start
cv.saveState()
cv.setStrokeColor(ARM_S)
cv.setLineWidth(0.9)
cv.line(PUSH_CX, OY + OH, PUSH_CX, OY + OE_H / 2 + OY + (OH - OE_H) / 2)
cv.line(PUSH_CX, OEY + OE_H / 2, M + 14, OEY + OE_H / 2)
cv.setFillColor(ARM_S)
p = cv.beginPath()
p.moveTo(M + 14, OEY + OE_H / 2)
p.lineTo(M + 14 + 7, OEY + OE_H / 2 + 3.5)
p.lineTo(M + 14 + 7, OEY + OE_H / 2 - 3.5)
p.close()
cv.drawPath(p, fill=1, stroke=0)
cv.restoreState()

# ══════════════════════════════════════════════════════════════════════════════
# DSP TIMING + PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════
section(cv, M, DY, SW, DH,
        "DUAL-DSP TIMING & PERFORMANCE METRICS  (from board run: 2283 frames, "
        "indian_road_curve_30fps.mp4, 91 MB mkv output)")

# ── DSP Gantt chart (left half) ───────────────────────────────────────────────
GANTT_X = M + 14
GANTT_W = 520
GANTT_H = DH - 28
GANTT_Y = DY + 14

# Background
rbox(cv, GANTT_X, GANTT_Y, GANTT_W, GANTT_H, GANTT_BG, SEC_S, r=4, lw=0.8)

# Title
cv.setFont("Helvetica-Bold", 7)
cv.setFillColor(MED)
cv.drawString(GANTT_X + 6, GANTT_Y + GANTT_H - 13,
    "Staggered-DSP timing — UFLDv2(N) ∥ YOLOX(N-1)  (frame period ≈ 78 ms)")

# Timeline axis
AXIS_Y  = GANTT_Y + 20    # y of time axis labels
ROW_H   = 28              # height of each DSP row
LANE1_Y = GANTT_Y + 30   # C7x-2 row top
LANE2_Y = LANE1_Y + ROW_H + 6  # C7x-1 row top
MS_SCALE = 3.8            # pts per ms
TOTAL_MS = 130            # timeline width in ms
T0_X    = GANTT_X + 54   # x of time=0

# Time axis
for t in range(0, TOTAL_MS + 1, 33):
    tx = T0_X + t * MS_SCALE
    cv.setStrokeColor(FAINT)
    cv.setLineWidth(0.5)
    cv.line(tx, LANE1_Y, tx, LANE2_Y + ROW_H)
    cv.setFont("Helvetica", 5.5)
    cv.setFillColor(MED)
    cv.drawCentredString(tx, AXIS_Y, "%d ms" % t)

# Row labels
for label_text, ry in [("C7x-2\n(UFLDv2\nLD)", LANE1_Y),
                        ("C7x-1\n(YOLOX\nOD)", LANE2_Y)]:
    cv.setFont("Helvetica-Bold", 5.5)
    cv.setFillColor(DARK)
    for i, part in enumerate(label_text.split("\n")):
        cv.drawRightString(T0_X - 4, ry + ROW_H - 8 - i * 7, part)

# Frame bars — 3 frames shown
# C7x-2 (UFLDv2): F0=0-55ms, F1=78-133ms
# C7x-1 (YOLOX):  F0=55-95ms (after F0 UFLDv2 result → post_pipeline starts YOLOX)
#                 F1=133-173ms
for frame_start, fname in [(0, "F0·55ms"), (78, "F1·55ms")]:
    fx = T0_X + frame_start * MS_SCALE
    fw = 55 * MS_SCALE
    rbox(cv, fx, LANE1_Y + 2, fw, ROW_H - 4, GANTT_LD, C7X_S, r=3, lw=0.7)
    cv.setFont("Helvetica-Bold", 6)
    cv.setFillColor(C7X_S)
    cv.drawCentredString(fx + fw / 2, LANE1_Y + ROW_H / 2 - 1, fname)

# YOLOX runs after UFLDv2 result arrives (staggered by ~55ms)
for frame_start, fname in [(55, "F0·40ms"), (133, "F1·40ms")]:
    if frame_start > TOTAL_MS:
        break
    fx = T0_X + frame_start * MS_SCALE
    fw = min(40 * MS_SCALE, (TOTAL_MS - frame_start) * MS_SCALE)
    rbox(cv, fx, LANE2_Y + 2, fw, ROW_H - 4, GANTT_OD, ROI_S, r=3, lw=0.7)
    cv.setFont("Helvetica-Bold", 6)
    cv.setFillColor(ROI_S)
    cv.drawCentredString(fx + fw / 2, LANE2_Y + ROW_H / 2 - 1, fname)

# Frame period markers
for ft in [0, 78]:
    fx = T0_X + ft * MS_SCALE
    cv.setStrokeColor(colors.HexColor("#2e86c1"))
    cv.setLineWidth(0.7)
    cv.setDash(3, 3)
    cv.line(fx, LANE1_Y, fx, LANE2_Y + ROW_H)
    cv.setDash()

cv.setFont("Helvetica-Oblique", 5.5)
cv.setFillColor(colors.HexColor("#2e86c1"))
cv.drawCentredString(T0_X + 39 * MS_SCALE, LANE2_Y + ROW_H + 8, "← 78 ms frame period →")

cv.setFont("Helvetica", 5.5)
cv.setFillColor(MED)
cv.drawString(GANTT_X + 6, GANTT_Y + 6,
    "UFLDv2(N) runs on C7x-2 while YOLOX(N-1) runs on C7x-1 — both DSPs busy simultaneously.")

# ── Performance metric cards (right half) ─────────────────────────────────────
CARD_X  = GANTT_X + GANTT_W + 14
CARD_W  = SW - GANTT_W - 28
CARD_H  = (DH - 22) / 3 - 4
CARD_GAP = 5

metrics = [
    ("ROI Latency  (ROIGenerator.step per frame)",
     [("mean", "5.29 ms"), ("p99", "21.17 ms"), ("max", "31.29 ms"), ("n", "2283 frames")],
     ROI_F, ROI_S),
    ("CPU Usage  (top samples during pipeline run)",
     [("usr", "~87 %"), ("sys", "~7 %"), ("python3", "~340 % per-core"), ("idle", "~6 %")],
     ARM_F, ARM_S),
    ("DSP Throughput  (measured on board)",
     [("UFLDv2", "~55 ms / frame"), ("YOLOX", "~40 ms / frame"),
      ("bottleneck", "UFLDv2 → ~12 fps"), ("output", "91 MB mkv")],
     C7X_F, C7X_S),
]

cy = DY + DH - 18
for (title, kvs, fill, stroke) in metrics:
    cy -= CARD_H
    rbox(cv, CARD_X, cy, CARD_W, CARD_H, fill, stroke, r=5, lw=0.9)
    cv.setFont("Helvetica-Bold", 6.5)
    cv.setFillColor(DARK)
    cv.drawString(CARD_X + 8, cy + CARD_H - 12, title)
    # KV pairs in 2 columns
    col_w = CARD_W / 2
    for i, (k, v) in enumerate(kvs):
        col = i % 2
        row = i // 2
        kx = CARD_X + 10 + col * col_w
        ky = cy + CARD_H - 22 - row * 14
        cv.setFont("Helvetica-Bold", 6)
        cv.setFillColor(stroke)
        cv.drawString(kx, ky, k + ":")
        cv.setFont("Helvetica", 6)
        cv.setFillColor(DARK)
        cv.drawString(kx + 40, ky, v)
    cy -= CARD_GAP

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    ("Wave5 VPU",    "decode + encode",         VPU_F, VPU_S),
    ("VPAC MSC",     "ROI crop · scale",        MSC_F, MSC_S),
    ("C7x + MMA",    "TIDL inference · dl kernels", C7X_F, C7X_S),
    ("Arm A55",      "bridge · OpenCV · I/O",   ARM_F, ARM_S),
    ("Python bridge", "queue · threads",         PY_F,  PY_S),
    ("ROI generator", "dynamic crop control",    ROI_F, ROI_S),
]

LBW = (SW - 10) / len(legend_items)
lx  = M + 5
for (name, desc, fill, stroke) in legend_items:
    rbox(cv, lx, LY + 4, LBW - 4, LH - 8, fill, stroke, r=4)
    cv.setFont("Helvetica-Bold", 6.5)
    cv.setFillColor(DARK)
    cv.drawCentredString(lx + (LBW - 4) / 2, LY + LH - 16, name)
    cv.setFont("Helvetica", 5.5)
    cv.setFillColor(MED)
    cv.drawCentredString(lx + (LBW - 4) / 2, LY + LH - 25, desc)
    lx += LBW

cv.save()
print("Saved: tools/pipeline_diagram.pdf")
