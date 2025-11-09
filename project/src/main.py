import argparse
import os
import cv2
import yaml
import time
import numpy as np
from preprocess import threshold_bev, crop_road_roi, threshold_bev_2
from warp import get_perspective_matrices
from lane_fit import fit_lanes_sliding_windows
from temporal import TemporalSmoother
from overlay import draw_overlay_and_hud, CSVWriter, estimate_lateral_offset_m

DEFAULT_CONF_TAU = 0.6

def parse_args():
    p = argparse.ArgumentParser(description="LKA baseline (classical CV).")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--out-video", default="outputs/annotated.mp4", help="Annotated MP4 output")
    p.add_argument("--out-csv", default="outputs/per_frame.csv", help="Per-frame CSV output")
    p.add_argument("--config", default="config.yaml", help="YAML config path")
    p.add_argument("--conf-tau", type=float, default=DEFAULT_CONF_TAU, help="Detection threshold for YES/NO")
    p.add_argument("--show", action="store_true", help="Preview while processing")
    p.add_argument("--src-norm",
                help="Override IPM src_norm as normalized ROI coords (order BL,BR,TR,TL). "
                     "Format: 'x1,y1;x2,y2;x3,y3;x4,y4' (each in [0,1]).")
    return p.parse_args()

def _parse_src_norm(s: str):
    pts = []
    for pair in s.strip().split(";"):
        x_str, y_str = pair.split(",")
        x, y = float(x_str), float(y_str)
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"src_norm point out of range [0,1]: {(x,y)}")
        pts.append([x, y])
    if len(pts) != 4:
        raise ValueError("src_norm needs exactly 4 points.")
    return pts

def _order_bl_br_tr_tl(pts):
    # pts: list[[x,y],...], normalized in ROI space
    # sort by y (descending): bottom row first
    by_y = sorted(pts, key=lambda p: p[1], reverse=True)
    bottom = sorted(by_y[:2], key=lambda p: p[0])  # left->right
    top    = sorted(by_y[2:], key=lambda p: p[0])  # left->right
    bl, br = bottom[0], bottom[1]
    tl, tr = top[0], top[1]
    return [bl, br, tr, tl]

def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.src_norm:
        pts = _parse_src_norm(args.src_norm)
        pts = _order_bl_br_tr_tl(pts)
        # write into config
        cfg.setdefault("ipm", {})["src_norm"] = pts

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # Read one frame to initialize# read one frame
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Empty video.")
    H, W = frame.shape[:2]

    # writers
    ensure_dir(args.out_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out_video, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25.0, (W, H))
    ensure_dir(args.out_csv)
    csv = CSVWriter(args.out_csv)

    # --- IMPORTANT: build ROI & IPM on the ROI ---
    roi0, roi_y0_init = crop_road_roi(frame, cfg.get("roi", {}))
    roiH, roiW = roi0.shape[:2]

    # Build IPM on the ROI (not full frame) and from config only
    M, Minv, warp_size = get_perspective_matrices((roiH, roiW), cfg.get("ipm", {}))

    smoother = TemporalSmoother(cfg.get("temporal", {}))

    frame_id = 0
    prev_model = None
    t0 = time.time()

    # Rewind to start (we used one frame to init)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) ROI
        roi, roi_y0 = crop_road_roi(frame, cfg.get("roi", {}))

        # 2) Warp to bird’s-eye
        bird_bgr = cv2.warpPerspective(roi, M, warp_size, flags=cv2.INTER_LINEAR)

        # 3) Threshold
        # bird_bin = threshold_bev(bird_bgr, cfg.get("thresholds_bev", {}))

        bird_bin = threshold_bev_2(bird_bgr, cfg.get("thresholds_bev", {}))

        vis = cv2.resize(bird_bin, (640, int(640 * bird_bin.shape[0] / bird_bin.shape[1])))
        cv2.imshow("BEV binary (every ~2s)", vis)

        # 4) Fit lanes in bird’s-eye
        model = fit_lanes_sliding_windows(
            bird_bin,
            prior=prev_model,
            cfg=cfg.get("fit", {})
        )

        # 5) Temporal smoothing
        model_s = smoother.update(model, prev_model)

        # 6) Confidence & flags
        left_conf, right_conf = model_s.left.confidence, model_s.right.confidence
        left_detected = int(left_conf > args.conf_tau)
        right_detected = int(right_conf > args.conf_tau)

        # 7) Lateral offset (meters, approx)
        lat_offset_m = estimate_lateral_offset_m(
            model_s, image_shape=(H, W), Minv=Minv, roi_y0=roi_y0,
            lane_width_m=cfg.get("metrics", {}).get("lane_width_m", 3.7)
        )

        # 8) Draw overlay & HUD
        annotated = draw_overlay_and_hud(
            frame, model_s, Minv, roi_y0,
            left_detected, right_detected,
            left_conf, right_conf,
            cfg.get("overlay", {})
        )

        # 9) Write CSV row and video frame
        csv.write(
            frame_id=frame_id,
            left_detected=left_detected,
            right_detected=right_detected,
            left_conf=left_conf,
            right_conf=right_conf,
            lat_offset_m=lat_offset_m
        )
        out.write(annotated)

        if args.show and frame_id == 0:
            dbg = roi.copy()
            src = cv2.perspectiveTransform(
                np.array([[[0,0],[warp_size[0]-1,0],[warp_size[0]-1,warp_size[1]-1],[0,warp_size[1]-1]]], dtype=np.float32),
                Minv
            ).reshape(-1,2).astype(int)
            cv2.polylines(dbg, [src.reshape(-1,1,2)], True, (0,255,255), 2)
            cv2.imshow("ROI + IPM trapezoid (first frame)", dbg)
            cv2.waitKey(1)


        if args.show:
            cv2.imshow("LKA Annotated", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        prev_model = model_s
        frame_id += 1

    out.release()
    cap.release()
    csv.close()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()