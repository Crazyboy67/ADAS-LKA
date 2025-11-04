import cv2
import numpy as np
from dataclasses import dataclass
from lane_fit import LaneModel

@dataclass
class Colors:
    left_good: tuple = (0, 200, 0)    # green
    right_good: tuple = (200, 100, 0) # blue-ish (BGR: (255,0,0) is blue; we’ll set a distinct tone)
    dashed_gray: tuple = (128, 128, 128)
    ego_fill: tuple = (0, 120, 200)   # polygon fill (BGR)

def _poly_points_from_coeffs(coeffs, y_vals):
    x = np.polyval(coeffs, y_vals)
    pts = np.stack([x, y_vals], axis=1).astype(np.float32)  # (x,y) in BEV
    return pts

def _warp_points_to_image(pts_bev, Minv, roi_y0):
    """
    pts_bev: Nx2 in warped (bird's-eye) coordinates (x, y)
    Minv: inverse perspective 3x3
    roi_y0: y-offset where ROI starts in original image
    Returns Nx2 points in original image coordinates.
    """
    if pts_bev.size == 0:
        return pts_bev
    pts = pts_bev.reshape(-1,1,2)
    pts = cv2.perspectiveTransform(pts, Minv).reshape(-1,2)
    # pts are in ROI coordinates; add offset back
    pts[:,1] += roi_y0
    return pts

def _draw_dashed_polyline(img, pts, color, thickness=3, dash_len=20, gap_len=15):
    """
    Draw dashed line along given polyline points.
    """
    if pts is None or len(pts) < 2:
        return img
    # Sample along the polyline with equal steps
    for i in range(0, len(pts)-1):
        p0 = tuple(np.round(pts[i]).astype(int))
        p1 = tuple(np.round(pts[i+1]).astype(int))
        # draw small segments by splitting long segment
        seg_len = max(abs(p1[0]-p0[0]), abs(p1[1]-p0[1]))
        if seg_len == 0: 
            continue
        n_dashes = seg_len // (dash_len + gap_len) + 1
        for k in range(int(n_dashes)):
            t0 = (k*(dash_len+gap_len))/max(1, seg_len)
            t1 = min(1.0, (k*(dash_len+gap_len)+dash_len)/max(1, seg_len))
            q0 = (int(p0[0] + (p1[0]-p0[0])*t0), int(p0[1] + (p1[1]-p0[1])*t0))
            q1 = (int(p0[0] + (p1[0]-p0[0])*t1), int(p0[1] + (p1[1]-p0[1])*t1))
            cv2.line(img, q0, q1, color, thickness, lineType=cv2.LINE_AA)
    return img

def _draw_polyline(img, pts, color, thickness=5):
    if pts is None or len(pts) < 2:
        return img
    pts_i = np.round(pts).astype(np.int32).reshape(-1,1,2)
    cv2.polylines(img, [pts_i], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return img

def _compute_lane_polygon_pts(left_pts, right_pts):
    if left_pts is None or right_pts is None or len(left_pts) < 2 or len(right_pts) < 2:
        return None
    # Order: left down->up, then right up->down to make a closed polygon
    lp = left_pts.copy()
    rp = right_pts.copy()
    # Ensure both sorted by y increasing
    lp = lp[np.argsort(lp[:,1])]
    rp = rp[np.argsort(rp[:,1])]
    poly = np.vstack([lp, rp[::-1]])
    return np.round(poly).astype(np.int32).reshape(-1,1,2)

def draw_overlay_and_hud(
    frame_bgr, model: LaneModel, Minv, roi_y0,
    left_detected, right_detected, left_conf, right_conf,
    overlay_cfg
):
    img = frame_bgr.copy()
    colors = Colors()
    # 1) Compute polylines in BEV then project back to image space
    H_w, W_w = model.warp_shape
    # y_vals already in [0 .. H_w-1]
    left_pts_img = None
    right_pts_img = None

    if model.left.coeffs is not None:
        l_pts_bev = _poly_points_from_coeffs(model.left.coeffs, model.y_vals)
        left_pts_img = _warp_points_to_image(l_pts_bev, Minv, roi_y0)
    if model.right.coeffs is not None:
        r_pts_bev = _poly_points_from_coeffs(model.right.coeffs, model.y_vals)
        right_pts_img = _warp_points_to_image(r_pts_bev, Minv, roi_y0)

    # 2) Ego-lane polygon (optional)
    if left_pts_img is not None and right_pts_img is not None and overlay_cfg.get("draw_ego", True):
        poly = _compute_lane_polygon_pts(left_pts_img, right_pts_img)
        if poly is not None:
            fill_alpha = float(overlay_cfg.get("ego_alpha", 0.25))
            fill_color = colors.ego_fill
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], fill_color)
            cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, dst=img)

    # 3) Draw lines (YES=solid color; NO=dashed gray)
    if left_detected and left_pts_img is not None:
        _draw_polyline(img, left_pts_img, colors.left_good, thickness=5)
    elif left_pts_img is not None:
        _draw_dashed_polyline(img, left_pts_img, colors.dashed_gray, thickness=3)
    if right_detected and right_pts_img is not None:
        _draw_polyline(img, right_pts_img, (255, 0, 0), thickness=5)  # pure blue in BGR
    elif right_pts_img is not None:
        _draw_dashed_polyline(img, right_pts_img, colors.dashed_gray, thickness=3)

    # 4) HUD
    hud = f"Left: {'YES' if left_detected else 'NO'} | Right: {'YES' if right_detected else 'NO'} | Conf: {0.5*(left_conf+right_conf):.2f}"
    cv2.rectangle(img, (10, 10), (10+560, 50), (0,0,0), thickness=-1)
    cv2.putText(img, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    return img

class CSVWriter:
    def __init__(self, path):
        self.f = open(path, "w", buffering=1)
        self.f.write("frame_id,left_detected,right_detected,left_conf,right_conf,lat_offset_m\n")

    def write(self, frame_id, left_detected, right_detected, left_conf, right_conf, lat_offset_m):
        self.f.write(f"{frame_id},{left_detected},{right_detected},{left_conf:.3f},{right_conf:.3f},{lat_offset_m:.3f}\n")

    def close(self):
        self.f.close()

def estimate_lateral_offset_m(model: LaneModel, image_shape, Minv, roi_y0, lane_width_m=3.7):
    """
    Approximate lateral offset at the bottom of the image (vehicle position).
    Without calibration we estimate meters-per-pixel from lane width in pixels.
    """
    H, W = image_shape
    # Use y near bottom in warped space -> map to image row ~ bottom
    y_idx = int(model.y_vals[-1])  # largest y in BEV
    def x_at_y(coeffs, y):
        if coeffs is None: return None
        return float(np.polyval(coeffs, y))
    xl = x_at_y(model.left.coeffs, y_idx)
    xr = x_at_y(model.right.coeffs, y_idx)
    if xl is None or xr is None:
        return 0.0
    lane_width_px = max(1.0, (xr - xl))
    xm_per_pix = lane_width_m / lane_width_px

    # Project these two BEV points to image, then measure center vs. image center
    pts_bev = np.array([[xl, y_idx], [xr, y_idx]], dtype=np.float32).reshape(-1,1,2)
    pts_img = cv2.perspectiveTransform(pts_bev, Minv).reshape(-1,2)
    pts_img[:,1] += roi_y0

    lane_center_x = 0.5*(pts_img[0,0] + pts_img[1,0])
    img_center_x = W / 2.0
    offset_px = img_center_x - lane_center_x  # left negative, right positive (we'll invert sign next)
    offset_m = -offset_px * xm_per_pix
    return float(offset_m)