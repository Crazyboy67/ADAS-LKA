import cv2
import numpy as np
from dataclasses import dataclass
from lane_fit import LaneModel

@dataclass
class Colors:
    left_good: tuple = (0, 200, 0)    # green
    right_good: tuple = (200, 100, 0) # blue
    dashed_gray: tuple = (128, 128, 128)
    ego_fill: tuple = (0, 120, 200)   # polygon fill (BGR)


def _safe_polyval(coeffs, y_vals):
    if coeffs is None or y_vals is None: 
        return None
    x = np.polyval(coeffs, y_vals.astype(np.float32))
    if not np.all(np.isfinite(x)):
        return None
    return x

def _sanitize_pts_img(pts, img_shape, margin=64, min_unique_y=8):
    """Keep points finite and not too far outside the image; ensure enough distinct y."""
    if pts is None: 
        return None
    H, W = img_shape[:2]
    pts = pts.astype(np.float32)
    ok = np.isfinite(pts).all(axis=1)
    pts = pts[ok]
    if pts.size == 0:
        return None
    
    x = np.clip(pts[:, 0], -margin, W + margin)
    y = np.clip(pts[:, 1], -margin, H + margin)
    pts = np.stack([x, y], axis=1)
    
    if np.unique(np.round(pts[:, 1])).size < min_unique_y:
        return None
    return pts

def _warp_points_to_image_safe(pts_bev, Minv, roi_y0, img_shape):
    if pts_bev is None or pts_bev.size == 0:
        return None
    pts = pts_bev.reshape(-1, 1, 2).astype(np.float32)
    pts_img = cv2.perspectiveTransform(pts, Minv).reshape(-1, 2)
    pts_img[:, 1] += roi_y0
    return _sanitize_pts_img(pts_img, img_shape)

def _lane_polygon_if_valid(left_pts_img, right_pts_img):
    if left_pts_img is None or right_pts_img is None:
        return None
    
    lp = left_pts_img[np.argsort(left_pts_img[:, 1])]
    rp = right_pts_img[np.argsort(right_pts_img[:, 1])]
    n = min(len(lp), len(rp))
    if n < 8:
        return None
    
    gap = rp[:n, 0] - lp[:n, 0]
    if np.median(gap) < 15 or np.sum(gap < 0) > n * 0.2:
        return None
    poly = np.vstack([lp[:n], rp[:n][::-1]])
    return np.round(poly).astype(np.int32).reshape(-1, 1, 2)

def _draw_dashed_polyline(img, pts, color, thickness=3, dash_len=20, gap_len=15):
    """
    Draw dashed line along given polyline points.
    """
    if pts is None or len(pts) < 2:
        return img
    
    for i in range(0, len(pts)-1):
        p0 = tuple(np.round(pts[i]).astype(int))
        p1 = tuple(np.round(pts[i+1]).astype(int))
        
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

def draw_overlay_and_hud(
    frame_bgr, model: LaneModel, Minv, roi_y0,
    left_detected, right_detected, left_conf, right_conf,
    overlay_cfg
):
    img = frame_bgr.copy()
    colors = Colors()
    left_pts_img = None
    right_pts_img = None

    if model.left and model.left.coeffs is not None:
        lx = _safe_polyval(model.left.coeffs, model.y_vals)
        l_pts_bev = None if lx is None else np.stack([lx, model.y_vals], axis=1).astype(np.float32)
    else:
        l_pts_bev = None

    if model.right and model.right.coeffs is not None:
        rx = _safe_polyval(model.right.coeffs, model.y_vals)
        r_pts_bev = None if rx is None else np.stack([rx, model.y_vals], axis=1).astype(np.float32)
    else:
        r_pts_bev = None

    left_pts_img  = _warp_points_to_image_safe(l_pts_bev, Minv, roi_y0, img.shape)   if l_pts_bev is not None else None
    right_pts_img = _warp_points_to_image_safe(r_pts_bev, Minv, roi_y0, img.shape)   if r_pts_bev is not None else None

    # 2) Ego-lane polygon
    poly = _lane_polygon_if_valid(left_pts_img, right_pts_img)
    if overlay_cfg.get("draw_ego", True) and poly is not None:
        alpha = float(overlay_cfg.get("ego_alpha", 0.25))
        tmp = img.copy()
        cv2.fillPoly(tmp, [poly], colors.ego_fill)
        cv2.addWeighted(tmp, alpha, img, 1 - alpha, 0, dst=img)

    # 3) Draw lines
    if left_pts_img is not None:
        if left_detected:
            _draw_polyline(img, left_pts_img, colors.left_good, thickness=5)      # green
        else:
            _draw_dashed_polyline(img, left_pts_img, colors.dashed_gray, 3)     # dashed gray

    if right_pts_img is not None:
        if right_detected:
            _draw_polyline(img, right_pts_img, colors.right_good, thickness=5)     # blue
        else:
            _draw_dashed_polyline(img, right_pts_img, colors.dashed_gray, 3)    # dashed gray

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
    """
    H, W = image_shape
    
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

    pts_bev = np.array([[xl, y_idx], [xr, y_idx]], dtype=np.float32).reshape(-1,1,2)
    pts_img = cv2.perspectiveTransform(pts_bev, Minv).reshape(-1,2)
    pts_img[:,1] += roi_y0

    lane_center_x = 0.5*(pts_img[0,0] + pts_img[1,0])
    img_center_x = W / 2.0
    offset_px = img_center_x - lane_center_x
    offset_m = -offset_px * xm_per_pix
    return float(offset_m)