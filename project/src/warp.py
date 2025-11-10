import cv2
import numpy as np

def _points_from_norm(img_shape, norm_pts):
    H, W = img_shape
    pts = np.stack([np.array(norm_pts)[:,0]*W, np.array(norm_pts)[:,1]*H], axis=1)
    return pts.astype(np.float32)

def get_perspective_matrices(img_shape, ipm_cfg, src_abs=None):
    H, W = img_shape
    warp_W = int(ipm_cfg.get("warp_W", W))
    warp_H = int(ipm_cfg.get("warp_H", H//2))

    # default normalized points
    src_norm = ipm_cfg.get("src_norm", [[0.42,0.98],[0.58,0.98],[0.53,0.66],[0.47,0.66]])
    dst_norm = ipm_cfg.get("dst_norm", [[0.25,1.00],[0.75,1.00],[0.75,0.00],[0.25,0.00]])

    if src_abs is None:
        src = _points_from_norm(img_shape, src_norm)
    else:
        src = np.array(src_abs, dtype=np.float32)

    dst = _points_from_norm((warp_H, warp_W), dst_norm)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, (warp_W, warp_H)