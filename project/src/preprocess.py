import cv2
import numpy as np

def crop_road_roi(frame, roi_cfg):
    H, W = frame.shape[:2]
    top_frac = roi_cfg.get("top_frac", 0.58)
    y0 = int(H * top_frac)
    roi = frame[y0:, :]
    return roi, y0

def threshold_bev(bev_bgr, thr):
    use_auto_L  = bool(thr.get("auto_L", True))
    L_percentile= float(thr.get("L_percentile", 90.0))
    L_offset    = float(thr.get("L_offset", 8.0))

    l_min_fixed = int(thr.get("l_white_min", 178))

    ksize      = int(thr.get("sobel_ksize", 3))
    mag_min    = int(thr.get("grad_mag_min", 20))

    band       = thr.get("bandpass_x", [0.40, 0.60])
    open_h     = int(thr.get("open_horiz", 7))
    close_v    = int(thr.get("close_vert", 21))
    keep_h_min = int(thr.get("keep_min_height", 50))
    keep_w_max = int(thr.get("keep_max_width", 54))
    area_min   = int(thr.get("small_blob_area", 160))

    spec_v_min = int(thr.get("spec_v_min", 248))
    clahe_clip = float(thr.get("clahe_clip", 2.0))
    clahe_grid = int(thr.get("clahe_grid", 8))

    use_white = bool(thr.get("use_white", True))
    use_grad  = bool(thr.get("use_grad",  True))

    lab = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    Lc  = clahe.apply(L)
    Lf  = cv2.bilateralFilter(Lc, d=5, sigmaColor=40, sigmaSpace=40)

    H, W = Lf.shape
    lo, hi = int(band[0]*W), int(band[1]*W)
    gate = np.zeros_like(Lf, dtype=np.uint8); gate[:, lo:hi] = 1

    if use_auto_L:
        vals = Lf[gate.astype(bool)]
        if vals.size > 500:
            p = np.percentile(vals, L_percentile)
            l_min = max(0, int(round(p - L_offset)))
        else:
            l_min = l_min_fixed
    else:
        l_min = l_min_fixed

    white = cv2.inRange(Lf, l_min, 255)

    # Grad + orientation (near-vertical)
    gx = cv2.Sobel(Lf, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(Lf, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)

    mag_mask = (mag >= mag_min).astype(np.uint8) * 255

    core = 255*np.ones_like(Lf, dtype=np.uint8)
    if use_white:
        core = cv2.bitwise_and(core, white)
    if use_grad:
        core = cv2.bitwise_and(core, mag_mask)

    # Specular removal
    V = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    spec = cv2.inRange(V, spec_v_min, 255)
    spec = cv2.morphologyEx(spec, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    num, lbl, st, _ = cv2.connectedComponentsWithStats(spec, 8)
    kill = np.zeros_like(core)
    for i in range(1, num):
        if st[i, cv2.CC_STAT_AREA] < area_min:
            x = st[i, cv2.CC_STAT_LEFT]; y = st[i, cv2.CC_STAT_TOP]
            w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
            kill[y:y+h, x:x+w] = 255
    core = cv2.bitwise_and(core, cv2.bitwise_not(kill))

    # Anisotropic morphology
    if open_h > 0:
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (open_h,1)), 1)
    if close_v > 0:
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,close_v)), 1)

    # Keep tall & thin components
    num, lbl, st, _ = cv2.connectedComponentsWithStats(core, 8)
    keep = np.zeros_like(core)
    for i in range(1, num):
        w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]; ar = st[i, cv2.CC_STAT_AREA]
        if h >= keep_h_min and w <= keep_w_max and ar >= area_min:
            keep[lbl == i] = 255

    return keep