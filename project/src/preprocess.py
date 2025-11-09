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


def threshold_bev_2(bev_bgr, thr):
    """
    CLAHE-free lane mask in BEV using only HLS + Sobel features.
    Returns a uint8 mask (0 or 255).
    """

    # ------------------------------
    # Parameters (minimal, Sobel-only)
    # ------------------------------
    # Optionally keep these off (defaults) to stay "HLS + Sobel only"
    use_exposure_balance = bool(thr.get("use_exposure_balance", False))
    use_white_balance    = bool(thr.get("use_white_balance",   False))
    use_highlight_shadow = bool(thr.get("use_highlight_shadow", False))

    # Sobel gates (triple-mask)
    thd_L_mag = int(thr.get("thd_L_mag", 20))
    thd_S_mag = int(thr.get("thd_S_mag", 25))
    thd_L_arg = int(thr.get("thd_L_arg",  0))
    thd_S_arg = int(thr.get("thd_S_arg",  0))
    thd_L_y   = int(thr.get("thd_L_y",   75))

    use_mag_gate = bool(thr.get("use_mag_gate", True))
    use_arg_gate = bool(thr.get("use_arg_gate", True))
    use_ly_gate  = bool(thr.get("use_ly_gate",  True))

    # Corridor (apply to final mask)
    bandpass_x = thr.get("bandpass_x", [0.40, 0.60])

    # Morphology and shape filtering (optional but helpful)
    open_horiz = int(thr.get("open_horiz", 7))
    close_vert = int(thr.get("close_vert", 21))
    keep_min_height = int(thr.get("keep_min_height", 50))
    keep_max_width  = int(thr.get("keep_max_width",  54))
    small_blob_area = int(thr.get("small_blob_area", 160))

    # (Optional) specular cleanup — OFF by default to keep “HLS+Sobel only”
    spec_v_min = int(thr.get("spec_v_min", 255))  # 255 => disabled

    # ------------------------------
    # 1) HLS conversion (no CLAHE/bilateral)
    # ------------------------------
    hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS).astype(np.uint8)

    if use_exposure_balance:
        _hls_balance_exposure_inplace(hls)
    if use_white_balance:
        _hls_balance_white_inplace(hls)
    if use_highlight_shadow:
        _remove_highlights_and_shadows_inplace(hls, 255, 0, 30, 50)

    L = hls[:, :, 1]  # Lightness
    S = hls[:, :, 2]  # Saturation

    # ------------------------------
    # 2) Sobel on L and S
    # ------------------------------
    Lx64 = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    Ly64 = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    Lmag = np.sqrt(Lx64 * Lx64 + Ly64 * Ly64)
    Larg = np.abs(np.degrees(np.arctan2(Ly64, Lx64)) - 90.0)

    Sx64 = cv2.Sobel(S, cv2.CV_64F, 1, 0, ksize=3)
    Sy64 = cv2.Sobel(S, cv2.CV_64F, 0, 1, ksize=3)
    Smag = np.sqrt(Sx64 * Sx64 + Sy64 * Sy64)
    Sarg = np.abs(np.degrees(np.arctan2(Sy64, Sx64)) - 90.0)

    # scale like before (centered around 127)
    Lx_u8, Ly_u8, Lmag_u8, Larg_u8 = _scale_sobel_to_u8(Lx64, Ly64, Lmag, Larg)
    Sx_u8, Sy_u8, Smag_u8, Sarg_u8 = _scale_sobel_to_u8(Sx64, Sy64, Smag, Sarg)

    # ------------------------------
    # 3) Triple-gate combine
    # ------------------------------
    if use_mag_gate:
        L_mag_cond = (np.abs(127 - Lmag_u8.astype(np.int16)) > thd_L_mag)
        S_mag_cond = (np.abs(127 - Smag_u8.astype(np.int16)) > thd_S_mag)
        mask_LS_mag = (L_mag_cond | S_mag_cond)
    else:
        mask_LS_mag = np.ones_like(L, dtype=bool)

    if use_arg_gate:
        L_arg_cond = (np.abs(127 - Larg_u8.astype(np.int16)) > thd_L_arg)
        S_arg_cond = (np.abs(127 - Sarg_u8.astype(np.int16)) > thd_S_arg)
        mask_LS_arg = (L_arg_cond & S_arg_cond)
    else:
        mask_LS_arg = np.ones_like(L, dtype=bool)

    if use_ly_gate:
        Ly_cond = (np.abs(127 - Ly_u8.astype(np.int16)) < thd_L_y)
    else:
        Ly_cond = np.ones_like(L, dtype=bool)

    votes = mask_LS_mag.astype(np.uint8) + mask_LS_arg.astype(np.uint8) + Ly_cond.astype(np.uint8)
    keep_mask = (votes >= 3)

    # make binary
    core = np.zeros_like(L, dtype=np.uint8)
    core[keep_mask] = 255

    # ------------------------------
    # 4) Corridor mask on final binary
    # ------------------------------
    H, W = core.shape
    lo = int(max(0, min(1, bandpass_x[0])) * W)
    hi = int(max(0, min(1, bandpass_x[1])) * W)
    gate = np.zeros_like(core, dtype=np.uint8); gate[:, lo:hi] = 255
    core = cv2.bitwise_and(core, gate)

    # ------------------------------
    # 5) (Optional) specular cleanup (disabled if spec_v_min==255)
    # ------------------------------
    if spec_v_min < 255:
        V = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        spec = cv2.inRange(V, spec_v_min, 255)
        spec = cv2.morphologyEx(spec, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
        num, lbl, st, _ = cv2.connectedComponentsWithStats(spec, 8)
        kill = np.zeros_like(core, dtype=np.uint8)
        for i in range(1, num):
            if st[i, cv2.CC_STAT_AREA] < small_blob_area:
                x = st[i, cv2.CC_STAT_LEFT]; y = st[i, cv2.CC_STAT_TOP]
                w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
                kill[y:y+h, x:x+w] = 255
        core = cv2.bitwise_and(core, cv2.bitwise_not(kill))

    # ------------------------------
    # 6) Morphology + shape filter (lane-like)
    # ------------------------------
    if open_horiz > 0:
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (open_horiz, 1)), 1)
    if close_vert > 0:
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, close_vert)), 1)

    num, lbl, st, _ = cv2.connectedComponentsWithStats(core, 8)
    keep = np.zeros_like(core, dtype=np.uint8)
    for i in range(1, num):
        w_i = st[i, cv2.CC_STAT_WIDTH]
        h_i = st[i, cv2.CC_STAT_HEIGHT]
        if (h_i >= keep_min_height) and (w_i <= keep_max_width) and (st[i, cv2.CC_STAT_AREA] >= small_blob_area):
            keep[lbl == i] = 255

    return keep


# ------------------------------
# Helpers (private)
# ------------------------------

def _hls_balance_exposure_inplace(hls):
    """
    reduce L departures from mean L; keeps dtype uint8.
    """
    L = hls[:, :, 1].astype(np.float32)
    mean_L = float(np.mean(L))
    max_L = float(np.max(L))
    min_L = float(np.min(L))
    if max_L != min_L:
        # exposure_correction_ratio = 1 in the reference
        L = L - np.abs((L - mean_L) / (max_L - min_L)) * (L - mean_L)
    hls[:, :, 1] = np.clip(L, 0, 255).astype(np.uint8)


def _hls_balance_white_inplace(hls):
    """
    push S so the least-saturated reference pixel goes to zero.
    """
    S = hls[:, :, 2].astype(np.int32)
    Lneg = 255 - hls[:, :, 1].astype(np.int32)
    SLn = S + Lneg  # ratios = 1 in the reference
    mask_min_SLn = (SLn == SLn.min())
    # among those, find the minimum Lneg
    Lneg_min = Lneg[mask_min_SLn].min()
    mask_max_Ln = (Lneg == Lneg_min)
    #min S is the S where both masks hold
    ref = S[np.logical_and(mask_min_SLn, mask_max_Ln)]
    if ref.size > 0:
        S = S - int(ref[0])
        S[S < 0] = 0
    hls[:, :, 2] = np.clip(S, 0, 255).astype(np.uint8)


def _remove_highlights_and_shadows_inplace(hls, thd_highlight_L, thd_highlight_S,
                                           thd_shadow_L, thd_shadow_S):
    """
    Zero out highlight/shadow pixels using HLS rules.
    """
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    # kill highlights: high L & low S
    LS_high = (L > thd_highlight_L) & (S < thd_highlight_S)
    # kill shadows:   low L  & high S
    LS_shad = (L < thd_shadow_L) & (S > thd_shadow_S)

    kill = np.logical_or(LS_high, LS_shad)
    hls[kill] = (0, 0, 0)


def _scale_sobel_to_u8(sx, sy, smag, sarg_deg_from_vertical):
    """
    Mimic lib_frame.frame_scale_uint8 behavior: scale each into uint8 around 127.
    """
    def _scale(arr):
        arr = arr.astype(np.float32)
        length = np.max(np.abs(arr))
        if length > 0:
            out = np.uint8(127 * (1.0 + arr / length))
        else:
            out = np.zeros_like(arr, dtype=np.uint8)
        return out

    return _scale(sx), _scale(sy), _scale(smag), _scale(sarg_deg_from_vertical)