import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class SideFit:
    ok: bool
    coeffs: Optional[np.ndarray]  # polynomial x=f(y): [a,b,c] for 2nd-order, or [d,c,b,a] for 3rd
    residual: float
    pixel_count: int
    confidence: float

@dataclass
class LaneModel:
    left: SideFit
    right: SideFit
    # store model-space params used for drawing back
    y_vals: np.ndarray  # in warped space
    warp_shape: Tuple[int, int]  # (H, W)

def _histogram_peaks(binary_warped, left_band=(0.43,0.50), right_band=(0.50,0.57), min_peak=60):
    H, W = binary_warped.shape
    bottom = binary_warped[int(H*0.65):, :]
    hist = (bottom//255).sum(axis=0)

    def peak(lo, hi):
        lo, hi = int(lo*W), int(hi*W)
        band = hist[lo:hi]
        if band.size == 0: return (lo+hi)//2
        m = band.max()
        return lo + int(np.argmax(band)) if m >= min_peak else (lo+hi)//2

    return peak(*left_band), peak(*right_band)

def _sliding_window(binary_warped, base_x, nwindows=9, margin=80, minpix=40):
    H, W = binary_warped.shape
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    window_height = H // nwindows
    x_current = base_x
    lane_inds = []

    for window in range(nwindows):
        win_y_low = H - (window + 1) * window_height
        win_y_high = H - window * window_height
        win_x_low = max(0, x_current - margin)
        win_x_high = min(W, x_current + margin)

        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        lane_inds.append(good_inds)

        if len(good_inds) > minpix:
            x_current = int(np.mean(nonzerox[good_inds]))

    lane_inds = np.concatenate(lane_inds) if len(lane_inds) > 0 else np.array([], dtype=int)
    return nonzerox, nonzeroy, lane_inds

def _fit_poly_from_inds(nonzerox, nonzeroy, inds, order=2):
    if inds.size < max(200, order*50):  # too few pixels
        return None, np.inf, inds.size
    x = nonzerox[inds]
    y = nonzeroy[inds]
    try:
        coeffs = np.polyfit(y, x, order)  # x = f(y)
        x_fit = np.polyval(coeffs, y)
        residual = float(np.mean((x_fit - x) ** 2))
        return coeffs, residual, inds.size
    except np.linalg.LinAlgError:
        return None, np.inf, inds.size

def _confidence(pixel_count, residual, coeff_delta_norm, cfg):
    # Normalize heuristics
    pc_norm = min(1.0, pixel_count / float(cfg.get("norm_pixel_count", 1500)))
    res_norm = min(1.0, residual / float(cfg.get("norm_residual", 100.0)))
    delta_norm = min(1.0, coeff_delta_norm / float(cfg.get("norm_coeff_delta", 50.0)))

    a = cfg.get("conf_a", 4.0)
    b = cfg.get("conf_b", 2.0)
    c = cfg.get("conf_c", 1.0)
    # Higher pixel_count boosts; residual & delta penalize
    score = a * pc_norm - b * res_norm - c * delta_norm
    return float(_sigmoid(score))

def fit_lanes_sliding_windows(binary_warped, prior: Optional[LaneModel], cfg):
    """
    Main fitting function for one frame in bird's-eye space.
    Returns a LaneModel with per-side confidence scores.
    """
    lb = cfg.get("hist_left_band", [0.40,0.50])
    rb = cfg.get("hist_right_band", [0.50,0.60])
    min_peak = cfg.get("hist_min_peak", 50)

    leftx_base, rightx_base = _histogram_peaks(binary_warped, tuple(lb), tuple(rb), min_peak)

    # --- Safe base-from-right for a weak left base ---
    H, W = binary_warped.shape
    def band_evidence(xc, rng=12, y0_frac=0.80):
        y0 = int(H * y0_frac)
        x0 = max(0, int(xc - rng)); x1 = min(W, int(xc + rng))
        strip = binary_warped[y0:, x0:x1]
        return int((strip // 255).sum())

    left_ev  = band_evidence(leftx_base,  rng=12, y0_frac=0.80)
    right_ev = band_evidence(rightx_base, rng=12, y0_frac=0.80)

    if left_ev < 50 and right_ev >= 120:                     # left is weak, right is solid
        width_px = int(cfg.get("expected_lane_width_px", 0.5 * W))
        pad = int(0.05 * W)
        leftx_base = int(np.clip(rightx_base - width_px, pad, W - pad))


    use_prior = cfg.get("use_prior_track", True)
    track_margin = cfg.get("track_margin", 70)


    nwindows = cfg.get("nwindows", 9)
    margin = cfg.get("margin", 80)
    minpix = cfg.get("minpix", 40)
    poly_order = cfg.get("poly_order", 2)
    max_curvature = cfg.get("max_curvature_px", 1e6)  # not used explicitly; placeholder

    prev_left = prior.left.coeffs if (prior and prior.left and prior.left.coeffs is not None) else None
    prev_right = prior.right.coeffs if (prior and prior.right and prior.right.coeffs is not None) else None


    # LEFT
    if use_prior and prev_left is not None:
        nonzeroxL, nonzeroyL, left_inds = _search_around_poly(binary_warped, prev_left, margin=track_margin)
    else:
        nonzeroxL, nonzeroyL, left_inds = _sliding_window(binary_warped, leftx_base, nwindows, margin, minpix)

    left_coeffs, left_res, left_cnt = _fit_poly_from_inds(nonzeroxL, nonzeroyL, left_inds, order=poly_order)

    # RIGHT
    if use_prior and prev_right is not None:
        nonzeroxR, nonzeroyR, right_inds = _search_around_poly(binary_warped, prev_right, margin=track_margin)
    else:
        nonzeroxR, nonzeroyR, right_inds = _sliding_window(binary_warped, rightx_base, nwindows, margin, minpix)

    right_coeffs, right_res, right_cnt = _fit_poly_from_inds(nonzeroxR, nonzeroyR, right_inds, order=poly_order)

    if cfg.get("hough_fallback", True):
        need_left = (left_coeffs is None) or (left_cnt < 120)
        need_right = (right_coeffs is None) or (right_cnt < 120)
        if need_left or need_right:
            hb_left, hb_right = _hough_bootstrap(binary_warped, cfg)
            if need_left and hb_left is not None:
                left_coeffs, left_res, left_cnt = hb_left, 120.0, 200
            if need_right and hb_right is not None:
                right_coeffs, right_res, right_cnt = hb_right, 120.0, 200

    # Coeff deltas vs prior (temporal consistency)
    def coeff_delta(cur, prev):
        if cur is None or prev is None:
            return 1e3
        # pad / match orders
        n = max(len(cur), len(prev))
        c = np.pad(cur, (n-len(cur), 0), mode='constant')
        p = np.pad(prev, (n-len(prev), 0), mode='constant')
        return float(np.linalg.norm(c - p))

    left_delta = coeff_delta(left_coeffs, prev_left)
    right_delta = coeff_delta(right_coeffs, prev_right)
    left_conf = _confidence(left_cnt, left_res, left_delta, cfg)
    right_conf = _confidence(right_cnt, right_res, right_delta, cfg)

    left_ok = left_coeffs is not None
    right_ok = right_coeffs is not None

    # break out confidence components explicitly
    def conf_parts(pixel_count, residual, delta, cfg):
        pc_norm = min(1.0, pixel_count / float(cfg.get("norm_pixel_count", 1500)))
        res_norm = min(1.0, residual / float(cfg.get("norm_residual", 100.0)))
        d_norm = min(1.0, delta / float(cfg.get("norm_coeff_delta", 50.0)))
        a = cfg.get("conf_a", 4.0); b = cfg.get("conf_b", 2.0); c = cfg.get("conf_c", 1.0)
        score = a*pc_norm - b*res_norm - c*d_norm
        conf = 1.0 / (1.0 + np.exp(-score))
        return (pc_norm, res_norm, d_norm, score, conf)

    lp = conf_parts(left_cnt, left_res, left_delta, cfg)
    rp = conf_parts(right_cnt, right_res, right_delta, cfg)
    left_conf = lp[-1]; right_conf = rp[-1]

    # Pack model
    y_vals = np.linspace(0, H - 1, num=H)  # dense curve
    return LaneModel(
        left=SideFit(ok=left_ok, coeffs=left_coeffs, residual=left_res, pixel_count=left_cnt, confidence=left_conf),
        right=SideFit(ok=right_ok, coeffs=right_coeffs, residual=right_res, pixel_count=right_cnt, confidence=right_conf),
        y_vals=y_vals,
        warp_shape=(H, W)
    )

def _hough_bootstrap(binary_warped, cfg):
    """Return (left_coeffs, right_coeffs) as 2nd-order polys if we can estimate straight-ish lines."""
    H, W = binary_warped.shape
    edges = cv2.Canny(binary_warped, 30, 90, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=int(cfg.get("hough_thresh",50)),
        minLineLength=int(cfg.get("hough_min_len",60)),
        maxLineGap=int(cfg.get("hough_max_gap",30))
    )
    if lines is None:
        return None, None

    midpoint = W // 2
    best_left, best_right = None, None
    best_len_L, best_len_R = 0, 0

    for (x1,y1,x2,y2) in lines[:,0,:]:
        dx, dy = abs(x2-x1), abs(y2-y1)
        if dy < 20:  # ignore near-horizontal
            continue
        seg_len = np.hypot(dx, dy)
        # Fit x = a*y + b (1st order)
        ys = np.array([y1,y2], dtype=np.float32)
        xs = np.array([x1,x2], dtype=np.float32)
        A = np.vstack([ys, np.ones_like(ys)]).T
        a,b = np.linalg.lstsq(A, xs, rcond=None)[0]
        coeffs = np.array([0.0, a, b], dtype=np.float64)  # make it look like 2nd order [a2, a1, a0]

        xm = (x1+x2)/2.0
        if xm < midpoint:
            if seg_len > best_len_L:
                best_len_L = seg_len
                best_left = coeffs
        else:
            if seg_len > best_len_R:
                best_len_R = seg_len
                best_right = coeffs

    return best_left, best_right

def _search_around_poly(binary_warped, coeffs, margin=70):
    if coeffs is None:
        return None, None, np.array([], dtype=int)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    x_fit = np.polyval(coeffs, nonzeroy)
    good_inds = np.where((nonzerox > x_fit - margin) & (nonzerox < x_fit + margin))[0]
    return nonzerox, nonzeroy, good_inds