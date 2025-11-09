# src/lane_fit.py
import cv2
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List


# -----------------------------
# Public data structures
# -----------------------------

@dataclass
class SideFit:
    ok: bool
    coeffs: Optional[np.ndarray]
    residual: float
    pixel_count: int
    confidence: float


@dataclass
class LaneModel:
    left: SideFit
    right: SideFit
    y_vals: np.ndarray
    warp_shape: Tuple[int, int]


# -----------------------------
# Config helpers (defaults)
# -----------------------------

def _fit_cfg(cfg: dict) -> dict:
    """Flatten/normalize the 'fit' dict and provide robust defaults."""
    f = dict(cfg or {})
    # primary knobs (aligned with your existing config names)
    f.setdefault("nwindows", 9)
    f.setdefault("margin", 120)
    f.setdefault("minpix", 30)
    f.setdefault("poly_order", 2)

    # histogram seeds + bands
    f.setdefault("hist_left_band", [0.18, 0.50])
    f.setdefault("hist_right_band", [0.50, 0.82])
    f.setdefault("hist_min_peak", 60)
    f.setdefault("hist_smooth_ksize", 31)  # odd

    # dashed-friendly tracking
    f.setdefault("max_gap_windows", 3)
    f.setdefault("min_windows_hit", 0.35)
    f.setdefault("min_total_pixels", 400)

    # prior tracking
    f.setdefault("use_prior_track", True)
    f.setdefault("track_margin", 70)

    # curve-slider style per-window sanity
    f.setdefault("max_pixel_inside", 1500)     # reject windows that grab too much (barriers)
    f.setdefault("max_width_not_a_line", 160)  # px spread inside a window to still call it a line
    f.setdefault("min_pixel_confindex", 12)    # window counted as "hit" for confindex
    f.setdefault("thd_confindex", 30)          # % windows hit to label as "Dashed" vs "Solid"

    # residual & confidence shaping (compatible with your earlier scheme)
    f.setdefault("norm_pixel_count", 2000)
    f.setdefault("norm_residual", 150.0)
    f.setdefault("norm_coeff_delta", 80.0)
    f.setdefault("conf_a", 4.5)
    f.setdefault("conf_b", 1.9)
    f.setdefault("conf_c", 1.0)

    # optional robust refit
    f.setdefault("use_ransac_refit", True)
    f.setdefault("ransac_iters", 200)
    f.setdefault("ransac_resid_thresh", 6.0)

    # expected lane width rescue (px)
    f.setdefault("expected_lane_width_px", 400)

    # optional Hough fallback (cheap bootstrap)
    f.setdefault("hough_fallback", False)

    return f


# -----------------------------
# Numeric helpers (stable polyfit)
# -----------------------------

def _polyfit_power_normalized(y, x, order=2):
    """
    Fit x = P(y) by first normalizing y into z=(y - y0)/s in [-1,1],
    then convert back to power basis coefficients for np.polyval.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.size == 0 or x.size == 0:
        return None

    ymin = float(np.min(y)); ymax = float(np.max(y))
    if ymax == ymin:
        # degenerate vertical extent -> constant fit
        return np.array([0.0, float(np.mean(x))]) if order >= 1 else np.array([float(np.mean(x))])

    y0 = 0.5 * (ymin + ymax)
    s  = 0.5 * (ymax - ymin)
    z  = (y - y0) / s

    # fit in z; suppress local warnings only
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', Warning)
        c_desc = np.polyfit(z, x, order)  # descending powers in z

    # convert z-polynomial to y-polynomial (power basis)
    n = len(c_desc) - 1
    # back-convert: x = sum_k c_k * ((y - y0)/s)^k
    # expand (y - y0)^k via binomial; accumulate as power of y
    c_asc = c_desc[::-1]  # ascending powers in z
    a_asc = np.zeros(n + 1, dtype=np.float64)
    # precompute powers of (-y0) for speed
    from math import comb
    for k in range(n + 1):
        ck_over = c_asc[k] / (s ** k)
        for j in range(k + 1):
            a_asc[j] += ck_over * comb(k, j) * ((-y0) ** (k - j))
    a_desc = a_asc[::-1]
    return a_desc


def _fit_poly_from_points_stable(y, x, order=2):
    if len(x) < order + 1:
        return None
    return _polyfit_power_normalized(y, x, order=order)


def _residual_mse(y, x, coeffs):
    if coeffs is None or len(y) == 0:
        return np.inf
    pred = np.polyval(coeffs, y)
    return float(np.mean((pred - x) ** 2))


# -----------------------------
# Confidence shaping
# -----------------------------

def _confidence(pixel_count, residual, coeff_delta, cfg_fit):
    n_pix = min(pixel_count, cfg_fit["norm_pixel_count"]) / float(cfg_fit["norm_pixel_count"])
    res   = min(residual, cfg_fit["norm_residual"]) / float(cfg_fit["norm_residual"])
    delt  = min(coeff_delta, cfg_fit["norm_coeff_delta"]) / float(cfg_fit["norm_coeff_delta"])
    a, b, c = cfg_fit["conf_a"], cfg_fit["conf_b"], cfg_fit["conf_c"]
    score = a * n_pix - b * res - c * delt
    # sigmoid
    return float(1. / (1. + np.exp(-score)))


def _coeff_delta(cur, prev):
    if cur is None or prev is None:
        return 1e3
    n = max(len(cur), len(prev))
    c = np.pad(cur, (n - len(cur), 0), mode='constant')
    p = np.pad(prev, (n - len(prev), 0), mode='constant')
    return float(np.linalg.norm(c - p))


# -----------------------------
# Histogram seeds
# -----------------------------

def _histogram_peaks(binary_warped: np.ndarray,
                     left_band=(0.40, 0.50),
                     right_band=(0.50, 0.60),
                     min_peak=60,
                     smooth_ksize=31) -> Tuple[int, int]:
    H, W = binary_warped.shape
    bottom = binary_warped[int(H * 0.65):, :]  # lower third
    hist = (bottom // 255).sum(axis=0).astype(np.float32)
    if smooth_ksize % 2 == 0:
        smooth_ksize += 1
    hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, smooth_ksize), 0).ravel()

    def peak(lo, hi):
        lo_px, hi_px = int(max(0.0, lo) * W), int(min(1.0, hi) * W)
        if hi_px <= lo_px:
            return (lo_px + hi_px) // 2
        band = hist[lo_px:hi_px]
        if band.size == 0:
            return (lo_px + hi_px) // 2
        m = band.max()
        return lo_px + int(np.argmax(band)) if m >= min_peak else (lo_px + hi_px) // 2

    return peak(*left_band), peak(*right_band)


# -----------------------------
# Search around prior
# -----------------------------

def _search_around_poly(binary_warped: np.ndarray, coeffs: np.ndarray, margin: int):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    xfit = np.polyval(coeffs, nonzeroy)
    inds = np.where(np.abs(nonzerox - xfit) < margin)[0]
    return nonzerox, nonzeroy, inds


# -----------------------------
# Sliding windows (curve-slider style)
# -----------------------------

def _window_slide_one_curve(
    binary_warped: np.ndarray,
    seed_x: int,
    used_inds_mask: np.ndarray,
    cfg_fit: dict
) -> dict:
    """
    Slide windows from bottom to top starting at seed_x.
    Returns a dict with pixel indices, per-window hits, drawn window boxes, and a few scores.
    """
    H, W = binary_warped.shape
    nwindows = cfg_fit["nwindows"]
    margin = cfg_fit["margin"]
    minpix = cfg_fit["minpix"]
    max_pix_inside = cfg_fit["max_pixel_inside"]
    max_width = cfg_fit["max_width_not_a_line"]

    # precompute nonzeros
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # exclude pixels already assigned to other curves
    if used_inds_mask is not None and used_inds_mask.size == nonzerox.size:
        available = np.where(~used_inds_mask)[0]
        nonzerox = nonzerox[available]
        nonzeroy = nonzeroy[available]
        base_index_map = available
    else:
        base_index_map = np.arange(nonzerox.size)

    window_height = H // max(1, nwindows)
    x_current = int(seed_x)

    lane_inds_list: List[np.ndarray] = []
    window_hits: List[int] = []
    drawn_windows: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    hits = 0
    misses = 0
    for w in range(nwindows):
        win_y_low = H - (w + 1) * window_height
        win_y_high = H - w * window_height
        win_x_low = max(0, x_current - margin)
        win_x_high = min(W - 1, x_current + margin)

        good = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

        # reject windows that are too crowded or too wide -> likely barrier/vehicle edge
        take = np.array([], dtype=int)
        if good.size > 0:
            width_span = int(np.max(nonzerox[good]) - np.min(nonzerox[good]))
            if (good.size <= max_pix_inside) and (width_span <= max_width):
                take = good

        if take.size > minpix:
            lane_inds_list.append(base_index_map[take])
            x_current = int(np.mean(nonzerox[take]))
            hits += 1
            misses = 0
            drawn_windows.append(((win_x_low, win_y_low), (win_x_high, win_y_high)))
        else:
            misses += 1
            if misses > cfg_fit["max_gap_windows"]:
                # keep x_current, still climb; just record an empty window
                pass
            drawn_windows.append(((win_x_low, win_y_low), (win_x_high, win_y_high)))

        window_hits.append(take.size)

    # flatten indices
    lane_inds = np.concatenate(lane_inds_list) if len(lane_inds_list) else np.array([], dtype=int)

    # basic "confidence index" (% of windows with enough pixels)
    hits_per_window = np.array(window_hits, dtype=int)
    confindex = int(100 * np.sum(hits_per_window >= cfg_fit["min_pixel_confindex"]) / max(1, len(hits_per_window)))

    # crude line type classification (optional, not used downstream)
    if confindex >= 85:
        linetype = "Solid"
    elif confindex >= cfg_fit["thd_confindex"]:
        linetype = "Dashed"
    else:
        linetype = "No Line"

    # compute a rough signed distance at y=H-1 (negative: right of center; positive: left)
    center_x = W / 2.0
    dist_at_0 = float(seed_x - center_x)

    return {
        "inds": lane_inds,                    # indices in the *full* nonzero list
        "drawn_windows": drawn_windows,       # rectangles
        "confindex": confindex,
        "linetype": linetype,
        "dist_at_0": dist_at_0,
        "hits_windows": hits_per_window
    }


# -----------------------------
# Hough (optional bootstrap)
# -----------------------------

def _hough_bootstrap(binary_warped: np.ndarray, side: str) -> Optional[np.ndarray]:
    """
    Very small helper: try to spot a strong line segment and fit a poly2.
    Returns coeffs or None.
    """
    H, W = binary_warped.shape
    edges = cv2.Canny(binary_warped, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=40, minLineLength=40, maxLineGap=20)
    if lines is None:
        return None
    # pick the segment whose midpoint is closest to left/right third
    if side == "left":
        target_x = W * 0.33
    else:
        target_x = W * 0.67

    best = None
    best_d = 1e9
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, l)
        xm = 0.5 * (x1 + x2)
        d = abs(xm - target_x)
        if d < best_d:
            best = (x1, y1, x2, y2)
            best_d = d
    if best is None:
        return None
    x1, y1, x2, y2 = best
    y = np.array([y1, y2], dtype=float)
    x = np.array([x1, x2], dtype=float)
    return _fit_poly_from_points_stable(y, x, order=2)


# -----------------------------
# Main entry
# -----------------------------

def fit_lanes_sliding_windows(binary_warped: np.ndarray, prior, cfg: dict) -> LaneModel:
    """
    Curve-slider inspired sliding-window lane fit.
    - Seed with histogram peaks (smoothed)
    - Slide gap-tolerant windows
    - Choose best left/right candidates by closeness to center and confindex
    - Stable polynomial fit (+ optional RANSAC)
    - Optional search-around-prior rescue
    """
    cfg = _fit_cfg(cfg or {})
    H, W = binary_warped.shape
    poly_order = cfg["poly_order"]

    # 1) histogram seeding
    lb = tuple(cfg["hist_left_band"])
    rb = tuple(cfg["hist_right_band"])
    leftx_base, rightx_base = _histogram_peaks(
        binary_warped, lb, rb, cfg["hist_min_peak"], cfg["hist_smooth_ksize"]
    )

    # 2) optional rescue: if only one seed strong, infer the other using expected lane width
    def _band_evidence(xc, rng=12, y0_frac=0.80):
        y0 = int(H * y0_frac)
        x0 = max(0, int(xc - rng)); x1 = min(W, int(xc + rng))
        strip = binary_warped[y0:, x0:x1]
        return int((strip // 255).sum())

    left_ev = _band_evidence(leftx_base)
    right_ev = _band_evidence(rightx_base)
    if left_ev < 50 and right_ev >= 120:
        wpx = int(cfg["expected_lane_width_px"])
        leftx_base = int(np.clip(rightx_base - wpx, 10, W - 10))
    if right_ev < 50 and left_ev >= 120:
        wpx = int(cfg["expected_lane_width_px"])
        rightx_base = int(np.clip(leftx_base + wpx, 10, W - 10))

    # 3) multi-curve discovery like lib_curve_slider (erase histogram around a found curve)
    #    We'll just attempt two curves (left/right); you can extend if needed.
    used_inds_mask = np.zeros(binary_warped.nonzero()[0].shape[0], dtype=bool)

    cand = []
    for seed_x in [leftx_base, rightx_base]:
        c = _window_slide_one_curve(binary_warped, int(seed_x), used_inds_mask, cfg)
        # Mark used pixels to avoid re-using on the other curve
        nonzero = binary_warped.nonzero()
        base_map = np.arange(nonzero[0].shape[0])
        if c["inds"].size:
            used_inds_mask[c["inds"]] = True
        cand.append(c)

    # 4) choose left/right by x position relative to center (closest to ego center)
    center_x = W / 2.0

    def side_of_seed(seed_x):
        return "left" if seed_x < center_x else "right"

    # if seeds crossed, reassign by distance to center
    # collect candidates explicitly for each side
    cands_left, cands_right = [], []
    for c, sx in zip(cand, [leftx_base, rightx_base]):
        (cands_left if side_of_seed(sx) == "left" else cands_right).append(c)

    # pick best per side by (1) higher confindex, (2) |dist_at_0| small (near center)
    def pick_best(cands):
        if not cands:
            return None
        cands = sorted(cands, key=lambda d: (-d["confindex"], abs(d["dist_at_0"])))
        return cands[0]

    best_L = pick_best(cands_left)
    best_R = pick_best(cands_right)

    # 5) Gather pixels (with optional prior rescue)
    nonzero = binary_warped.nonzero()
    nonzeroy_all = np.array(nonzero[0])
    nonzerox_all = np.array(nonzero[1])

    def collect_points(cside, pcoeff):
        if cside is None:
            # try prior-only
            if cfg["use_prior_track"] and pcoeff is not None:
                nx, ny, inds = _search_around_poly(binary_warped, pcoeff, cfg["track_margin"])
                return ny[inds], nx[inds]
            return np.array([], dtype=int), np.array([], dtype=int)
        inds = cside["inds"]
        return nonzeroy_all[inds], nonzerox_all[inds]

    prev_left = getattr(prior.left, "coeffs", None) if prior else None
    prev_right = getattr(prior.right, "coeffs", None) if prior else None

    yl, xl = collect_points(best_L, prev_left)
    yr, xr = collect_points(best_R, prev_right)

    # prior rescue if too few points
    if yl.size < cfg["min_total_pixels"] and cfg["use_prior_track"] and prev_left is not None:
        nx, ny, inds = _search_around_poly(binary_warped, prev_left, cfg["track_margin"])
        yl = np.concatenate([yl, ny[inds]]) if inds.size else yl
        xl = np.concatenate([xl, nx[inds]]) if inds.size else xl

    if yr.size < cfg["min_total_pixels"] and cfg["use_prior_track"] and prev_right is not None:
        nx, ny, inds = _search_around_poly(binary_warped, prev_right, cfg["track_margin"])
        yr = np.concatenate([yr, ny[inds]]) if inds.size else yr
        xr = np.concatenate([xr, nx[inds]]) if inds.size else xr

    # 6) Fit polynomials (stable), RANSAC optional
    left_coeffs = _fit_poly_from_points_stable(yl, xl, order=poly_order) if yl.size else None
    right_coeffs = _fit_poly_from_points_stable(yr, xr, order=poly_order) if yr.size else None

    if cfg["use_ransac_refit"]:
        lc = _robust_polyfit_ransac(yl, xl, order=poly_order,
                                    iters=cfg["ransac_iters"], thresh=cfg["ransac_resid_thresh"])
        if lc is not None:
            left_coeffs = lc
        rc = _robust_polyfit_ransac(yr, xr, order=poly_order,
                                    iters=cfg["ransac_iters"], thresh=cfg["ransac_resid_thresh"])
        if rc is not None:
            right_coeffs = rc

    # 7) Residuals and confidence
    left_res = _residual_mse(yl, xl, left_coeffs)
    right_res = _residual_mse(yr, xr, right_coeffs)

    left_cnt = int(yl.size)
    right_cnt = int(yr.size)

    left_delta = _coeff_delta(left_coeffs, prev_left)
    right_delta = _coeff_delta(right_coeffs, prev_right)

    left_conf = _confidence(left_cnt, left_res, left_delta, cfg) if left_coeffs is not None else 0.0
    right_conf = _confidence(right_cnt, right_res, right_delta, cfg) if right_coeffs is not None else 0.0

    left_ok = left_coeffs is not None
    right_ok = right_coeffs is not None

    y_vals = np.linspace(0, H - 1, num=H)

    return LaneModel(
        left=SideFit(ok=left_ok, coeffs=left_coeffs, residual=left_res, pixel_count=left_cnt, confidence=left_conf),
        right=SideFit(ok=right_ok, coeffs=right_coeffs, residual=right_res, pixel_count=right_cnt, confidence=right_conf),
        y_vals=y_vals,
        warp_shape=(H, W)
    )


# -----------------------------
# RANSAC (stable)
# -----------------------------

def _robust_polyfit_ransac(y, x, order=2, iters=200, thresh=6.0):
    """RANSAC using the stable fitter; suppress warnings inside the tiny loops."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.size < order + 1:
        return None
    rng = np.random.default_rng(123)
    idx = np.arange(x.size)
    best_fit = None
    best_inliers = -1

    for _ in range(iters):
        if idx.size < order + 1:
            break
        samp = rng.choice(idx, size=(order + 1,), replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', Warning)
            fit = _fit_poly_from_points_stable(y[samp], x[samp], order=order)
        if fit is None:
            continue
        resid = np.abs(x - np.polyval(fit, y))
        inliers = int(np.sum(resid < thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_fit = fit

    if best_fit is None:
        return None

    resid = np.abs(x - np.polyval(best_fit, y))
    keep = resid < thresh
    if int(np.sum(keep)) >= order + 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', Warning)
            best_fit = _fit_poly_from_points_stable(y[keep], x[keep], order=order)
    return best_fit