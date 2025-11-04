# src/metrics.py
# Metrics for classical LKA: curve error, detection accuracy, stability.
# No external deps beyond numpy + stdlib.

from __future__ import annotations
import argparse
import csv
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Polynomial helpers (x = a*y^2 + b*y + c)
# -----------------------------
def poly_eval_x(coeffs: Tuple[float, float, float], y: np.ndarray) -> np.ndarray:
    a, b, c = coeffs
    return a * y * y + b * y + c

def residuals_x(
    xs: np.ndarray, ys: np.ndarray, coeffs: Tuple[float, float, float]
) -> np.ndarray:
    """Horizontal residuals in pixels: x_obs - x_fit(y)."""
    return xs - poly_eval_x(coeffs, ys)

def residual_stats(resid: np.ndarray) -> Dict[str, float]:
    resid = resid.astype(np.float64)
    if resid.size == 0:
        return {"rmse": math.nan, "mae": math.nan, "max": math.nan}
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    mxx = float(np.max(np.abs(resid)))
    return {"rmse": rmse, "mae": mae, "max": mxx}

def curvature_px(coeffs: Tuple[float, float, float], y: float) -> float:
    """
    Curvature kappa(y) in pixel units for x=f(y).
    kappa = |2a| / (1 + (2a*y + b)^2)^(3/2)
    """
    a, b, _ = coeffs
    denom = (1.0 + (2.0 * a * y + b) ** 2) ** 1.5
    return abs(2.0 * a) / max(denom, 1e-9)


# -----------------------------
# Curve error (use online or offline)
# -----------------------------
@dataclass
class CurveError:
    rmse: float
    mae: float
    max_abs: float
    n_points: int

def compute_curve_error(
    xs: np.ndarray,
    ys: np.ndarray,
    coeffs: Tuple[float, float, float],
) -> CurveError:
    """
    Compute horizontal residual stats between inlier lane pixels (xs, ys)
    and polynomial x=f(y) in the same (BEV) pixel space.
    """
    if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
        return CurveError(math.nan, math.nan, math.nan, 0)
    r = residuals_x(xs, ys, coeffs)
    stats = residual_stats(r)
    return CurveError(stats["rmse"], stats["mae"], stats["max"], int(xs.size))


# -----------------------------
# Detection accuracy and stability from CSV
# -----------------------------
@dataclass
class DetectionMetrics:
    frames: int
    tau: float
    left_rate: float
    right_rate: float
    left_mean_conf: float
    right_mean_conf: float
    left_runs_median: float
    right_runs_median: float
    left_dropouts_per_1k: float
    right_dropouts_per_1k: float

@dataclass
class StabilityMetrics:
    # lateral offset stability (meters)
    offset_mean_m: float
    offset_std_m: float
    offset_abs_diff_med_m: float
    offset_abs_diff_p95_m: float
    # optional polynomial coefficient jitter (if columns exist)
    have_poly_coeffs: bool
    coeff_jitter_left: Optional[Dict[str, float]] = None
    coeff_jitter_right: Optional[Dict[str, float]] = None

def _run_lengths(bits: List[int]) -> List[int]:
    """Lengths of consecutive 1-runs in a 0/1 list."""
    runs: List[int] = []
    cur = 0
    for b in bits:
        if b:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    return runs

def _dropouts(bits: List[int]) -> int:
    """Count 1->0 transitions (end of a detected run)."""
    cnt = 0
    prev = bits[0] if bits else 0
    for b in bits[1:]:
        if prev == 1 and b == 0:
            cnt += 1
        prev = b
    return cnt

def detection_metrics_from_csv(csv_path: str, tau: float = 0.6) -> DetectionMetrics:
    left_det: List[int] = []
    right_det: List[int] = []
    left_conf: List[float] = []
    right_conf: List[float] = []

    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # If your CSV already has 0/1 flags, we can trust them, otherwise derive from conf
            if "left_detected" in row and "right_detected" in row:
                ld = int(float(row["left_detected"]))
                rd = int(float(row["right_detected"]))
            else:
                ld = 1 if float(row["left_conf"]) > tau else 0
                rd = 1 if float(row["right_conf"]) > tau else 0

            left_det.append(ld)
            right_det.append(rd)
            left_conf.append(float(row.get("left_conf", "0.0")))
            right_conf.append(float(row.get("right_conf", "0.0")))

    n = len(left_det)
    if n == 0:
        return DetectionMetrics(0, tau, 0, 0, 0, 0, 0, 0, 0.0, 0.0)

    # Rates & mean conf
    l_rate = float(sum(left_det) / n)
    r_rate = float(sum(right_det) / n)
    l_mconf = float(np.mean(left_conf)) if left_conf else 0.0
    r_mconf = float(np.mean(right_conf)) if right_conf else 0.0

    # Continuity: median run length of detected streaks
    l_runs = _run_lengths(left_det)
    r_runs = _run_lengths(right_det)
    l_runs_med = float(np.median(l_runs)) if l_runs else 0.0
    r_runs_med = float(np.median(r_runs)) if r_runs else 0.0

    # Dropouts per 1k frames
    l_dropouts = _dropouts(left_det)
    r_dropouts = _dropouts(right_det)
    norm = 1000.0 / max(n, 1)
    l_drop_rate = float(l_dropouts * norm)
    r_drop_rate = float(r_dropouts * norm)

    return DetectionMetrics(
        frames=n, tau=tau,
        left_rate=l_rate, right_rate=r_rate,
        left_mean_conf=l_mconf, right_mean_conf=r_mconf,
        left_runs_median=l_runs_med, right_runs_median=r_runs_med,
        left_dropouts_per_1k=l_drop_rate, right_dropouts_per_1k=r_drop_rate,
    )

def _safe_float_series(rows: List[dict], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        if key in row and row[key] not in (None, "", "nan"):
            try:
                out.append(float(row[key]))
            except Exception:
                pass
    return out

def stability_from_csv(csv_path: str) -> StabilityMetrics:
    rows: List[dict] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    offsets = _safe_float_series(rows, "lat_offset_m")
    if offsets:
        off = np.array(offsets, dtype=np.float64)
        off_mean = float(np.mean(off))
        off_std = float(np.std(off))
        d = np.diff(off)
        ad = np.abs(d)
        off_abs_diff_med = float(np.median(ad)) if ad.size else 0.0
        off_abs_diff_p95 = float(np.percentile(ad, 95)) if ad.size else 0.0
    else:
        off_mean = off_std = off_abs_diff_med = off_abs_diff_p95 = math.nan

    # Optional: coefficient jitter if you logged them in CSV
    keysL = ("left_a", "left_b", "left_c")
    keysR = ("right_a", "right_b", "right_c")
    haveL = all(k in rows[0] for k in keysL) if rows else False
    haveR = all(k in rows[0] for k in keysR) if rows else False

    coeffL = {k: _safe_float_series(rows, k) for k in keysL} if haveL else {}
    coeffR = {k: _safe_float_series(rows, k) for k in keysR} if haveR else {}

    def jitter(coeff: Dict[str, List[float]]) -> Dict[str, float]:
        if not coeff:
            return {}
        out: Dict[str, float] = {}
        for k, v in coeff.items():
            if len(v) >= 2:
                dv = np.diff(np.array(v, dtype=np.float64))
                out[f"{k}_std"] = float(np.std(dv))
                out[f"{k}_p95"] = float(np.percentile(np.abs(dv), 95))
            else:
                out[f"{k}_std"] = math.nan
                out[f"{k}_p95"] = math.nan
        return out

    return StabilityMetrics(
        offset_mean_m=off_mean,
        offset_std_m=off_std,
        offset_abs_diff_med_m=off_abs_diff_med,
        offset_abs_diff_p95_m=off_abs_diff_p95,
        have_poly_coeffs=(haveL and haveR),
        coeff_jitter_left=jitter(coeffL) if haveL else None,
        coeff_jitter_right=jitter(coeffR) if haveR else None,
    )


# -----------------------------
# CLI
# -----------------------------
def _fmt_pct(x: float) -> str:
    return "nan" if math.isnan(x) else f"{100.0 * x:.1f}%"

def _fmt(x: float, nd=3) -> str:
    return "nan" if math.isnan(x) else f"{x:.{nd}f}"

def main():
    ap = argparse.ArgumentParser("metrics")
    ap.add_argument("--csv", required=True, help="per_frame.csv path")
    ap.add_argument("--tau", type=float, default=0.6, help="confidence threshold")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()

    det = detection_metrics_from_csv(args.csv, tau=args.tau)
    stab = stability_from_csv(args.csv)

    if args.json:
        import json
        print(json.dumps({"detection": asdict(det), "stability": asdict(stab)}, indent=2))
        return

    # Pretty text
    print(f"Frames: {det.frames} | τ={det.tau}")
    print(f"Detection rate L/R:  {_fmt_pct(det.left_rate)} / {_fmt_pct(det.right_rate)}")
    print(f"Mean conf     L/R:   {_fmt(det.left_mean_conf)} / {_fmt(det.right_mean_conf)}")
    print(f"Median run    L/R:   {det.left_runs_median:.1f} / {det.right_runs_median:.1f} frames")
    print(f"Dropouts/1k   L/R:   {det.left_dropouts_per_1k:.1f} / {det.right_dropouts_per_1k:.1f}")
    print()
    print(f"Lateral offset mean/std [m]:  {_fmt(stab.offset_mean_m)} / {_fmt(stab.offset_std_m)}")
    print(f"|Δ offset| median/p95  [m]:   {_fmt(stab.offset_abs_diff_med_m)} / {_fmt(stab.offset_abs_diff_p95_m)}")
    if stab.have_poly_coeffs:
        print("\nCoeff jitter (Δ per frame; pixels):")
        print("  Left :", stab.coeff_jitter_left)
        print("  Right:", stab.coeff_jitter_right)

if __name__ == "__main__":
    main()
