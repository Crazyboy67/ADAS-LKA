from dataclasses import dataclass
import numpy as np
from lane_fit import LaneModel, SideFit

@dataclass
class TemporalConfig:
    alpha: float = 0.9  # EMA for coefficients
    keep_last_on_drop: bool = True
    min_conf_for_update: float = 0.6

class TemporalSmoother:
    def __init__(self, cfg_dict):
        self.cfg = TemporalConfig(
            alpha=cfg_dict.get("alpha", 0.8),
            keep_last_on_drop=cfg_dict.get("keep_last_on_drop", True),
            min_conf_for_update=cfg_dict.get("min_conf_for_update", 0.4),
        )

    def _smooth_coeffs(self, prev, cur):
        if prev is None:
            return cur
        if cur is None:
            return prev
        # match vector lengths by padding on left
        n = max(len(prev), len(cur))
        p = np.pad(prev, (n - len(prev), 0))
        c = np.pad(cur, (n - len(cur), 0))
        a = self.cfg.alpha
        s = a * p + (1 - a) * c
        return s

    def _smooth_side(self, prev_side: SideFit, cur_side: SideFit) -> SideFit:
        if cur_side.coeffs is None and self.cfg.keep_last_on_drop and prev_side and prev_side.coeffs is not None:
            # keep previous, but decay confidence slightly
            new_conf = max(0.0, prev_side.confidence * 0.9)
            return SideFit(ok=True, coeffs=prev_side.coeffs.copy(), residual=prev_side.residual,
                           pixel_count=prev_side.pixel_count, confidence=new_conf)

        if prev_side is None or prev_side.coeffs is None or cur_side.coeffs is None:
            return cur_side

        # Only update if current confidence is reasonable
        if cur_side.confidence < self.cfg.min_conf_for_update:
            # small nudge towards current
            coeffs = self._smooth_coeffs(prev_side.coeffs, cur_side.coeffs)
            return SideFit(ok=True, coeffs=coeffs, residual=cur_side.residual,
                           pixel_count=cur_side.pixel_count, confidence=cur_side.confidence*0.95)
        else:
            coeffs = self._smooth_coeffs(prev_side.coeffs, cur_side.coeffs)
            return SideFit(ok=True, coeffs=coeffs, residual=cur_side.residual,
                           pixel_count=cur_side.pixel_count, confidence=cur_side.confidence)

    def update(self, current: LaneModel, prev: LaneModel) -> LaneModel:
        if prev is None:
            return current
        left_s = self._smooth_side(prev.left, current.left)
        right_s = self._smooth_side(prev.right, current.right)
        return LaneModel(left=left_s, right=right_s, y_vals=current.y_vals, warp_shape=current.warp_shape)