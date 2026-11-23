from __future__ import annotations

from typing import Callable


class PRBController:

    def __init__(self, kp: float, prb_min: float, prb_max: float):
        self.kp = kp
        self.prb_min = prb_min
        self.prb_max = prb_max

    def step(self, target_mbps: float, prb_fraction: float,
             deliver: Callable[[float], float]) -> tuple[float, float]:
        delivered = deliver(prb_fraction)
        denom = target_mbps if target_mbps > 1e-6 else 1.0
        error = (target_mbps - delivered) / denom
        new_prb = prb_fraction + self.kp * error
        new_prb = max(self.prb_min, min(self.prb_max, new_prb))
        return round(new_prb, 4), round(deliver(new_prb), 2)
