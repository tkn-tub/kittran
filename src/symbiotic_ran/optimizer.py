from __future__ import annotations

from typing import List, Tuple

from .config import Tenant


def weighted_target_mbps(
    tenants: List[Tenant],
    intents: dict,
    achievable_mbps: float,
) -> float:
    num = 0.0
    den = 0.0
    for t in tenants:
        w = t.utility_weight * t.willingness_to_pay
        num += w * intents[t.id]
        den += w
    target = num / den if den > 0 else 0.0
    target = max(target, sum(t.min_acceptable_mbps for t in tenants))
    return min(target, achievable_mbps)


def confidence_interval(
    center_mbps: float,
    half_width_mbps: float,
    achievable_mbps: float,
) -> Tuple[float, float]:
    low = max(0.0, center_mbps - half_width_mbps)
    high = min(achievable_mbps, center_mbps + half_width_mbps)
    if high < low:
        high = low
    return round(low, 2), round(high, 2)


def compute_interval(
    tenants: List[Tenant],
    intents: dict,
    achievable_mbps: float,
    half_width_mbps: float,
) -> Tuple[float, Tuple[float, float]]:
    center = weighted_target_mbps(tenants, intents, achievable_mbps)
    interval = confidence_interval(center, half_width_mbps, achievable_mbps)
    return round(center, 2), interval
