from __future__ import annotations

from .channel import snr_db_to_spectral_efficiency

PRIORITY_WEIGHTS = {"low": 0.5, "normal": 1.0, "high": 3.0}


def ue_demand_mbps(ue: dict, bandwidth_mhz: float, prb_fraction: float) -> float:
    eff = snr_db_to_spectral_efficiency(ue["actual_snr_db"])
    return eff * bandwidth_mhz * max(0.0, min(1.0, prb_fraction))


def allocate_throughput(ues: dict, slices: dict, total_capacity_mbps: float,
                        bandwidth_mhz: float, prb_fraction: float) -> dict:
    members: dict = {}
    for ue_id, ue in ues.items():
        members.setdefault(ue["slice"], []).append(ue_id)

    raw = {ue_id: ue_demand_mbps(ue, bandwidth_mhz, prb_fraction)
           for ue_id, ue in ues.items()}

    demand = {}
    weight = {}
    for name, ue_ids in members.items():
        sl = slices[name]
        slice_raw = sum(raw[u] for u in ue_ids)
        budget = min(slice_raw, sl["throttle_mbps"])
        w = PRIORITY_WEIGHTS.get(sl["priority"], 1.0) / len(ue_ids)
        for ue_id in ue_ids:
            frac = (raw[ue_id] / slice_raw) if slice_raw > 0 else 1.0 / len(ue_ids)
            demand[ue_id] = budget * frac
            weight[ue_id] = w

    total_demand = sum(demand.values())
    if total_demand <= total_capacity_mbps or total_demand == 0:
        return demand

    total_weight = sum(weight.values()) or 1.0
    allocation = {}
    for ue_id in ues:
        fair_share = total_capacity_mbps * (weight[ue_id] / total_weight)
        allocation[ue_id] = min(demand[ue_id], max(fair_share, 0.0))

    leftover = total_capacity_mbps - sum(allocation.values())
    if leftover > 1e-9:
        needy = {u: demand[u] - allocation[u]
                 for u in ues if allocation[u] < demand[u] - 1e-6}
        needy_weight_sum = sum(weight[u] for u in needy) or 1.0
        for u, need in needy.items():
            extra = leftover * (weight[u] / needy_weight_sum)
            allocation[u] += min(need, extra)

    return allocation


def allocate_pf(ues: dict, slice_weights: dict, bandwidth_mhz: float,
                prb_fraction: float) -> dict:
    members: dict = {}
    for ue_id, ue in ues.items():
        members.setdefault(ue["slice"], []).append(ue_id)
    weight = {}
    for name, ue_ids in members.items():
        w = max(0.0, float(slice_weights.get(name, 1.0)))
        for ue_id in ue_ids:
            weight[ue_id] = w / len(ue_ids)
    total = sum(weight.values())
    if total <= 0:
        weight = {ue_id: 1.0 for ue_id in ues}
        total = float(len(ues))
    return {ue_id: (weight[ue_id] / total) * ue_demand_mbps(ue, bandwidth_mhz, prb_fraction)
            for ue_id, ue in ues.items()}


def slice_delivered_mbps(ues: dict) -> dict:
    totals: dict = {}
    for ue in ues.values():
        name = ue["slice"]
        totals[name] = totals.get(name, 0.0) + ue.get("throughput_mbps", 0.0)
    return totals


def slice_throughput(ues: dict, allocation: dict) -> dict:
    totals: dict = {}
    for ue_id, mbps in allocation.items():
        totals.setdefault(ues[ue_id]["slice"], 0.0)
        totals[ues[ue_id]["slice"]] += mbps
    return {k: round(v, 2) for k, v in totals.items()}
