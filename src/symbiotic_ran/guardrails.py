from __future__ import annotations

import re

from . import energy
from .schemas import MediatorDecision, PCDecision, ULRADecision

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers_in(text: str | None) -> set[float]:
    return {float(m) for m in _NUMBER_RE.findall(text or "")}


def _close_to_any(value: float, pool: set[float], tol: float = 0.05) -> bool:
    return any(abs(value - p) < tol for p in pool)


def clamp_pc(decision: PCDecision, ues: dict, guardrails: dict,
             energy_approved: list | None = None,
             energy_accepted: dict | None = None) -> dict:
    g = guardrails["power_control"]
    approved = set(energy_approved or ())
    accepted = energy_accepted or {}
    out = {}
    for ue_id, (proposed, reasoning) in decision.per_ue().items():
        ue = ues[ue_id]
        if ue_id in approved:
            floor = min(ue["snr_ceiling_db"], g["snr_max_db"])
            if float(proposed) < floor:
                reasoning = (f"[ENERGY APPROVED: target raised to the {floor:.1f} dB "
                             f"ceiling] {reasoning}")
                proposed = floor
        elif ue_id in accepted and ue.get("battery_pct") is not None:
            floor = min(energy.accepted_snr_db(ue, accepted[ue_id]), g["snr_max_db"])
            if float(proposed) < floor:
                reasoning = (f"[POWER ACCEPTED: target raised to {floor:.1f} dB, what the "
                             f"accepted transmit power buys] {reasoning}")
                proposed = floor
        current = ues[ue_id]["target_snr_db"]
        delta = max(-g["max_delta_db_per_cycle"],
                    min(g["max_delta_db_per_cycle"], float(proposed) - current))
        new_target = max(g["snr_min_db"], min(g["snr_max_db"], current + delta))
        out[ue_id] = (round(new_target, 2), reasoning)
    return out


def clamp_ulra(decision: ULRADecision, slices: dict, guardrails: dict,
               sla_mbps: float | None = None, trusted_text: str | None = None,
               untrusted_text: str | None = None) -> dict:
    g = guardrails["ul_ra"]
    out = {}
    trusted_numbers = _numbers_in(trusted_text) if trusted_text is not None else None
    for name, (throttle, priority, reasoning) in decision.per_slice().items():
        throttle = float(throttle)
        if throttle <= 0:
            throttle = g["throttle_max_mbps"]
        throttle = max(g["throttle_min_mbps"], min(g["throttle_max_mbps"], throttle))
        if priority not in g["allowed_priorities"]:
            priority = slices[name]["priority"]
        if trusted_numbers is not None:
            desc_numbers = _numbers_in(slices.get(name, {}).get("description")
                                       if untrusted_text is None else untrusted_text)
            untraced = {n for n in desc_numbers if not _close_to_any(n, trusted_numbers)}
            if untraced and _close_to_any(throttle, untraced):
                reasoning = (
                    f"[BLOCKED: {throttle:.1f} Mbit/s traces only to the slice's "
                    f"own description, not the operator intent -- kept the "
                    f"current throttle] {reasoning}")
                throttle = slices[name]["throttle_mbps"]
        out[name] = (round(throttle, 2), priority, reasoning)
    return _restore_throttle_headroom(out, g, sla_mbps)


def _restore_throttle_headroom(applied: dict, g: dict,
                               sla_mbps: float | None) -> dict:
    headroom = g.get("throttle_sum_headroom")
    if not headroom or not sla_mbps or sla_mbps <= 0 or not applied:
        return applied

    total = sum(throttle for throttle, _, _ in applied.values())
    target = sla_mbps * headroom
    if total <= 0 or total >= target:
        return applied

    scale = target / total
    return {
        name: (round(min(g["throttle_max_mbps"], throttle * scale), 2),
               priority, reasoning)
        for name, (throttle, priority, reasoning) in applied.items()
    }


def clamp_diff(proposed: dict, applied: dict, *, tolerance: float = 1e-6) -> dict:
    out = {}
    for key, before in proposed.items():
        after = applied.get(key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            changed = abs(float(before) - float(after)) > tolerance
        else:
            changed = before != after
        if changed:
            out[key] = {"proposed": before, "applied": after}
    return out


def floor_report(shares: dict, sla_mbps: float, floors: dict) -> dict:
    per_tenant = {tid: shares.get(tid, 0.0) * sla_mbps for tid in floors}
    short = {tid: round(floors[tid] - got, 2)
             for tid, got in per_tenant.items() if got < floors[tid] - 0.05}
    required = sum(floors.values())
    return {
        "floors_required_mbps": round(required, 2),
        "floors_fit_in_sla": required <= sla_mbps + 0.05,
        "tenants_below_floor": sorted(short),
        "shortfall_mbps": {tid: short[tid] for tid in sorted(short)},
    }


def honour_floors(shares: dict, sla_mbps: float, floors: dict,
                  min_share: float = 0.02) -> dict:
    if sla_mbps <= 0 or not floors:
        return shares
    if sum(floors.values()) > sla_mbps + 0.05:
        return shares

    need = {tid: floors[tid] / sla_mbps for tid in floors}
    lifted = dict(shares)
    deficit = 0.0
    for tid, want in need.items():
        have = lifted.get(tid, 0.0)
        if have < want:
            deficit += want - have
            lifted[tid] = want
    if deficit <= 0.0:
        return shares

    surplus = {tid: lifted[tid] - need.get(tid, 0.0) for tid in lifted}
    donors = {tid: v for tid, v in surplus.items() if v > 0}
    pool = sum(donors.values())
    if pool <= 0:
        return shares
    for tid, have in donors.items():
        lifted[tid] -= deficit * (have / pool)

    total = sum(lifted.values())
    return {tid: v / total for tid, v in lifted.items()} if total > 0 else shares


def clamp_negotiation(sla_mbps: float, ran_off: bool, shares: dict,
                      achievable_mbps: float, guardrails: dict,
                      tenant_ids=None, floors: dict | None = None
                      ) -> tuple[float, bool, dict]:
    g = guardrails.get("negotiation", {})
    sla = float(sla_mbps)
    if g.get("clamp_sla_to_achievable", True):
        sla = max(0.0, min(achievable_mbps, sla))
    if tenant_ids is not None:
        known = set(tenant_ids)
        shares = {k: v for k, v in shares.items() if k in known}
    min_share = g.get("min_share_fraction", 0.02)
    out = normalise_shares(shares, min_share)
    if floors and not ran_off:
        required = min(sum(floors.values()), achievable_mbps)
        sla = max(sla, required)
        out = honour_floors(out, round(sla, 2), floors, min_share)
    return round(sla, 2), bool(ran_off), out


def clamp_mediator_decision(decision: MediatorDecision, achievable_mbps: float,
                            guardrails: dict,
                            tenant_ids=None) -> tuple[float, bool, dict]:
    return clamp_negotiation(
        decision.suggested_sla_mbps, decision.switch_ran_off, decision.shares(),
        achievable_mbps, guardrails, tenant_ids=tenant_ids)


def normalise_shares(shares: dict, min_share: float = 0.02) -> dict:
    clean = {k: max(0.0, float(v)) for k, v in shares.items()}
    n = len(clean)
    if n == 0:
        return {}
    total = sum(clean.values())
    if total <= 0:
        clean = {k: 1.0 / n for k in clean}
    else:
        clean = {k: v / total for k, v in clean.items()}

    if min_share > 0 and min_share * n < 1.0:
        room = 1.0 - min_share * n
        clean = {k: min_share + room * v for k, v in clean.items()}

    rounded = {k: round(v, 4) for k, v in clean.items()}
    residual = round(1.0 - sum(rounded.values()), 4)
    if residual:
        biggest = max(rounded, key=lambda k: rounded[k])
        rounded[biggest] = round(rounded[biggest] + residual, 4)
    return rounded
