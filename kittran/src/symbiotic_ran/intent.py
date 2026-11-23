from __future__ import annotations

_SLICE_BLURB = {
    "FWA": "fixed wireless access, grid-powered, throughput-sensitive",
    "MTC": "battery-powered telemetry sensors",
    "BE": "best-effort battery-powered IoT, cost-sensitive",
}


def slice_targets(*, sla_mbps: float, shares: dict, tenant_slice_map: dict,
                  tenants_by_id: dict) -> dict:
    out = {}
    for tid, share in shares.items():
        name = tenant_slice_map.get(tid)
        if name is None:
            continue
        share_mbps = share * sla_mbps
        tenant = tenants_by_id.get(tid)
        floor_mbps = tenant.min_acceptable_mbps if tenant is not None else 0.0
        out[name] = max(share_mbps, floor_mbps)
    return out


def render_operator_intent(
    *,
    sla_mbps: float,
    ran_off: bool,
    shares: dict,
    achievable_mbps: float,
    tenant_slice_map: dict,
    tenants_by_id: dict,
    phase_label: str,
    reason: str,
    energy_approved_ues: list | None = None,
    energy_report: dict | None = None,
) -> str:
    if ran_off:
        return (
            f"The tenants have agreed to switch the RAN OFF (phase "
            f"'{phase_label}', triggered by {reason}). The channel can deliver "
            f"only {achievable_mbps:.1f} Mbit/s at full PRB, which is not worth "
            f"operating for. Hold all transmit power at its minimum and do not "
            f"attempt to serve traffic; no slice should be favoured over another."
        )

    split = ", ".join(
        f"{tenant_slice_map[tid]} {share * 100:.0f}%"
        for tid, share in sorted(shares.items())
    )
    headroom = achievable_mbps - sla_mbps
    if headroom < 2.0:
        pressure = (
            f"The cell is capacity-tight: the agreed SLA is within "
            f"{max(0.0, headroom):.1f} Mbit/s of everything the channel can "
            f"deliver ({achievable_mbps:.1f} Mbit/s at full PRB), so expect "
            f"contention and make the priority split do real work."
        )
    else:
        pressure = (
            f"The channel can deliver {achievable_mbps:.1f} Mbit/s at full PRB "
            f"against an agreed {sla_mbps:.1f} Mbit/s, so there is "
            f"{headroom:.1f} Mbit/s of headroom; the binding constraint is what "
            f"the tenants want, not what the radio can do."
        )

    detail = []
    by_slice = slice_targets(sla_mbps=sla_mbps, shares=shares,
                             tenant_slice_map=tenant_slice_map,
                             tenants_by_id=tenants_by_id)
    targets = {tid: by_slice[tenant_slice_map[tid]] for tid in shares}
    for tid, share in sorted(shares.items(), key=lambda kv: -kv[1]):
        slice_name = tenant_slice_map[tid]
        tenant = tenants_by_id.get(tid)
        share_mbps = share * sla_mbps
        if tenant is None:
            detail.append(
                f"  - slice {slice_name} "
                f"({_SLICE_BLURB.get(slice_name, 'tenant slice')}) is owed "
                f"{share * 100:.0f}% of delivered throughput, i.e. about "
                f"{share_mbps:.1f} Mbit/s."
            )
            continue

        floor_mbps = tenant.min_acceptable_mbps
        target = targets[tid]
        which = ("the FLOOR binds" if floor_mbps > share_mbps
                 else "the share binds")
        detail.append(
            f"  - slice {slice_name} "
            f"({_SLICE_BLURB.get(slice_name, 'tenant slice')}) is owed "
            f"{share * 100:.0f}% of delivered throughput, i.e. about "
            f"{share_mbps:.1f} Mbit/s, against a contractual floor of "
            f"{floor_mbps:.1f} Mbit/s -- {which}, so TARGET "
            f"{target:.1f} Mbit/s."
        )

    ranked = [tid for tid, _ in sorted(shares.items(), key=lambda kv: -kv[1])]
    order = " > ".join(
        f"{tenant_slice_map[tid]} ({shares[tid] * 100:.0f}%)" for tid in ranked)
    standing = (
        f"Priority order agreed by the tenants this phase: {order}. The slice "
        f"with the largest agreed share is the one they decided matters most "
        f"right now, and it is not necessarily the one that mattered last phase."
    )

    required = sum(targets.values())
    if required > sla_mbps + 0.05:
        short = ", ".join(
            f"{tenant_slice_map[tid]} {targets[tid]:.1f}" for tid in ranked)
        feasibility = (
            f"WARNING: these targets sum to {required:.1f} Mbit/s against an "
            f"agreed SLA of {sla_mbps:.1f} Mbit/s ({short}). They cannot all be "
            f"met. This is a triage decision: choose which slice absorbs the "
            f"{required - sla_mbps:.1f} Mbit/s shortfall and say so explicitly, "
            f"rather than spreading it evenly and missing every target. Take it "
            f"from the SMALLEST agreed shares first and protect the largest -- "
            f"cutting the top of the order to satisfy the bottom of it inverts "
            f"the decision the tenants just made."
        )
    else:
        feasibility = (
            f"These targets sum to {required:.1f} Mbit/s against an agreed SLA "
            f"of {sla_mbps:.1f} Mbit/s, so every one of them can be met at once."
        )

    top = tenant_slice_map[ranked[0]]
    rest = ", ".join(tenant_slice_map[tid] for tid in ranked[1:])
    ladder = (
        "Only three priority levels exist (low=0.5, normal=1.0, high=3.0, "
        "shared out in proportion), so the closest available ordering divides "
        "the cell about 67/22/11 whatever was agreed above. The throttle is "
        "the only continuous lever you have, and it can only take away."
    )
    if required > sla_mbps + 0.05:
        ceiling = (
            f"{ladder} Under this shortfall the TARGET is NOT a ceiling for "
            f"{top}: its target is not reachable at all, so capping it there "
            f"only hands the surplus to the slices that were supposed to absorb "
            f"the loss. Put the caps on {rest} instead, and leave {top} "
            f"uncapped so it can take everything the ordering gives it. Do NOT "
            f"correct a slice by demoting it a rung when a cap on its "
            f"competitors would do."
        )
    else:
        ceiling = (
            f"The TARGET is a CEILING as well as a goal. {ladder} The ordering "
            f"will over-serve the slice at the top of it -- here {top}. Cap "
            f"{top} AT its TARGET and let the surplus fall through to {rest}; "
            f"trimming the top is what the throttle is for. Leave {rest} capped "
            f"well ABOVE their own targets -- they are absorbing what the trim "
            f"frees, and caps that merely sum to the SLA are scaled back up by "
            f"the guardrail, undoing the trim. Do NOT correct an over-delivering "
            f"slice by demoting its priority: the rung below is far under its "
            f"target, and that turns a small overshoot into a large shortfall."
        )

    return (
        f"The tenants have negotiated a shared SLA of {sla_mbps:.1f} Mbit/s of "
        f"total cell throughput (phase '{phase_label}', triggered by {reason}). "
        f"Delivered capacity is to be divided {split}.\n"
        f"{pressure}\n"
        f"{standing}\n"
        f"Per-slice obligations (largest first):\n" + "\n".join(detail)
        + f"\n{feasibility}\n{ceiling}" + _energy_sentence(energy_approved_ues, energy_report)
    )


def _energy_sentence(approved: list | None, report: dict | None) -> str:
    if approved:
        return (f"\nENERGY APPROVED: the tenants accept the transmit-power cost of "
                f"driving {', '.join(approved)} to their channel ceiling to keep "
                f"this SLA. Power Control must hold them there; do not ask to "
                f"hold them low to save battery.")
    if report is not None:
        return (f"\nNo extra battery power was approved: the SLA is set within the "
                f"{report['held_mbps']:.1f} Mbit/s the current Power Control "
                f"targets can carry, and battery-powered UEs may stay where they "
                f"are.")
    return ""


def evaluate_triggers(state, *, new_phase_started: bool, achievable_mbps: float,
                      neg_cfg: dict, rrm_cfg: dict, guardrails: dict,
                      slices: dict, slice_delivered_mbps: dict) -> tuple[bool, str, int]:
    exhausted = state.get("rrm_exhausted_cycles", 0)

    if new_phase_started:
        return True, "phase_change", 0

    if state["ran_off"]:
        return False, "", 0

    drift_gap = state["sla_mbps"] - achievable_mbps
    if drift_gap > neg_cfg["renegotiation_trigger_gap_mbps"]:
        return True, "sla_drift", 0

    rise_gap = neg_cfg.get("capacity_rise_trigger_mbps")
    agreed_at = state.get("negotiated_achievable_mbps")
    if (rise_gap is not None and agreed_at is not None and state["sla_mbps"] > 0
            and achievable_mbps - agreed_at > rise_gap
            and achievable_mbps - state["sla_mbps"] > rise_gap):
        return True, "capacity_rise", 0

    prb_pinned = state["prb_fraction"] >= rrm_cfg["prb_pinned_fraction"]
    short = (state["sla_mbps"] - state["delivered_mbps"]) > rrm_cfg["exhaustion_gap_mbps"]

    binding = rrm_cfg["throttle_binding_fraction"]
    throttle_binds = any(
        slice_delivered_mbps.get(name, 0.0) >= binding * sl["throttle_mbps"]
        for name, sl in slices.items()
    )

    best = state.get("best_delivered_mbps", 0.0)
    progress = state["delivered_mbps"] - best
    stalled = progress < rrm_cfg["progress_epsilon_mbps"]

    if state["sla_mbps"] > 0 and prb_pinned and short and stalled and not throttle_binds:
        exhausted += 1
        if exhausted >= rrm_cfg["exhaustion_k"]:
            return True, "rrm_exhaustion", 0
        return False, "", exhausted

    return False, "", 0
