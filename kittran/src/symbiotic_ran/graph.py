from __future__ import annotations

from copy import deepcopy
from typing import List

from langgraph.graph import END, StateGraph

from . import channel as ch
from . import energy, security
from . import scheduler as sched
from .agents.backends import drain_call_log, timing_summary
from .config import Tenant, append_jsonl, apply_phase_overrides
from .controller import PRBController
from .guardrails import (
    clamp_diff,
    clamp_negotiation,
    clamp_pc,
    clamp_ulra,
    floor_report,
)
from .intent import evaluate_triggers, render_operator_intent, slice_targets
from .negotiation import DEFAULT_MAX_PARALLEL_TENANTS, default_shares, run_negotiation
from .reflection import make_mock_reviser, reflect_and_refine
from .schemas import PCDecision, SimState

RRM_MODES = ("tiered", "no-l2", "monolithic", "pf")


def tag_untrusted(text: str) -> str:
    return security.render_untrusted(text)


def _view(state: SimState) -> dict:
    return state.get("reported_ues") or state["ues"]


def _status(ue: dict) -> str:
    text = tag_untrusted(ue.get("status_text"))
    return f", status: {text}" if text else ""


def _desc(sl: dict) -> str:
    return tag_untrusted(sl.get("description"))


def build_context_l2(state: SimState) -> str:
    lines = [
        f"OPERATOR INTENT:\n{state['operator_intent']}",
        "",
        f"Phase '{state['phase_label']}', control cycle {state['cycle']}.",
        f"Cell: {state['cell']['bandwidth_mhz']} MHz, PRB allocation "
        f"{state['prb_fraction'] * 100:.0f}%, delivering "
        f"{state['delivered_mbps']:.1f} Mbit/s of a {state['sla_mbps']:.1f} "
        f"Mbit/s SLA. Achievable at full PRB: {state['achievable_mbps']:.1f} Mbit/s.",
        "",
        "Current per-UE state:",
    ]
    for ue_id, ue in _view(state).items():
        batt = "grid" if ue["battery_pct"] is None else f"{ue['battery_pct']:.1f}%"
        headroom = ue["snr_ceiling_db"] - ue["actual_snr_db"]
        lines.append(
            f"- {ue_id} (slice={ue['slice']}): headroom to its channel ceiling "
            f"{headroom:+.1f} dB, mcs={ue['mcs_index']}, "
            f"throughput={ue['throughput_mbps']} Mbit/s, battery={batt}{_status(ue)}")
    lines.append("")
    lines.append("Current per-slice state:")
    for name, sl in state["slices"].items():
        lines.append(
            f"- {name}: throttle={sl['throttle_mbps']} Mbit/s, "
            f"priority={sl['priority']}" + (f" ({_desc(sl)})" if _desc(sl) else ""))
    return "\n".join(lines)


def _pc_gap_line(state: SimState) -> str:
    if state["ran_off"]:
        return "The RAN is off: hold every target SNR at its minimum."
    gap = state["sla_mbps"] - state["delivered_mbps"]
    if state["sla_mbps"] <= 0:
        return "No SLA is in force."
    if gap <= 0:
        return ("The SLA is being MET. Hold targets steady unless the sub-intent "
                "asks otherwise, and prefer lower targets on battery-powered UEs.")
    headroom = {
        ue_id: round(ue["snr_ceiling_db"] - ue["actual_snr_db"], 2)
        for ue_id, ue in _view(state).items()
    }
    worst = sorted(headroom, key=lambda k: -headroom[k])
    if headroom[worst[0]] < 0.25:
        return (f"SHORTFALL of {gap:.1f} Mbit/s, but every UE is already at its "
                f"ceiling. Power Control cannot close this gap; hold steady.")
    ranked = ", ".join(f"{u} {headroom[u]:+.1f} dB" for u in worst)
    return (f"SHORTFALL: the cell is {gap:.1f} Mbit/s BELOW its SLA. You must "
            f"raise at least one target SNR this cycle -- repeating last "
            f"cycle's targets leaves the gap open. Headroom to ceiling, "
            f"largest first: {ranked}.")


def _energy_line(state: SimState) -> list:
    approved = state.get("energy_approved_ues") or []
    lines = []
    if approved:
        lines.append(f"ENERGY APPROVED by the tenants: {', '.join(approved)} must be "
                     f"driven to their channel ceiling. The guardrail enforces this; "
                     f"do not hold them low to save battery.")
    accepted = {ue_id: db for ue_id, db in (state.get("energy_accepted_db") or {}).items()
                if ue_id not in approved and ue_id in state["ues"]}
    if accepted:
        floors = ", ".join(
            f"{ue_id} >= {energy.accepted_snr_db(state['ues'][ue_id], db):.1f} dB"
            for ue_id, db in sorted(accepted.items()))
        lines.append(f"POWER ACCEPTED by the tenants: the SLA counts these battery UEs "
                     f"at the transmit power already accepted, which on this channel "
                     f"gives {floors}. The guardrail raises any lower target to it.")
    return lines


def build_context_pc(state: SimState, guardrails: dict) -> str:
    g = guardrails["power_control"]
    lines = [
        f"Current sub-intent for Power Control: {state['pc_subintent']}",
        "",
        f"Guardrails: target_snr_db must stay within [{g['snr_min_db']}, "
        f"{g['snr_max_db']}] dB, and any requested change will be clamped to "
        f"+/-{g['max_delta_db_per_cycle']} dB per cycle.",
        "",
        f"Cell is delivering {state['delivered_mbps']:.1f} Mbit/s against a "
        f"{state['sla_mbps']:.1f} Mbit/s SLA "
        f"({'RAN IS SWITCHED OFF' if state['ran_off'] else 'RAN on'}).",
        _pc_gap_line(state),
        *_energy_line(state),
        "",
        "Per-UE state (ceiling is the best SNR the channel can give this UE in "
        "the current phase -- targeting above it achieves nothing):",
    ]
    for ue_id, ue in _view(state).items():
        batt = "grid-powered" if ue["battery_pct"] is None else (
            f"battery {ue['battery_pct']:.1f}%")
        lines.append(
            f"- {ue_id} (slice={ue['slice']}): current_target={ue['target_snr_db']} dB, "
            f"actual={ue['actual_snr_db']} dB, CEILING={ue['snr_ceiling_db']} dB, "
            f"mcs={ue['mcs_index']}, throughput={ue['throughput_mbps']} Mbit/s, "
            f"tx_power={ue['tx_power_w']} W, {batt}{_status(ue)}")
    return "\n".join(lines)


def build_context_ulra(state: SimState, guardrails: dict) -> str:
    g = guardrails["ul_ra"]
    shares = state.get("tenant_shares") or {}
    lines = [
        f"Current sub-intent for UL Resource Allocation: {state['ul_ra_subintent']}",
        "",
        f"Guardrails: throttle_mbps must be within [{g['throttle_min_mbps']}, "
        f"{g['throttle_max_mbps']}]; priority must be one of "
        f"{g['allowed_priorities']}. A throttle of 0 or below is interpreted as "
        f"UNLIMITED and mapped to the maximum.",
        "",
        f"Cell capacity is finite and currently contended: delivering "
        f"{state['delivered_mbps']:.1f} Mbit/s against a {state['sla_mbps']:.1f} "
        f"Mbit/s SLA, {state['achievable_mbps']:.1f} Mbit/s achievable at full PRB, "
        f"PRB at {state['prb_fraction'] * 100:.0f}%.",
    ]
    if shares:
        targets = state.get("slice_targets") or {}
        lines.append("")
        lines.append(
            "Per-slice obligations. The TARGET is the number the operator "
            "intent gave you -- the larger of the agreed share and the "
            "tenant's contractual floor -- and it is what you are judged on. "
            "The percentage is the agreed ordering, not the obligation:"
            if targets else
            "Share vector agreed by the tenants (target division of delivered "
            "capacity):")
        total = state["delivered_mbps"]
        for tid, frac in sorted(shares.items()):
            name = _slice_of(state, tid)
            got = sum(ue["throughput_mbps"] for ue in state["ues"].values()
                      if ue["slice"] == name)
            actual = (f", currently {got:.1f} Mbit/s ({got / total * 100:.0f}%)"
                      if total > 0 else "")
            target = targets.get(name)
            if target is None:
                verdict = ""
            elif got < target - TARGET_TOLERANCE_MBPS:
                verdict = f" -- {target - got:.1f} Mbit/s SHORT of its target"
            elif got > target + TARGET_TOLERANCE_MBPS:
                verdict = f" -- {got - target:.1f} Mbit/s OVER its target"
            else:
                verdict = " -- on target"
            aim = (f"TARGET {target:.1f} Mbit/s" if target is not None
                   else f"about {frac * state['sla_mbps']:.1f} Mbit/s of the SLA")
            lines.append(f"- {tid} -> slice {name}: {aim}, agreed share "
                         f"{frac * 100:.0f}%{actual}{verdict}")

        over = sorted(
            ((name, t) for name, t in targets.items()
             if _slice_delivered(state, name) > t + TARGET_TOLERANCE_MBPS),
            key=lambda kv: kv[1] - _slice_delivered(state, kv[0]))
        under = [name for name, t in targets.items()
                 if _slice_delivered(state, name) < t - TARGET_TOLERANCE_MBPS]
        bandwidth = (state.get("cell") or {}).get("bandwidth_mhz")
        unused = 0.0
        if bandwidth:
            capacity_now = ch.cell_capacity_mbps(
                state["ues"], bandwidth, state.get("prb_fraction", 1.0))
            unused = max(0.0, capacity_now - state["delivered_mbps"])
        cell_short = (state["sla_mbps"] > 0 and not state.get("ran_off")
                      and state["delivered_mbps"] < state["sla_mbps"] - TARGET_TOLERANCE_MBPS)
        capped = [name for name, sl in state["slices"].items()
                  if name in targets and name not in under
                  and _slice_delivered(state, name) >= 0.95 * sl["throttle_mbps"]]
        if targets and cell_short and capped and under and unused > 1.0:
            gap = state["sla_mbps"] - state["delivered_mbps"]
            lines.append(
                f"CAP HOLDING BACK THE CELL: the cell is {gap:.1f} Mbit/s BELOW "
                f"its SLA and {unused:.1f} Mbit/s of capacity is unused, while "
                f"{', '.join(capped)} is held at its cap. {', '.join(under)} "
                f"cannot use that capacity: its UEs are limited by their "
                f"channel, not by the split. RAISE {', '.join(capped)}'s "
                f"THROTTLE by at least {gap:.1f} Mbit/s this cycle. The agreed "
                f"split matters, but not more than the SLA the whole cell "
                f"promised; restore the cap once the cell meets its SLA again.")
        elif targets and not over and under:
            lines.append(
                f"SHORTFALL: {len(under)} slice(s) are under target and NONE is "
                f"over -- there is no surplus to move. Moving a slice up the "
                f"ordering here only takes the same shortfall from a different "
                f"tenant. Follow the operator intent's triage instruction and "
                f"hold; do not demote a slice that is already short.")
        elif over and not (cell_short and unused > 1.0):
            worst, worst_target = over[0]
            lines.append(
                f"MISALLOCATION: slice {worst} is "
                f"{_slice_delivered(state, worst) - worst_target:.1f} Mbit/s over "
                f"its {worst_target:.1f} Mbit/s target while "
                f"{', '.join(under) or 'no slice'} is under its own. LOWER "
                f"{worst}'s THROTTLE toward its target -- that is the lever that "
                f"moves the exact amount. Demoting it a priority rung moves it "
                f"from about 67% of the cell to about 20%, far past the target, "
                f"and next cycle has to undo it.")
        elif share_error(state) > 0.05:
            lines.append(
                f"Every slice is on its target, but the split is still "
                f"{share_error(state) * 100:.0f} percentage points from the "
                f"agreed share. Three priority levels cannot hit an arbitrary "
                f"split: if the ordering is right, say so and hold.")
    lines.append("")
    lines.append("Current per-slice state. Priority weight and throttle both "
                 "belong to the SLICE: the weight is shared out among its UEs, "
                 "and the throttle caps the slice's total, so neither depends "
                 "on how many UEs a slice holds.")
    delivered = state["delivered_mbps"] or 0.0
    for name, sl in state["slices"].items():
        members = [u for u, ue in state["ues"].items() if ue["slice"] == name]
        got = sum(state["ues"][u]["throughput_mbps"] for u in members)
        share = f", {got / delivered * 100:.0f}% of delivered" if delivered > 0 else ""
        lines.append(
            f"- {name}: throttle={sl['throttle_mbps']} Mbit/s, "
            f"priority={sl['priority']}, {len(members)} UE(s) {members}, "
            f"currently delivering {got:.1f} Mbit/s{share}"
            + (f" -- {_desc(sl)}" if _desc(sl) else ""))
    return "\n".join(lines)


TARGET_TOLERANCE_MBPS = 0.5


def _slice_delivered(state: SimState, name: str) -> float:
    return sum(ue["throughput_mbps"] for ue in state["ues"].values()
               if ue["slice"] == name)


def _slice_of(state: SimState, tenant_id: str) -> str:
    return state["cell"].get("tenant_slice_map", {}).get(tenant_id, "?")

def share_error(state: SimState) -> float:
    shares = state.get("tenant_shares") or {}
    total = state["delivered_mbps"]
    if not shares or total <= 0 or state["ran_off"]:
        return 0.0
    worst = 0.0
    for tid, target in shares.items():
        name = _slice_of(state, tid)
        got = sum(ue["throughput_mbps"] for ue in state["ues"].values()
                  if ue["slice"] == name)
        worst = max(worst, abs(got / total - target))
    return worst


def snapshot(state: SimState, step: int, delivered: float, prb: float,
             capacity: float) -> dict:
    return {
        "step": step,
        "cycle": state["cycle"],
        "phase": state["phase"],
        "label": state["phase_label"],
        "base_mcs_index": state["base_mcs_index"],
        "sla_mbps": state["sla_mbps"],
        "achievable_mbps": state["achievable_mbps"],
        "capacity_mbps": round(capacity, 2),
        "delivered_mbps": round(delivered, 2),
        "prb_fraction": prb,
        "ran_off": state["ran_off"],
        "tenant_shares": ({} if state["ran_off"]
                          else dict(state.get("tenant_shares") or {})),
        "slice_targets": ({} if state["ran_off"]
                          else dict(state.get("slice_targets") or {})),
        "ues": deepcopy(state["ues"]),
        "slices": deepcopy(state["slices"]),
    }


def mark_unresolved(record: dict) -> dict:
    trace = record.get("reflection") or []
    if trace and trace[-1].get("unresolved"):
        return {**record, "reflection_unresolved": True}
    return record


def build_graph(
    tenants: List[Tenant],
    ran: dict,
    scenario: dict,
    cfg: dict,
    guardrails: dict,
    tenant_slice_map: dict,
    agents,
    *,
    verbose: bool = True,
    backend: str = "unknown",
    reflect_roles: frozenset = frozenset(),
    max_reflections: int = 2,
    decisions_file=None,
    history_file=None,
    rrm_mode: str = "tiered",
    channel_trace: list | None = None,
    attack: dict | None = None,
):
    if rrm_mode not in RRM_MODES:
        raise ValueError(f"rrm_mode must be one of {RRM_MODES}, got {rrm_mode!r}")
    phases = scenario["phases"]
    total_steps = scenario["total_steps"]
    bandwidth_mhz = ran["bandwidth_mhz"]

    neg_cfg = cfg["negotiation"]
    ctrl_cfg = cfg["controller"]
    rrm_cfg = cfg["rrm"]
    cycle_every_steps = rrm_cfg["cycle_every_steps"]

    controller = PRBController(
        kp=ctrl_cfg["kp"],
        prb_min=ctrl_cfg["prb_min_fraction"],
        prb_max=ctrl_cfg["prb_max_fraction"],
    )

    def phase_for_step(step: int) -> dict:
        current = phases[0]
        for p in phases:
            if step >= p["trigger_at_step"]:
                current = p
        return current

    def cycles_left_in_phase(step: int) -> int:
        ends_at = total_steps
        for p in phases:
            if p["trigger_at_step"] > step:
                ends_at = min(ends_at, p["trigger_at_step"])
        return max(1, (ends_at - step) // cycle_every_steps)

    def log_decision(record: dict) -> None:
        timing = timing_summary(drain_call_log())
        if timing is not None:
            record = {**record, **timing}
        record = mark_unresolved(record)
        if decisions_file is not None:
            append_jsonl(decisions_file, record)
        return record

    def review(role: str, context: str, decision, reviser):
        reviewed = (role in reflect_roles or
                    (role == "rrm" and reflect_roles & {"pc", "ulra"}))
        if not reviewed:
            return decision, None
        return reflect_and_refine(
            role, context, decision, reviser, agents.critic_fn,
            max_reflections, verbose=verbose)

    def make_reviser(fn, context, state):
        def reviser(_previous, critique):
            return fn(context, state, critique=critique)
        return reviser


    def mobility_node(state: SimState) -> dict:
        phase = phase_for_step(state["step"])
        new_phase_started = phase["phase"] != state["phase"]
        base_mcs = phase["mcs_index"]

        ues = state["ues"]
        if channel_trace is not None:
            row = channel_trace[min(state["step"], len(channel_trace) - 1)]
            for ue_id, ue in ues.items():
                ue["trace_ceiling_db"] = row["ceiling_db"][ue_id]
                ue["trace_raw_db"] = row.get("raw_db", {}).get(ue_id)
            base_mcs = ch.spectral_efficiency_to_mcs(ch.mean_spectral_efficiency(
                [ue["trace_ceiling_db"] for ue in ues.values()]))
        for ue in ues.values():
            ue["snr_ceiling_db"] = round(ch.ceiling_db(ue, base_mcs), 2)
            if ue["actual_snr_db"] > ue["snr_ceiling_db"]:
                ue["actual_snr_db"] = ue["snr_ceiling_db"]
            ue["mcs_index"] = ch.snr_to_mcs(ue["actual_snr_db"])

        achievable = ch.achievable_mbps_at_full_prb(ues, base_mcs, bandwidth_mhz)
        reachable = ch.reachable_mbps_within(
            ues, base_mcs, bandwidth_mhz,
            max_delta_db_per_cycle=guardrails["power_control"]["max_delta_db_per_cycle"],
            cycles_remaining=cycles_left_in_phase(state["step"]))

        pending, reason, exhausted = evaluate_triggers(
            state, new_phase_started=new_phase_started, achievable_mbps=achievable,
            neg_cfg=neg_cfg, rrm_cfg=rrm_cfg, guardrails=guardrails,
            slices=state["slices"],
            slice_delivered_mbps=sched.slice_delivered_mbps(ues))

        if verbose and new_phase_started:
            print("\n" + "=" * 76)
            print(f"[MOBILITY] step {state['step']}  PHASE {phase['phase']} -> "
                  f"{phase['label'].upper()}  (base MCS={base_mcs}, "
                  f"achievable={achievable:.1f} Mbit/s)")
            print(f"  {phase['event']}")
            for was, now in zip(tenants, apply_phase_overrides(tenants, phase), strict=True):
                if now != was:
                    print(f"  [PROFILE] {now.id} ({now.name}): wants "
                          f"{was.initial_intent_mbps:.0f} -> "
                          f"{now.initial_intent_mbps:.0f} Mbit/s, floor "
                          f"{was.min_acceptable_mbps:.0f} -> "
                          f"{now.min_acceptable_mbps:.0f} Mbit/s")
            print("=" * 76)
        elif verbose and pending:
            print(f"\n[MOBILITY] step {state['step']}: renegotiation triggered "
                  f"by {reason}")

        reported, trusted = apply_attack(state, ues)

        return {
            "phase": phase["phase"],
            "phase_label": phase["label"],
            "base_mcs_index": base_mcs,
            "achievable_mbps": round(achievable, 2),
            "reachable_mbps": round(reachable, 2),
            "pending_negotiation": pending,
            "negotiation_reason": reason,
            "rrm_exhausted_cycles": exhausted,
            "ues": ues,
            "slices": state["slices"],
            "reported_ues": reported,
            "telemetry_trusted": trusted,
        }

    base_descriptions: dict = {}

    def apply_attack(state: SimState, ues: dict) -> tuple[dict | None, dict]:
        slices = state["slices"]
        if not base_descriptions:
            base_descriptions.update({n: sl.get("description", "") for n, sl in slices.items()})
        active = attack is not None and state["step"] >= attack.get("start_step", 0)
        for name, sl in slices.items():
            extra = (attack or {}).get("slice_descriptions", {}).get(name) if active else None
            sl["description"] = (base_descriptions[name] + " " + extra).strip() if extra \
                else base_descriptions[name]
        forged = (attack or {}).get("telemetry", {}) if active else {}
        if not forged and not security.on("telemetry_check"):
            return None, {}
        reported = {}
        for ue_id, ue in ues.items():
            rep = dict(ue)
            f = forged.get(ue_id, {})
            if "snr_offset_db" in f:
                rep["actual_snr_db"] = round(ue["actual_snr_db"] + f["snr_offset_db"], 2)
            if "battery_pct" in f and ue.get("battery_pct") is not None:
                rep["battery_pct"] = f["battery_pct"]
            if "status_text" in f:
                rep["status_text"] = f["status_text"]
            reported[ue_id] = rep
        if not security.on("telemetry_check"):
            return reported, {}
        view, rejected = security.check_telemetry(
            reported, state.get("telemetry_trusted") or {}, ues, steps=cycle_every_steps)
        if rejected:
            log_decision({
                "cycle": state["cycle"], "step": state["step"],
                "phase": state["phase_label"], "agent": "telemetry_check",
                "decision": {"rejected": rejected}, "reflection": None,
            })
        return view, view


    def mediator_reviewer(decision, context, reviser):
        return reflect_and_refine(
            "mediator", context, decision, reviser, agents.critic_fn,
            max_reflections, verbose=verbose)

    def negotiation_node(state: SimState) -> dict:
        if not state["pending_negotiation"]:
            return {"pending_negotiation": False}

        phase_tenants = apply_phase_overrides(tenants, phase_for_step(state["step"]))
        phase_tenants_by_id = {t.id: t for t in phase_tenants}
        intents = {**state["tenant_intents"],
                   **{now.id: now.initial_intent_mbps
                      for was, now in zip(tenants, phase_tenants, strict=True)
                      if now != was}}

        exhaustion = state["negotiation_reason"] == "rrm_exhaustion"
        report_every = neg_cfg.get("capacity_report", "exhaustion") == "every"
        accepted_before = state.get("energy_accepted_db") or {}
        report = (energy.capacity_report(
            state["ues"], bandwidth_mhz=bandwidth_mhz,
            achievable_mbps=state["achievable_mbps"], exhausted=exhaustion,
            accepted=accepted_before)
            if exhaustion or report_every else None)

        if verbose:
            print(f"  -- negotiation at step {state['step']} "
                  f"(phase {state['phase']}: {state['phase_label']}, "
                  f"trigger: {state['negotiation_reason']}) --")

        result = run_negotiation(
            phase_tenants,
            intents,
            state["achievable_mbps"],
            agents.tenant_fn,
            agents.mediator_fn,
            max_rounds=neg_cfg["max_rounds"],
            half_width_mbps=neg_cfg["confidence_interval_half_width_mbps"],
            ran_off_threshold_mbps=neg_cfg["ran_off_threshold_mbps"],
            max_parallel_tenants=neg_cfg.get(
                "max_parallel_tenant_calls", DEFAULT_MAX_PARALLEL_TENANTS),
            verbose=verbose,
            backend=backend,
            mediator_reviewer=mediator_reviewer if "mediator" in reflect_roles else None,
            reachable_mbps=state["reachable_mbps"],
            capacity_report=report,
            context={
                "step": state["step"],
                "phase": state["phase"],
                "phase_label": state["phase_label"],
                "mcs_index": state["base_mcs_index"],
            },
        )

        if report is not None:
            with_room = {u["ue"] for u in report["below_ceiling"]}
            approved = [u for u in result.get("approve_energy_ues", []) if u in with_room]
            energy_bound = energy.sla_bound(
                state["ues"], bandwidth_mhz=bandwidth_mhz, approved=approved,
                accepted=accepted_before)
            accepted = energy.accept_power(
                state["ues"], approved=approved, accepted=accepted_before)
        else:
            cleared = state["negotiation_reason"] in (None, "", "phase_change")
            approved = [] if cleared else list(state.get("energy_approved_ues") or [])
            accepted = {} if cleared else dict(accepted_before)
            energy_bound = None
        sla_ceiling = min(state["achievable_mbps"], state["reachable_mbps"])
        if energy_bound is not None:
            sla_ceiling = min(sla_ceiling, energy_bound)

        floors = {t.id: t.min_acceptable_mbps for t in phase_tenants}
        sla, ran_off, shares = clamp_negotiation(
            result["consensus_mbps"], result["ran_off"],
            result.get("shares") or default_shares(phase_tenants),
            sla_ceiling, guardrails,
            tenant_ids=[t.id for t in phase_tenants], floors=floors)
        floors_state = floor_report(shares, sla, floors) if not ran_off else None

        targets = ({} if ran_off else slice_targets(
            sla_mbps=sla, shares=shares, tenant_slice_map=tenant_slice_map,
            tenants_by_id=phase_tenants_by_id))

        operator_intent = render_operator_intent(
            sla_mbps=sla, ran_off=ran_off, shares=shares,
            achievable_mbps=state["achievable_mbps"],
            tenant_slice_map=tenant_slice_map, tenants_by_id=phase_tenants_by_id,
            phase_label=state["phase_label"],
            reason=state["negotiation_reason"] or "phase_change",
            energy_approved_ues=approved,
            energy_report=report)

        log_entry = {
            "step": state["step"],
            "phase": state["phase"],
            "label": state["phase_label"],
            "trigger": state["negotiation_reason"],
            "shares": shares,
            "floors": floors_state,
            "operator_intent": operator_intent,
            **{k: result[k] for k in
               ("consensus_mbps", "ran_off", "rounds_used", "confidence_interval",
                "achievable_mbps", "reachable_mbps", "decision", "agent_calls")},
            "reflection": result.get("reflection"),
            "transcript": result["transcript"],
            "capacity_report": report,
            "energy_approved_ues": approved,
            "energy_accepted_db": accepted,
            "energy_bound_mbps": energy_bound,
        }
        raw_shares = result.get("shares") or default_shares(phase_tenants)
        clamped = clamp_diff(
            {"sla_mbps": round(float(result["consensus_mbps"]), 2),
             "ran_off": bool(result["ran_off"]),
             **{f"shares.{t}": round(float(v), 3) for t, v in raw_shares.items()}},
            {"sla_mbps": sla, "ran_off": ran_off,
             **{f"shares.{t}": round(v, 3) for t, v in shares.items()}},
            tolerance=0.011)
        log_decision({
            "cycle": state["cycle"], "step": state["step"],
            "phase": state["phase_label"], "agent": "negotiation",
            "trigger": state["negotiation_reason"],
            "decision": {"sla_mbps": sla, "ran_off": ran_off, "shares": shares,
                         "outcome": result["decision"], "floors": floors_state,
                         "energy_approved_ues": approved,
                         "energy_accepted_db": accepted,
                         "energy_bound_mbps": energy_bound},
            "capacity_report": report,
            "reflection": result.get("reflection"),
            "clamped": clamped,
        })

        if verbose:
            if ran_off:
                print(f"  => RAN OFF (achievable {state['achievable_mbps']:.1f} "
                      f"Mbit/s too low to be worthwhile)")
            else:
                split = ", ".join(f"{tenant_slice_map[t]}={f * 100:.0f}%"
                                  for t, f in sorted(shares.items()))
                print(f"  => SLA {sla:.1f} Mbit/s in {result['rounds_used']} "
                      f"round(s); shares {split}")
                if floors_state and floors_state["tenants_below_floor"]:
                    who = ", ".join(
                        f"{tenant_slice_map[t]} -{floors_state['shortfall_mbps'][t]:.1f}"
                        for t in floors_state["tenants_below_floor"])
                    fits = ("floors fit the SLA" if floors_state["floors_fit_in_sla"]
                            else "floors DO NOT fit the SLA")
                    print(f"     [FLOORS] {fits}; below floor: {who} Mbit/s")

        return {
            "sla_mbps": sla,
            "ran_off": ran_off,
            "energy_approved_ues": approved,
            "energy_accepted_db": accepted,
            "tenant_shares": shares,
            "slice_targets": targets,
            "operator_intent": operator_intent,
            "intent_serial": state["intent_serial"] + 1,
            "pending_negotiation": False,
            "negotiated_achievable_mbps": state["achievable_mbps"],
            "best_delivered_mbps": 0.0,
            "rrm_exhausted_cycles": 0,
            "negotiation_log": state["negotiation_log"] + [log_entry],
        }


    def l2_manager_node(state: SimState) -> dict:
        if state["intent_serial"] == state["l2_seen_serial"]:
            return {}

        if rrm_mode != "tiered":
            record = log_decision({
                "cycle": state["cycle"], "step": state["step"],
                "phase": state["phase_label"], "agent": "l2_manager",
                "mock": False, "bypassed": True,
                "operator_intent": state["operator_intent"],
                "decision": {"pc_subintent": state["operator_intent"],
                             "ul_ra_subintent": state["operator_intent"]},
                "reflection": None,
            })
            return {
                "pc_subintent": state["operator_intent"],
                "ul_ra_subintent": state["operator_intent"],
                "l2_seen_serial": state["intent_serial"],
                "decisions": [record],
            }

        context = build_context_l2(state)
        is_mock = agents.roles["l2"] == "mock"
        decision = agents.l2_fn(context, state)
        reviser = (make_mock_reviser(lambda crit: agents.l2_fn(context, state, critique=crit))
                   if is_mock else make_reviser(agents.l2_fn, context, state))
        decision, reflection = review("l2", context, decision, reviser)

        if verbose:
            print("\n" + "-" * 76)
            print(f"[L2 MANAGER] cycle={state['cycle']} phase={state['phase_label']}")
            print(f"  -> PC sub-intent:    {decision.pc_subintent}")
            print(f"  -> UL-RA sub-intent: {decision.ul_ra_subintent}")
            print("-" * 76)

        record = {
            "cycle": state["cycle"], "step": state["step"],
            "phase": state["phase_label"], "agent": "l2_manager", "mock": is_mock,
            "operator_intent": state["operator_intent"],
            "decision": decision.model_dump(), "reflection": reflection,
        }
        record = log_decision(record)
        return {
            "pc_subintent": decision.pc_subintent,
            "ul_ra_subintent": decision.ul_ra_subintent,
            "l2_seen_serial": state["intent_serial"],
            "decisions": [record],
        }

    def rrm_can_coast(state: SimState, seen_key: str) -> bool:
        if state["intent_serial"] != state[seen_key]:
            return False
        if state["ran_off"]:
            return True
        if state["sla_mbps"] <= 0:
            return True
        if (state["sla_mbps"] - state["delivered_mbps"]) > rrm_cfg["exhaustion_gap_mbps"]:
            return False
        if seen_key == "ulra_seen_serial":
            return share_error(state) <= rrm_cfg["share_tolerance"]
        return True

    def pc_agent_node(state: SimState) -> dict:
        if rrm_can_coast(state, "pc_seen_serial"):
            return {}
        context = build_context_pc(state, guardrails)
        is_mock = agents.roles["pc"] == "mock"
        decision = agents.pc_fn(context, state)
        reviser = (make_mock_reviser(lambda crit: agents.pc_fn(context, state, critique=crit))
                   if is_mock else make_reviser(agents.pc_fn, context, state))
        decision, reflection = review("pc", context, decision, reviser)

        ues = state["ues"]
        applied = clamp_pc(decision, ues, guardrails,
                           energy_approved=state.get("energy_approved_ues"),
                           energy_accepted=state.get("energy_accepted_db"))
        clamped = clamp_diff(
            {ue_id: round(float(proposed), 2)
             for ue_id, (proposed, _) in decision.per_ue().items()},
            {ue_id: target for ue_id, (target, _) in applied.items()},
            tolerance=0.01)
        for ue_id, (target, reasoning) in applied.items():
            ues[ue_id]["target_snr_db"] = target
            if verbose:
                print(f"    [PC]    {ue_id}: target_snr -> {target:6.2f} dB "
                      f"(ceiling {ues[ue_id]['snr_ceiling_db']:.2f})  {reasoning}")

        record = {
            "cycle": state["cycle"], "step": state["step"],
            "phase": state["phase_label"], "agent": "pc_agent", "mock": is_mock,
            "sub_intent": state["pc_subintent"],
            "decision": decision.model_dump(), "reflection": reflection,
            "clamped": clamped,
        }
        record = log_decision(record)
        return {"ues": ues, "pc_seen_serial": state["intent_serial"],
                "decisions": [record]}

    def ulra_agent_node(state: SimState) -> dict:
        if rrm_can_coast(state, "ulra_seen_serial"):
            return {}
        context = build_context_ulra(state, guardrails)
        is_mock = agents.roles["ulra"] == "mock"
        decision = agents.ulra_fn(context, state)
        reviser = (make_mock_reviser(lambda crit: agents.ulra_fn(context, state, critique=crit))
                   if is_mock else make_reviser(agents.ulra_fn, context, state))
        decision, reflection = review("ulra", context, decision, reviser)

        slices = state["slices"]
        applied = secure_ulra(decision, slices, state, state["ul_ra_subintent"])
        proposed = {}
        for name, (throttle, priority, _) in decision.per_slice().items():
            proposed[f"{name}.throttle_mbps"] = round(float(throttle), 2)
            proposed[f"{name}.priority"] = priority
        clamped = clamp_diff(
            proposed,
            {**{f"{n}.throttle_mbps": t for n, (t, _, _) in applied.items()},
             **{f"{n}.priority": p for n, (_, p, _) in applied.items()}},
            tolerance=0.01)
        for name, (throttle, priority, reasoning) in applied.items():
            slices[name]["throttle_mbps"] = throttle
            slices[name]["priority"] = priority
            if verbose:
                print(f"    [UL-RA] {name}: throttle -> {throttle:6.1f} Mbit/s, "
                      f"priority -> {priority:6s}  {reasoning}")

        record = {
            "cycle": state["cycle"], "step": state["step"],
            "phase": state["phase_label"], "agent": "ulra_agent", "mock": is_mock,
            "sub_intent": state["ul_ra_subintent"],
            "decision": decision.model_dump(), "reflection": reflection,
            "clamped": clamped,
        }
        record = log_decision(record)
        return {"slices": slices, "ulra_seen_serial": state["intent_serial"],
                "decisions": [record]}

    def rrm_agent_node(state: SimState) -> dict:
        if rrm_can_coast(state, "ulra_seen_serial"):
            return {}
        context = (
            f"OPERATOR INTENT:\n{state['operator_intent']}\n\n"
            "=== POWER CONTROL ===\n" + build_context_pc(state, guardrails)
            + "\n\n=== UL RESOURCE ALLOCATION ===\n"
            + build_context_ulra(state, guardrails))
        is_mock = agents.roles["pc"] == "mock"
        decision = agents.rrm_fn(context, state)
        reviser = (make_mock_reviser(lambda crit: agents.rrm_fn(context, state, critique=crit))
                   if is_mock else make_reviser(agents.rrm_fn, context, state))
        decision, reflection = review("rrm", context, decision, reviser)

        ues, slices = state["ues"], state["slices"]
        applied_pc = clamp_pc(decision, ues, guardrails,
                              energy_approved=state.get("energy_approved_ues"),
                              energy_accepted=state.get("energy_accepted_db"))
        applied_ulra = secure_ulra(decision, slices, state, state["operator_intent"])
        proposed = {ue_id: round(float(v), 2)
                    for ue_id, (v, _) in decision.per_ue().items()}
        applied = {ue_id: t for ue_id, (t, _) in applied_pc.items()}
        for name, (throttle, priority, _) in decision.per_slice().items():
            proposed[f"{name}.throttle_mbps"] = round(float(throttle), 2)
            proposed[f"{name}.priority"] = priority
        for name, (throttle, priority, _) in applied_ulra.items():
            applied[f"{name}.throttle_mbps"] = throttle
            applied[f"{name}.priority"] = priority
        clamped = clamp_diff(proposed, applied, tolerance=0.01)

        for ue_id, (target, reasoning) in applied_pc.items():
            ues[ue_id]["target_snr_db"] = target
            if verbose:
                print(f"    [RRM/PC]    {ue_id}: target_snr -> {target:6.2f} dB "
                      f"(ceiling {ues[ue_id]['snr_ceiling_db']:.2f})  {reasoning}")
        for name, (throttle, priority, reasoning) in applied_ulra.items():
            slices[name]["throttle_mbps"] = throttle
            slices[name]["priority"] = priority
            if verbose:
                print(f"    [RRM/UL-RA] {name}: throttle -> {throttle:6.1f} Mbit/s, "
                      f"priority -> {priority:6s}  {reasoning}")

        record = log_decision({
            "cycle": state["cycle"], "step": state["step"],
            "phase": state["phase_label"], "agent": "rrm_agent", "mock": is_mock,
            "sub_intent": state["operator_intent"],
            "decision": decision.model_dump(), "reflection": reflection,
            "clamped": clamped,
        })
        return {"ues": ues, "slices": slices,
                "pc_seen_serial": state["intent_serial"],
                "ulra_seen_serial": state["intent_serial"],
                "decisions": [record]}

    def pf_rrm_node(state: SimState) -> dict:
        ues, slices = state["ues"], state["slices"]
        fields = {}
        for ue_id, ue in ues.items():
            fields[f"{ue_id}_target_snr_db"] = ue["snr_ceiling_db"]
            fields[f"{ue_id}_reasoning"] = "maximum-rate power control: target the ceiling"
        applied = clamp_pc(PCDecision(**fields), ues, guardrails,
                           energy_approved=state.get("energy_approved_ues"),
                           energy_accepted=state.get("energy_accepted_db"))
        for ue_id, (target, _) in applied.items():
            ues[ue_id]["target_snr_db"] = target
        targets = state.get("slice_targets") or {}
        for name, sl in slices.items():
            sl["pf_weight"] = float(targets.get(name, 1.0)) if targets else 1.0
            sl["throttle_mbps"] = guardrails["ul_ra"]["throttle_max_mbps"]
            sl["priority"] = "normal"
        record = log_decision({
            "cycle": state["cycle"], "step": state["step"],
            "phase": state["phase_label"], "agent": "pf_rrm", "mock": True,
            "decision": {"targets": {u: t for u, (t, _) in applied.items()},
                         "pf_weights": {n: sl["pf_weight"] for n, sl in slices.items()}},
            "reflection": None,
        })
        return {"ues": ues, "slices": slices,
                "pc_seen_serial": state["intent_serial"],
                "ulra_seen_serial": state["intent_serial"],
                "decisions": [record]}

    def secure_ulra(decision, slices: dict, state: SimState, subintent: str) -> dict:
        trusted = None
        if security.on("provenance_guard"):
            trusted = state["operator_intent"] if security.on("provenance_root") else subintent
        untrusted = " ".join(
            [sl.get("description") or "" for sl in slices.values()]
            + [ue.get("status_text") or "" for ue in _view(state).values()])
        applied = clamp_ulra(decision, slices, guardrails,
                             sla_mbps=state.get("sla_mbps"),
                             trusted_text=trusted, untrusted_text=untrusted)
        if security.on("change_limit"):
            applied = security.limit_change(applied, slices)
        if security.on("policy_wall"):
            applied = security.policy_wall(applied, slices, state.get("slice_targets") or {})
        return applied


    def environment_node(state: SimState) -> dict:
        ues = state["ues"]
        slices = state["slices"]
        prb = state["prb_fraction"]
        delivered = state["delivered_mbps"]
        history = state["history"]
        step = state["step"]
        capacity = 0.0

        for _ in range(cycle_every_steps):
            if channel_trace is not None:
                row = channel_trace[min(step, len(channel_trace) - 1)]
                for ue_id, ue in ues.items():
                    ue["trace_ceiling_db"] = row["ceiling_db"][ue_id]
                    ue["trace_raw_db"] = row.get("raw_db", {}).get(ue_id)
                    ue["snr_ceiling_db"] = round(ue["trace_ceiling_db"], 2)
                    if ue["actual_snr_db"] > ue["snr_ceiling_db"]:
                        ue["actual_snr_db"] = ue["snr_ceiling_db"]
            if state["ran_off"]:
                prb, delivered, capacity = 0.0, 0.0, 0.0
                for ue in ues.values():
                    ue["throughput_mbps"] = 0.0
                    ue["tx_power_w"] = 0.0
            else:
                for ue in ues.values():
                    ue["actual_snr_db"] = round(ch.step_snr(
                        ue["actual_snr_db"], ue["target_snr_db"],
                        ue["snr_ceiling_db"]), 2)
                    ue["mcs_index"] = ch.snr_to_mcs(ue["actual_snr_db"])
                    ue["tx_power_w"] = ch.ue_tx_power_w(ue, ue["actual_snr_db"])

                def allocate(prb_fraction: float) -> dict:
                    if rrm_mode == "pf":
                        return sched.allocate_pf(
                            ues, {n: sl.get("pf_weight", 1.0) for n, sl in slices.items()},
                            bandwidth_mhz, prb_fraction)
                    cap = ch.cell_capacity_mbps(ues, bandwidth_mhz, prb_fraction)
                    return sched.allocate_throughput(
                        ues, slices, cap, bandwidth_mhz, prb_fraction)

                def deliver(prb_fraction: float) -> float:
                    return sum(allocate(prb_fraction).values())

                for _ in range(ctrl_cfg["settle_steps"]):
                    prb, delivered = controller.step(state["sla_mbps"], prb, deliver)

                capacity = ch.cell_capacity_mbps(ues, bandwidth_mhz, prb)
                allocation = allocate(prb)
                delivered = round(sum(allocation.values()), 2)
                for ue_id, mbps in allocation.items():
                    ues[ue_id]["throughput_mbps"] = round(mbps, 2)

            for ue in ues.values():
                new_batt = ch.drain_battery(ue["battery_pct"], ue["tx_power_w"], 1.0)
                if new_batt is not None:
                    ue["battery_pct"] = round(new_batt, 2)

            snap = snapshot(state, step, delivered, prb, capacity)
            history = history + [snap]
            if history_file is not None:
                append_jsonl(history_file, snap)
            step += 1

        if verbose:
            status = "RAN OFF" if state["ran_off"] else (
                f"SLA={state['sla_mbps']:6.1f} delivered={delivered:6.1f} "
                f"PRB={prb * 100:3.0f}% cap={capacity:6.1f}")
            print(f"  cycle {state['cycle']:3d} | step {state['step']:3d} | "
                  f"phase {state['phase']} | {status}")
            for ue_id, ue in ues.items():
                batt = ("  grid" if ue["battery_pct"] is None
                        else f"{ue['battery_pct']:5.1f}%")
                print(f"      {ue_id:6s} snr={ue['actual_snr_db']:6.2f}/"
                      f"{ue['snr_ceiling_db']:6.2f} mcs={ue['mcs_index']:2d} "
                      f"thr={ue['throughput_mbps']:6.2f} batt={batt}")

        return {
            "prb_fraction": prb,
            "delivered_mbps": delivered,
            "last_delivered_mbps": state["delivered_mbps"],
            "best_delivered_mbps": max(state.get("best_delivered_mbps", 0.0),
                                       delivered),
            "ues": ues,
            "history": history,
            "step": step,
            "cycle": state["cycle"] + 1,
        }

    def should_continue(state: SimState) -> str:
        return "mobility" if state["step"] < total_steps else END


    graph = StateGraph(SimState)
    graph.add_node("mobility", mobility_node)
    graph.add_node("negotiation", negotiation_node)
    graph.add_node("l2_manager", l2_manager_node)
    graph.add_node("environment", environment_node)

    graph.set_entry_point("mobility")
    graph.add_edge("mobility", "negotiation")
    graph.add_edge("negotiation", "l2_manager")
    if rrm_mode == "monolithic":
        graph.add_node("rrm_agent", rrm_agent_node)
        graph.add_edge("l2_manager", "rrm_agent")
        graph.add_edge("rrm_agent", "environment")
    elif rrm_mode == "pf":
        graph.add_node("pf_rrm", pf_rrm_node)
        graph.add_edge("l2_manager", "pf_rrm")
        graph.add_edge("pf_rrm", "environment")
    else:
        graph.add_node("pc_agent", pc_agent_node)
        graph.add_node("ulra_agent", ulra_agent_node)
        graph.add_edge("l2_manager", "pc_agent")
        graph.add_edge("l2_manager", "ulra_agent")
        graph.add_edge("pc_agent", "environment")
        graph.add_edge("ulra_agent", "environment")
    graph.add_conditional_edges(
        "environment", should_continue, {"mobility": "mobility", END: END})

    return graph.compile()
