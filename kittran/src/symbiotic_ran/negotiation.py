from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .agents.backends import drain_call_log, record_calls
from .config import Tenant
from .optimizer import compute_interval
from .records import NegotiationRecord, TenantState
from .schemas import MediatorDecision, TenantProposal

DEFAULT_MAX_PARALLEL_TENANTS = 3


def format_profile(t: Tenant) -> str:
    return (
        f"- name: {t.name}\n"
        f"- id: {t.id}\n"
        f"- current intent: {t.initial_intent_mbps} Mbps (may change across phases)\n"
        f"- minimum acceptable: {t.min_acceptable_mbps} Mbps\n"
        f"- utility weight: {t.utility_weight}\n"
        f"- willingness to pay: {t.willingness_to_pay}\n"
        f"- description: {t.description}"
    )


def _gather_proposals(
    tenants: List[Tenant],
    tenant_fn,
    *,
    interval,
    achievable_mbps: float,
    mediator_suggestion: float,
    round_idx: int,
    executor: Optional[ThreadPoolExecutor],
) -> List[TenantProposal]:
    def ask(t: Tenant) -> TenantProposal:
        return tenant_fn(t, format_profile(t), interval, achievable_mbps,
                         mediator_suggestion, round_idx)

    if executor is None:
        return [ask(t) for t in tenants]

    def ask_and_account(t: Tenant) -> tuple[TenantProposal, list]:
        drain_call_log()
        proposal = ask(t)
        return proposal, drain_call_log()

    proposals = []
    for proposal, calls in executor.map(ask_and_account, tenants):
        record_calls(calls)
        proposals.append(proposal)
    return proposals


def default_shares(tenants: List[Tenant]) -> dict:
    weights = {t.id: max(0.0, t.utility_weight * t.willingness_to_pay) for t in tenants}
    total = sum(weights.values())
    if total <= 0:
        return {t.id: 1.0 / len(tenants) for t in tenants}
    return {tid: w / total for tid, w in weights.items()}


def _review_context(round_idx, proposals, tenants, interval, achievable_mbps,
                    ran_off_threshold_mbps) -> str:
    lines = [
        f"NEGOTIATION ROUND {round_idx}.",
        f"Achievable throughput at 100% PRB: {achievable_mbps:.1f} Mbps.",
        f"Recommended interval: [{interval[0]:.1f}, {interval[1]:.1f}] Mbps.",
        f"RAN switch-off threshold: {ran_off_threshold_mbps:.1f} Mbps.",
        "",
        "Tenants and their positions this round:",
    ]
    for t, p in zip(tenants, proposals, strict=True):
        starved = (" -- the channel CANNOT reach this tenant's floor"
                   if achievable_mbps < t.min_acceptable_mbps else "")
        lines.append(
            f"- {t.id} ({t.name}): proposes {p.proposed_mbps:.1f} Mbps, "
            f"accept={p.accept}, floor {t.min_acceptable_mbps:.1f} Mbps, "
            f"utility weight {t.utility_weight}, willingness to pay "
            f"{t.willingness_to_pay}{starved}. {p.reasoning}"
        )
    return "\n".join(lines)


def _result_from_record(
    record: NegotiationRecord,
    *,
    transcript: Optional[List[dict]] = None,
    shares: Optional[dict] = None,
    reflection: Optional[List[dict]] = None,
) -> dict:
    return {
        "consensus_mbps": record.consensus_mbps,
        "ran_off": record.ran_off,
        "rounds_used": record.rounds_used,
        "confidence_interval": list(record.confidence_interval),
        "achievable_mbps": record.achievable_mbps,
        "reachable_mbps": record.reachable_mbps,
        "transcript": transcript or [],
        "decision": record.decision,
        "agent_calls": record.agent_calls,
        "record": record,
        "shares": shares,
        "reflection": reflection,
    }


def run_negotiation(
    tenants: List[Tenant],
    intents: dict,
    achievable_mbps: float,
    tenant_fn,
    mediator_fn,
    *,
    max_rounds: int,
    half_width_mbps: float,
    ran_off_threshold_mbps: float,
    verbose: bool = True,
    context: Optional[dict] = None,
    backend: str = "unknown",
    max_parallel_tenants: int = DEFAULT_MAX_PARALLEL_TENANTS,
    mediator_reviewer=None,
    reachable_mbps: Optional[float] = None,
    capacity_report: Optional[dict] = None,
) -> dict:
    context = context or {}
    ceiling = (achievable_mbps if reachable_mbps is None
               else min(achievable_mbps, reachable_mbps))
    center, interval = compute_interval(tenants, intents, ceiling, half_width_mbps)

    if verbose:
        ramp = ("" if reachable_mbps is None or ceiling >= achievable_mbps - 0.05
                else f", reachable this phase {ceiling:.1f} Mbps")
        print(f"    interval=[{interval[0]:.1f}, {interval[1]:.1f}] Mbps "
              f"(center {center:.1f}), achievable {achievable_mbps:.1f} Mbps{ramp}")

    mediator_suggestion = center
    decision: MediatorDecision | None = None
    tenant_states = {
        t.id: TenantState(
            id=t.id,
            name=t.name,
            intent_mbps=intents.get(t.id, t.initial_intent_mbps),
            min_acceptable_mbps=t.min_acceptable_mbps,
            utility_weight=t.utility_weight,
            willingness_to_pay=t.willingness_to_pay,
        )
        for t in tenants
    }
    transcript: List[dict] = []
    reflection: List[dict] = []
    agent_calls = 0

    workers = min(max_parallel_tenants, len(tenants))
    executor = (ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tenant")
                if workers > 1 else None)

    rounds_used = 0
    try:
        for round_idx in range(1, max_rounds + 1):
            rounds_used = round_idx
            proposals = _gather_proposals(
                tenants, tenant_fn, interval=interval,
                achievable_mbps=ceiling,
                mediator_suggestion=mediator_suggestion, round_idx=round_idx,
                executor=executor,
            )
            agent_calls += len(proposals)
            for t, proposal in zip(tenants, proposals, strict=True):
                tenant_states[t.id].observe(proposal.proposed_mbps, proposal.accept)

            extra = ({"capacity_report": capacity_report}
                     if capacity_report is not None else {})
            round_decision: MediatorDecision = mediator_fn(
                proposals, interval, ceiling, tenants, **extra)
            agent_calls += 1

            if mediator_reviewer is not None:
                review_context = _review_context(
                    round_idx, proposals, tenants, interval, ceiling,
                    ran_off_threshold_mbps)

                def _revise(_previous, critique, _p=proposals, _x=extra):
                    return mediator_fn(_p, interval, ceiling, tenants,
                                       critique=critique, **_x)

                round_decision, round_trace = mediator_reviewer(
                    round_decision, review_context, _revise)
                if round_trace:
                    reflection.extend(
                        {"round": round_idx, **entry} for entry in round_trace)
                    agent_calls += len(round_trace)

            decision = round_decision
            mediator_suggestion = round_decision.suggested_sla_mbps
            transcript.append({
                "round": round_idx,
                "proposals": [
                    {"tenant": t.id, "mbps": p.proposed_mbps, "accept": p.accept,
                     "reasoning": p.reasoning}
                    for t, p in zip(tenants, proposals, strict=True)
                ],
                "mediator": {
                    "suggested_mbps": round_decision.suggested_sla_mbps,
                    "consensus": round_decision.consensus_reached,
                    "switch_off": round_decision.switch_ran_off,
                    "shares": round_decision.shares(),
                    "reasoning": round_decision.reasoning,
                },
            })

            if verbose:
                props = ", ".join(
                    f"{t.id}={p.proposed_mbps:.0f}"
                    for t, p in zip(tenants, proposals, strict=True))
                print(f"    round {round_idx}: {props}  ->  mediator "
                      f"{round_decision.suggested_sla_mbps:.1f} "
                      f"Mbps {'[CONSENSUS]' if round_decision.consensus_reached else ''}"
                      f"{' [RAN OFF]' if round_decision.switch_ran_off else ''}")

            if round_decision.consensus_reached:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if decision is None:
        raise ValueError("negotiation ran zero rounds; max_rounds must be >= 1")

    worth_operating = achievable_mbps >= ran_off_threshold_mbps
    ran_off = bool(decision.switch_ran_off or
                   decision.suggested_sla_mbps < ran_off_threshold_mbps)
    if ran_off and worth_operating and ceiling < achievable_mbps:
        ran_off = False
        consensus_hint = max(decision.suggested_sla_mbps, 0.0)
        if verbose:
            print(f"    [ramp] switch-off declined: {achievable_mbps:.1f} Mbps is "
                  f"worth operating; only the ramp to it is slow "
                  f"(reachable {ceiling:.1f} Mbps this phase)")
        decision.suggested_sla_mbps = max(consensus_hint, ran_off_threshold_mbps)
    consensus_mbps = 0.0 if ran_off else round(decision.suggested_sla_mbps, 2)

    if ran_off:
        outcome = "ran_off"
    elif decision.consensus_reached:
        outcome = "consensus"
    else:
        outcome = "round_cap"

    for state in tenant_states.values():
        state.finalize(ceiling, consensus_mbps, ran_off)

    record = NegotiationRecord(
        step=context.get("step", -1),
        phase=context.get("phase", -1),
        phase_label=context.get("phase_label", ""),
        achievable_mbps=round(achievable_mbps, 2),
        reachable_mbps=round(ceiling, 2),
        confidence_interval=list(interval),
        interval_center_mbps=center,
        tenant_states=[tenant_states[t.id] for t in tenants],
        rounds_used=rounds_used,
        consensus_mbps=consensus_mbps,
        ran_off=ran_off,
        decision=outcome,
        decision_reasoning=decision.reasoning,
        max_rounds=max_rounds,
        half_width_mbps=half_width_mbps,
        ran_off_threshold_mbps=ran_off_threshold_mbps,
        backend=backend,
        agent_calls=agent_calls,
        shares=decision.shares(),
    )

    result = _result_from_record(
        record, transcript=transcript,
        shares=decision.shares(), reflection=reflection or None)
    result["approve_energy_ues"] = (
        list(decision.approve_energy_ues or []) if capacity_report is not None else [])
    return result
