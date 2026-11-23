from __future__ import annotations

import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

from . import security
from .agents.backends import (
    DEFAULT_BASE_URL,
    ROLES,
    backend_id_for,
    build_agents,
    configure_failure_injection,
)
from .config import (
    DATA_DIR,
    load_config,
    load_guardrails,
    load_initial_radio,
    load_scenario,
    load_tenant_slice_map,
    load_tenants,
)
from .graph import build_graph
from .run_table import write_run_table
from .schemas import SimState
from .uma import load_trace

DEFAULT_REFLECT_ROLES = ("l2", "pc", "ulra")

REFLECTABLE_ROLES = ("l2", "pc", "ulra", "mediator")


def build_initial_state(radio: dict, tenants, tenant_slice_map: dict) -> SimState:
    ues = deepcopy(radio["ues"])
    for ue in ues.values():
        ue.setdefault("snr_ceiling_db", ue["actual_snr_db"])
        ue.setdefault("mcs_index", 0)
    cell = deepcopy(radio["cell"])
    cell["tenant_slice_map"] = dict(tenant_slice_map)

    return SimState(
        step=0,
        cycle=0,
        phase=0,
        phase_label="init",
        base_mcs_index=0,
        ues=ues,
        slices=deepcopy(radio["slices"]),
        cell=cell,
        prb_fraction=1.0,
        delivered_mbps=0.0,
        last_delivered_mbps=0.0,
        best_delivered_mbps=0.0,
        achievable_mbps=0.0,
        reachable_mbps=0.0,
        operator_intent="",
        pc_subintent="Maintain current target SNR for all UEs.",
        ul_ra_subintent="Keep normal priority and maximum throttle for all slices.",
        intent_serial=0,
        l2_seen_serial=0,
        pc_seen_serial=0,
        ulra_seen_serial=0,
        rrm_exhausted_cycles=0,
        negotiated_achievable_mbps=0.0,
        energy_approved_ues=[],
        energy_accepted_db={},
        reported_ues=None,
        telemetry_trusted={},
        sla_mbps=0.0,
        tenant_shares={},
        slice_targets={},
        tenant_intents={t.id: t.initial_intent_mbps for t in tenants},
        ran_off=False,
        pending_negotiation=False,
        negotiation_reason="",
        history=[],
        negotiation_log=[],
        decisions=[],
    )


ATTACKS_FILE = DATA_DIR.parent / "scenarios" / "attacks" / "attacks.json"


def load_attack(name: str) -> dict:
    attacks = json.loads(ATTACKS_FILE.read_text())["attacks"]
    if name not in attacks:
        raise ValueError(f"unknown attack {name!r}; known: {', '.join(sorted(attacks))}")
    return {"name": name, **attacks[name]}


def run(
    *,
    model: str = "qwen3.6:35b-a3b-mtp-q4_K_M",
    mock: bool = False,
    agent_roles: dict | None = None,
    mock_profile: str = "good",
    reflect_roles=None,
    max_reflections: int | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
    thinking: bool | None = None,
    engine: str = "ollama",
    base_url: str | None = None,
    total_steps: int | None = None,
    cycle_every_steps: int | None = None,
    seed: int | None = 12345,
    rrm_mode: str = "tiered",
    capacity_report: str | None = None,
    inject_failures: dict | None = None,
    attack: str | dict | None = None,
    security_overrides: dict | None = None,
) -> dict:
    if seed is not None:
        random.seed(seed)
    configure_failure_injection(inject_failures, seed=seed)

    data_dir = data_dir or DATA_DIR
    output_dir = output_dir or (data_dir.parent / "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    tenants, ran = load_tenants(data_dir)
    scenario = load_scenario(data_dir)
    cfg = load_config(data_dir)
    guardrails = load_guardrails(data_dir)
    radio = load_initial_radio(data_dir)
    tenant_slice_map = load_tenant_slice_map(data_dir)

    if total_steps is not None:
        scenario = dict(scenario, total_steps=total_steps)
    if capacity_report is not None:
        cfg = dict(cfg, negotiation=dict(cfg["negotiation"], capacity_report=capacity_report))
    if cycle_every_steps is not None:
        cfg = dict(cfg, rrm=dict(cfg["rrm"], cycle_every_steps=cycle_every_steps))
    security_settings = security.configure(
        {**cfg.get("security", {}), **(security_overrides or {})})
    attack_spec = (dict(attack) if isinstance(attack, dict)
                   else load_attack(attack) if attack else None)
    (output_dir / "security.json").write_text(json.dumps(
        {"security": security_settings, "attack": attack_spec}, indent=2))
    ran = dict(ran, bandwidth_mhz=radio["cell"]["bandwidth_mhz"])

    channel_trace = None
    if cfg.get("channel_trace"):
        channel_trace = load_trace(
            data_dir / cfg["channel_trace"].format(seed=seed if seed is not None else 0))

    if mock:
        agent_roles = {role: "mock" for role in ROLES}
        reflect_roles = ()
    if reflect_roles is None:
        reflect_roles = cfg["reflection"].get("enabled_roles", DEFAULT_REFLECT_ROLES)
    reflect_roles = frozenset(reflect_roles)
    unknown = reflect_roles - set(REFLECTABLE_ROLES)
    if unknown:
        raise ValueError(
            f"cannot reflect on {sorted(unknown)}; "
            f"reviewable roles: {', '.join(REFLECTABLE_ROLES)}")
    if max_reflections is None:
        max_reflections = cfg["reflection"]["max_reflections"]

    agents = build_agents(
        model=model,
        roles=agent_roles,
        ran_off_threshold_mbps=cfg["negotiation"]["ran_off_threshold_mbps"],
        thinking=thinking,
        mock_profile=mock_profile,
        need_critic=bool(reflect_roles),
        engine=engine,
        base_url=base_url,
    )

    backend_id = backend_id_for(agents.roles, model, engine)

    decisions_file = output_dir / "decisions.jsonl"
    decisions_file.unlink(missing_ok=True)

    if verbose:
        print("=" * 76)
        print("SymbioticRAN -- three-tier agentic RAN simulation")
        print(f"  tenants   = {[t.id for t in tenants]} -> slices "
              f"{[tenant_slice_map[t.id] for t in tenants]}")
        print(f"  UEs       = {list(radio['ues'])}")
        print(f"  phases    = {[p['label'] for p in scenario['phases']]}")
        print(f"  steps     = {scenario['total_steps']} env steps, "
              f"{cfg['rrm']['cycle_every_steps']} per control cycle "
              f"(~{scenario['total_steps'] // cfg['rrm']['cycle_every_steps']} cycles)")
        print("  agents    = " + ", ".join(
            f"{r}:{agents.roles[r].upper()}" for r in ROLES))
        if rrm_mode != "tiered":
            print(f"  rrm       = {rrm_mode}")
        if inject_failures:
            print(f"  inject    = {inject_failures} (share of calls that fail and hold)")
        print(f"  reflect   = {sorted(reflect_roles) or 'off'} "
              f"(max {max_reflections} round(s), critic always LLM)")
        print(f"  mocks     = {mock_profile}")
        needs_server = agents.uses_llm() or reflect_roles
        print(f"  model     = {model if needs_server else '(none needed)'}")
        if needs_server:
            print(f"  engine    = {engine} @ "
                  f"{base_url or DEFAULT_BASE_URL.get(engine, '?')}")
        print(f"  backend   = {backend_id}")
        print("=" * 76)

    history_file = output_dir / "history.jsonl"
    history_file.write_text("")

    graph = build_graph(
        tenants, ran, scenario, cfg, guardrails, tenant_slice_map, agents,
        verbose=verbose, backend=backend_id, reflect_roles=reflect_roles,
        max_reflections=max_reflections, decisions_file=decisions_file,
        history_file=history_file, rrm_mode=rrm_mode, channel_trace=channel_trace,
        attack=attack_spec)

    initial = build_initial_state(radio, tenants, tenant_slice_map)
    cycles = scenario["total_steps"] // cfg["rrm"]["cycle_every_steps"] + 2
    try:
        final_state = graph.invoke(
            initial, config={"recursion_limit": cycles * 6 + 20})
    finally:
        configure_failure_injection(None)

    with open(history_file, "w") as f:
        for snap in final_state["history"]:
            f.write(json.dumps(snap) + "\n")

    negotiation_file = output_dir / "negotiation_log.json"
    with open(negotiation_file, "w") as f:
        json.dump(final_state["negotiation_log"], f, indent=2, default=str)

    table_csv, table_md = write_run_table(
        final_state["history"], output_dir,
        tenant_slice_map=tenant_slice_map,
        title=f"Run table — {data_dir.name} / {backend_id}")

    if verbose:
        _print_summary(final_state, history_file, negotiation_file,
                       decisions_file, table_csv, table_md)

    return final_state


def _print_summary(state, history_file, negotiation_file, decisions_file,
                   table_csv=None, table_md=None) -> None:
    print("\n" + "=" * 76)
    print("Simulation finished.")
    print(f"  history        -> {history_file}")
    print(f"  negotiations   -> {negotiation_file}")
    print(f"  decisions      -> {decisions_file}")
    if table_csv is not None:
        print(f"  run table      -> {table_csv}")
        print(f"                    {table_md}")
    rrm = [d for d in state["decisions"] if d["agent"] != "negotiation"]
    print(f"  {len(state['negotiation_log'])} negotiation(s), "
          f"{len(rrm)} RRM decision(s) across "
          f"{len(state['history'])} env steps / {state['cycle']} control cycles")

    triggers = {}
    for entry in state["negotiation_log"]:
        triggers[entry.get("trigger") or "?"] = triggers.get(entry.get("trigger") or "?", 0) + 1
    if triggers:
        print("  renegotiation triggers: " + ", ".join(
            f"{k}={v}" for k, v in sorted(triggers.items())))

    rounds = approvals = rejections = 0
    for record in state["decisions"]:
        for entry in record.get("reflection") or []:
            rounds += 1
            approvals += bool(entry["approved"])
            rejections += not entry["approved"]
    for entry in state["negotiation_log"]:
        for r in entry.get("reflection") or []:
            rounds += 1
            approvals += bool(r["approved"])
            rejections += not r["approved"]
    unresolved = [r for r in state["decisions"] if r.get("reflection_unresolved")]
    if rounds:
        print(f"  reflection: {rounds} round(s), {rejections} rejection(s) / "
              f"{approvals} approval(s)")
    if unresolved:
        by_role = Counter(r["agent"] for r in unresolved)
        who = ", ".join(f"{role} x{n}" for role, n in sorted(by_role.items()))
        print(f"  UNRESOLVED: {len(unresolved)} decision(s) shipped still "
              f"rejected ({who})")

