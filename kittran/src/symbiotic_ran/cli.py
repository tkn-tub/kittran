from __future__ import annotations

import argparse
from pathlib import Path

from .agents.backends import (
    DEFAULT_BASE_URL,
    ENGINES,
    INJECTABLE_ROLES,
    ROLES,
    parse_injection_spec,
)
from .graph import RRM_MODES
from .plotting import plot_run
from .simulation import REFLECTABLE_ROLES, run


def _roles(value: str, what: str) -> list:
    parsed = [r.strip() for r in value.split(",") if r.strip()]
    unknown = [r for r in parsed if r not in ROLES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {what} role(s) {unknown}; known: {', '.join(ROLES)}")
    return parsed


def _reflect_roles(value: str) -> list:
    if value.strip().lower() in ("none", "off", ""):
        return []
    parsed = [r.strip() for r in value.split(",") if r.strip()]
    unknown = [r for r in parsed if r not in REFLECTABLE_ROLES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"cannot reflect on {unknown}; reviewable roles: "
            f"{', '.join(REFLECTABLE_ROLES)}")
    return parsed


def parse_security_spec(text: str) -> dict:
    out = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        key, _, value = item.partition("=")
        low = value.strip().lower()
        if low in ("true", "on", "1", "yes"):
            out[key.strip()] = True
        elif low in ("false", "off", "0", "no"):
            out[key.strip()] = False
        else:
            try:
                out[key.strip()] = float(value)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"bad security setting {item!r}") from exc
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="symbiotic-ran",
        description="Three-tier agentic RAN simulation: multi-tenant SLA "
                    "negotiation -> intent decomposition -> radio resource "
                    "management -> deterministic enforcement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen3.6:35b-a3b-mtp-q4_K_M",
                   help="model name for every LLM-backed role. Ollama wants a "
                        "tag; vLLM wants the full repo path it was started with")
    p.add_argument("--engine", choices=ENGINES, default="ollama",
                   help="inference server to drive the LLM roles")
    p.add_argument("--base-url", default=None,
                   help="override the engine's endpoint (defaults: "
                        + ", ".join(f"{k} {v}" for k, v in DEFAULT_BASE_URL.items())
                        + ")")
    p.add_argument("--mock", action="store_true",
                   help="Mock every role AND turn reflection off. The only "
                        "configuration that needs no Ollama server.")
    p.add_argument("--llm", type=lambda v: _roles(v, "llm"), default=None,
                   metavar="ROLES",
                   help=f"comma-separated roles to run on the LLM ({', '.join(ROLES)})")
    p.add_argument("--mock-agents", type=lambda v: _roles(v, "mock"), default=None,
                   metavar="ROLES",
                   help="comma-separated roles to run as deterministic mocks")
    p.add_argument("--mock-profile", choices=("good", "flawed"), default="good",
                   help="'flawed' mocks reproduce the documented failure modes "
                        "on their first proposal, then self-correct under critique")
    p.add_argument("--reflect", type=_reflect_roles, default=None, metavar="ROLES",
                   help=f"roles to run the Reflection critic on "
                        f"({', '.join(REFLECTABLE_ROLES)}), or 'none'")
    p.add_argument("--no-reflect", action="store_true", help="disable reflection entirely")
    p.add_argument("--max-reflections", type=int, default=None,
                   help="max critique/revise rounds per decision")
    p.add_argument("--cycle-every-steps", type=int, default=None,
                   help="environment steps per control cycle")
    p.add_argument("--total-steps", type=int, default=None,
                   help="override the scenario length (useful for short LLM runs)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--thinking", dest="thinking", action="store_true", default=None)
    p.add_argument("--no-thinking", dest="thinking", action="store_false")
    p.add_argument("--capacity-report", choices=("exhaustion", "every"), default=None,
                   help="when a negotiation receives the radio capacity report: "
                        "only after RRM exhaustion, or at every renegotiation "
                        "(default: the pack's config.json)")
    p.add_argument("--rrm", choices=RRM_MODES, default="tiered",
                   help="shape of the RRM tier: 'tiered' is the system; "
                        "'no-l2' bypasses the manager so PC and UL-RA read the "
                        "operator intent directly; 'monolithic' replaces all "
                        "three with one agent deciding every value in one call "
                        "(takes the pc role's backend); 'pf' uses no RRM agent: "
                        "maximum-rate power control and a weighted "
                        "proportional-fair scheduler. All three are baselines.")
    p.add_argument("--attack", default=None, metavar="NAME",
                   help="apply one attack from scenarios/attacks/attacks.json "
                        "(forged slice text or UE telemetry)")
    p.add_argument("--security", type=parse_security_spec, default=None,
                   metavar="KEY=VALUE[,KEY=VALUE]",
                   help="override config.json's security switches, e.g. "
                        "tag=false,policy_wall=true")
    p.add_argument("--inject-failures", type=parse_injection_spec, default=None,
                   metavar="ROLE:RATE[,ROLE:RATE]",
                   help="make this share of a role's LLM calls fail every "
                        "attempt and hold, to exercise the fallback on purpose "
                        f"(roles: {', '.join(INJECTABLE_ROLES)}). Mocks are "
                        "never poisoned.")
    p.add_argument("--seed", type=int, default=12345,
                   help="seed for the channel's SNR noise. Runs at one seed are "
                        "comparable across code changes; a sweep over seeds "
                        "(symbiotic_ran.sweep) is how a result becomes a "
                        "distribution. A negative value leaves the RNG unseeded.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data-dir", default=None,
                   help="alternative scenario pack: a directory holding the same "
                        "six JSON files as data/ (see scenarios/contention/). "
                        "Output defaults beside it, so pass --output-dir too if "
                        "you want the results somewhere else.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    roles = {}
    for role in args.llm or []:
        roles[role] = "llm"
    for role in args.mock_agents or []:
        roles[role] = "mock"
    both = set(args.llm or []) & set(args.mock_agents or [])
    if both:
        raise SystemExit(f"role(s) {sorted(both)} given to both --llm and --mock-agents")

    reflect_roles = [] if args.no_reflect else args.reflect

    output_dir = Path(args.output_dir) if args.output_dir else None
    data_dir = Path(args.data_dir) if args.data_dir else None

    state = run(
        model=args.model,
        mock=args.mock,
        agent_roles=roles or None,
        mock_profile=args.mock_profile,
        reflect_roles=reflect_roles,
        max_reflections=args.max_reflections,
        output_dir=output_dir,
        verbose=not args.quiet,
        thinking=args.thinking,
        total_steps=args.total_steps,
        cycle_every_steps=args.cycle_every_steps,
        data_dir=data_dir,
        engine=args.engine,
        base_url=args.base_url,
        seed=args.seed if args.seed >= 0 else None,
        rrm_mode=args.rrm,
        attack=args.attack,
        security_overrides=args.security,
        capacity_report=args.capacity_report,
        inject_failures=args.inject_failures,
    )

    if not args.no_plot:
        from .config import DATA_DIR
        out = output_dir or ((data_dir or DATA_DIR).parent / "output")
        path = plot_run(out / "history.jsonl", out / "simulation_results.png",
                        data_dir=data_dir)
        if not args.quiet:
            print(f"  plot           -> {path}")

    return 0 if state else 1


if __name__ == "__main__":
    raise SystemExit(main())
