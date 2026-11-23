"""The network KPIs the attainment number cannot see, plus what a cycle cost."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from runs import (
    group_by_phase,
    live_steps,
    load_decisions,
    load_history,
    parser,
    phases_in_order,
    rule,
)

FALLBACK = {"cycle_every_steps": 5, "step_seconds": 1.0, "prb_max_fraction": 1.0,
            "ran_off_threshold_mbps": 6.0}
STARVED_FRACTION = 0.5
SHORT_TOLERANCE = 0.10


def pack_settings(data_dir: Path | None) -> dict:
    out = dict(FALLBACK)
    path = (data_dir or Path("data")) / "config.json"
    if not path.exists():
        return out
    cfg = json.loads(path.read_text())
    out["cycle_every_steps"] = cfg["rrm"].get("cycle_every_steps", out["cycle_every_steps"])
    out["step_seconds"] = cfg["rrm"].get("step_seconds", out["step_seconds"])
    out["prb_max_fraction"] = cfg["controller"].get("prb_max_fraction", out["prb_max_fraction"])
    out["ran_off_threshold_mbps"] = cfg["negotiation"].get(
        "ran_off_threshold_mbps", out["ran_off_threshold_mbps"])
    return out


def slice_delivered(snap: dict) -> dict:
    out: dict[str, float] = defaultdict(float)
    for ue in snap["ues"].values():
        out[ue["slice"]] += ue["throughput_mbps"]
    return dict(out)


def jain(values) -> float | None:
    values = [max(0.0, float(v)) for v in values]
    if not values or sum(values) <= 0:
        return None
    return sum(values) ** 2 / (len(values) * sum(v * v for v in values))


def fairness_of(snap: dict) -> float | None:
    targets = snap.get("slice_targets") or {}
    if not targets:
        return None
    delivered = slice_delivered(snap)
    ratios = [delivered.get(name, 0.0) / t for name, t in targets.items() if t > 0]
    return jain(ratios) if len(ratios) > 1 else None


def starving(snap: dict) -> list:
    targets = snap.get("slice_targets") or {}
    if not targets:
        return []
    delivered = slice_delivered(snap)
    served = [name for name, t in targets.items()
              if t > 0 and delivered.get(name, 0.0) >= 0.95 * t]
    if not served:
        return []
    return [name for name, t in targets.items()
            if t > 0 and delivered.get(name, 0.0) < STARVED_FRACTION * t]


def headroom_db(snap: dict) -> float:
    ues = snap["ues"].values()
    return mean(ue["snr_ceiling_db"] - ue["actual_snr_db"] for ue in ues) if ues else 0.0


def is_short(snap: dict) -> bool:
    return snap["delivered_mbps"] < snap["sla_mbps"] * (1.0 - SHORT_TOLERANCE)


def cycle_cost(decisions: list[dict]) -> dict:
    rrm: dict[int, float] = defaultdict(float)
    tier2: dict[int, float] = defaultdict(float)
    tokens: dict[int, int] = defaultdict(int)
    for d in decisions:
        if "llm_seconds" not in d:
            continue
        bucket = tier2 if d["agent"] == "negotiation" else rrm
        bucket[d["cycle"]] += d["llm_seconds"]
        tokens[d["cycle"]] += d.get("llm_tokens_in", 0) + d.get("llm_tokens_out", 0)
    return {"rrm": dict(rrm), "negotiation": dict(tier2), "tokens": dict(tokens)}


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=None,
                   help="the scenario pack this run used, for its timescale and "
                        "thresholds (default: ./data)")
    args = p.parse_args()

    history = load_history(args.run_dir)
    decisions = load_decisions(args.run_dir)
    settings = pack_settings(args.data_dir)
    live = live_steps(history)
    if not live:
        raise SystemExit("no step in this run carried a live SLA")

    rule(f"Fairness and starvation -- {args.run_dir}")
    print(f"  {'phase':<22} {'steps':>5} {'jain':>6} {'worst':>6}  starved slices")
    all_fair, all_starved = [], []
    for label in phases_in_order(history):
        steps = [s for s in group_by_phase(history)[label] if s["sla_mbps"] > 0]
        fairs = [f for f in (fairness_of(s) for s in steps) if f is not None]
        starved = [name for s in steps for name in starving(s)]
        all_fair += fairs
        all_starved += starved
        if not fairs:
            print(f"  {label:<22} {len(steps):>5}     --      --  (RAN off)")
            continue
        counts = ", ".join(f"{n} x{starved.count(n)}" for n in dict.fromkeys(starved)) or "-"
        print(f"  {label:<22} {len(steps):>5} {mean(fairs):>6.2f} {min(fairs):>6.2f}  {counts}")
    print(f"\n  Jain's index over the three slices' delivered/target ratios: 1.00 is every\n"
          f"  slice at the same fraction of what it was owed, 0.33 is one slice taking\n"
          f"  everything. starved: a slice under {STARVED_FRACTION:.0%} of its target while the\n"
          f"  cell still had radio to spare -- a division failure, not a capacity one.")
    print(f"\n  run mean {mean(all_fair):.3f}, worst step {min(all_fair):.3f}, "
          f"{len(all_starved)} starvation event(s)")

    rule("Radio: headroom, PRB, outage")
    short = [s for s in live if is_short(s)]
    if short:
        print(f"  unused headroom while short   {mean(headroom_db(s) for s in short):>6.2f} dB "
              f"mean over {len(short)} short step(s)")
    else:
        print("  unused headroom while short       -- (no step was short)")
    print(f"  unused headroom, all live     {mean(headroom_db(s) for s in live):>6.2f} dB")
    prbs = [s["prb_fraction"] for s in live]
    pinned = sum(1 for x in prbs if x >= settings["prb_max_fraction"] - 1e-6)
    print(f"  PRB allocation                {mean(prbs) * 100:>6.1f}% mean, "
          f"pinned at max on {pinned}/{len(prbs)} step(s)")
    off = [s for s in history if s["ran_off"]]
    right = sum(1 for s in off if s["achievable_mbps"] < settings["ran_off_threshold_mbps"])
    print(f"  outage                        {len(off):>6} step(s) with the RAN off, "
          f"{right} correct, {len(off) - right} not")
    print(f"\n  headroom is the mean dB between a UE's actual SNR and its channel ceiling:\n"
          f"  radio that was there and unasked for. The first row counts only steps more\n"
          f"  than {SHORT_TOLERANCE:.0%} under the SLA -- that is capacity the cell needed and did not\n"
          f"  take. An outage is correct when achievable was under the "
          f"{settings['ran_off_threshold_mbps']:.1f} Mbit/s\n  switch-off threshold.")

    rule("Energy")
    power = [sum(ue["tx_power_w"] for ue in s["ues"].values()) for s in live]
    delivered = [s["delivered_mbps"] for s in live]
    mean_power, mean_rate = mean(power), mean(delivered)
    print(f"  mean radio power              {mean_power * 1000:>6.1f} mW over {len(live)} live step(s)")
    if mean_rate > 0:
        print(f"  energy per delivered Mbit     {mean_power / mean_rate * 1000:>6.2f} mJ/Mbit")
    first, last = history[0]["ues"], history[-1]["ues"]
    drains = {u: first[u]["battery_pct"] - last[u]["battery_pct"]
              for u in first if first[u]["battery_pct"] is not None}
    if drains:
        print("  battery drain over the run    " +
              ", ".join(f"{u} {d:.2f} pp" for u, d in drains.items()))
    print("\n  power is the sum of the four UEs' transmit power. mJ/Mbit divides mean power\n"
          "  by mean delivered throughput, so it does not depend on how long a step is\n"
          "  declared to be. pp is percentage points of battery.")

    rule("Control loop and cost")
    budget = settings["cycle_every_steps"] * settings["step_seconds"]
    cost = cycle_cost(decisions)
    rrm_seconds = cost["rrm"]
    if not rrm_seconds:
        print("  every RRM role was mocked in this run, so no cycle spent inference time.")
    else:
        over = sorted(c for c, s in rrm_seconds.items() if s > budget)
        print(f"  cycle budget                  {budget:>6.1f} s "
              f"({settings['cycle_every_steps']} steps x {settings['step_seconds']} s declared)")
        print(f"  RRM inference per cycle       {mean(rrm_seconds.values()):>6.2f} s mean, "
              f"{max(rrm_seconds.values()):.2f} s worst, over {len(rrm_seconds)} acting cycle(s)")
        print(f"  cycles within budget          {len(rrm_seconds) - len(over):>6}/{len(rrm_seconds)}"
              + (f"   over: {over}" if over else ""))
        negotiation = cost["negotiation"]
        if negotiation:
            print(f"  negotiation, off that clock   {mean(negotiation.values()):>6.2f} s mean "
                  f"over {len(negotiation)} negotiation(s)")
        tokens = cost["tokens"]
        if any(tokens.values()):
            print(f"  tokens per acting cycle       {mean(tokens.values()):>6.0f} mean")
    holds = sum(d.get("llm_fallbacks", 0) for d in decisions)
    unresolved = sum(1 for d in decisions for x in (d.get("reflection") or [])
                     if x.get("unresolved"))
    print(f"  autonomy proxy                {holds + unresolved:>6} event(s) "
          f"({holds} hold(s), {unresolved} critique(s) shipped unresolved)")
    print("\n  a cycle is over budget when the RRM tier's inference took longer than the\n"
          "  cycle it was steering -- the network would have waited. Negotiation is\n"
          "  listed apart because it fires on a trigger, not on the cycle clock.\n"
          "  the autonomy proxy counts the moments a human would have been called:\n"
          "  a call that gave up, or a decision that shipped with the reviewer\n"
          "  still objecting.")


if __name__ == "__main__":
    main()
