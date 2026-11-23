"""Mean, confidence interval, pass@1 and pass^k over a sweep of seeds."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from statistics import mean

from kpis import cycle_cost, fairness_of, headroom_db, is_short, pack_settings, starving
from runs import (
    group_by_phase,
    live_steps,
    load_decisions,
    load_history,
    phases_in_order,
    rule,
)

BOOTSTRAP_RESAMPLES = 2000


def met(step: dict, tolerance: float) -> bool:
    return step["delivered_mbps"] >= step["sla_mbps"] * (1.0 - tolerance)


def attainment(steps: list[dict], tolerance: float) -> float | None:
    live = [s for s in steps if s["sla_mbps"] > 0]
    return (sum(met(s, tolerance) for s in live) / len(live)) if live else None


def bootstrap_ci(values: list[float], resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0], values[0]) if values else (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def one_run(run_dir: Path, tolerance: float, settings: dict) -> dict:
    history = load_history(run_dir)
    live = live_steps(history)
    misses = [s for s in live if not met(s, tolerance)]
    try:
        decisions = load_decisions(run_dir)
    except SystemExit:
        decisions = []

    fairs = [f for f in (fairness_of(s) for s in live) if f is not None]
    short = [s for s in live if is_short(s)]
    power = [sum(ue["tx_power_w"] for ue in s["ues"].values()) for s in live]
    rate = [s["delivered_mbps"] for s in live]
    budget = settings["cycle_every_steps"] * settings["step_seconds"]
    rrm_seconds = cycle_cost(decisions)["rrm"]
    return {
        "sla": mean(s["sla_mbps"] for s in live) if live else None,
        "delivered": mean(rate) if rate else None,
        "fairness": mean(fairs) if fairs else None,
        "starvation": sum(len(starving(s)) for s in live),
        "headroom_short": mean(headroom_db(s) for s in short) if short else 0.0,
        "mj_per_mbit": (mean(power) / mean(rate) * 1000) if rate and mean(rate) > 0 else None,
        "cycles_over": (sum(1 for v in rrm_seconds.values() if v > budget) / len(rrm_seconds)
                        if rrm_seconds else None),
        "autonomy_events": (sum(d.get("llm_fallbacks", 0) for d in decisions)
                            + sum(1 for d in decisions for x in (d.get("reflection") or [])
                                  if x.get("unresolved"))),
        "attainment": attainment(history, tolerance),
        "system_misses": sum(1 for s in misses if s["achievable_mbps"] >= s["sla_mbps"]),
        "channel_misses": sum(1 for s in misses if s["achievable_mbps"] < s["sla_mbps"]),
        "retries": sum(d.get("llm_retries", 0) for d in decisions),
        "holds": sum(d.get("llm_fallbacks", 0) for d in decisions),
        "calls": sum(d.get("llm_calls", 0) for d in decisions),
        "seconds": sum(d.get("llm_seconds", 0.0) for d in decisions),
        "tokens": sum(d.get("llm_tokens_in", 0) + d.get("llm_tokens_out", 0)
                      for d in decisions),
        "clamped": _clamp_rate(decisions),
        "phases": {label: attainment(steps, tolerance)
                   for label, steps in group_by_phase(history).items()},
        "phase_order": phases_in_order(history),
    }


def _clamp_rate(decisions: list[dict]) -> float | None:
    graded = [d for d in decisions if "clamped" in d]
    if not graded:
        return None
    return sum(1 for d in graded if d["clamped"]) / len(graded)


def summarise(sweep_dir: Path, tolerance: float, threshold: float,
              data_dir: Path | None = None) -> dict:
    run_dirs = sorted(p for p in sweep_dir.glob("seed_*") if (p / "history.jsonl").exists())
    if not run_dirs:
        raise SystemExit(f"{sweep_dir}: no finished seed_*/history.jsonl")
    settings = pack_settings(data_dir)
    runs = [one_run(p, tolerance, settings) for p in run_dirs]
    att = [r["attainment"] for r in runs if r["attainment"] is not None]
    lo, hi = bootstrap_ci(att)
    phases = runs[0]["phase_order"]
    per_phase = {}
    for label in phases:
        vals = [r["phases"].get(label) for r in runs]
        vals = [v for v in vals if v is not None]
        if not vals:
            per_phase[label] = None
            continue
        passed = [v >= threshold for v in vals]
        per_phase[label] = {
            "mean": mean(vals), "min": min(vals), "max": max(vals),
            "pass_at_1": sum(passed) / len(passed), "pass_all": all(passed),
        }
    return {
        "name": sweep_dir.name, "k": len(runs),
        "attainment_mean": mean(att), "ci_lo": lo, "ci_hi": hi,
        "attainment_min": min(att), "attainment_max": max(att),
        "system_misses": mean(r["system_misses"] for r in runs),
        "channel_misses": mean(r["channel_misses"] for r in runs),
        "retries": mean(r["retries"] for r in runs),
        "holds": mean(r["holds"] for r in runs),
        "calls": mean(r["calls"] for r in runs),
        "seconds": mean(r["seconds"] for r in runs),
        "tokens": mean(r["tokens"] for r in runs),
        "clamped": (mean(r["clamped"] for r in runs if r["clamped"] is not None)
                    if any(r["clamped"] is not None for r in runs) else None),
        **{k: (mean(r[k] for r in runs if r[k] is not None)
               if any(r[k] is not None for r in runs) else None)
           for k in ("sla", "delivered", "fairness", "headroom_short", "mj_per_mbit", "cycles_over")},
        "starvation": mean(r["starvation"] for r in runs),
        "autonomy_events": mean(r["autonomy_events"] for r in runs),
        "phases": per_phase,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sweeps", type=Path, nargs="+", help="sweep directories")
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="a step counts as met within this fraction of the SLA")
    p.add_argument("--strict-tolerance", type=float, default=0.05,
                   help="a second, tighter rung reported beside the first (0 to skip)")
    p.add_argument("--pass-threshold", type=float, default=0.80,
                   help="a phase passes when this share of its live steps is met")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="the scenario pack these sweeps used, for the cycle budget "
                        "(default: ./data)")
    p.add_argument("--csv", type=Path, default=None,
                   help="also write the per-configuration table here")
    args = p.parse_args()

    results = [summarise(s, args.tolerance, args.pass_threshold, args.data_dir)
               for s in args.sweeps]

    rule(f"Attainment within {args.tolerance * 100:.0f}%, over seeds")
    print(f"  {'config':<14} {'k':>2} {'mean':>6} {'95% CI':>15} {'min':>6} {'max':>6} "
          f"{'sys':>5} {'chan':>5} {'retry':>5} {'hold':>4} {'calls':>5} {'s/run':>6} "
          f"{'ktok':>6} {'clamp':>6}")
    for r in results:
        clamp = f"{r['clamped'] * 100:5.0f}%" if r["clamped"] is not None else "    --"
        print(f"  {r['name']:<14} {r['k']:>2} {r['attainment_mean'] * 100:>5.1f}% "
              f"[{r['ci_lo'] * 100:5.1f}, {r['ci_hi'] * 100:5.1f}] "
              f"{r['attainment_min'] * 100:>5.1f}% {r['attainment_max'] * 100:>5.1f}% "
              f"{r['system_misses']:>5.1f} {r['channel_misses']:>5.1f} "
              f"{r['retries']:>5.1f} {r['holds']:>4.1f} {r['calls']:>5.0f} {r['seconds']:>6.0f} "
              f"{r['tokens'] / 1000:>6.0f} {clamp}")
    print("\n  sys/chan: missed steps per run with capacity to spare / without; "
          "retry, hold, calls, s/run, ktok: LLM means per run (ktok = thousand "
          "tokens in+out); clamp: share of proposals the guardrail rewrote")

    if args.strict_tolerance > 0:
        strict = [summarise(s, args.strict_tolerance, args.pass_threshold, args.data_dir)
                  for s in args.sweeps]
        t0, t1 = args.tolerance * 100, args.strict_tolerance * 100
        rule(f"Both rungs -- within {t0:.0f}% beside within {t1:.0f}%, over seeds")
        print(f"  {'config':<14} {f'@{t0:.0f}%':>6} {'95% CI':>15} {f'@{t1:.0f}%':>6} "
              f"{'95% CI':>15} {'pass^k':>7} {'pass^k':>7}")
        print(f"  {'':<14} {'':>6} {'':>15} {'':>6} {'':>15} {f'@{t0:.0f}%':>7} {f'@{t1:.0f}%':>7}")
        for a, b in zip(results, strict, strict=True):
            def passed(r):
                phases = [ph for ph in r["phases"].values() if ph is not None]
                return f"{sum(1 for ph in phases if ph['pass_all'])}/{len(phases)}"
            print(f"  {a['name']:<14} {a['attainment_mean'] * 100:>5.1f}% "
                  f"[{a['ci_lo'] * 100:5.1f}, {a['ci_hi'] * 100:5.1f}] "
                  f"{b['attainment_mean'] * 100:>5.1f}% "
                  f"[{b['ci_lo'] * 100:5.1f}, {b['ci_hi'] * 100:5.1f}] "
                  f"{passed(a):>7} {passed(b):>7}")
        print("\n  the same runs scored twice; pass^k: live phases passed in every seed\n"
              "  out of live phases. A gap that opens between the rungs is the tolerance\n"
              "  hiding a difference, not the seeds")

    rule("Network KPIs and cost, over seeds")
    print(f"  {'config':<14} {'sla':>6} {'deliv':>6} {'jain':>6} {'starve':>6} {'hdrm':>6} {'mJ/Mbit':>8} "
          f"{'over':>6} {'human':>6}")
    def cell(row, key, fmt, scale=1.0, suffix=""):
        v = row[key]
        return f"{v * scale:{fmt}}{suffix}" if v is not None else "    --"

    for r in results:
        print(f"  {r['name']:<14} {cell(r, 'sla', '6.1f')} {cell(r, 'delivered', '6.1f')} "
              f"{cell(r, 'fairness', '6.3f')} {r['starvation']:>6.1f} "
              f"{cell(r, 'headroom_short', '6.2f')} {cell(r, 'mj_per_mbit', '8.2f')} "
              f"{cell(r, 'cycles_over', '5.0f', 100, '%')} {r['autonomy_events']:>6.1f}")
    print("\n  sla, deliv: mean SLA and mean delivered rate in Mbit/s over live steps;\n"
          "  jain: fairness over the slices' delivered/target ratios, 1.00 is even;\n"
          "  starve: slices under half their target per run while another was served\n"
          "  whole; hdrm: mean dB of unused radio on steps the cell was short;\n"
          "  mJ/Mbit: mean transmit power over mean delivered throughput; over: share\n"
          "  of acting cycles whose RRM inference exceeded the cycle budget; human:\n"
          "  holds plus critiques shipped unresolved -- where an operator would be called")

    for r in results:
        rule(f"{r['name']} by phase -- pass = >= {args.pass_threshold * 100:.0f}% of "
             f"steps within {args.tolerance * 100:.0f}%")
        print(f"  {'phase':<22} {'mean':>6} {'min':>6} {'max':>6} {'pass@1':>7} {'pass^k':>7}")
        for label, ph in r["phases"].items():
            if ph is None:
                print(f"  {label:<22}     --  RAN off in every seed")
                continue
            print(f"  {label:<22} {ph['mean'] * 100:>5.1f}% {ph['min'] * 100:>5.1f}% "
                  f"{ph['max'] * 100:>5.1f}% {ph['pass_at_1'] * 100:>6.0f}% "
                  f"{'yes' if ph['pass_all'] else 'no':>7}")

    if args.csv:
        fields = ["name", "k", "attainment_mean", "ci_lo", "ci_hi", "attainment_min",
                  "attainment_max", "system_misses", "channel_misses", "retries",
                  "holds", "calls", "seconds", "tokens", "clamped", "fairness",
                  "starvation", "headroom_short", "mj_per_mbit", "cycles_over",
                  "autonomy_events"]
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({k: r[k] for k in fields})
        print(f"\n  table -> {args.csv}")


if __name__ == "__main__":
    main()
