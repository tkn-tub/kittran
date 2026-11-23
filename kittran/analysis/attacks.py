"""Score attack sweeps: success at the model and after the defences, per class."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from aggregate import bootstrap_ci

from symbiotic_ran.attack_score import one_run


def _fmt(values: list, pct: bool = True) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "     n/a"
    m = mean(vals)
    if len(vals) < 2:
        return f"{m:8.1%}" if pct else f"{m:8.1f}"
    lo, hi = bootstrap_ci(vals)
    return (f"{m:6.1%} [{lo:.0%},{hi:.0%}]" if pct else f"{m:5.1f} [{lo:.1f},{hi:.1f}]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sweeps", nargs="+", type=Path)
    args = p.parse_args()
    print(f"  {'sweep':<34} {'attack':<16} {'model':>17} {'applied':>17} "
          f"{'benign':>7} {'attacked':>8} {'holds':>6} {'attain':>7} {'rej':>4}")
    for sweep in args.sweeps:
        runs = [one_run(r) for r in sorted(sweep.glob("seed_*"))
                if (r / "security.json").exists() and (r / "history.jsonl").exists()]
        if not runs:
            print(f"  {sweep.name:<34} (no runs)")
            continue
        col = {k: [r[k] for r in runs] for k in runs[0]}
        b = [v for v in col["benign"] if v is not None]
        a = [v for v in col["attacked"] if v is not None]
        print(f"  {sweep.name:<34} {runs[0]['attack']:<16} {_fmt(col['model']):>17} "
              f"{_fmt(col['applied']):>17} {mean(b) if b else float('nan'):7.1%} "
              f"{mean(a) if a else float('nan'):8.1%} {mean(col['holds']):6.1f} "
              f"{mean(col['attain']):7.1%} {mean(col['rejected']):4.1f}")
    print("\n  model: share of RRM proposals after the attack starts that met its canary\n"
          "  (after the critic, before deterministic checks); applied: share of those\n"
          "  steps where it reached the network; [..] is the 95% bootstrap CI over\n"
          "  seeds. benign/attacked: first-round critic send-backs before and after\n"
          "  the attack starts. holds per run; attain: within 10% over the run; rej:\n"
          "  telemetry reports rejected per run.")


if __name__ == "__main__":
    main()
