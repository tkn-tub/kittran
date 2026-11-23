"""Table of adaptive-attack results: fixed text against the best adaptive text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("root", type=Path)
    args = p.parse_args()
    rows = sorted((json.loads(f.read_text()) for f in args.root.glob("*/summary.json")),
                  key=lambda s: (s["attack"], s["condition"]))
    print("| attack | condition | round 0 applied (train) | best round | best applied (train) "
          "| held-out model | held-out applied | held-out attainment |")
    print("|---|---|---|---|---|---|---|---|")
    for s in rows:
        e = s["eval"]
        print(f"| {s['attack']} | {s['condition']} | {s['round0_train']['applied']:.0%} | "
              f"{s['best_round']} | {s['train_best']['applied']:.0%} | {e['model']:.0%} | "
              f"{e['applied']:.0%} | {e['attain']:.0%} |")
    print("\nRound 0 is the fixed text of attacks.json on the training seeds (1, 2). The best "
          "round is the attacker's most successful text on those seeds; the held-out columns "
          "run that text on seeds 6 to 10, which the attacker never saw. model: share of RRM "
          "proposals that met the goal; applied: share of steps where it reached the network.\n")
    for s in rows:
        print(f"- **{s['attack']} / {s['condition']}** best text (round {s['best_round']}): "
              f"{s['best_injection']!r}")


if __name__ == "__main__":
    main()
