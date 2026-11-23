"""Run one configuration over many seeds, so a result becomes a distribution."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .cli import main as run_cli
from .config import DATA_DIR


def seed_list(count: int, explicit: str | None) -> list[int]:
    if explicit:
        return [int(s) for s in explicit.split(",") if s.strip()]
    return list(range(1, count + 1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m symbiotic_ran.sweep",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name", required=True,
                   help="configuration label; becomes the sweep directory")
    p.add_argument("--seeds", type=int, default=10, help="how many seeds, 1..k")
    p.add_argument("--seed-list", default=None,
                   help="explicit comma-separated seeds instead of 1..k")
    p.add_argument("--root", type=Path, default=None,
                   help="where sweeps live (default: <data>/../output/sweeps)")
    p.add_argument("cli_args", nargs=argparse.REMAINDER,
                   help="everything after -- is passed to symbiotic_ran.cli")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    passthrough = [a for a in args.cli_args if a != "--"]
    for forbidden in ("--seed", "--output-dir"):
        if any(a == forbidden or a.startswith(forbidden + "=") for a in passthrough):
            raise SystemExit(f"{forbidden} is set by the sweep; drop it from the CLI args")

    root = args.root or (DATA_DIR.parent / "output" / "sweeps")
    sweep_dir = root / args.name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "sweep.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else {"name": args.name, "cli_args": passthrough, "runs": {}})
    if manifest["cli_args"] != passthrough:
        raise SystemExit(
            f"{manifest_path} was made with {manifest['cli_args']}; "
            f"a sweep is one configuration -- use another --name")

    seeds = seed_list(args.seeds, args.seed_list)
    print(f"sweep {args.name}: {len(seeds)} seed(s) -> {sweep_dir}")
    for seed in seeds:
        run_dir = sweep_dir / f"seed_{seed}"
        if (run_dir / "history.jsonl").exists():
            print(f"  seed {seed:>3}  already done, skipping")
            continue
        started = time.perf_counter()
        rc = run_cli(passthrough + ["--seed", str(seed), "--output-dir", str(run_dir),
                                    "--no-plot", "--quiet"])
        elapsed = round(time.perf_counter() - started, 1)
        manifest["runs"][str(seed)] = {
            "seconds": elapsed, "ok": rc == 0,
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  seed {seed:>3}  {'ok' if rc == 0 else 'FAILED'}  {elapsed:>7.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
