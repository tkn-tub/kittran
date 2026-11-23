from __future__ import annotations

import argparse
import json
from pathlib import Path


def live_steps(history: list[dict]) -> list[dict]:
    return [s for s in history if s["sla_mbps"] > 0]


def load_history(run_dir: Path) -> list[dict]:
    return _jsonl(run_dir / "history.jsonl")


def load_decisions(run_dir: Path) -> list[dict]:
    return _jsonl(run_dir / "decisions.jsonl")


def load_negotiations(run_dir: Path) -> list[dict]:
    path = run_dir / "negotiation_log.json"
    return json.loads(path.read_text()) if path.exists() else []


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found -- is that a run directory?")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def phases_in_order(history: list[dict]) -> list[str]:
    return list(dict.fromkeys(s["label"] for s in history))


def group_by_phase(history: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for snap in history:
        out.setdefault(snap["label"], []).append(snap)
    return out


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("run_dir", type=Path, nargs="?", default=Path("output"),
                   help="a run's output directory (default: ./output)")
    return p


def rule(title: str, width: int = 78) -> None:
    print(f"\n{title}")
    print("-" * width)
