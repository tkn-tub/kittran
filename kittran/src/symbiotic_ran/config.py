from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Tenant:

    id: str
    name: str
    initial_intent_mbps: float
    min_acceptable_mbps: float
    utility_weight: float
    willingness_to_pay: float
    description: str = ""


OVERRIDABLE_TENANT_FIELDS = frozenset({
    "initial_intent_mbps",
    "min_acceptable_mbps",
    "utility_weight",
    "willingness_to_pay",
    "description",
})


def apply_phase_overrides(tenants: List[Tenant], phase: dict | None) -> List[Tenant]:
    overrides = (phase or {}).get("tenant_overrides") or {}
    known = {t.id for t in tenants}
    unknown = set(overrides) - known
    if unknown:
        raise KeyError(f"tenant_overrides names unknown tenant(s): {sorted(unknown)}")

    out = []
    for t in tenants:
        fields = overrides.get(t.id)
        if not fields:
            out.append(t)
            continue
        bad = set(fields) - OVERRIDABLE_TENANT_FIELDS
        if bad:
            raise KeyError(
                f"tenant_overrides for {t.id} names non-overridable field(s): "
                f"{sorted(bad)}")
        out.append(replace(t, **fields))
    return out


def _read_json(data_dir: Path, filename: str) -> dict:
    with open(Path(data_dir) / filename) as f:
        return json.load(f)


def load_tenants(data_dir: Path | None = None) -> Tuple[List[Tenant], dict]:
    data = _read_json(data_dir or DATA_DIR, "tenants.json")
    tenants = [Tenant(**t) for t in data["tenants"]]
    return tenants, data["ran"]


def load_scenario(data_dir: Path | None = None) -> dict:
    return _read_json(data_dir or DATA_DIR, "mobility_scenario.json")


def load_config(data_dir: Path | None = None) -> dict:
    return _read_json(data_dir or DATA_DIR, "config.json")


def load_guardrails(data_dir: Path | None = None) -> dict:
    return _read_json(data_dir or DATA_DIR, "guardrails.json")


def load_initial_radio(data_dir: Path | None = None) -> dict:
    return _read_json(data_dir or DATA_DIR, "initial_state.json")


def load_tenant_slice_map(data_dir: Path | None = None) -> dict:
    data = _read_json(data_dir or DATA_DIR, "tenant_slice_map.json")
    return {tid: spec["slice"] for tid, spec in data["tenants"].items()}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
