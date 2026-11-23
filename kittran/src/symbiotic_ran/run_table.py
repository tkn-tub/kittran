from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = (
    "step", "cycle", "phase", "phase_label", "ran_off",
    "slice", "tenant", "ues",
    "priority", "cap_mbps",
    "target_mbps", "agreed_share_pct", "share_mbps",
    "delivered_mbps", "delivered_pct", "gap_mbps", "over_target_mbps",
    "mcs_mean", "snr_mean_db", "snr_ceiling_mean_db",
    "sla_mbps", "cell_delivered_mbps", "achievable_mbps", "capacity_mbps",
    "prb_pct", "base_mcs",
)

LEGEND = (
    ("step / cycle / phase", "environment step, control cycle, phase index and label"),
    ("slice / tenant / ues", "the slice, the tenant that owns it, how many UEs it holds"),
    ("priority", "scheduler rung set by UL-RA: low=0.5, normal=1.0, high=3.0, proportional"),
    ("cap_mbps", "the slice's throttle -- an UPPER cap on its total, never a reservation"),
    ("target_mbps", "what the slice is OWED: max(share x SLA, contractual floor). "
                    "The figure the operator intent states as TARGET and the one the "
                    "agents are scored against"),
    ("agreed_share_pct", "the tenants' agreed division of the cell -- the ORDERING, not the obligation"),
    ("share_mbps", "share x SLA alone. Equals target_mbps unless a floor binds; "
                   "where one does, the two disagree in sign"),
    ("delivered_mbps", "what the slice's UEs actually carried, summed"),
    ("delivered_pct", "delivered_mbps as a percentage of the whole cell's delivery"),
    ("gap_mbps", "target_mbps - delivered_mbps when short, else 0 -- how far under its obligation"),
    ("over_target_mbps", "delivered_mbps - target_mbps when over, else 0"),
    ("mcs_mean / snr_mean_db", "mean realised MCS index and SNR across the slice's UEs"),
    ("snr_ceiling_mean_db", "mean channel-imposed SNR ceiling -- what Power Control is climbing toward"),
    ("sla_mbps", "the cell-wide total the tenants negotiated"),
    ("cell_delivered_mbps", "what the whole cell carried"),
    ("achievable_mbps", "cell ceiling at 100% PRB with every UE at its ceiling"),
    ("capacity_mbps", "cell capacity at the PRB fraction actually in use"),
    ("prb_pct", "PRB allocation set by the Tier 0 controller"),
    ("base_mcs", "the phase's base MCS index -- the channel regime, set by the scenario"),
)


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def build_rows(history: list[dict],
               tenant_slice_map: dict | None = None) -> list[dict]:
    owner = {name: tid for tid, name in (tenant_slice_map or {}).items()}
    rows = []
    for snap in history:
        shares = snap.get("tenant_shares") or {}
        targets = snap.get("slice_targets") or {}
        cell_delivered = snap.get("delivered_mbps") or 0.0
        sla = snap.get("sla_mbps") or 0.0

        for name, sl in snap["slices"].items():
            members = [ue for ue in snap["ues"].values() if ue["slice"] == name]
            got = sum(ue["throughput_mbps"] for ue in members)
            tenant = owner.get(name)
            share = shares.get(tenant) if tenant else None
            target = targets.get(name)
            if target is None and share is not None and sla > 0:
                target = share * sla
            rows.append({
                "step": snap["step"],
                "cycle": snap.get("cycle"),
                "phase": snap.get("phase"),
                "phase_label": snap.get("label", ""),
                "ran_off": snap.get("ran_off", False),
                "slice": name,
                "tenant": tenant or "",
                "ues": len(members),
                "priority": sl.get("priority", ""),
                "cap_mbps": round(sl.get("throttle_mbps", 0.0), 2),
                "target_mbps": round(target, 2) if target is not None else "",
                "agreed_share_pct": round(share * 100, 1) if share is not None else "",
                "share_mbps": (round(share * sla, 2)
                               if share is not None and sla > 0 else ""),
                "delivered_mbps": round(got, 2),
                "delivered_pct": (round(got / cell_delivered * 100, 1)
                                  if cell_delivered > 0 else ""),
                "gap_mbps": (round(max(0.0, target - got), 2)
                             if target is not None else ""),
                "over_target_mbps": (round(max(0.0, got - target), 2)
                                     if target is not None else ""),
                "mcs_mean": _mean([ue.get("mcs_index") for ue in members]),
                "snr_mean_db": _mean([ue.get("actual_snr_db") for ue in members]),
                "snr_ceiling_mean_db": _mean(
                    [ue.get("snr_ceiling_db") for ue in members]),
                "sla_mbps": round(sla, 2),
                "cell_delivered_mbps": round(cell_delivered, 2),
                "achievable_mbps": round(snap.get("achievable_mbps") or 0.0, 2),
                "capacity_mbps": round(snap.get("capacity_mbps") or 0.0, 2),
                "prb_pct": round((snap.get("prb_fraction") or 0.0) * 100, 1),
                "base_mcs": snap.get("base_mcs_index"),
            })
    return rows


def cycle_rows(rows: list[dict]) -> list[dict]:
    last_step = {}
    for row in rows:
        last_step[row["cycle"]] = max(last_step.get(row["cycle"], -1), row["step"])
    return [r for r in rows if r["step"] == last_step[r["cycle"]]]


_MD_COLUMNS = (
    ("step", "step"), ("phase_label", "phase"), ("slice", "slice"),
    ("priority", "prio"), ("cap_mbps", "cap"), ("target_mbps", "target"),
    ("delivered_mbps", "got"), ("gap_mbps", "short"),
    ("over_target_mbps", "over"), ("agreed_share_pct", "share%"),
    ("delivered_pct", "got%"), ("mcs_mean", "mcs"),
    ("sla_mbps", "cell SLA"), ("cell_delivered_mbps", "cell got"),
    ("achievable_mbps", "achiev"), ("prb_pct", "PRB%"),
)


def render_markdown(rows: list[dict], *, title: str = "Run table") -> str:
    picked = cycle_rows(rows)
    head = "| " + " | ".join(label for _, label in _MD_COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in _MD_COLUMNS) + "|"
    body = []
    prev_cycle = None
    for row in picked:
        if prev_cycle is not None and row["cycle"] != prev_cycle:
            body.append("|" + "|".join("" for _ in _MD_COLUMNS) + "|")
        prev_cycle = row["cycle"]
        body.append("| " + " | ".join(
            str(row[key]) if row[key] != "" else "-" for key, _ in _MD_COLUMNS)
            + " |")

    legend = "\n".join(f"- **{name}** — {what}" for name, what in LEGEND)
    return (
        f"# {title}\n\n"
        f"One row per slice per control cycle, at the last step of the cycle — "
        f"the state as it stood when the agents last acted. Every step of every "
        f"slice, with all {len(COLUMNS)} columns, is in `run_table.csv` beside "
        f"this file.\n\n"
        + head + "\n" + rule + "\n" + "\n".join(body) + "\n\n"
        f"## What the columns mean\n\n{legend}\n\n"
        f"**`target` vs `share%` is the distinction to watch.** The share is the "
        f"ordering the tenants agreed; the target is `max(share × SLA, "
        f"contractual floor)`, which is what they are owed. Where a floor binds "
        f"the two disagree in sign: a slice can hold 67% of the cell against an "
        f"agreed 49% and still be 4 Mbit/s under its obligation.\n"
    )


def write_run_table(history: list[dict], output_dir: Path, *,
                    tenant_slice_map: dict | None = None,
                    title: str = "Run table") -> tuple[Path, Path]:
    rows = build_rows(history, tenant_slice_map)
    csv_file = output_dir / "run_table.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    md_file = output_dir / "run_table.md"
    md_file.write_text(render_markdown(rows, title=title))
    return csv_file, md_file
