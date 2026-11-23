from __future__ import annotations

import json
from pathlib import Path
from typing import List

_PRIORITY_Y = {"low": 0, "normal": 1, "high": 2}


def load_history(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _off_spans(history: List[dict]):
    spans, start = [], None
    for snap in history:
        if snap["ran_off"] and start is None:
            start = snap["step"]
        elif not snap["ran_off"] and start is not None:
            spans.append((start, snap["step"]))
            start = None
    if start is not None:
        spans.append((start, history[-1]["step"]))
    return spans


def _slice_owner(data_dir: Path | None = None) -> dict:
    try:
        from .config import load_tenant_slice_map
        return {slice_name: tenant
                for tenant, slice_name in load_tenant_slice_map(data_dir).items()}
    except Exception:
        return {}


def agreed_slice_mbps(snap: dict, slice_name: str, owner: dict) -> float | None:
    tenant = owner.get(slice_name)
    shares = snap.get("tenant_shares") or {}
    if tenant is None or tenant not in shares or snap["sla_mbps"] <= 0:
        return None
    return snap["sla_mbps"] * shares[tenant]


def target_slice_mbps(snap: dict, slice_name: str, owner: dict,
                      floors: dict | None = None) -> float | None:
    recorded = (snap.get("slice_targets") or {}).get(slice_name)
    if recorded is not None:
        return recorded
    share_mbps = agreed_slice_mbps(snap, slice_name, owner)
    if share_mbps is None:
        return None
    floor = (floors or {}).get(slice_name, 0.0)
    return max(share_mbps, floor)


def _floors_by_phase(data_dir: Path | None, owner: dict) -> dict:
    try:
        from .config import apply_phase_overrides, load_scenario, load_tenants
        tenants, _ = load_tenants(data_dir)
        phases = load_scenario(data_dir)["phases"]
    except Exception:
        return {}
    tenant_of = {slice_name: tid for slice_name, tid in owner.items()}
    out = {}
    for idx, phase in enumerate(phases, start=1):
        by_id = {t.id: t for t in apply_phase_overrides(tenants, phase)}
        out[idx] = {name: by_id[tid].min_acceptable_mbps
                    for name, tid in tenant_of.items() if tid in by_id}
    return out


def slice_delivered_mbps(snap: dict, slice_name: str) -> float:
    return sum(ue["throughput_mbps"] for ue in snap["ues"].values()
               if ue["slice"] == slice_name)


def plot_run(history_file: Path, out_file: Path, *,
             data_dir: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = load_history(history_file)
    if not history:
        raise ValueError(f"no history to plot in {history_file}")

    steps = [s["step"] for s in history]
    spans = _off_spans(history)
    ue_ids = list(history[0]["ues"])
    slice_names = list(history[0]["slices"])
    owner = _slice_owner(data_dir)

    floors_by_phase = _floors_by_phase(data_dir, owner)

    fig, axes = plt.subplots(6, 1, figsize=(13, 18), sharex=True)

    def shade(ax):
        for start, end in spans:
            ax.axvspan(start, end, color="0.85", zorder=0)

    ax = axes[0]
    ax.plot(steps, [s["sla_mbps"] for s in history],
            label="cell agreed SLA", lw=5, color="0.15", alpha=0.25,
            solid_capstyle="butt")
    ax.plot(steps, [s["delivered_mbps"] for s in history],
            label="cell delivered", lw=1.8, color="0.15")
    ax.plot(steps, [s["achievable_mbps"] for s in history],
            label="achievable @100% PRB", ls="--", lw=1, color="0.4")

    for idx, name in enumerate(slice_names):
        colour = f"C{idx}"
        tenant = owner.get(name, "")
        agreed = [agreed_slice_mbps(s, name, owner) for s in history]
        ax.plot(steps, [slice_delivered_mbps(s, name) for s in history],
                lw=1.5, color=colour,
                label=f"{name} delivered" + (f" ({tenant})" if tenant else ""))
        if any(v is not None for v in agreed):
            ax.plot(steps, agreed, lw=1.2, ls=":", color=colour, alpha=0.95,
                    label=f"{name} agreed")

    ax.set_ylabel("Mbit/s")
    ax.margins(y=0.42)
    ax.set_ylim(bottom=0)
    ax.set_title("Tier 2 -- negotiated SLA (dotted / band) vs delivered (solid), "
                 "for the cell and for each tenant\n(shaded = RAN switched off)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), fontsize=6.5,
              ncol=3, framealpha=0.92)
    shade(ax)
    ax.grid(alpha=0.3)

    ax = axes[1]
    any_target = False
    for idx, name in enumerate(slice_names):
        colour = f"C{idx}"
        tenant = owner.get(name, "")
        target = [target_slice_mbps(s_, name, owner,
                                    floors_by_phase.get(s_["phase"]))
                  for s_ in history]
        got = [slice_delivered_mbps(s_, name) for s_ in history]
        if not any(v is not None for v in target):
            continue
        any_target = True
        ax.plot(steps, got, lw=1.5, color=colour,
                label=f"{name} delivered" + (f" ({tenant})" if tenant else ""))
        ax.plot(steps, target, lw=1.3, ls="-.", color=colour, alpha=0.95,
                label=f"{name} TARGET")
        ax.fill_between(
            steps, got, [t if t is not None else g for t, g in zip(target, got, strict=True)],
            where=[t is not None and g < t for t, g in zip(target, got, strict=True)],
            color=colour, alpha=0.12, interpolate=True, linewidth=0)
    ax.set_ylabel("Mbit/s")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.42)
    ax.set_title("Tier 2 -- per-slice OBLIGATION: max(share x SLA, contractual "
                 "floor) (dash-dot) vs delivered (solid)\n"
                 "shaded gap = the slice is under its target; compare with panel "
                 "1, which draws the share alone")
    if any_target:
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), fontsize=6.5,
                  ncol=3, framealpha=0.92)
    shade(ax)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(steps, [s["prb_fraction"] * 100 for s in history], color="tab:green", lw=2)
    ax.set_ylabel("PRB %")
    ax.set_ylim(-5, 105)
    ax.set_title("Tier 0 -- PRB allocation (the fast deterministic controller)")
    shade(ax)
    ax.grid(alpha=0.3)

    ax = axes[3]
    for ue_id in ue_ids:
        ax.plot(steps, [s["ues"][ue_id]["actual_snr_db"] for s in history],
                lw=1.5, label=f"{ue_id} actual")
        ax.plot(steps, [s["ues"][ue_id]["snr_ceiling_db"] for s in history],
                lw=0.8, ls=":", alpha=0.6)
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Tier 1 -- per-UE SNR set by the Power Control agent "
                 "(dotted = channel ceiling)")
    ax.legend(loc="upper right", fontsize=7, ncol=len(ue_ids))
    shade(ax)
    ax.grid(alpha=0.3)

    ax = axes[4]
    for idx, name in enumerate(slice_names):
        members = [u for u, ue in history[0]["ues"].items() if ue["slice"] == name]
        colour = f"C{idx}"
        ax.plot(steps,
                [sum(s["ues"][u]["throughput_mbps"] for u in members) for s in history],
                lw=1.8, color=colour, label=f"{name} delivered")
        ax.plot(steps, [s["slices"][name]["throttle_mbps"] for s in history],
                lw=0.8, ls="--", alpha=0.7, color=colour, label=f"{name} throttle")
        agreed = [agreed_slice_mbps(s, name, owner) for s in history]
        if any(v is not None for v in agreed):
            tenant = owner.get(name, "")
            ax.plot(steps, agreed, lw=1.4, ls=":", color=colour, alpha=0.95,
                    label=f"{name} agreed ({tenant})")
    ax.set_ylabel("Mbit/s")
    ax.set_title("Tier 1 -- per-slice throughput vs throttle (UL-RA agent), "
                 "with each tenant's agreed share (dotted)")
    ax.margins(y=0.30)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=6.5, ncol=len(slice_names),
              framealpha=0.92)
    shade(ax)
    ax.grid(alpha=0.3)

    ax = axes[5]
    ax.plot(steps, [s["base_mcs_index"] for s in history],
            color="tab:purple", lw=2, label="base MCS (phase)")
    for ue_id in ue_ids:
        ax.plot(steps, [s["ues"][ue_id]["mcs_index"] for s in history],
                lw=0.8, alpha=0.6, label=f"{ue_id} MCS")
    ax.set_ylabel("MCS index")
    ax.set_xlabel("environment step")
    ax.set_title("Tier 0 -- channel: phase base MCS and realised per-UE MCS")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    shade(ax)
    ax.grid(alpha=0.3)

    seen = set()
    for snap in history:
        if snap["phase"] not in seen:
            seen.add(snap["phase"])
            for a in axes:
                a.axvline(snap["step"], color="0.6", lw=0.7, ls="-", alpha=0.7)
            axes[0].annotate(
                snap["label"], (snap["step"], axes[0].get_ylim()[1]),
                fontsize=6.5, rotation=90, va="top", ha="right", color="0.35")

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=130)
    plt.close(fig)
    return out_file


def main() -> None:
    from .config import DATA_DIR
    output_dir = DATA_DIR.parent / "output"
    print(plot_run(output_dir / "history.jsonl", output_dir / "simulation_results.png"))


if __name__ == "__main__":
    main()
