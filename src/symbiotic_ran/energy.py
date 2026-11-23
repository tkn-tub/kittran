from __future__ import annotations

from . import channel as ch

MIN_GAP_DB = 0.5


def power_below_full_db(ue: dict, snr_db: float) -> float:
    return snr_db - ch.snr_at_full_power_db(ue)


def accepted_snr_db(ue: dict, accepted_db: float) -> float:
    return min(ue["snr_ceiling_db"], ch.snr_at_full_power_db(ue) + accepted_db)


def _held_snr_db(ue: dict, accepted_db: float | None = None) -> float:
    if ue.get("battery_pct") is None:
        return ue["snr_ceiling_db"]
    held = min(ue["target_snr_db"], ue["snr_ceiling_db"])
    if accepted_db is not None:
        held = max(held, accepted_snr_db(ue, accepted_db))
    return held


def capacity_report(ues: dict, *, bandwidth_mhz: float, achievable_mbps: float,
                    exhausted: bool = True, accepted: dict | None = None) -> dict:
    accepted = accepted or {}
    held_snr = {ue_id: _held_snr_db(ue, accepted.get(ue_id)) for ue_id, ue in ues.items()}
    held = ch.mean_spectral_efficiency(list(held_snr.values())) * bandwidth_mhz
    below = []
    for ue_id, ue in ues.items():
        gap = ue["snr_ceiling_db"] - held_snr[ue_id]
        if gap < MIN_GAP_DB:
            continue
        extra_w = (ch.ue_tx_power_w(ue, ue["snr_ceiling_db"])
                   - ch.ue_tx_power_w(ue, held_snr[ue_id]))
        below.append({
            "ue": ue_id,
            "slice": ue["slice"],
            "battery": ue.get("battery_pct") is not None,
            "held_snr_db": round(held_snr[ue_id], 2),
            "ceiling_db": round(ue["snr_ceiling_db"], 2),
            "gap_db": round(gap, 2),
            "extra_power_w": round(extra_w, 3),
        })
    return {
        "exhausted": exhausted,
        "held_mbps": round(held, 2),
        "achievable_mbps": round(achievable_mbps, 2),
        "headroom_mbps": round(max(0.0, achievable_mbps - held), 2),
        "below_ceiling": below,
    }


def sla_bound(ues: dict, *, bandwidth_mhz: float, approved: list[str],
              accepted: dict | None = None) -> float:
    accepted = accepted or {}
    snrs = [ue["snr_ceiling_db"] if ue_id in approved
            else _held_snr_db(ue, accepted.get(ue_id))
            for ue_id, ue in ues.items()]
    return round(ch.mean_spectral_efficiency(snrs) * bandwidth_mhz, 2)


def accept_power(ues: dict, *, approved: list[str], accepted: dict | None) -> dict:
    out = dict(accepted or {})
    for ue_id, ue in ues.items():
        if ue.get("battery_pct") is None:
            continue
        snr = (ue["snr_ceiling_db"] if ue_id in approved
               else _held_snr_db(ue, out.get(ue_id)))
        now = round(power_below_full_db(ue, snr), 2)
        out[ue_id] = max(now, out.get(ue_id, now))
    return out


def render_report(report: dict) -> str:
    why = ("the radio tier ran out of the levers it uses on its own, and the "
           "cell is still below its SLA" if report.get("exhausted", True) else
           "sent with every renegotiation, so the tenants decide the battery "
           "trade-off together with the SLA")
    lines = [
        f"RADIO CAPACITY REPORT ({why}):",
        f"- With the battery-powered UEs at the transmit power already accepted "
        f"(or at the targets Power Control holds now, if higher) and the "
        f"grid-powered UEs at their ceiling, the cell can carry "
        f"{report['held_mbps']:.1f} Mbit/s at full PRB.",
        f"- With every UE at its channel ceiling it could carry "
        f"{report['achievable_mbps']:.1f} Mbit/s, i.e. "
        f"{report['headroom_mbps']:.1f} Mbit/s more.",
    ]
    if report["below_ceiling"]:
        lines.append("- That headroom costs battery power on these UEs:")
        for u in report["below_ceiling"]:
            power = "battery-powered" if u["battery"] else "grid-powered"
            lines.append(
                f"    {u['ue']} (slice {u['slice']}, {power}): held at "
                f"{u['held_snr_db']:.1f} dB, ceiling {u['ceiling_db']:.1f} dB "
                f"({u['gap_db']:.1f} dB below), about {u['extra_power_w']:.2f} W "
                f"more to reach it.")
    lines.append(
        "Decide: either keep the SLA at or below the held capacity (no extra "
        "battery), or list the UEs whose energy cost the tenants accept in "
        "approve_energy_ues and set a higher SLA. An SLA above what your "
        "approval pays for is clamped to it.")
    return "\n".join(lines)
