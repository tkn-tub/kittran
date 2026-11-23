from __future__ import annotations

import math
import random

import numpy as np

_MCS_ANCHORS_IDX = np.array([0, 4, 9, 14, 19, 24, 28])
_MCS_ANCHORS_EFF = np.array([0.23, 0.60, 1.18, 1.91, 2.73, 3.61, 4.52])

TPC_STEPS = (-1.0, 0.0, 1.0, 3.0)

SNR_NOISE_STD_DB = 0.2
BATTERY_DRAIN_PER_WATT_PER_TICK = 0.05


def mcs_to_spectral_efficiency(mcs_index: float) -> float:
    mcs_index = float(np.clip(mcs_index, 0, 28))
    return float(np.interp(mcs_index, _MCS_ANCHORS_IDX, _MCS_ANCHORS_EFF))


def spectral_efficiency_to_mcs(eff: float) -> int:
    idx = float(np.interp(eff, _MCS_ANCHORS_EFF, _MCS_ANCHORS_IDX))
    return int(round(min(28.0, max(0.0, idx))))


def snr_db_to_spectral_efficiency(snr_db: float) -> float:
    eff = math.log2(1.0 + 10.0 ** (snr_db / 10.0))
    return max(float(_MCS_ANCHORS_EFF[0]), eff)


def snr_to_mcs(snr_db: float) -> int:
    return spectral_efficiency_to_mcs(snr_db_to_spectral_efficiency(snr_db))


def mcs_to_snr_db(mcs_index: float) -> float:
    eff = mcs_to_spectral_efficiency(mcs_index)
    linear = max(1e-9, 2.0 ** eff - 1.0)
    return 10.0 * math.log10(linear)


def ue_snr_ceiling_db(base_mcs_index: float, snr_offset_db: float) -> float:
    return mcs_to_snr_db(base_mcs_index) + snr_offset_db


def ceiling_db(ue: dict, base_mcs_index: float) -> float:
    traced = ue.get("trace_ceiling_db")
    if traced is not None:
        return traced
    return ue_snr_ceiling_db(base_mcs_index, ue.get("snr_offset_db", 0.0))


def snr_at_full_power_db(ue: dict) -> float:
    raw = ue.get("trace_raw_db")
    return raw if raw is not None else ue["snr_ceiling_db"]


def tpc_command(actual_snr_db: float, target_snr_db: float) -> float:
    diff = target_snr_db - actual_snr_db
    if diff >= 3.0:
        return 3.0
    if diff > 0.3:
        return 1.0
    if diff < -0.3:
        return -1.0
    return 0.0


def step_snr(actual_snr_db: float, target_snr_db: float, ceiling_db: float,
             floor_db: float = -25.0) -> float:
    step = tpc_command(actual_snr_db, target_snr_db)
    noise = random.gauss(0.0, SNR_NOISE_STD_DB)
    return max(floor_db, min(ceiling_db, actual_snr_db + step + noise))


def snr_to_tx_power_w(snr_db: float) -> float:
    snr_linear = 10 ** (snr_db / 10)
    return round(0.05 + 0.05 * snr_linear, 4)


UE_MAX_TX_DBM = 23.0


def ue_tx_power_w(ue: dict, snr_db: float) -> float:
    raw = ue.get("trace_raw_db")
    if raw is None:
        return snr_to_tx_power_w(snr_db)
    dbm = UE_MAX_TX_DBM + min(0.0, snr_db - raw)
    return round(10 ** (dbm / 10) / 1000, 6)


def drain_battery(battery_pct, tx_power_w: float, dt_s: float):
    if battery_pct is None:
        return None
    drained = tx_power_w * BATTERY_DRAIN_PER_WATT_PER_TICK * dt_s
    return max(0.0, battery_pct - drained)


def mean_spectral_efficiency(snr_values_db) -> float:
    effs = [snr_db_to_spectral_efficiency(s) for s in snr_values_db]
    return sum(effs) / len(effs) if effs else 0.0


def cell_capacity_mbps(ues: dict, bandwidth_mhz: float, prb_fraction: float,
                       key: str = "actual_snr_db") -> float:
    prb = max(0.0, min(1.0, prb_fraction))
    eff = mean_spectral_efficiency([ue[key] for ue in ues.values()])
    return eff * bandwidth_mhz * prb


def achievable_mbps_at_full_prb(ues: dict, base_mcs_index: float,
                                bandwidth_mhz: float) -> float:
    ceilings = [ceiling_db(ue, base_mcs_index) for ue in ues.values()]
    return mean_spectral_efficiency(ceilings) * bandwidth_mhz


def reachable_mbps_within(ues: dict, base_mcs_index: float, bandwidth_mhz: float,
                          *, max_delta_db_per_cycle: float,
                          cycles_remaining: int) -> float:
    reach = max(0.0, max_delta_db_per_cycle) * max(0, cycles_remaining)
    bounded = []
    for ue in ues.values():
        ceiling = ceiling_db(ue, base_mcs_index)
        base = max(ue["actual_snr_db"], ue.get("target_snr_db", ue["actual_snr_db"]))
        bounded.append(min(ceiling, base + reach))
    return mean_spectral_efficiency(bounded) * bandwidth_mhz
