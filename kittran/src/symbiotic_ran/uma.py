"""Per-UE SNR-ceiling traces from the 3GPP TR 38.901 UMa large-scale model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SPEED_OF_LIGHT = 299_792_458.0
CARRIER_HZ = 3.5e9
BANDWIDTH_HZ = 20e6
UE_TX_DBM = 23.0
BS_GAIN_DBI = 8.0
UE_GAIN_DBI = 0.0
NOISE_FIGURE_DB = 5.0
NOISE_DBM = -174.0 + 10.0 * math.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB
BS_HEIGHT_M = 25.0
UE_HEIGHT_M = 1.5
EFFECTIVE_ENV_HEIGHT_M = 1.0

STEPS = 100
STEP_S = 1.0
SPEED_MPS = 10.0
ROUTE_HALF_M = SPEED_MPS * STEP_S * STEPS / 2
ROAD_OFFSET_M = 40.0

SIGMA_SF_DB = {True: 4.0, False: 6.0}
CORR_DIST_SF_M = {True: 37.0, False: 50.0}
LOS_CORR_DIST_M = 50.0

_FC_DS_GHZ = max(CARRIER_HZ / 1e9, 6.0)
K_FACTOR_DB = (9.0, 3.5)
CORR_DIST_K_M = 12.0
LG_DS = {True: (-6.955 - 0.0963 * math.log10(_FC_DS_GHZ), 0.66),
         False: (-6.28 - 0.204 * math.log10(_FC_DS_GHZ), 0.39)}
CORR_DIST_DS_M = {True: 30.0, False: 40.0}

CEILING_CAP_DB = 18.0

LAYOUT = {
    "fwa_1": {"start": (150.0, 120.0), "direction": 0},
    "fwa_2": {"start": (-350.0, 200.0), "direction": 0},
    "mtc_1": {"start": (-ROUTE_HALF_M, ROAD_OFFSET_M), "direction": +1},
    "iot_1": {"start": (ROUTE_HALF_M, -ROAD_OFFSET_M), "direction": -1},
}


def los_probability(d2d_m: np.ndarray) -> np.ndarray:
    d = np.maximum(d2d_m, 1e-3)
    p = 18.0 / d + np.exp(-d / 63.0) * (1.0 - 18.0 / d)
    return np.where(d <= 18.0, 1.0, p)


def pathloss_db(d2d_m: np.ndarray, *, los: np.ndarray) -> np.ndarray:
    fc_ghz = CARRIER_HZ / 1e9
    d2d = np.maximum(d2d_m, 10.0)
    d3d = np.hypot(d2d, BS_HEIGHT_M - UE_HEIGHT_M)
    h_bs = BS_HEIGHT_M - EFFECTIVE_ENV_HEIGHT_M
    h_ut = UE_HEIGHT_M - EFFECTIVE_ENV_HEIGHT_M
    d_bp = 4.0 * h_bs * h_ut * CARRIER_HZ / SPEED_OF_LIGHT

    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * math.log10(fc_ghz)
    pl2 = (28.0 + 40.0 * np.log10(d3d) + 20.0 * math.log10(fc_ghz)
           - 9.0 * np.log10(d_bp**2 + (BS_HEIGHT_M - UE_HEIGHT_M) ** 2))
    pl_los = np.where(d2d <= d_bp, pl1, pl2)
    pl3 = (13.54 + 39.08 * np.log10(d3d) + 20.0 * math.log10(fc_ghz)
           - 0.6 * (UE_HEIGHT_M - 1.5))
    pl_nlos = np.maximum(pl_los, pl3)
    return np.where(los, pl_los, pl_nlos)


def _positions(start: tuple, direction: int) -> np.ndarray:
    t = np.arange(STEPS) * STEP_S
    pos = np.tile(np.asarray(start, dtype=float), (STEPS, 1))
    pos[:, 0] += direction * SPEED_MPS * t
    return pos


def _correlated_normal(travelled_m: np.ndarray, corr_dist_m: float,
                       rng: np.random.Generator) -> np.ndarray:
    out = np.empty(travelled_m.size)
    out[0] = rng.standard_normal()
    for i in range(1, travelled_m.size):
        rho = math.exp(-(travelled_m[i] - travelled_m[i - 1]) / corr_dist_m)
        out[i] = rho * out[i - 1] + math.sqrt(1.0 - rho * rho) * rng.standard_normal()
    return out


def ue_trace(start: tuple, direction: int, rng: np.random.Generator) -> dict:
    pos = _positions(start, direction)
    d2d = np.hypot(pos[:, 0], pos[:, 1])
    travelled = SPEED_MPS * STEP_S * np.arange(STEPS) * abs(direction)

    g_los = _correlated_normal(travelled, LOS_CORR_DIST_M, rng)
    u_los = 0.5 * (1.0 + np.array([math.erf(x / math.sqrt(2.0)) for x in g_los]))
    los = u_los < los_probability(d2d)

    shadow = {state: SIGMA_SF_DB[state]
              * _correlated_normal(travelled, CORR_DIST_SF_M[state], rng)
              for state in (True, False)}
    sf = np.where(los, shadow[True], shadow[False])

    snr_raw = (UE_TX_DBM + BS_GAIN_DBI + UE_GAIN_DBI
               - pathloss_db(d2d, los=los) - sf - NOISE_DBM)
    return {"ceiling_db": np.minimum(snr_raw, CEILING_CAP_DB), "raw_db": snr_raw,
            "los": los, "distance_m": d2d}


def fading_db(los: np.ndarray, snr_db: np.ndarray, *, moving: bool,
              rng: np.random.Generator) -> np.ndarray:
    travelled = SPEED_MPS * STEP_S * np.arange(STEPS) * (1 if moving else 0)
    k_db = K_FACTOR_DB[0] + K_FACTOR_DB[1] * _correlated_normal(travelled, CORR_DIST_K_M, rng)
    ds = {state: 10.0 ** (LG_DS[state][0] + LG_DS[state][1]
                          * _correlated_normal(travelled, CORR_DIST_DS_M[state], rng))
          for state in (True, False)}
    out = np.empty(STEPS)
    draws = None
    for t in range(STEPS):
        delay_spread = ds[bool(los[t])][t]
        n_sub = int(np.clip(round(BANDWIDTH_HZ * 5.0 * delay_spread), 1, 100))
        if moving or draws is None:
            draws = rng.standard_normal((100, 2)) / math.sqrt(2.0)
        scatter = draws[:n_sub, 0] + 1j * draws[:n_sub, 1]
        if los[t]:
            k = 10.0 ** (k_db[t] / 10.0)
            h = math.sqrt(k / (k + 1.0)) + math.sqrt(1.0 / (k + 1.0)) * scatter
        else:
            h = scatter
        snr = 10.0 ** (snr_db[t] / 10.0)
        eff = 2.0 ** np.mean(np.log2(1.0 + snr * np.abs(h) ** 2)) - 1.0
        out[t] = 10.0 * math.log10(max(eff, 1e-12)) - snr_db[t]
    return out


def make_trace(seed: int, *, small_scale: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    per_ue = {ue: ue_trace(spec["start"], spec["direction"], rng)
              for ue, spec in LAYOUT.items()}
    if small_scale:
        rng_ss = np.random.default_rng([seed, 38901])
        for ue, tr in per_ue.items():
            fade = fading_db(tr["los"], tr["raw_db"], rng=rng_ss,
                             moving=LAYOUT[ue]["direction"] != 0)
            tr["large_scale_db"] = tr["raw_db"]
            tr["raw_db"] = tr["raw_db"] + fade
            tr["ceiling_db"] = np.minimum(tr["raw_db"], CEILING_CAP_DB)
    steps = [{
        "step": t,
        "ceiling_db": {ue: round(float(tr["ceiling_db"][t]), 2) for ue, tr in per_ue.items()},
        "raw_db": {ue: round(float(tr["raw_db"][t]), 2) for ue, tr in per_ue.items()},
        "los": {ue: bool(tr["los"][t]) for ue, tr in per_ue.items()},
        "distance_m": {ue: round(float(tr["distance_m"][t]), 1) for ue, tr in per_ue.items()},
    } for t in range(STEPS)]
    meta = {
        "model": ("3GPP TR 38.901 UMa, large-scale (path loss, LoS, shadow fading)"
                  + (" + wideband small-scale fading" if small_scale else "")),
        "seed": seed, "carrier_hz": CARRIER_HZ, "bandwidth_hz": BANDWIDTH_HZ,
        "ue_tx_dbm": UE_TX_DBM, "bs_gain_dbi": BS_GAIN_DBI,
        "noise_dbm": round(NOISE_DBM, 2), "ceiling_cap_db": CEILING_CAP_DB,
        "speed_mps": SPEED_MPS, "steps": STEPS, "step_s": STEP_S,
        "layout": {ue: {"start": list(s["start"]), "direction": s["direction"]}
                   for ue, s in LAYOUT.items()},
    }
    return {"meta": meta, "steps": steps}


def load_trace(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())["steps"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="1-10", help="a range such as 1-10, or one seed")
    p.add_argument("--out", type=Path, required=True, help="directory for uma_seed<n>.json")
    p.add_argument("--small-scale", action="store_true",
                   help="add wideband small-scale fading (Rician/Rayleigh, delay-spread sub-bands)")
    args = p.parse_args()
    lo, _, hi = args.seeds.partition("-")
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in range(int(lo), int(hi or lo) + 1):
        trace = make_trace(seed, small_scale=args.small_scale)
        (args.out / f"uma_seed{seed}.json").write_text(json.dumps(trace, indent=1))
        raw = np.array([list(s["raw_db"].values()) for s in trace["steps"]])
        cap = np.array([list(s["ceiling_db"].values()) for s in trace["steps"]])
        print(f"seed {seed:>2}: raw SNR {raw.min():6.1f} .. {raw.max():5.1f} dB, "
              f"median {np.median(raw):5.1f}; capped ceiling median {np.median(cap):5.1f} dB")


if __name__ == "__main__":
    main()
