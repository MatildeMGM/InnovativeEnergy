# simulation.py
from __future__ import annotations
import pandas as pd
import numpy as np

TZ = "Europe/Copenhagen"

def read_consumption_scaled(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    if "load_kwh" not in df.columns:
        raise ValueError("Consumption file must contain 'load_kwh'")
    return df[["load_kwh"]].copy()

def read_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    req = ["buy_price_dkk_per_kwh", "sell_price_dkk_per_kwh"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Price file missing columns: {missing}")
    return df[req].copy()

def energy_kwh_from_power(power_w: pd.Series) -> pd.Series:
    idx = power_w.index
    dt_h = (idx.to_series().shift(-1) - idx.to_series()).dt.total_seconds() / 3600.0
    dt_h.iloc[-1] = 0.0
    e_kwh = (power_w * dt_h) / 1000.0
    e_kwh.name = "pv_kwh"
    return e_kwh

def simulate_no_battery(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pv = out["pv_kwh"].to_numpy()
    load = out["load_kwh"].to_numpy()

    out["pv_used_kwh"] = np.minimum(pv, load)
    out["import_kwh"] = np.maximum(load - out["pv_used_kwh"].to_numpy(), 0.0)
    out["export_kwh"] = np.maximum(pv - out["pv_used_kwh"].to_numpy(), 0.0)
    return out

def compute_costs(df: pd.DataFrame) -> dict:
    baseline = (df["load_kwh"] * df["buy_price_dkk_per_kwh"]).sum()
    system = (df["import_kwh"] * df["buy_price_dkk_per_kwh"]).sum() - (df["export_kwh"] * df["sell_price_dkk_per_kwh"]).sum()

    pv_sum = df["pv_kwh"].sum()
    load_sum = df["load_kwh"].sum()
    import_sum = df["import_kwh"].sum()
    pv_used_sum = df["pv_used_kwh"].sum()

    return {
        "baseline_cost_dkk": float(baseline),
        "system_cost_dkk": float(system),
        "savings_dkk": float(baseline - system),
        "pv_kwh": float(pv_sum),
        "load_kwh": float(load_sum),
        "import_kwh": float(import_sum),
        "export_kwh": float(df["export_kwh"].sum()),
        "self_consumption_ratio": float(pv_used_sum / pv_sum) if pv_sum > 0 else None,
        "self_sufficiency": float(1.0 - import_sum / load_sum) if load_sum > 0 else None,
    }

def _dt_hours(index: pd.DatetimeIndex) -> pd.Series:
    dt = index.to_series().diff().dt.total_seconds().div(3600.0)
    dt.iloc[0] = dt.median() if len(dt) > 1 else 1.0
    # guard
    dt = dt.clip(lower=0)
    return dt

def simulate_greedy_battery(
    df: pd.DataFrame,
    capacity_kwh: float,
    soc_init: float = 0.5,        # fraction of usable range
    soc_min: float = 0.1,         # 0..1
    soc_max: float = 0.9,         # 0..1
    p_charge_kw: float | None = None,     # inverter charge limit (kW)
    p_discharge_kw: float | None = None,  # inverter discharge limit (kW)
    eta_charge: float = 0.95,
    eta_discharge: float = 0.95,
) -> pd.DataFrame:
    """
    Greedy battery dispatch.

    Assumes df has:
      - load_kwh : energy in interval (kWh)
      - pv_kwh   : PV energy in interval (kWh)

    Returns df with:
      - import_kwh, export_kwh
      - pv_used_direct_kwh, batt_charge_in_kwh
      - batt_discharge_out_kwh, pv_used_total_kwh
      - soc_kwh, soc
    """
    out = df.copy()

    if capacity_kwh <= 0:
        # fallback to no-battery behavior
        load = out["load_kwh"].to_numpy(float)
        pv = out["pv_kwh"].to_numpy(float)
        pv_used_direct = np.minimum(load, pv)
        surplus = pv - pv_used_direct
        deficit = load - pv_used_direct

        out["pv_used_direct_kwh"] = pv_used_direct
        out["batt_charge_in_kwh"] = 0.0
        out["batt_discharge_out_kwh"] = 0.0
        out["import_kwh"] = np.maximum(deficit, 0.0)
        out["export_kwh"] = np.maximum(surplus, 0.0)
        out["pv_used_total_kwh"] = pv_used_direct
        out["soc_kwh"] = 0.0
        out["soc"] = 0.0
        out["pv_used_kwh"] = out["pv_used_total_kwh"]
        return out

    # clamp SoC limits
    soc_min = float(np.clip(soc_min, 0.0, 1.0))
    soc_max = float(np.clip(soc_max, 0.0, 1.0))
    if soc_max <= soc_min:
        raise ValueError("soc_max must be > soc_min")

    dt_h = _dt_hours(out.index).to_numpy(float)

    load = out["load_kwh"].to_numpy(float)
    pv = out["pv_kwh"].to_numpy(float)

    n = len(out)

    # Energy limits in kWh (absolute)
    e_min = soc_min * capacity_kwh
    e_max = soc_max * capacity_kwh

    # Initial stored energy: soc_init interpreted within [soc_min, soc_max]
    soc_init = float(np.clip(soc_init, 0.0, 1.0))
    e = e_min + soc_init * (e_max - e_min)

    # Power limits -> energy per interval
    # If None: unlimited (within SoC bounds)
    charge_limit_kwh = np.full(n, np.inf)
    discharge_limit_kwh = np.full(n, np.inf)
    if p_charge_kw is not None:
        charge_limit_kwh = np.maximum(0.0, float(p_charge_kw)) * dt_h
    if p_discharge_kw is not None:
        discharge_limit_kwh = np.maximum(0.0, float(p_discharge_kw)) * dt_h

    pv_used_direct = np.minimum(load, pv)
    surplus = pv - pv_used_direct
    deficit = load - pv_used_direct

    batt_charge_in = np.zeros(n, dtype=float)      # energy taken from PV into inverter (kWh)
    batt_discharge_out = np.zeros(n, dtype=float)  # energy delivered from inverter to load (kWh)
    export_kwh = np.zeros(n, dtype=float)
    import_kwh = np.zeros(n, dtype=float)
    soc_kwh = np.zeros(n, dtype=float)

    for i in range(n):
        # ---- charge if surplus ----
        if surplus[i] > 0:
            # available headroom in stored-energy domain
            headroom = max(e_max - e, 0.0)

            # If we take x kWh from PV into charger, stored increases by x*eta_charge
            # so x <= headroom/eta_charge
            x_max_by_soc = headroom / max(eta_charge, 1e-9)

            x = min(
                surplus[i],
                charge_limit_kwh[i],
                x_max_by_soc,
            )
            x = max(x, 0.0)

            batt_charge_in[i] = x
            e += x * eta_charge

            export_kwh[i] = max(surplus[i] - x, 0.0)
            import_kwh[i] = 0.0

        # ---- discharge if deficit ----
        elif deficit[i] > 0:
            available = max(e - e_min, 0.0)

            # If we deliver y kWh to load, stored decreases by y/eta_discharge
            # so y <= available*eta_discharge
            y_max_by_soc = available * max(eta_discharge, 1e-9)

            y = min(
                deficit[i],
                discharge_limit_kwh[i],
                y_max_by_soc,
            )
            y = max(y, 0.0)

            batt_discharge_out[i] = y
            e -= y / max(eta_discharge, 1e-9)

            import_kwh[i] = max(deficit[i] - y, 0.0)
            export_kwh[i] = 0.0

        else:
            # exactly matched
            import_kwh[i] = 0.0
            export_kwh[i] = 0.0

        soc_kwh[i] = e

    out["pv_used_direct_kwh"] = pv_used_direct
    out["batt_charge_in_kwh"] = batt_charge_in
    out["batt_discharge_out_kwh"] = batt_discharge_out
    out["export_kwh"] = export_kwh
    out["import_kwh"] = import_kwh

    out["pv_used_total_kwh"] = out["pv_used_direct_kwh"] + out["batt_discharge_out_kwh"]
    out["soc_kwh"] = soc_kwh
    out["soc"] = out["soc_kwh"] / capacity_kwh
    out["pv_used_kwh"] = out["pv_used_total_kwh"]

    return out
