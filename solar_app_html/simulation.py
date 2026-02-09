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
    out["import_kwh"] = np.maximum(load - pv, 0.0)
    out["export_kwh"] = np.maximum(pv - load, 0.0)
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
