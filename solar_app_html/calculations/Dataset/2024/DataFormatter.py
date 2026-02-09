#%% # format_2024_datasets.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = Path(__file__).parent
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Input files (raw downloads)
CONSUMPTION_IN = BASE_DIR / "ConsumptionConsumerCategoryHour.csv"
ELSPOT_IN      = BASE_DIR / "Elspotprices.csv"

# Output files (clean)
CONSUMPTION_OUT = BASE_DIR / "consumption_private_scaled_hourly.csv"
ELSPOT_CLEAN_OUT = BASE_DIR / "elspot_DK1_clean_hourly.csv"
ELSPOT_TOTAL_OUT = BASE_DIR / "elspot_DK1_with_total_prices.csv"

# Annual target scaling for household consumption (kWh/year)
TARGET_ANNUAL_KWH = 4200.0

# Tariffs etc. (DKK/kWh)
GRID_TARIFF   = 0.45
SYSTEM_TARIFF = 0.10
ELAFGIFT      = 0.90
SUPPLIER_FEE  = 0.10
VAT           = 0.25


# ----------------------------
# HELPERS
# ----------------------------
def _require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")

def _to_utc_index(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = df.dropna(subset=[col]).sort_values(col).set_index(col)
    df.index.name = col
    return df

def _ensure_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[col])


# ----------------------------
# CONSUMPTION FORMATTING
# ----------------------------
def format_consumption(
    in_path: Path,
    out_path: Path,
    target_annual_kwh: float,
) -> None:
    _require_exists(in_path)

    df = pd.read_csv(in_path, sep=";", decimal=",")

    # Required cols check (fail early with clear message)
    required = {"TimeUTC", "ConsumptionkWh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Consumption file missing columns: {sorted(missing)}")

    # Time parsing
    df = _to_utc_index(df, "TimeUTC")

    # Numeric consumption
    df = _ensure_numeric(df, "ConsumptionkWh")

    annual_sum = float(df["ConsumptionkWh"].sum())
    if annual_sum <= 0:
        raise ValueError("Consumption annual sum <= 0; check input file.")

    scale = target_annual_kwh / annual_sum

    out = pd.DataFrame(index=df.index)
    out["load_kwh"] = df["ConsumptionkWh"] * scale

    # Optional columns to keep (only if present)
    for col in ["TimeDK", "RegionName", "ConsumerCategory3", "ConsumerCategory2"]:
        if col in df.columns:
            out[col] = df[col].values

    out.to_csv(out_path, index=True)

    print("Consumption formatting done")
    print(f"  Input:  {in_path}")
    print(f"  Output: {out_path}")
    print(f"  Annual sum before scaling: {annual_sum:.3f} kWh")
    print(f"  Target annual kWh:         {target_annual_kwh:.3f} kWh")
    print(f"  Scale factor:             {scale:.12e}")


# ----------------------------
# ELSPOT FORMATTING
# ----------------------------
def format_elspot(
    in_path: Path,
    clean_out_path: Path,
    total_out_path: Path,
    grid_tariff: float,
    system_tariff: float,
    elafgift: float,
    supplier_fee: float,
    vat: float,
) -> None:
    _require_exists(in_path)

    df = pd.read_csv(in_path, sep=";", decimal=",")
    df.columns = df.columns.str.strip()

    required = {"HourUTC", "HourDK", "SpotPriceDKK", "PriceArea"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Elspot file missing columns: {sorted(missing)}")

    # Parse time
    df = _to_utc_index(df, "HourUTC")
    df["HourDK"] = pd.to_datetime(df["HourDK"], errors="coerce")

    # Prices numeric
    df = _ensure_numeric(df, "SpotPriceDKK")

    # Unit conversion: DKK/MWh -> DKK/kWh (+ convenience ore/kWh)
    df["price_dkk_per_kwh"] = df["SpotPriceDKK"] / 1000.0
    df["price_ore_per_kwh"] = df["price_dkk_per_kwh"] * 100.0

    clean = df[["HourDK", "PriceArea", "price_dkk_per_kwh", "price_ore_per_kwh", "SpotPriceDKK"]].copy()
    clean.to_csv(clean_out_path, index=True)

    # Add tariffs (household buy price)
    addons = grid_tariff + system_tariff + elafgift + supplier_fee
    total = clean.copy()
    total["buy_price_dkk_per_kwh"] = (total["price_dkk_per_kwh"] + addons) * (1.0 + vat)
    total["sell_price_dkk_per_kwh"] = total["price_dkk_per_kwh"]  # spot-only export assumption
    total.to_csv(total_out_path, index=True)

    print("Elspot formatting done")
    print(f"  Input:          {in_path}")
    print(f"  Clean output:   {clean_out_path}")
    print(f"  Total output:   {total_out_path}")
    print(f"  Spot price (øre/kWh) min/median/max: "
          f"{clean['price_ore_per_kwh'].min():.2f} / "
          f"{clean['price_ore_per_kwh'].median():.2f} / "
          f"{clean['price_ore_per_kwh'].max():.2f}")
    print(f"  Avg buy price (DKK/kWh): {total['buy_price_dkk_per_kwh'].mean():.3f}")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    format_consumption(
        in_path=CONSUMPTION_IN,
        out_path=CONSUMPTION_OUT,
        target_annual_kwh=TARGET_ANNUAL_KWH,
    )

    format_elspot(
        in_path=ELSPOT_IN,
        clean_out_path=ELSPOT_CLEAN_OUT,
        total_out_path=ELSPOT_TOTAL_OUT,
        grid_tariff=GRID_TARIFF,
        system_tariff=SYSTEM_TARIFF,
        elafgift=ELAFGIFT,
        supplier_fee=SUPPLIER_FEE,
        vat=VAT,
    )
