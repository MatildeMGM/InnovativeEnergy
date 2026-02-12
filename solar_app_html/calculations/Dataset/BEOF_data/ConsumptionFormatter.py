#%%
# consumption_formatter_2025.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = Path(__file__).parent
CONSUMPTION_IN = BASE_DIR / "ConsumptionConsumerCategoryHour.csv"
CONSUMPTION_OUT = BASE_DIR / "consumption_private_scaled_hourly_2025.csv"

DK_TZ = "Europe/Copenhagen"

# Price timeframe (DK local time)
START_DK = "2025-01-01 00:00:00"
END_DK_EXCL = "2026-01-01 00:00:00"  # exclusive end

TARGET_ANNUAL_KWH = 4200.0


# ----------------------------
# HELPERS
# ----------------------------
def _require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")


def _read_consumption_csv(path: Path) -> pd.DataFrame:
    # Typical Danish export: semicolon + decimal comma
    try:
        df = pd.read_csv(path, sep=";", decimal=",")
    except Exception:
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _ensure_timeutc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a timezone-aware UTC datetime column named TimeUTC.
    Accepts input with either TimeUTC or TimeDK.
    """
    if "TimeUTC" in df.columns:
        df["TimeUTC"] = pd.to_datetime(df["TimeUTC"], utc=True, errors="coerce")
        return df

    if "TimeDK" in df.columns:
        df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")
        dt_dk = df["TimeDK"].dt.tz_localize(
            DK_TZ,
            ambiguous=False,            # deterministic for fall-back hour
            nonexistent="shift_forward" # deterministic for spring-forward gap
        )
        df["TimeUTC"] = dt_dk.dt.tz_convert("UTC")
        return df

    raise ValueError("Input must contain either 'TimeUTC' or 'TimeDK' column.")


def _ensure_consumption_col(df: pd.DataFrame) -> pd.DataFrame:
    if "ConsumptionkWh" in df.columns:
        return df
    # Attempt to find a likely alternative column name
    candidates = [c for c in df.columns if "consum" in c.lower() or "kwh" in c.lower()]
    if not candidates:
        raise ValueError("Input missing 'ConsumptionkWh' column (and no obvious alternative found).")
    df = df.rename(columns={candidates[0]: "ConsumptionkWh"})
    return df


# ----------------------------
# MAIN
# ----------------------------
def main() -> None:
    _require_exists(CONSUMPTION_IN)

    df = _read_consumption_csv(CONSUMPTION_IN)
    df = _ensure_timeutc(df)
    df = _ensure_consumption_col(df)

    # Parse/clean
    df = df.dropna(subset=["TimeUTC", "ConsumptionkWh"]).copy()
    df["ConsumptionkWh"] = pd.to_numeric(df["ConsumptionkWh"], errors="coerce")
    df = df.dropna(subset=["ConsumptionkWh"]).copy()

    # Restrict to price timeframe (DK boundaries converted to UTC)
    start_utc = pd.Timestamp(START_DK).tz_localize(DK_TZ).tz_convert("UTC")
    end_utc = pd.Timestamp(END_DK_EXCL).tz_localize(DK_TZ).tz_convert("UTC")
    df = df[(df["TimeUTC"] >= start_utc) & (df["TimeUTC"] < end_utc)].copy()

    if df.empty:
        raise RuntimeError("No consumption rows remain after applying the 2025 timeframe filter.")

    # Hourly aggregate (sum)
    df = df.sort_values("TimeUTC").set_index("TimeUTC")
    df.index.name = "HourUTC"

    hourly = df["ConsumptionkWh"].resample("1H").sum(min_count=1).to_frame(name="ConsumptionkWh")

    # Fill missing hours with 0 (so you can merge cleanly with prices)
    missing_hours = int(hourly["ConsumptionkWh"].isna().sum())
    if missing_hours:
        hourly["ConsumptionkWh"] = hourly["ConsumptionkWh"].fillna(0.0)

    # Scale to target annual kWh
    annual_sum = float(hourly["ConsumptionkWh"].sum())
    if annual_sum <= 0:
        raise RuntimeError("Annual sum is <= 0 after hourly aggregation; check input data.")

    scale = TARGET_ANNUAL_KWH / annual_sum

    out = pd.DataFrame(index=hourly.index)
    out["load_kwh"] = hourly["ConsumptionkWh"] * scale
    out["HourDK"] = out.index.tz_convert(DK_TZ).tz_localize(None)

    # Save
    out.to_csv(CONSUMPTION_OUT, index=True)

    print("Consumption formatting done")
    print(f"  Input:   {CONSUMPTION_IN}")
    print(f"  Output:  {CONSUMPTION_OUT}")
    print(f"  Hours:   {len(out)}")
    print(f"  Missing hours filled with 0: {missing_hours}")
    print(f"  Annual sum before scaling: {annual_sum:.3f} kWh")
    print(f"  Target annual kWh:         {TARGET_ANNUAL_KWH:.3f} kWh")
    print(f"  Scale factor:              {scale:.12e}")


if __name__ == "__main__":
    main()
# %%
