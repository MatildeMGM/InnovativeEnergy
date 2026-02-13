#%%
from __future__ import annotations

import json
from urllib.parse import quote

import requests
import pandas as pd


# ----------------------------
# CONFIG
# ----------------------------
PRICE_AREA = "DK2"
DK_TZ = "Europe/Copenhagen"

# Cutover: last hour available as Elspotprices hourly (DK time)
ELSPOT_LAST_HOUR_DK = pd.Timestamp("2025-09-30 23:00:00")  # naive, DK local

# API URLs you provided (kept “link style”)
ELSPOT_BASE = (
    "https://api.energidataservice.dk/dataset/Elspotprices"
    f"?start={quote('2025-01-01')}&end={quote('2026-01-01')}"
    f"&filter={quote(json.dumps({'PriceArea': PRICE_AREA}, separators=(',', ':')))}"
    f"&sort={quote('HourDK asc')}"
)
DAYAHEAD_15MIN_BASE = (
    "https://api.energidataservice.dk/dataset/DayAheadPrices"
    f"?start={quote('2025-01-01')}&end={quote('2026-01-01')}"
    f"&filter={quote(json.dumps({'PriceArea': PRICE_AREA}, separators=(',', ':')))}"
    f"&sort={quote('TimeDK asc')}"
)
BEOF_URL = "https://beof.dk/wp-json/beof/v1/tariffs?start_date=2025-01-01T00:00&end_date=2025-12-31T24:00"

OUT_CSV = "DK1_2025_hourly_for_program.csv"

# JS-style “Nordpool” transformation (from your JS)

VAT_FACTOR = 1.25
FIXED_ADDON_DKK_PER_KWH = 0.0625  # <-- use this if you want “spot incl moms + addon” like original JS


# BEOF sell payout deductions (øre/kWh) -> total 8.82 øre/kWh
BEOF_SELL_DEDUCTION_DKK_PER_KWH = 0.0882


# ----------------------------
# HTTP helpers
# ----------------------------
def _get_json(url: str, timeout: int = 120) -> dict | list:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _fetch_records_paged(base_url_with_query: str, limit: int = 20000) -> pd.DataFrame:
    """Fetches Energi Data Service endpoints returning {records:[...]} using limit/offset paging."""
    all_rows = []
    offset = 0

    while True:
        url = f"{base_url_with_query}&limit={limit}&offset={offset}"
        data = _get_json(url)

        if not isinstance(data, dict) or "records" not in data:
            raise RuntimeError(f"Unexpected JSON shape from URL: {url}")

        batch = data.get("records", [])
        all_rows.extend(batch)

        if len(batch) < limit:
            break

        offset += limit

    return pd.DataFrame(all_rows)


# ----------------------------
# Load spot prices
# ----------------------------
def load_elspot_hourly_until_cutoff() -> pd.DataFrame:
    df = _fetch_records_paged(ELSPOT_BASE)
    if df.empty:
        raise RuntimeError(f"No Elspotprices rows returned. Try opening:\n{ELSPOT_BASE}")

    required = {"HourUTC", "HourDK", "PriceArea", "SpotPriceDKK"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Elspotprices missing columns: {sorted(missing)}")

    df["HourUTC"] = pd.to_datetime(df["HourUTC"], utc=True, errors="coerce")
    df["HourDK"] = pd.to_datetime(df["HourDK"], errors="coerce")  # naive local
    df = df.dropna(subset=["HourUTC", "HourDK"]).copy()

    df = df[df["PriceArea"] == PRICE_AREA].copy()
    df["SpotPriceDKK"] = pd.to_numeric(df["SpotPriceDKK"], errors="coerce")
    df = df.dropna(subset=["SpotPriceDKK"]).copy()

    df = df[df["HourDK"] <= ELSPOT_LAST_HOUR_DK].copy()

    df = df.sort_values("HourUTC").drop_duplicates("HourUTC", keep="last")
    df = df.set_index("HourUTC")
    df.index.name = "HourUTC"

    return df[["HourDK", "PriceArea", "SpotPriceDKK"]].copy()


def load_dayahead_15min_after_cutoff_to_hourly() -> pd.DataFrame:
    df = _fetch_records_paged(DAYAHEAD_15MIN_BASE)
    if df.empty:
        raise RuntimeError(f"No DayAheadPrices rows returned. Try opening:\n{DAYAHEAD_15MIN_BASE}")

    required = {"TimeUTC", "TimeDK", "PriceArea", "DayAheadPriceDKK"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"DayAheadPrices missing columns: {sorted(missing)}")

    df["TimeUTC"] = pd.to_datetime(df["TimeUTC"], utc=True, errors="coerce")
    df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")  # naive local
    df = df.dropna(subset=["TimeUTC", "TimeDK"]).copy()

    df = df[df["PriceArea"] == PRICE_AREA].copy()
    df["DayAheadPriceDKK"] = pd.to_numeric(df["DayAheadPriceDKK"], errors="coerce")
    df = df.dropna(subset=["DayAheadPriceDKK"]).copy()

    # Convert cutoff DK->UTC safely
    cutoff_utc = ELSPOT_LAST_HOUR_DK.tz_localize(DK_TZ).tz_convert("UTC")

    df = df.sort_values("TimeUTC").set_index("TimeUTC")
    df.index.name = "HourUTC"

    # Keep strictly after the last hourly point
    df = df[df.index > cutoff_utc].copy()

    # Resample to hourly mean (15-min -> hourly)
    hourly_mean = df["DayAheadPriceDKK"].resample("1H").mean()
    hourly_count = df["DayAheadPriceDKK"].resample("1H").count()
    hourly_mean[hourly_count < 4] = pd.NA  # incomplete hours

    out = hourly_mean.to_frame(name="SpotPriceDKK")
    out["PriceArea"] = PRICE_AREA
    out["HourDK"] = out.index.tz_convert(DK_TZ).tz_localize(None)

    return out[["HourDK", "PriceArea", "SpotPriceDKK"]].copy()


# ----------------------------
# Load BEOF tariffs hourly
# ----------------------------
def load_beof_tariffs_hourly() -> pd.DataFrame:
    rows = _get_json(BEOF_URL)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No BEOF tariff rows returned. Try opening:\n{BEOF_URL}")

    required = {"time", "elnet", "energinet", "elafgift"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"BEOF tariffs missing columns: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()

    dt_dk = df["time"].dt.tz_localize(DK_TZ, ambiguous=False, nonexistent="shift_forward")
    df["HourUTC"] = dt_dk.dt.tz_convert("UTC")
    df = df.sort_values("HourUTC").drop_duplicates("HourUTC", keep="last").set_index("HourUTC")

    for c in ["elnet", "energinet", "elafgift"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["elnet", "energinet", "elafgift"]).copy()

    return df[["elnet", "energinet", "elafgift"]].copy()



# ----------------------------
# Build final CSV with required headers
# ----------------------------
def main() -> None:
    # Spot: hourly then 15-min->hourly
    s1 = load_elspot_hourly_until_cutoff()
    s2 = load_dayahead_15min_after_cutoff_to_hourly()

    spot = pd.concat([s2, s1]).sort_index()
    spot = spot[~spot.index.duplicated(keep="last")].copy()

    # Restrict to all of 2025 (DK local boundaries)
    start_utc = pd.Timestamp("2025-01-01 00:00:00").tz_localize(DK_TZ).tz_convert("UTC")
    end_utc = pd.Timestamp("2026-01-01 00:00:00").tz_localize(DK_TZ).tz_convert("UTC")
    spot = spot[(spot.index >= start_utc) & (spot.index < end_utc)].copy()

    # Derived spot columns required by your program
    spot["price_dkk_per_kwh"] = spot["SpotPriceDKK"] / 1000.0
    spot["price_ore_per_kwh"] = spot["price_dkk_per_kwh"] * 100.0

    # Tariffs
    tariffs = load_beof_tariffs_hourly()
    out = spot.merge(tariffs, left_index=True, right_index=True, how="left")


    # Spot component (DKK/kWh) used in the final price:
    # Option A (closest to "Nordpool incl moms" style): spot*VAT + addon
    out["spot_component_dkk_per_kwh"] = out["price_dkk_per_kwh"] * VAT_FACTOR + FIXED_ADDON_DKK_PER_KWH

    # Option B (pure spot without VAT/addon): uncomment this instead
    # out["spot_component_dkk_per_kwh"] = out["price_dkk_per_kwh"]

    # Buy price = spot component + BEoF tariffs
    out["buy_price_dkk_per_kwh"] = (
        out["spot_component_dkk_per_kwh"]
        + out["elnet"]
        + out["energinet"]
        + out["elafgift"]
    )

    # Sell price per your BEoF payout adjustments (spot minus 0.0882 DKK/kWh)
    out["sell_price_dkk_per_kwh"] = out["price_dkk_per_kwh"] - BEOF_SELL_DEDUCTION_DKK_PER_KWH

    # Ensure HourDK column exists (already in spot) and enforce required output headers
    out = out.reset_index()

    out = out[[
        "HourUTC",
        "HourDK",
        "PriceArea",

        # raw spot
        "SpotPriceDKK",
        "price_dkk_per_kwh",
        "price_ore_per_kwh",

        # tariffs needed for scenario recalculation in the app
        "elnet",
        "energinet",
        "elafgift",

        # recommended (so app doesn't need to recompute)
        "spot_component_dkk_per_kwh",

        # final buy/sell (still useful)
        "buy_price_dkk_per_kwh",
        "sell_price_dkk_per_kwh",
    ]].sort_values("HourUTC")


    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()

# %%
