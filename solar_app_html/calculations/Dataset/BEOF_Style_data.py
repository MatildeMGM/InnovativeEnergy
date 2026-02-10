#%%
import json
import time
from urllib.parse import quote

import requests
import pandas as pd


# ----------------------------
# CONFIG
# ----------------------------
PRICE_AREA = "DK2"  # "DK1" or "DK2"

# Your working tariff endpoint (link style)
TARIFFS_URL = "https://beof.dk/wp-json/beof/v1/tariffs"

# Use Elspotprices for historical DK spot prices (works for 2024/2025 etc.)
EDS_BASE = "https://api.energidataservice.dk/dataset/Elspotprices"

# Time interval: MUST be 'YYYY-MM-DDTHH:MM'
START = "2023-12-31T23:00"
END = "2024-12-31T22:00"

OUT_CSV = "BEOF_prices_tariffs.csv"

# Matches the JS calculation
VAT_FACTOR = 1.25
FIXED_ADDON_DKK_PER_KWH = 0.0625


# ----------------------------
# HELPERS
# ----------------------------
def _get_json(url: str, timeout: int = 60):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ----------------------------
# FETCHERS (LINK-STYLE)
# ----------------------------
def fetch_tariffs(start: str, end: str) -> pd.DataFrame:
    # Link style: .../tariffs?start_date=...&end_date=...
    url = f"{TARIFFS_URL}?start_date={quote(start)}&end_date={quote(end)}"
    rows = _get_json(url, timeout=60)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No tariff rows returned. URL: {url}")

    required = {"time", "elnet", "energinet", "elafgift"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Tariff endpoint missing columns: {missing}. URL: {url}")

    df["time"] = pd.to_datetime(df["time"])
    df["hour_key"] = df["time"].dt.floor("H")

    # Ensure numeric
    for c in ["elnet", "energinet", "elafgift"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If multiple rows per hour exist, keep the last
    df = df.sort_values("time").drop_duplicates("hour_key", keep="last")

    return df[["hour_key", "elnet", "energinet", "elafgift"]]


def fetch_spot_prices_elspot(start: str, end: str, price_area: str) -> pd.DataFrame:
    """
    Fetches Elspotprices and returns hourly points.
    Elspotprices typically contains:
      - HourDK (or HourUTC)
      - SpotPriceDKK (and/or SpotPriceEUR)
      - PriceArea
    """
    # Some examples online use filter={"PriceArea":"DK1"} (string), not array. We'll follow that style.
    filter_str = json.dumps({"PriceArea": price_area}, separators=(",", ":"))
    filter_enc = quote(filter_str, safe="")

    limit = 20000
    offset = 0
    all_records = []

    while True:
        url = (
            f"{EDS_BASE}"
            f"?start={quote(start)}"
            f"&end={quote(end)}"
            f"&filter={filter_enc}"
            f"&sort={quote('HourDK ASC')}"
            f"&limit={limit}"
            f"&offset={offset}"
        )

        data = _get_json(url, timeout=60)
        batch = data.get("records", [])
        all_records.extend(batch)

        if len(batch) < limit:
            break

        offset += limit
        time.sleep(0.5)  # be polite (rate limits are documented)  :contentReference[oaicite:5]{index=5}

    df = pd.DataFrame(all_records)
    if df.empty:
        raise RuntimeError(
            "No Elspotprices records returned.\n"
            f"Try this URL in your browser:\n{url}\n"
            f"Check PRICE_AREA={price_area} and time range."
        )

    # Determine the datetime column name
    if "HourDK" in df.columns:
        time_col = "HourDK"
    elif "HourUTC" in df.columns:
        time_col = "HourUTC"
    else:
        raise RuntimeError(f"Elspotprices response did not include HourDK/HourUTC. Columns: {list(df.columns)}")

    # Determine the DKK price column
    if "SpotPriceDKK" in df.columns:
        price_col = "SpotPriceDKK"
    elif "DayAheadPriceDKK" in df.columns:
        # (fallback if schema differs)
        price_col = "DayAheadPriceDKK"
    else:
        raise RuntimeError(f"Elspotprices response did not include SpotPriceDKK. Columns: {list(df.columns)}")

    df[time_col] = pd.to_datetime(df[time_col])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Elspotprices is normally hourly already; enforce hourly key anyway
    df["hour_key"] = df[time_col].dt.floor("H")

    # If there are duplicates inside the same hour, keep the last
    df = df.sort_values(time_col).drop_duplicates("hour_key", keep="last")

    # Match JS transformation
    df["spot_dkk_per_kwh_raw"] = df[price_col] / 1000.0
    df["spot_dkk_per_kwh_shown"] = df["spot_dkk_per_kwh_raw"] * VAT_FACTOR + FIXED_ADDON_DKK_PER_KWH

    out = df[["hour_key", time_col, price_col, "spot_dkk_per_kwh_raw", "spot_dkk_per_kwh_shown"]].copy()
    out = out.rename(columns={time_col: "TimeDK", price_col: "SpotPriceDKK"})
    out = out.sort_values("TimeDK")

    return out


# ----------------------------
# MAIN
# ----------------------------
def main():
    spot = fetch_spot_prices_elspot(START, END, PRICE_AREA)
    tariffs = fetch_tariffs(START, END)

    out = spot.merge(tariffs, on="hour_key", how="left")

    out["tariffs_total"] = out[["elnet", "energinet", "elafgift"]].sum(axis=1, skipna=True)
    out["total_dkk_per_kwh"] = out["spot_dkk_per_kwh_shown"] + out["tariffs_total"]
    out["PriceArea"] = PRICE_AREA

    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
