from __future__ import annotations
from functools import lru_cache
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Any
from pathlib import Path
from shapely.geometry import Point
import re
import csv
import json
import geopandas as gpd
import pandas as pd
import pvlib
import numpy as np
import math
from pathlib import Path
import simulation





# ----------------------------
# Base settings (same as original)
# ----------------------------
TZ = "Europe/Copenhagen"
ALTITUDE_M = 50

PANEL_PRESETS = {
    "1 kWp": 1.0,
    "2 kWp": 2.0,
    "5 kWp": 5.0,
    "Custom": 1.0,
}

FREQ_OPTIONS = {
    "Hourly": "1h",
    "15 min": "15min",
    "5 min": "5min",
}

LAYERS_DIR = Path(__file__).resolve().parent / "layers"
LAYERS_CONFIG_PATH = LAYERS_DIR / "layers_config.json"

BUILTIN_LAYER_IDS = {"bornholm", "planer"}  # served from memory, not from file

DATA_DIR = Path(__file__).resolve().parent / "calculations/Dataset/2024"
CONSUMPTION_PATH = DATA_DIR / "consumption_private_scaled_hourly.csv"
PRICES_PATH = DATA_DIR / "elspot_DK1_with_total_prices.csv"


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

DETAILS_CSV_PATH = Path(__file__).resolve().parent / "theMASTERplan - Lokal_Plan.csv"

def _extract_planid_from_doklink(doklink: str) -> int | None:
    if not isinstance(doklink, str):
        return None
    m = re.search(r"20_(\d+)_", doklink)
    return int(m.group(1)) if m else None

def _df_row_to_jsonable(d: dict) -> dict:
    # Convert pandas/numpy types + NaN to JSON-friendly python types
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
            continue
        if pd.isna(v):
            out[k] = None
            continue
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
            continue
        if isinstance(v, (np.floating,)):
            out[k] = float(v)
            continue
        if isinstance(v, pd.Timestamp):
            out[k] = v.date().isoformat()
            continue
        out[k] = v
    return out

def load_plan_details(csv_path: Path) -> dict[int, list[dict]]:
    """
    Returns mapping: planid -> [details_row_dict, ...]
    Using list allows multiple rows per planid (future-proof).
    """
    if not csv_path.exists():
        print(f"WARNING: details CSV not found at {csv_path.resolve()}. Continuing without extra plan details.")
        return {}

    
    df = pd.read_csv(
        csv_path,
        sep=",",                  # IMPORTANT: your file is comma-separated
        encoding="utf-8-sig",
        engine="python",          # tolerant, but still consistent
        na_values=["null", "NULL", ""],  # convert "null" strings to NaN
        keep_default_na=True,
        quoting=csv.QUOTE_MINIMAL
    )

    # If you want datovedt as proper date:
    if "datovedt" in df.columns:
        df["datovedt"] = pd.to_datetime(df["datovedt"], errors="coerce")

    if "doklink" not in df.columns:
        print("WARNING: details CSV has no 'doklink' column; cannot extract planid.")
        return {}

    df["planid"] = df["doklink"].apply(_extract_planid_from_doklink)
    df = df[df["planid"].notna()].copy()
    df["planid"] = df["planid"].astype(int)

    details_by_planid: dict[int, list[dict]] = {}
    for _, row in df.iterrows():
        r = row.to_dict()
        planid = int(r.pop("planid"))
        details_by_planid.setdefault(planid, []).append(_df_row_to_jsonable(r))

    return details_by_planid

def _layer_id_from_path(p: Path) -> str:
    return p.stem  # filename without extension

# ----------------------------
# Solar computation helpers (mirrors MapAndSolar_V2.py)
# ----------------------------

def kwp_from_area_eff(area_m2: float, eff: float) -> float:
    # kWp under STC (1000 W/m²)
    return area_m2 * eff

def power_w_from_poa_kwp(poa_wm2: pd.Series, kwp: float, pr: float = 1.0) -> pd.Series:
    # Standard industry scaling from plane-of-array irradiance to DC power
    return poa_wm2.clip(lower=0) * kwp * pr

def _pv_timeseries(lat, lon, kwp, tilt_deg, azimuth_deg, start_ts, end_ts, freq, pr: float = 1.0):
    if end_ts <= start_ts:
        raise ValueError("End must be after start.")

    times = pd.date_range(start_ts, end_ts, freq=freq, tz=TZ, inclusive="left")
    if len(times) < 2:
        raise ValueError("Date range too short for the selected resolution.")

    location = pvlib.location.Location(lat, lon, tz=TZ, altitude=ALTITUDE_M)
    cs = location.get_clearsky(times)
    solpos = location.get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        dni_extra=dni_extra,
        model="perez",
    )

    power_w = power_w_from_poa_kwp(poa["poa_global"], kwp=kwp, pr=pr)

    return pd.DataFrame(
        {"poa_global": poa["poa_global"].astype(float),
         "power_w": power_w.astype(float)},
        index=times,
    )

def _energy_wh_series_from_power(power_w: pd.Series) -> pd.Series:
    dt_h = power_w.index.to_series().diff().dt.total_seconds().div(3600.0)
    dt_h.iloc[0] = 0.0
    return power_w * dt_h

@lru_cache(maxsize=256)
def _get_pvgis_tmy_irradiance_cached(lat_r: float, lon_r: float) -> pd.DataFrame:
    tmy, meta = pvlib.iotools.get_pvgis_tmy(lat_r, lon_r, map_variables=True)
    if tmy.index.tz is None:
        tmy.index = tmy.index.tz_localize("UTC")
    tmy = tmy.tz_convert(TZ)
    return tmy[["ghi", "dni", "dhi"]].astype(float)

def _pv_timeseries_from_irradiance(lat, lon, kwp, tilt_deg, azimuth_deg, times, ghi, dni, dhi, pr: float = 1.0):
    location = pvlib.location.Location(lat, lon, tz=TZ, altitude=ALTITUDE_M)
    solpos = location.get_solarposition(times)

    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model="perez",
    )

    power_w = power_w_from_poa_kwp(poa["poa_global"], kwp=kwp, pr=pr)
    return pd.DataFrame({"poa_global": poa["poa_global"].astype(float),
                         "power_w": power_w.astype(float)}, index=times)

def daily_profile(lat, lon, kwp, tilt_deg, azimuth_deg, day_date, freq, pr: float = 1.0):
    start_ts = pd.Timestamp(day_date).tz_localize(TZ)
    end_ts = (pd.Timestamp(day_date) + pd.Timedelta(days=1)).tz_localize(TZ)
    return _pv_timeseries(lat, lon, kwp, tilt_deg, azimuth_deg, start_ts, end_ts, freq, pr=pr)

def yearly_daily_energy(lat, lon, kwp, tilt_deg, azimuth_deg, year: int, pr: float = 1.0):
    tmy = _get_pvgis_tmy_irradiance_cached(round(lat, 4), round(lon, 4))

    times = pd.DatetimeIndex([ts.replace(year=year) for ts in tmy.index]).tz_convert(TZ)

    df = _pv_timeseries_from_irradiance(
        lat, lon, kwp, tilt_deg, azimuth_deg,
        times=times,
        ghi=tmy["ghi"].to_numpy(),
        dni=tmy["dni"].to_numpy(),
        dhi=tmy["dhi"].to_numpy(),
        pr=pr,
    )

    e_wh = _energy_wh_series_from_power(df["power_w"])
    return e_wh.resample("1D").sum()


# ----------------------------
# Geo / plan helpers (mirrors MapAndSolar_V2.py)
# ----------------------------
def sanitize_for_geojson(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    for col in gdf.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = gdf[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            continue
        if gdf[col].dtype == "object":
            gdf[col] = gdf[col].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)
    gdf = gdf.where(pd.notna(gdf), None)
    return gdf

def _planer_prepare_for_query(planer: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    planer = planer.copy()

    if planer.crs is None:
        planer = planer.set_crs(4326)
    planer = planer.to_crs(4326)

    planer = planer[planer.geometry.notna()].copy()
    planer = planer[planer.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if hasattr(planer.geometry, "make_valid"):
        planer["geometry"] = planer.geometry.make_valid()
    else:
        planer["geometry"] = planer.buffer(0)

    planer = planer.explode(index_parts=False)
    return planer

def _pick_summary_fields(planer_cols):
    preferred = [
        "plannavn", "plan_navn", "navn", "name", "titel", "title",
        "planid", "plan_id", "id",
        "plantype", "type", "status",
        "vedtaget", "vedtagelsesdato", "dato", "startdato", "slutdato",
        "link", "url", "doklink"
    ]

    chosen = [c for c in preferred if c in planer_cols and c != "geometry"]
    if not chosen:
        chosen = [c for c in planer_cols if c != "geometry"][:10]

    chosen = chosen[:12]

    if "doklink" in planer_cols and "doklink" not in chosen:
        chosen[-1] = "doklink"

    return chosen


def find_plans_at_point(planer_query: gpd.GeoDataFrame, lat: float, lon: float) -> gpd.GeoDataFrame:
    """
    Return ALL plan polygons that contain the point (lat, lon).
    Output is in the same CRS as planer_query (whatever you pass in).
    """
    proj_crs = 25833  # ETRS89 / UTM zone 33N

    planer_proj = planer_query.to_crs(proj_crs)
    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(proj_crs).iloc[0]

    sidx = planer_proj.sindex
    candidates_idx = list(sidx.intersection(pt.bounds))
    if not candidates_idx:
        return planer_proj.iloc[0:0]  # empty GeoDataFrame

    candidates = planer_proj.iloc[candidates_idx]
    hits = candidates[candidates.contains(pt)]
    if hits.empty:
        return hits

    # Optional: sort for stable / nicer display (smallest area first)
    hits = hits.copy()
    hits["_area_m2"] = hits.geometry.area
    hits = hits.sort_values("_area_m2").drop(columns=["_area_m2"])
    return hits

# ----------------------------
# Load data once (like notebook script)
# ----------------------------
gdf = gpd.read_file(LAYERS_DIR / "municipalities.geojson").to_crs(4326)
bornholm = gdf[gdf["label_dk"].str.contains("Bornholm", case=False)].copy()
if bornholm.empty:
    raise ValueError("Could not find Bornholm in 'label_dk'.")

planer_raw = gpd.read_file(LAYERS_DIR / "planer.geojson")
planer_query = _planer_prepare_for_query(planer_raw)
planer_query_proj = planer_query.to_crs(25833)
planer_map = sanitize_for_geojson(planer_query)

PLAN_SUMMARY_FIELDS = _pick_summary_fields(planer_map.columns)
PLAN_DETAILS_BY_ID = load_plan_details(DETAILS_CSV_PATH)
print(f"Plan details loaded: {len(PLAN_DETAILS_BY_ID)} unique planids")
HIDDEN_LAYER_IDS = {"municipalities", "planer"}  # built-ins, not optional overlays


bornholm_proj = bornholm.to_crs(25833)
centroid_wgs84 = bornholm_proj.geometry.centroid.to_crs(4326).iloc[0]
center_lat = float(centroid_wgs84.y)
center_lon = float(centroid_wgs84.x)

def _json_safe(x):
    # None stays None
    if x is None:
        return None

    # pandas Timestamp -> string
    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    # numpy scalar -> python scalar
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)

    # python float NaN -> None
    if isinstance(x, float) and math.isnan(x):
        return None

    # pandas/GeoPandas may store missing as NA objects
    if pd.isna(x):
        return None

    # basic types pass through
    if isinstance(x, (str, int, float, bool)):
        return x

    # fallback to string (safe for weird objects)
    return str(x)


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/api/center")
def api_center():
    return {"lat": center_lat, "lon": center_lon}

@app.get("/api/presets")
def api_presets():
    return {"presets": PANEL_PRESETS, "freq_options": FREQ_OPTIONS}

@app.get("/api/bornholm")
def api_bornholm():
    return JSONResponse(bornholm.__geo_interface__)

@app.get("/api/planer")
def api_planer():
    # Full plans, like original (performance later)
    return JSONResponse(planer_map.__geo_interface__)

@app.get("/api/layers")
def api_layers():
    """
    Returns layer metadata.
    Priority:
      1) layers_config.json if present
      2) fallback: all *.geojson files in ./layers
    """
    # ---- 1) If config exists, use it ----
    if LAYERS_CONFIG_PATH.exists():
        try:
            cfg = json.loads(LAYERS_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            return JSONResponse({"error": f"Failed to parse layers_config.json: {e}"}, status_code=500)

        # Allow both formats:
        # A) list[dict]
        # B) {"layers": list[dict]}
        if isinstance(cfg, dict):
            entries = cfg.get("layers", [])
        else:
            entries = cfg

        if not isinstance(entries, list):
            return JSONResponse({"error": "layers_config.json must be a list OR {\"layers\": [...] }"}, status_code=500)

        out = []
        for entry in entries:
            # Robust: skip non-dicts (this prevents your current crash)
            if not isinstance(entry, dict):
                continue

            layer_id = entry.get("id")
            if not layer_id or not isinstance(layer_id, str):
                continue

            # Must exist as a geojson file in ./layers
            path = (LAYERS_DIR / f"{layer_id}.geojson")
            if not path.exists():
                continue

            out.append({
                "id": layer_id,
                "name": entry.get("name", layer_id.replace("_", " ").title()),
                "style": entry.get("style"),
                "show_in_control": bool(entry.get("show_in_control", True)),
                "enabled_by_default": bool(entry.get("enabled_by_default", False)),
            })

        return {"layers": out}

    # ---- 2) Fallback: auto-discover geojson files ----
    if not LAYERS_CONFIG_PATH.exists():
        return {"layers": []}

    cfg = json.loads(LAYERS_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_layers = cfg.get("layers", [])

    out = []
    for entry in raw_layers:
        if not isinstance(entry, dict):
            continue

        layer_id = entry.get("id")
        if not layer_id:
            continue

        # only expose layers that actually exist
        p = (LAYERS_DIR / f"{layer_id}.geojson")
        if not p.exists():
            continue

        out.append({
            "id": layer_id,
            "name": entry.get("name", layer_id.replace("_", " ").title()),
            "style": entry.get("style") or None,
            "show_in_control": bool(entry.get("show_in_control", True)),
            "enabled_by_default": bool(entry.get("enabled_by_default", False)),
        })

    return {"layers": out}


@app.get("/api/layers/{layer_id}")
def api_layer_geojson(layer_id: str):
    """
    Returns GeoJSON for either:
    - built-in layers: "bornholm", "planer"
    - file layers: ./layers/{layer_id}.geojson
    """
    layer_id = str(layer_id).strip()

    if layer_id == "bornholm":
        return JSONResponse(bornholm.__geo_interface__)

    if layer_id == "planer":
        return JSONResponse(planer_map.__geo_interface__)

    # file layer
    path = (LAYERS_DIR / f"{layer_id}.geojson").resolve()

    if not str(path).startswith(str(LAYERS_DIR.resolve())):
        return JSONResponse({"error": "Invalid layer id"}, status_code=400)

    if not path.exists():
        return JSONResponse({"error": "Layer not found"}, status_code=404)

    return FileResponse(path, media_type="application/geo+json")


@app.get("/api/summary")
def api_summary(
    lat: float,
    lon: float,

    # New style
    kwp: float | None = Query(None),
    pr: float = Query(1.0),

    # Legacy style (keep frontend working)
    area_m2: float | None = Query(None),
    eff: float | None = Query(None),

    tilt: float = Query(40.0),
    az: float = Query(180.0),
    day: str = Query(...),
    freq: str = Query("1h"),
):
    if kwp is None:
        if area_m2 is None or eff is None:
            # final fallback (prevents crash if frontend sends neither)
            area_m2 = 10.0 if area_m2 is None else area_m2
            eff = 0.20 if eff is None else eff
        kwp = kwp_from_area_eff(area_m2, eff)

    # Day: clear-sky
    df_24h = daily_profile(lat, lon, kwp, tilt, az, day, freq, pr=pr)
    e_day_kwh = float(_energy_wh_series_from_power(df_24h["power_w"]).sum() / 1000.0)

    peak_power_w = float(df_24h["power_w"].max())
    peak_poa_wm2 = float(df_24h["poa_global"].max())

    # Year: TMY/PVGIS
    year = int(pd.Timestamp(day).year)
    daily_wh = yearly_daily_energy(lat, lon, kwp, tilt, az, year, pr=pr)
    total_year_kwh = float(daily_wh.sum() / 1000.0)

    # Plan lookup (projected/sindex, same as original)
    hits = find_plans_at_point(planer_query, lat, lon)

    plans_data = []
    if not hits.empty:
        for _, row in hits.iterrows():
            d = {}

            # Collect plan fields
            for c in PLAN_SUMMARY_FIELDS:
                if c in row.index and c != "geometry":
                    d[c] = _json_safe(row[c])

            # Compute planid for joining (prefer doklink extraction)
            planid_int = None
            if d.get("doklink"):
                planid_int = _extract_planid_from_doklink(str(d["doklink"]))

            # Fallbacks if doklink missing
            if planid_int is None:
                for key in ("planid", "plan_id", "id"):
                    val = d.get(key)
                    try:
                        if val is not None:
                            planid_int = int(val)
                            break
                    except (TypeError, ValueError):
                        pass

            # Attach CSV details
            if planid_int is not None:
                details = PLAN_DETAILS_BY_ID.get(planid_int)
                if details:
                    d["details"] = details

            plans_data.append(d)

    plan_data = plans_data if plans_data else None



    # Series for plots
    series_day = {
        "t": [t.isoformat() for t in df_24h.index],
        "power_w": df_24h["power_w"].tolist(),
    }
    series_year = {
        "d": [d.date().isoformat() for d in daily_wh.index],
        "kwh": (daily_wh / 1000.0).tolist(),
    }

    return {
        "location": {"lat": lat, "lon": lon},
        "inputs": {
            "kwp": float(kwp),
            "pr": float(pr),
            "area_m2": None if area_m2 is None else float(area_m2),
            "eff": None if eff is None else float(eff),
            "tilt": tilt, "az": az,
            "day": day, "freq": freq,
        },

        "peak_poa_wm2": peak_poa_wm2,
        "peak_power_w": peak_power_w,
        "energy_day_kwh_clear_sky": e_day_kwh,
        "year": year,
        "energy_year_kwh_tmy": total_year_kwh,
        "plan": plan_data,
        "series_day": series_day,
        "series_year": series_year,
    }

@app.get("/api/simulate_day")
def api_simulate_day(
    lat: float,
    lon: float,
    kwp: float = Query(1.0),
    pr: float = Query(1.0),
    tilt: float = Query(40.0),
    az: float = Query(180.0),
    day: str = Query(...),
    freq: str = Query("1h"),
):
    # PV power series (local TZ)
    df_pv = daily_profile(lat, lon, kwp, tilt, az, day, freq, pr=pr)

    # Convert to pv_kwh per interval, then convert index to UTC for joining
    pv_kwh_local = simulation.energy_kwh_from_power(df_pv["power_w"])
    pv_df = pd.DataFrame({"pv_kwh": pv_kwh_local})
    pv_df.index = pv_df.index.tz_convert("UTC")
    pv_df.index.name = "TimeUTC"

    # Interpret "day" as DK local day and convert to UTC window
    start_local = pd.Timestamp(day).tz_localize(TZ)
    end_local = start_local + pd.Timedelta(days=1)
    start_utc = start_local.tz_convert("UTC")
    end_utc = end_local.tz_convert("UTC")

    # Load consumption + prices (UTC indexed)
    load_df = simulation.read_consumption_scaled(str(CONSUMPTION_PATH))
    price_df = simulation.read_prices(str(PRICES_PATH))

    load_day = load_df.loc[start_utc:end_utc]
    price_day = price_df.loc[start_utc:end_utc]

    # Align strictly on timestamps
    df = load_day.join(price_day, how="inner").join(pv_df, how="left")
    df["pv_kwh"] = df["pv_kwh"].fillna(0.0)

    df_sim = simulation.simulate_no_battery(df)
    summary = simulation.compute_costs(df_sim)

    return {
        "inputs": {"day": day, "freq": freq},
        "summary": summary,
        "series": {
            "t": [t.isoformat() for t in df_sim.index],
            "load_kwh": df_sim["load_kwh"].tolist(),
            "pv_kwh": df_sim["pv_kwh"].tolist(),
            "import_kwh": df_sim["import_kwh"].tolist(),
            "export_kwh": df_sim["export_kwh"].tolist(),
            "buy_price": df_sim["buy_price_dkk_per_kwh"].tolist(),
            "sell_price": df_sim["sell_price_dkk_per_kwh"].tolist(),
        },
    }

@app.get("/api/simulate_year")
def api_simulate_year(
    lat: float,
    lon: float,
    kwp: float = Query(1.0),
    pr: float = Query(1.0),
    tilt: float = Query(40.0),
    az: float = Query(180.0),
):
    year = 2024  # fixed for now

    # PV: hourly from PVGIS TMY
    tmy = _get_pvgis_tmy_irradiance_cached(round(lat, 4), round(lon, 4))
    times_local = pd.DatetimeIndex([ts.replace(year=year) for ts in tmy.index]).tz_convert(TZ)

    df_pv_local = _pv_timeseries_from_irradiance(
        lat, lon, kwp, tilt, az,
        times=times_local,
        ghi=tmy["ghi"].to_numpy(),
        dni=tmy["dni"].to_numpy(),
        dhi=tmy["dhi"].to_numpy(),
        pr=pr,
    )

    pv_kwh_local = simulation.energy_kwh_from_power(df_pv_local["power_w"])
    pv_df = pd.DataFrame({"pv_kwh": pv_kwh_local})
    pv_df.index = pv_df.index.tz_convert("UTC")
    pv_df.index.name = "TimeUTC"

    # Load consumption + prices (UTC indexed)
    load_df = simulation.read_consumption_scaled(str(CONSUMPTION_PATH))
    price_df = simulation.read_prices(str(PRICES_PATH))

    start_utc = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    end_utc = pd.Timestamp("2025-01-01 00:00", tz="UTC")

    load_year = load_df.loc[start_utc:end_utc].reindex(pv_df.index)
    price_year = price_df.loc[start_utc:end_utc].reindex(pv_df.index)

    df = load_year.join(price_year, how="inner").join(pv_df, how="left")
    df["pv_kwh"] = df["pv_kwh"].fillna(0.0)

    df_sim = simulation.simulate_no_battery(df)
    summary = simulation.compute_costs(df_sim)

    # Monthly aggregates
    monthly_kwh = df_sim[["load_kwh","pv_kwh","import_kwh","export_kwh"]].resample("MS").sum()

    monthly_baseline = (df_sim["load_kwh"] * df_sim["buy_price_dkk_per_kwh"]).resample("MS").sum()
    monthly_system = (
        (df_sim["import_kwh"] * df_sim["buy_price_dkk_per_kwh"])
        - (df_sim["export_kwh"] * df_sim["sell_price_dkk_per_kwh"])
    ).resample("MS").sum()

    pv_used = df_sim["pv_used_kwh"].resample("MS").sum()
    pv_prod = df_sim["pv_kwh"].resample("MS").sum()
    load_sum = df_sim["load_kwh"].resample("MS").sum()
    import_sum = df_sim["import_kwh"].resample("MS").sum()

    self_consumption = (pv_used / pv_prod).replace([np.inf, -np.inf], np.nan)
    self_sufficiency = (1.0 - import_sum / load_sum).replace([np.inf, -np.inf], np.nan)

    # Convert to JSON-friendly lists
    months = [d.strftime("%Y-%m") for d in monthly_kwh.index.tz_convert("UTC")]
    monthly = {
        "month": months,
        "load_kwh": monthly_kwh["load_kwh"].fillna(0).tolist(),
        "pv_kwh": monthly_kwh["pv_kwh"].fillna(0).tolist(),
        "import_kwh": monthly_kwh["import_kwh"].fillna(0).tolist(),
        "export_kwh": monthly_kwh["export_kwh"].fillna(0).tolist(),
        "baseline_cost_dkk": monthly_baseline.fillna(0).tolist(),
        "system_cost_dkk": monthly_system.fillna(0).tolist(),
        "savings_dkk": (monthly_baseline - monthly_system).fillna(0).tolist(),
        "self_consumption": self_consumption.fillna(0).tolist(),
        "self_sufficiency": self_sufficiency.fillna(0).tolist(),
    }

    return {"inputs": {"year": year}, "summary": summary, "monthly": monthly}
