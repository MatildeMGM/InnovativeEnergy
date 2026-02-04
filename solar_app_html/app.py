from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import geopandas as gpd
import pandas as pd
import pvlib
import numpy as np
import math

from shapely.geometry import Point


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


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


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
        "planid", "plan_id", "lokalplan", "lokalplan_nr", "nummer",
        "plantype", "type", "status",
        "vedtaget", "vedtagelsesdato", "dato", "startdato", "slutdato",
        "link", "url", "doklink"
    ]
    chosen = [c for c in preferred if c in planer_cols and c != "geometry"]
    if not chosen:
        chosen = [c for c in planer_cols if c != "geometry"][:10]
    return chosen[:12]

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
gdf = gpd.read_file("municipalities.geojson").to_crs(4326)
bornholm = gdf[gdf["label_dk"].str.contains("Bornholm", case=False)].copy()
if bornholm.empty:
    raise ValueError("Could not find Bornholm in 'label_dk'.")

planer_raw = gpd.read_file("planer.geojson")
planer_query = _planer_prepare_for_query(planer_raw)
planer_query_proj = planer_query.to_crs(25833)
planer_map = sanitize_for_geojson(planer_query)

PLAN_SUMMARY_FIELDS = _pick_summary_fields(planer_map.columns)

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
            for c in PLAN_SUMMARY_FIELDS:
                if c in row.index and c != "geometry":
                    d[c] = _json_safe(row[c])
            # Only append non-empty dicts (optional)
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
