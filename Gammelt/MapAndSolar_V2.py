#%%
import geopandas as gpd
import pandas as pd
import pvlib
import matplotlib.pyplot as plt

from shapely.geometry import Point

from ipyleaflet import Map, GeoData, basemaps, LayersControl, Marker
from ipywidgets import (
    Output, VBox, HBox, Dropdown, FloatSlider, DatePicker, Button, Label, Layout
)
from IPython.display import display


# ----------------------------
# Base settings
# ----------------------------
TZ = "Europe/Copenhagen"
ALTITUDE_M = 50

PANEL_PRESETS = {
    "Standard (20%, 10 m²)": (10.0, 0.20),
    "Small (20%, 5 m²)": (5.0, 0.20),
    "High-eff (22%, 10 m²)": (10.0, 0.22),
    "Custom": (10.0, 0.20),
}

FREQ_OPTIONS = {
    "Hourly": "1h",
    "15 min": "15min",
    "5 min": "5min",
}


# ----------------------------
# Solar computation helpers
# ----------------------------
def _pv_timeseries(lat, lon, area_m2, eff, tilt_deg, azimuth_deg, start_ts, end_ts, freq):
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

    power_w = poa["poa_global"].clip(lower=0) * area_m2 * eff

    df = pd.DataFrame(
        {"poa_global": poa["poa_global"].astype(float),
         "power_w": power_w.astype(float)},
        index=times,
    )
    return df

def _energy_wh_from_power(power_w: pd.Series) -> float:
    dt_h = pd.Series(power_w.index).diff().dt.total_seconds().div(3600).fillna(0).to_numpy()
    return float((power_w.to_numpy() * dt_h).sum())

def daily_profile(lat, lon, area_m2, eff, tilt_deg, azimuth_deg, day_date, freq):
    start_ts = pd.Timestamp(day_date).tz_localize(TZ)
    end_ts = (pd.Timestamp(day_date) + pd.Timedelta(days=1)).tz_localize(TZ)
    return _pv_timeseries(lat, lon, area_m2, eff, tilt_deg, azimuth_deg, start_ts, end_ts, freq)

def yearly_daily_energy(lat, lon, area_m2, eff, tilt_deg, azimuth_deg, year: int):
    start_ts = pd.Timestamp(f"{year}-01-01").tz_localize(TZ)
    end_ts = pd.Timestamp(f"{year+1}-01-01").tz_localize(TZ)

    df = _pv_timeseries(lat, lon, area_m2, eff, tilt_deg, azimuth_deg, start_ts, end_ts, freq="1h")
    daily_wh = df["power_w"].resample("1D").sum()  # hourly -> Wh/day
    return daily_wh


# ----------------------------
# GeoJSON sanitation (ipyleaflet requires JSON-serializable properties)
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


# ----------------------------
# Plan polygon lookup helpers
# ----------------------------
def _planer_prepare_for_query(planer: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    planer = planer.copy()

    if planer.crs is None:
        planer = planer.set_crs(4326)
    planer = planer.to_crs(4326)

    planer = planer[planer.geometry.notna()].copy()
    planer = planer[planer.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # Fix invalid geometries
    if hasattr(planer.geometry, "make_valid"):
        planer["geometry"] = planer.geometry.make_valid()
    else:
        planer["geometry"] = planer.buffer(0)

    # Explode multipolygons to help matching
    planer = planer.explode(index_parts=False)

    return planer

def _pick_summary_fields(planer_cols):
    preferred = [
        "plannavn", "plan_navn", "navn", "name", "titel", "title",
        "planid", "plan_id", "lokalplan", "lokalplan_nr", "nummer",
        "plantype", "type", "status",
        "vedtaget", "vedtagelsesdato", "dato", "startdato", "slutdato",
        "link", "url"
    ]
    chosen = [c for c in preferred if c in planer_cols and c != "geometry"]
    # If none match, fall back to first few columns (excluding geometry)
    if not chosen:
        chosen = [c for c in planer_cols if c != "geometry"][:10]
    return chosen[:12]

def find_plan_at_point(planer_query: gpd.GeoDataFrame, lat: float, lon: float) -> pd.Series | None:
    # Use projected CRS for accurate geometry ops
    proj_crs = 25833  # ETRS89 / UTM zone 33N (good for Bornholm)

    planer_proj = planer_query.to_crs(proj_crs)
    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(proj_crs).iloc[0]

    sidx = planer_proj.sindex
    candidates_idx = list(sidx.intersection(pt.bounds))
    if not candidates_idx:
        return None

    candidates = planer_proj.iloc[candidates_idx]
    hits = candidates[candidates.contains(pt)]
    if hits.empty:
        return None

    # If multiple: choose smallest area polygon
    if len(hits) > 1:
        best_idx = hits.geometry.area.idxmin()
        return hits.loc[best_idx]
    return hits.iloc[0]


# ----------------------------
# Load Bornholm polygon + planer polygons
# ----------------------------
url = "municipalities.geojson"
gdf = gpd.read_file(url).to_crs(4326)

bornholm = gdf[gdf["label_dk"].str.contains("Bornholm", case=False)].copy()
if bornholm.empty:
    raise ValueError("Could not find Bornholm in 'label_dk'.")

planer_raw = gpd.read_file("planer.geojson")
planer_query = _planer_prepare_for_query(planer_raw)

# Display layer must be JSON safe
planer_map = sanitize_for_geojson(planer_query)

PLAN_SUMMARY_FIELDS = _pick_summary_fields(planer_map.columns)


# ----------------------------
# Create map + add layers
# ----------------------------
center_lat = float(bornholm.geometry.centroid.y.iloc[0])
center_lon = float(bornholm.geometry.centroid.x.iloc[0])

m = Map(center=(center_lat, center_lon), zoom=10, basemap=basemaps.OpenStreetMap.Mapnik)

m.add_layer(GeoData(
    geo_dataframe=bornholm,
    name="Bornholm",
    style={"fillOpacity": 0.05, "weight": 2},
))

m.add_layer(GeoData(
    geo_dataframe=planer_map,
    name="Lokal Planer",
    style={"fillOpacity": 0.10, "weight": 1},
))

m.add_control(LayersControl(position="topright"))

marker = Marker(location=(center_lat, center_lon), draggable=False, name="Markør")
m.add_layer(marker)


# ----------------------------
# Widgets
# ----------------------------
w_preset = Dropdown(
    options=list(PANEL_PRESETS.keys()),
    value="Standard (20%, 10 m²)",
    description="Preset:",
    layout=Layout(width="320px"),
)

w_area = FloatSlider(
    value=PANEL_PRESETS[w_preset.value][0],
    min=1.0, max=100.0, step=0.5,
    description="Area (m²):",
    readout_format=".1f",
    layout=Layout(width="420px"),
)

w_eff = FloatSlider(
    value=PANEL_PRESETS[w_preset.value][1],
    min=0.05, max=1.00, step=0.005,
    description="Eff.:",
    readout_format=".3f",
    layout=Layout(width="420px"),
)

w_tilt = FloatSlider(
    value=40.0, min=0.0, max=90.0, step=1.0,
    description="Tilt (°):",
    readout_format=".0f",
    layout=Layout(width="420px"),
)

w_az = FloatSlider(
    value=180.0, min=0.0, max=360.0, step=1.0,
    description="Azimuth (°):",
    readout_format=".0f",
    layout=Layout(width="420px"),
)

w_freq = Dropdown(
    options=list(FREQ_OPTIONS.keys()),
    value="Hourly",
    description="24h res:",
    layout=Layout(width="320px"),
)

today = pd.Timestamp.now(tz=TZ).date()
w_day = DatePicker(description="Day:", value=today)

btn_refresh_summary = Button(description="Refresh summary", button_style="primary")
btn_show_plots = Button(description="Show plots", button_style="info")

status = Label(value="Click map to set location. Use Refresh summary / Show plots afterwards.")

def on_preset_change(change):
    if change["name"] != "value":
        return
    preset = change["new"]
    area, eff = PANEL_PRESETS[preset]
    if preset != "Custom":
        w_area.value = area
        w_eff.value = eff

w_preset.observe(on_preset_change, names="value")

controls = VBox(
    [
        HBox([w_preset, w_freq, btn_refresh_summary, btn_show_plots]),
        HBox([w_area, w_eff]),
        HBox([w_tilt, w_az]),
        HBox([w_day]),
        status,
    ],
    layout=Layout(border="1px solid #ddd", padding="10px", width="fit-content"),
)


# ----------------------------
# Outputs + state
# ----------------------------
out_text = Output()
out_plot = Output()

clicked = {"lat": None, "lon": None}

_year_cache = {"key": None, "daily_wh": None}

def _year_key(lat, lon, area, eff, tilt, az, year):
    return (round(lat, 6), round(lon, 6), round(area, 3), round(eff, 4), round(tilt, 2), round(az, 2), int(year))

def _get_settings():
    if w_day.value is None:
        raise ValueError("Select a day.")
    return {
        "area_m2": float(w_area.value),
        "eff": float(w_eff.value),
        "tilt": float(w_tilt.value),
        "az": float(w_az.value),
        "day": w_day.value,
        "freq_24h": FREQ_OPTIONS[w_freq.value],
        "year": int(pd.Timestamp(w_day.value).year),
    }

def compute_summary(lat, lon):
    s = _get_settings()
    df_24h = daily_profile(lat, lon, s["area_m2"], s["eff"], s["tilt"], s["az"], s["day"], s["freq_24h"])
    energy_24h_wh = _energy_wh_from_power(df_24h["power_w"])

    peak_power_w = float(df_24h["power_w"].max())
    peak_poa_wm2 = float(df_24h["poa_global"].max())

    key = _year_key(lat, lon, s["area_m2"], s["eff"], s["tilt"], s["az"], s["year"])
    if _year_cache["key"] != key:
        daily_wh = yearly_daily_energy(lat, lon, s["area_m2"], s["eff"], s["tilt"], s["az"], s["year"])
        _year_cache["key"] = key
        _year_cache["daily_wh"] = daily_wh
    else:
        daily_wh = _year_cache["daily_wh"]

    total_year_kwh = float(daily_wh.sum() / 1000.0)

    # Plan lookup (uses planer_query which is NOT sanitized-away)
    plan_row = find_plan_at_point(planer_query, lat, lon)

    return {
        "settings": s,
        "peak_power_w": peak_power_w,
        "peak_poa_wm2": peak_poa_wm2,
        "energy_24h_wh": float(energy_24h_wh),
        "total_year_kwh": total_year_kwh,
        "plan_row": plan_row,
    }

def compute_plot_data(lat, lon):
    s = _get_settings()
    df_24h = daily_profile(lat, lon, s["area_m2"], s["eff"], s["tilt"], s["az"], s["day"], s["freq_24h"])

    key = _year_key(lat, lon, s["area_m2"], s["eff"], s["tilt"], s["az"], s["year"])
    if _year_cache["key"] != key:
        daily_wh = yearly_daily_energy(lat, lon, s["area_m2"], s["eff"], s["tilt"], s["az"], s["year"])
        _year_cache["key"] = key
        _year_cache["daily_wh"] = daily_wh
    else:
        daily_wh = _year_cache["daily_wh"]

    return df_24h, daily_wh, s


def render_summary(lat, lon):
    result = compute_summary(lat, lon)
    s = result["settings"]
    plan_row = result["plan_row"]

    with out_text:
        out_text.clear_output(wait=True)

        print(f"Location: {lat:.5f}, {lon:.5f}")
        print(f"Settings: area={s['area_m2']:.1f} m², eff={s['eff']:.3f}, tilt={s['tilt']:.0f}°, az={s['az']:.0f}°")
        print(f"Selected day: {pd.Timestamp(s['day']).date()} (24h step={s['freq_24h']})")
        print(f"Peak POA: {result['peak_poa_wm2']:.0f} W/m²")
        print(f"Peak PV power: {result['peak_power_w']:.0f} W")
        print(f"Energy for selected day (clear-sky): {result['energy_24h_wh']/1000.0:.2f} kWh")
        print(f"Year {s['year']} total (daily clear-sky): {result['total_year_kwh']:.0f} kWh")

        print("\nPlan at clicked location:")
        if plan_row is None:
            print("  None (clicked point is not inside any plan polygon).")
        else:
            # Print selected fields only
            for c in PLAN_SUMMARY_FIELDS:
                if c in plan_row.index and c != "geometry":
                    print(f"  {c}: {plan_row[c]}")

def render_plots(lat, lon):
    df_24h, daily_wh, s = compute_plot_data(lat, lon)

    with out_plot:
        out_plot.clear_output(wait=True)

        fig1 = plt.figure(figsize=(10, 3))
        plt.plot(df_24h.index, df_24h["power_w"])
        plt.title(f"24-hour PV power (clear-sky) — {pd.Timestamp(s['day']).date()}")
        plt.xlabel("Time")
        plt.ylabel("Power (W)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        display(fig1)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(10, 3))
        plt.plot(daily_wh.index, daily_wh.to_numpy() / 1000.0)
        plt.title(f"Yearly PV energy per day (clear-sky) — {s['year']}")
        plt.xlabel("Date")
        plt.ylabel("Daily energy (kWh/day)")
        plt.tight_layout()
        display(fig2)
        plt.close(fig2)


# ----------------------------
# Map click: sets location + auto summary (includes plan lookup)
# ----------------------------
def handle_interaction(**kwargs):
    if kwargs.get("type") != "click":
        return

    lat, lon = kwargs["coordinates"]
    lat = float(lat)
    lon = float(lon)

    clicked["lat"] = lat
    clicked["lon"] = lon
    marker.location = (lat, lon)

    try:
        render_summary(lat, lon)
    except Exception as e:
        with out_text:
            out_text.clear_output(wait=True)
            print(f"Error: {e}")


# ----------------------------
# Buttons
# ----------------------------
def on_refresh_summary(_):
    if clicked["lat"] is None:
        with out_text:
            out_text.clear_output(wait=True)
            print("Click on the map first to select a location.")
        return
    try:
        render_summary(clicked["lat"], clicked["lon"])
    except Exception as e:
        with out_text:
            out_text.clear_output(wait=True)
            print(f"Error: {e}")

def on_show_plots(_):
    if clicked["lat"] is None:
        with out_text:
            out_text.clear_output(wait=True)
            print("Click on the map first to select a location.")
        return
    try:
        render_plots(clicked["lat"], clicked["lon"])
    except Exception as e:
        with out_plot:
            out_plot.clear_output(wait=True)
            print(f"Error: {e}")

btn_refresh_summary.on_click(on_refresh_summary)
btn_show_plots.on_click(on_show_plots)

m.on_interaction(handle_interaction)


# ----------------------------
# Display
# ----------------------------
display(controls)
display(out_text)
display(out_plot)
display(m)

# %%
