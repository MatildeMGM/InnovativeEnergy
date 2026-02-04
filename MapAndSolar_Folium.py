import streamlit as st
import geopandas as gpd
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import folium
import json
from shapely.geometry import Point
from streamlit_folium import st_folium


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

    return pd.DataFrame(
        {"poa_global": poa["poa_global"].astype(float),
         "power_w": power_w.astype(float)},
        index=times,
    )

def _energy_wh_from_power(power_w: pd.Series) -> float:
    dt_h = pd.Series(power_w.index).diff().dt.total_seconds().div(3600).fillna(0).to_numpy()
    return float((power_w.to_numpy() * dt_h).sum())

def _energy_wh_series_from_power(power_w: pd.Series) -> pd.Series:
    dt_h = power_w.index.to_series().diff().dt.total_seconds().div(3600.0)
    dt_h.iloc[0] = 0.0
    return power_w * dt_h

@st.cache_data(show_spinner=False, ttl=24*3600)
def _get_pvgis_tmy_irradiance(lat: float, lon: float) -> pd.DataFrame:
    tmy, meta = pvlib.iotools.get_pvgis_tmy(lat, lon, map_variables=True)

    if tmy.index.tz is None:
        tmy.index = tmy.index.tz_localize("UTC")

    tmy = tmy.tz_convert(TZ)
    return tmy[["ghi", "dni", "dhi"]].astype(float)

def _pv_timeseries_from_irradiance(lat, lon, area_m2, eff, tilt_deg, azimuth_deg, times, ghi, dni, dhi):
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

    power_w = poa["poa_global"].clip(lower=0) * area_m2 * eff

    return pd.DataFrame(
        {"poa_global": poa["poa_global"].astype(float), "power_w": power_w.astype(float)},
        index=times,
    )

def daily_profile(lat, lon, area_m2, eff, tilt_deg, azimuth_deg, day_date, freq):
    start_ts = pd.Timestamp(day_date).tz_localize(TZ)
    end_ts = (pd.Timestamp(day_date) + pd.Timedelta(days=1)).tz_localize(TZ)
    return _pv_timeseries(lat, lon, area_m2, eff, tilt_deg, azimuth_deg, start_ts, end_ts, freq)

def yearly_daily_energy(lat, lon, area_m2, eff, tilt_deg, azimuth_deg, year: int):
    tmy = _get_pvgis_tmy_irradiance(lat, lon)

    times = tmy.index.map(lambda ts: ts.replace(year=year))
    times = pd.DatetimeIndex(times).tz_convert(TZ)

    df = _pv_timeseries_from_irradiance(
        lat, lon, area_m2, eff, tilt_deg, azimuth_deg,
        times=times,
        ghi=tmy["ghi"].to_numpy(),
        dni=tmy["dni"].to_numpy(),
        dhi=tmy["dhi"].to_numpy(),
    )

    e_wh = _energy_wh_series_from_power(df["power_w"])
    daily_wh = e_wh.resample("1D").sum()
    return daily_wh


# ----------------------------
# Geo helpers
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
        "link", "url"
    ]
    chosen = [c for c in preferred if c in planer_cols and c != "geometry"]
    if not chosen:
        chosen = [c for c in planer_cols if c != "geometry"][:10]
    return chosen[:12]

def find_plan_at_point(planer_query: gpd.GeoDataFrame, lat: float, lon: float):
    proj_crs = 25833  # ETRS89 / UTM zone 33N
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

    if len(hits) > 1:
        best_idx = hits.geometry.area.idxmin()
        return hits.loc[best_idx]
    return hits.iloc[0]


# ----------------------------
# Cached data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    gdf = gpd.read_file("municipalities.geojson").to_crs(4326)
    bornholm = gdf[gdf["label_dk"].str.contains("Bornholm", case=False)].copy()
    if bornholm.empty:
        raise ValueError("Could not find Bornholm in 'label_dk'.")

    planer_raw = gpd.read_file("planer.geojson")
    planer_query = _planer_prepare_for_query(planer_raw)

    planer_map = sanitize_for_geojson(planer_query)
    plan_fields = _pick_summary_fields(planer_map.columns)

    center_lat = float(bornholm.geometry.centroid.y.iloc[0])
    center_lon = float(bornholm.geometry.centroid.x.iloc[0])
    return bornholm, planer_query, planer_map, plan_fields, center_lat, center_lon

@st.cache_data(show_spinner=False)
def make_display_geojson(bornholm_geojson_text: str, planer_geojson_text: str, tol_deg: float):
    bornholm = gpd.GeoDataFrame.from_features(json.loads(bornholm_geojson_text), crs=4326)
    planer = gpd.GeoDataFrame.from_features(json.loads(planer_geojson_text), crs=4326)

    bornholm["geometry"] = bornholm.geometry.simplify(tol_deg, preserve_topology=True)
    planer["geometry"] = planer.geometry.simplify(tol_deg, preserve_topology=True)

    return bornholm.__geo_interface__, planer.__geo_interface__



# ----------------------------
# Cached PV computations
# ----------------------------
@st.cache_data(show_spinner=False)
def cached_daily_profile(lat, lon, area_m2, eff, tilt, az, day, freq_24h):
    return daily_profile(lat, lon, area_m2, eff, tilt, az, day, freq_24h)

@st.cache_data(show_spinner=False)
def cached_yearly_daily_energy(lat, lon, area_m2, eff, tilt, az, year):
    return yearly_daily_energy(lat, lon, area_m2, eff, tilt, az, year)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PV Irradiance Map (Streamlit)", layout="wide")
st.title("PV irradiance / production estimator (Streamlit)")

bornholm, planer_query, planer_map, PLAN_SUMMARY_FIELDS, center_lat, center_lon = load_data()

# Session state initialization
if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = center_lat
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = center_lon

if "pending_lat" not in st.session_state:
    st.session_state.pending_lat = st.session_state.selected_lat
if "pending_lon" not in st.session_state:
    st.session_state.pending_lon = st.session_state.selected_lon

if "applied_settings" not in st.session_state:
    st.session_state.applied_settings = {
        "preset": "Standard (20%, 10 m²)",
        "area_m2": PANEL_PRESETS["Standard (20%, 10 m²)"][0],
        "eff": PANEL_PRESETS["Standard (20%, 10 m²)"][1],
        "tilt": 40.0,
        "az": 180.0,
        "freq_label": "Hourly",
        "day": pd.Timestamp.now(tz=TZ).date(),
    }

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_key" not in st.session_state:
    st.session_state.last_key = None
if "show_plots" not in st.session_state:
    st.session_state.show_plots = False

# Sidebar form: changes only apply on submit (prevents stutter while editing)
st.sidebar.header("Settings")

with st.sidebar.form("settings_form", clear_on_submit=False):
    s0 = st.session_state.applied_settings

    preset = st.selectbox("Preset", list(PANEL_PRESETS.keys()),
                          index=list(PANEL_PRESETS.keys()).index(s0["preset"]))

    preset_area, preset_eff = PANEL_PRESETS[preset]

    if preset == "Custom":
        area_m2 = st.slider("Area (m²)", 1.0, 100.0, float(s0["area_m2"]), 0.5)
        eff = st.slider("Efficiency", 0.05, 1.00, float(s0["eff"]), 0.005)
    else:
        # Display fixed values but still show as sliders (disabled)
        area_m2 = st.slider("Area (m²)", 1.0, 100.0, float(preset_area), 0.5, disabled=True)
        eff = st.slider("Efficiency", 0.05, 1.00, float(preset_eff), 0.005, disabled=True)

    tilt = st.slider("Tilt (°)", 0.0, 90.0, float(s0["tilt"]), 1.0)
    az = st.slider("Azimuth (°)", 0.0, 360.0, float(s0["az"]), 1.0)
    freq_label = st.selectbox("24h resolution", list(FREQ_OPTIONS.keys()),
                              index=list(FREQ_OPTIONS.keys()).index(s0["freq_label"]))
    day = st.date_input("Day", value=s0["day"])

    simplify_tol = st.slider("Map detail (simplify)", 0.0001, 0.0020, 0.0003, 0.0001,
                             help="Higher = faster map, less detail. Display only; lookup uses full geometry.")

    apply_settings = st.form_submit_button("Apply settings")

# Buttons (outside form so they don't interfere with form submits)
refresh_summary = st.sidebar.button("Refresh summary")
toggle_plots = st.sidebar.button("Show plots" if not st.session_state.show_plots else "Hide plots")

if toggle_plots:
    st.session_state.show_plots = not st.session_state.show_plots

# Apply settings only when button pressed
if apply_settings:
    if preset != "Custom":
        area_m2_committed, eff_committed = float(preset_area), float(preset_eff)
    else:
        area_m2_committed, eff_committed = float(area_m2), float(eff)

    st.session_state.applied_settings = {
        "preset": preset,
        "area_m2": area_m2_committed,
        "eff": eff_committed,
        "tilt": float(tilt),
        "az": float(az),
        "freq_label": freq_label,
        "day": day,
        "simplify_tol": float(simplify_tol),
    }

# Always read from applied settings for computation
s = st.session_state.applied_settings
area_m2_use = float(s["area_m2"])
eff_use = float(s["eff"])
tilt_use = float(s["tilt"])
az_use = float(s["az"])
freq_24h_use = FREQ_OPTIONS[s["freq_label"]]
day_use = s["day"]
tol_use = float(s.get("simplify_tol", 0.0003))

# Build display geojson (simplified) with caching
bornholm_geojson_text = bornholm.to_json()
planer_geojson_text = planer_map.to_json()

bornholm_geojson, planer_geojson = make_display_geojson(bornholm_geojson_text, planer_geojson_text, tol_use)


# ----------------------------
# Map section (click -> pending -> confirm)
# ----------------------------
st.subheader("Map")

m = folium.Map(
    location=[st.session_state.pending_lat, st.session_state.pending_lon],
    zoom_start=10,
    tiles="OpenStreetMap",
    control_scale=True
)

folium.GeoJson(
    data=bornholm_geojson,
    name="Bornholm",
    style_function=lambda x: {"fillOpacity": 0.05, "weight": 2},
).add_to(m)

folium.GeoJson(
    data=planer_geojson,
    name="Lokal Planer",
    style_function=lambda x: {"fillOpacity": 0.10, "weight": 1},
).add_to(m)

folium.LayerControl().add_to(m)

folium.Marker(
    [st.session_state.selected_lat, st.session_state.selected_lon],
    tooltip="Selected location"
).add_to(m)

folium.Marker(
    [st.session_state.pending_lat, st.session_state.pending_lon],
    tooltip="Pending location"
).add_to(m)

map_state = st_folium(m, width=1100, height=600)

# Update pending location on click only (does NOT change selected)
if map_state and map_state.get("last_clicked"):
    st.session_state.pending_lat = float(map_state["last_clicked"]["lat"])
    st.session_state.pending_lon = float(map_state["last_clicked"]["lng"])

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    st.write(f"Selected: {st.session_state.selected_lat:.5f}, {st.session_state.selected_lon:.5f}")
with colB:
    st.write(f"Pending: {st.session_state.pending_lat:.5f}, {st.session_state.pending_lon:.5f}")

with colC:
    use_location = st.button("Use this location", type="primary",
                             help="Commit the pending marker as the selected point (triggers recompute).")

if use_location:
    st.session_state.selected_lat = st.session_state.pending_lat
    st.session_state.selected_lon = st.session_state.pending_lon


# ----------------------------
# Summary + plots (compute gated)
# ----------------------------
def compute_summary(lat, lon):
    df_24h = cached_daily_profile(lat, lon, area_m2_use, eff_use, tilt_use, az_use, day_use, freq_24h_use)
    energy_24h_wh = _energy_wh_from_power(df_24h["power_w"])

    peak_power_w = float(df_24h["power_w"].max())
    peak_poa_wm2 = float(df_24h["poa_global"].max())

    year = int(pd.Timestamp(day_use).year)
    daily_wh = cached_yearly_daily_energy(lat, lon, area_m2_use, eff_use, tilt_use, az_use, year)
    total_year_kwh = float(daily_wh.sum() / 1000.0)

    plan_row = find_plan_at_point(planer_query, lat, lon)

    return {
        "df_24h": df_24h,
        "daily_wh": daily_wh,
        "year": year,
        "peak_power_w": peak_power_w,
        "peak_poa_wm2": peak_poa_wm2,
        "energy_24h_wh": float(energy_24h_wh),
        "total_year_kwh": total_year_kwh,
        "plan_row": plan_row,
    }

def current_compute_key():
    return (
        round(float(st.session_state.selected_lat), 6),
        round(float(st.session_state.selected_lon), 6),
        round(area_m2_use, 3),
        round(eff_use, 4),
        round(tilt_use, 2),
        round(az_use, 2),
        freq_24h_use,
        str(day_use),
        int(pd.Timestamp(day_use).year),
    )

st.subheader("Summary")

key_now = current_compute_key()

# Recompute only when:
# - first run, OR
# - user commits location, OR
# - user applies settings, OR
# - user explicitly refreshes, OR
# - key changed due to those actions (rare but safe)
should_recompute = (
    st.session_state.last_result is None
    or refresh_summary
    or use_location
    or apply_settings
    or (st.session_state.last_key != key_now)
)

if should_recompute:
    with st.spinner("Computing… (PVGIS yearly may take a moment on first run)"):
        try:
            st.session_state.last_result = compute_summary(
                float(st.session_state.selected_lat),
                float(st.session_state.selected_lon)
            )
            st.session_state.last_key = key_now
        except Exception as e:
            st.session_state.last_result = None
            st.error(f"Error: {e}")

result = st.session_state.last_result

if result is None:
    st.info("Click the map to set a pending point, then press **Use this location**.")
else:
    st.write(
        f"Settings: preset={s['preset']}, area={area_m2_use:.1f} m², eff={eff_use:.3f}, "
        f"tilt={tilt_use:.0f}°, az={az_use:.0f}° | Day={pd.Timestamp(day_use).date()} | 24h step={freq_24h_use}"
    )
    st.write(f"Peak POA: {result['peak_poa_wm2']:.0f} W/m²")
    st.write(f"Peak PV power: {result['peak_power_w']:.0f} W")
    st.write(f"Energy for selected day (clear-sky): {result['energy_24h_wh']/1000.0:.2f} kWh")
    st.write(f"Year {result['year']} total (TMY/PVGIS): {result['total_year_kwh']:.0f} kWh")

    st.markdown("**Plan at selected location:**")
    plan_row = result["plan_row"]
    if plan_row is None:
        st.write("None (selected point is not inside any plan polygon).")
    else:
        plan_dict = {}
        for c in PLAN_SUMMARY_FIELDS:
            if c in plan_row.index and c != "geometry":
                plan_dict[c] = plan_row[c]
        st.json(plan_dict)


st.subheader("Plots")
if result is None:
    st.info("Compute a summary first.")
elif not st.session_state.show_plots:
    st.write("Press **Show plots** in the sidebar to render the figures.")
else:
    df_24h = result["df_24h"]
    daily_wh = result["daily_wh"]

    fig1 = plt.figure(figsize=(10, 3))
    plt.plot(df_24h.index, df_24h["power_w"])
    plt.title(f"24-hour PV power (clear-sky) — {pd.Timestamp(day_use).date()}")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig1, clear_figure=True)

    fig2 = plt.figure(figsize=(10, 3))
    plt.plot(daily_wh.index, daily_wh.to_numpy() / 1000.0)
    plt.title(f"Yearly PV energy per day (TMY/PVGIS) — {result['year']}")
    plt.xlabel("Date")
    plt.ylabel("Daily energy (kWh/day)")
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)
