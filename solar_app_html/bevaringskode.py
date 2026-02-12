# bevaringskode.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import requests

BASE_URL = "https://drift.kortinfo.net/Feature.aspx"


@dataclass(frozen=True)
class KortinfoQueryConfig:
    site: str = "bornholm"
    page: str = "dkplan_kommuneplan2020"
    layer: str = "TL931496"
    srs: str = "EPSG:4326"
    output: str = "geojson"
    branch: str = "null"
    geojsoncallback: Optional[str] = None


def _build_bbox_polygon_wkt(lon: float, lat: float, d: float) -> str:
    lon_min, lon_max = lon - d, lon + d
    lat_min, lat_max = lat - d, lat + d
    return (
        "POLYGON (("
        f"{lon_min} {lat_min},"
        f"{lon_min} {lat_max},"
        f"{lon_max} {lat_max},"
        f"{lon_max} {lat_min},"
        f"{lon_min} {lat_min}"
        "))"
    )


def _build_filter_xml_for_intersects(wkt_polygon: str, epsg: int = 4326) -> str:
    return f"<intersects><wkt epsg='{epsg}'>{wkt_polygon}</wkt></intersects>"


def build_query_params(
    lon: float,
    lat: float,
    d: float = 0.0001,
    config: KortinfoQueryConfig = KortinfoQueryConfig(),
) -> Dict[str, str]:
    wkt = _build_bbox_polygon_wkt(lon=lon, lat=lat, d=d)
    filter_xml = _build_filter_xml_for_intersects(wkt_polygon=wkt, epsg=4326)

    params: Dict[str, str] = {
        "srs": config.srs,
        "output": config.output,
        "site": config.site,
        "page": config.page,
        "layer": config.layer,
        "branch": config.branch,
        "filter": filter_xml,
    }
    if config.geojsoncallback:
        params["geojsoncallback"] = config.geojsoncallback
    return params


def fetch_geojson(
    lon: float,
    lat: float,
    d: float = 0.0001,
    config: KortinfoQueryConfig = KortinfoQueryConfig(),
    timeout: float = 10.0,
) -> Dict[str, Any]:
    params = build_query_params(lon=lon, lat=lat, d=d, config=config)

    headers = {
        "User-Agent": "Mozilla/5.0 (BevaringskodeFetcher/1.0)",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }

    resp = requests.get(BASE_URL, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()

    text = resp.text.strip()
    if config.geojsoncallback:
        cb = config.geojsoncallback
        prefix = f"{cb}("
        suffix = ")"
        if text.startswith(prefix) and text.endswith(suffix):
            text = text[len(prefix) : -len(suffix)].strip()
        return requests.models.complexjson.loads(text)

    return resp.json()


def get_bevaringskode_unique(
    lon: float,
    lat: float,
    d: float = 0.0001,
    config: KortinfoQueryConfig = KortinfoQueryConfig(),
    timeout: float = 10.0,
) -> Optional[str]:
    data = fetch_geojson(lon=lon, lat=lat, d=d, config=config, timeout=timeout)
    features: List[Dict[str, Any]] = data.get("features", []) or []
    if not features:
        return None

    codes: list[str] = []
    for f in features:
        props = f.get("properties", {}) or {}
        c = props.get("Bevaringskode")
        if c is not None:
            codes.append(str(c))

    if not codes:
        return None

    unique = set(codes)
    if len(unique) == 1:
        return next(iter(unique))

    return None
