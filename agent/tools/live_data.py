"""Live-data tools for the Mohafiz agent — all keyless/free APIs.

Each function returns a compact dict ready to inject into the LLM context.
Run this file directly for a smoke test of all four tools.
"""
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

UA = {"User-Agent": "mohafiz-climate-rag (github.com/MuteeullahBaig/mohafiz_climate_rag)"}
TIMEOUT = 20.0

# ReliefWeb V2 (V1 decommissioned) requires a pre-approved appname since 2025-11-01.
# Request one at https://apidoc.reliefweb.int/parameters#appname and set it in .env.
RELIEFWEB_APPNAME = os.environ.get("RELIEFWEB_APPNAME", "").strip()

# Pakistan bounding box for earthquake queries
PK_BBOX = {"minlatitude": 23, "maxlatitude": 38, "minlongitude": 60, "maxlongitude": 78}

CITY_COORDS = {
    "karachi": (24.86, 67.01), "lahore": (31.55, 74.34), "islamabad": (33.69, 73.06),
    "peshawar": (34.01, 71.58), "quetta": (30.18, 66.98), "multan": (30.20, 71.47),
    "hyderabad": (25.39, 68.37), "faisalabad": (31.42, 73.08), "rawalpindi": (33.60, 73.04),
    "gilgit": (35.92, 74.31), "skardu": (35.30, 75.63), "hunza": (36.32, 74.65),
    "muzaffarabad": (34.37, 73.47), "sukkur": (27.70, 68.86), "gwadar": (25.13, 62.32),
}


def get_weather(city: str = None, lat: float = None, lon: float = None,
                include_soil: bool = False) -> dict:
    """Open-Meteo forecast (keyless). 3-day daily summary + current conditions."""
    if city:
        c = CITY_COORDS.get(city.strip().lower())
        if not c:
            return {"error": f"unknown city '{city}'; pass lat/lon", "known": sorted(CITY_COORDS)}
        lat, lon = c
    params = {
        "latitude": lat, "longitude": lon, "timezone": "Asia/Karachi", "forecast_days": 3,
        "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
    }
    if include_soil:
        params["hourly"] = "soil_moisture_0_to_1cm,et0_fao_evapotranspiration"
    r = httpx.get("https://api.open-meteo.com/v1/forecast", params=params, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    out = {"source": "open-meteo.com", "lat": lat, "lon": lon,
           "current": d.get("current"), "daily": d.get("daily")}
    if include_soil and "hourly" in d:
        h = d["hourly"]
        out["soil_now"] = {
            "soil_moisture_0_1cm": h["soil_moisture_0_to_1cm"][:24],
            "et0_mm": h["et0_fao_evapotranspiration"][:24],
        }
    return out


def get_earthquakes(days: int = 7, min_magnitude: float = 4.0) -> dict:
    """USGS FDSN event query (keyless), Pakistan bbox."""
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {"format": "geojson", "starttime": start, "minmagnitude": min_magnitude,
              "orderby": "time", **PK_BBOX}
    r = httpx.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params=params,
                  headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    feats = r.json().get("features", [])
    quakes = [{
        "magnitude": f["properties"]["mag"],
        "place": f["properties"]["place"],
        "time_utc": datetime.fromtimestamp(f["properties"]["time"] / 1000, timezone.utc).isoformat(timespec="minutes"),
        "depth_km": f["geometry"]["coordinates"][2],
    } for f in feats[:10]]
    return {"source": "earthquake.usgs.gov", "window_days": days,
            "min_magnitude": min_magnitude, "count": len(feats), "events": quakes}


def get_gdacs_alerts() -> dict:
    """GDACS global alert RSS, filtered to Pakistan-relevant items."""
    r = httpx.get("https://www.gdacs.org/xml/rss.xml", headers=UA, timeout=TIMEOUT,
                  follow_redirects=True)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    items = list(root.iter("item"))
    alerts = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        blob = f"{title} {desc}".lower()
        if "pakistan" in blob or "india-pakistan" in blob:
            alerts.append({"title": title, "published": (item.findtext("pubDate") or "").strip(),
                           "link": (item.findtext("link") or "").strip()})
    # total_feed_items distinguishes "no PK alerts right now" from "parser broke"
    return {"source": "gdacs.org", "pakistan_alerts": alerts, "count": len(alerts),
            "total_feed_items": len(items)}


def get_sitreps(limit: int = 5) -> dict:
    """ReliefWeb V2 latest Pakistan situation reports (needs an approved appname).

    Degrades gracefully when RELIEFWEB_APPNAME is unset — the agent simply omits
    sitreps rather than erroring, so the other three tools stay usable.
    """
    if not RELIEFWEB_APPNAME:
        return {"source": "reliefweb.int", "status": "not_configured",
                "note": "set RELIEFWEB_APPNAME in .env (request approval at "
                        "apidoc.reliefweb.int/parameters#appname)", "reports": []}
    payload = {
        "filter": {"field": "country", "value": "Pakistan"},
        "sort": ["date:desc"],
        "limit": limit,
        "fields": {"include": ["title", "date.created", "url", "format.name", "source.shortname"]},
    }
    url = f"https://api.reliefweb.int/v2/reports?appname={RELIEFWEB_APPNAME}"
    r = httpx.post(url, json=payload, headers=UA, timeout=TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    items = [{
        "title": d["fields"].get("title"),
        "date": (d["fields"].get("date") or {}).get("created", "")[:10],
        "format": ((d["fields"].get("format") or [{}])[0]).get("name"),
        "source": ((d["fields"].get("source") or [{}])[0]).get("shortname"),
        "url": d["fields"].get("url"),
    } for d in r.json().get("data", [])]
    return {"source": "reliefweb.int", "status": "ok", "reports": items}


if __name__ == "__main__":
    import json
    for name, fn in [("weather(lahore)", lambda: get_weather("lahore")),
                     ("earthquakes(7d)", get_earthquakes),
                     ("gdacs", get_gdacs_alerts),
                     ("sitreps", get_sitreps)]:
        try:
            out = fn()
            print(f"\n=== {name} OK ===")
            print(json.dumps(out, indent=1, ensure_ascii=False)[:600])
        except Exception as e:
            print(f"\n=== {name} FAILED: {type(e).__name__}: {e}")
