"""
Environmental data acquisition from NASA POWER (MERRA-2 reanalysis).

Why this module exists
----------------------
The CPCB water-quality files contain no rainfall or ambient-temperature data,
but the project scope requires both. NASA POWER provides daily precipitation
(PRECTOTCORR) and 2 m air temperature (T2M) for any coordinate back to 1981.

IMPORTANT SCIENTIFIC CAVEAT
---------------------------
POWER data are *modelled reanalysis* (MERRA-2), not gauge observations. They are
gridded (~0.5 deg) and should be described as "modelled" wherever they appear in
reports. They are used here because no IMD/gauge series was supplied.

Caching
-------
Requests are made per unique 0.5-degree grid cell rather than per station
(1096 stations collapse to 255 cells), and results are cached to CSV so that
re-running the pipeline does not re-hit the API.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=PRECTOTCORR,T2M&community=AG"
    "&longitude={lon}&latitude={lat}&start=20210101&end=20211231&format=JSON"
)

GRID = 0.5  # degrees; MERRA-2 native resolution is 0.5 x 0.625


def grid_key(lat: float, lon: float) -> tuple[float, float]:
    """Snap a coordinate to its 0.5-degree grid cell centre."""
    return round(round(lat / GRID) * GRID, 2), round(round(lon / GRID) * GRID, 2)


def fetch_cell(lat: float, lon: float, retries: int = 3) -> list[dict]:
    """Fetch one grid cell's daily rainfall and temperature for 2021."""
    url = POWER_URL.format(lat=lat, lon=lon)
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
            params = payload["properties"]["parameter"]
            precip, t2m = params["PRECTOTCORR"], params["T2M"]
            fill = payload["header"].get("fill_value", -999.0)
            rows = []
            for day in sorted(precip):
                p, t = precip[day], t2m.get(day)
                rows.append(
                    {
                        "grid_lat": lat,
                        "grid_lon": lon,
                        "date": pd.to_datetime(day, format="%Y%m%d"),
                        # -999 is POWER's fill value for missing days
                        "rainfall_mm": None if p == fill else p,
                        "t2m_c": None if t == fill else t,
                    }
                )
            return rows
        except Exception as exc:  # network hiccup, rate limit, etc.
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"NASA POWER failed for {lat},{lon}: {last_err}")


def build_environmental_dataset(
    stations: pd.DataFrame, cache_path: str, workers: int = 12
) -> pd.DataFrame:
    """Fetch (or load from cache) daily environmental data for all grid cells.

    Parameters
    ----------
    stations : DataFrame with columns ['Station', 'Latitude', 'Longitude']
    cache_path : CSV path used to persist results between runs
    """
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        return cached

    cells = (
        stations.assign(
            grid_lat=(stations["Latitude"] / GRID).round() * GRID,
            grid_lon=(stations["Longitude"] / GRID).round() * GRID,
        )[["grid_lat", "grid_lon"]]
        .drop_duplicates()
        .round(2)
    )
    pairs = list(cells.itertuples(index=False, name=None))
    print(f"Fetching NASA POWER daily data for {len(pairs)} grid cells...")

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_cell, lat, lon): (lat, lon) for lat, lon in pairs}
        for i, fut in enumerate(as_completed(futures), 1):
            all_rows.extend(fut.result())
            if i % 25 == 0:
                print(f"  {i}/{len(pairs)} cells done")

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} station-days to {cache_path}")
    return df


if __name__ == "__main__":
    import glob

    chem = pd.concat(
        [pd.read_csv(f, dtype=str, keep_default_na=False)
         for f in glob.glob("data/raw/*chemical*.csv")],
        ignore_index=True,
    )
    chem["Latitude"] = pd.to_numeric(chem["Latitude"], errors="coerce")
    chem["Longitude"] = pd.to_numeric(chem["Longitude"], errors="coerce")
    stations = chem.dropna(subset=["Latitude", "Longitude"]).drop_duplicates("Station")
    build_environmental_dataset(stations, "data/raw/nasa_power_daily_2021.csv")
