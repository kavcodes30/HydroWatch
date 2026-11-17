"""
Phase 2 - Data cleaning and preprocessing.

Design principles
-----------------
1. Every transformation is logged with a row count so the cleaning report is
   reproducible rather than narrative.
2. Physically implausible values are FLAGGED and set to missing, never silently
   deleted. Extreme-but-plausible readings (heavy rain, high turbidity, high TDS
   in coastal water) are deliberately retained: they are signal, not noise.
3. Nothing is imputed here. Imputation happens inside the scikit-learn pipeline
   fitted on training folds only, so no test-set information leaks backwards.
4. The '-' character is used by CPCB as a missing-value placeholder; it must be
   converted to NaN before any numeric coercion, otherwise it becomes a string
   and the column appears "complete".
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column dictionaries
# --------------------------------------------------------------------------
META_MAP = {
    "SlNo": "sl_no",
    "Station": "station",
    "Agency": "agency",
    "State LGD Code": "state_lgd",
    "State": "state",
    "District LGD Code": "district_lgd",
    "District": "district",
    "Tehsil": "tehsil",
    "Block": "block",
    "Village": "village",
    "River": "river",
    "Basin": "basin",
    "Tributary": "tributary",
    "Subtributary": "subtributary",
    "SubSubtributary": "sub_sub_tributary",
    "Local River": "local_river",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Data Acquisition Time": "observed_at",
}

CHEM_MAP = {
    "Potential of Hydrogen (pH)": "ph",
    "Dissolved oxygen (mg/L)": "dissolved_oxygen",
    "Total Dissolved Solids (mg/L)": "tds",
    "Amonia N (mgN/L)": "ammonia_n",
    "Nitrate N (mgN/L)": "nitrate_n",
    "Nitrite N+Nitrate N (mgN/L)": "nitrite_nitrate_n",
    "Chloride (mg/L)": "chloride",
    "Total Alkalinity (mg/L as CaCO3)": "alkalinity",
    "Iron(mg/L)": "iron",
    "Zinc (mg/L)": "zinc",
    "Lead (mg/L)": "lead",
    "Nickel (mg/L)": "nickel",
    "Chromium (mg/L)": "chromium",
    "Cadmium (mg/L)": "cadmium",
    "Copper (mg/L)": "copper",
    "Arsenic (mg/L)": "arsenic",
    "Mercury(mg/L)": "mercury",
    "Manganese (mg/L)": "manganese",
    "Fluoride (mg/L)": "fluoride",
    "Boron (mg/L)": "boron",
    "Calcium (mg/L)": "calcium",
    "Magnesium (mg/L)": "magnesium",
    "Sodium (mg/L)": "sodium",
    "Potassium (mg/L)": "potassium",
    "Carbonate (mg/L)": "carbonate",
    "Bicarbonate (mg/L)": "bicarbonate",
    "Sulphate (mg/L)": "sulphate",
    "Total Phosphorus (mgP/L)": "total_phosphorus",
    "Hardness Calcium (mgCaCO3/L)": "hardness_calcium",
    "Hardness_Magnesium (mg/L as CaCO3)": "hardness_magnesium",
    "Total Hardness (mgCaCO3/L)": "hardness_total",
    "Sodium Adsorption Ratio (%)": "sodium_adsorption_ratio",
    "Nitrate (mg/L)": "nitrate_as_no3",
}

PHYS_MAP = {
    "Electric Conductivity (μS/cm)": "electrical_conductivity",
    "Colour": "colour",
    "Odour": "odour",
    "Turbidity (NTU)": "turbidity",
    "Temperature (ºC)": "water_temperature",
    "Total Solids (mg/L)": "total_solids",
}

# Physically implausible bounds. Values outside these are data-entry or unit
# errors, not real extremes. Bounds are deliberately generous so that genuine
# severe readings survive.
IMPLAUSIBLE_BOUNDS = {
    "ph": (3.0, 12.0),
    "dissolved_oxygen": (0.0, 20.0),
    "tds": (0.0, 10_000.0),
    "ammonia_n": (0.0, 100.0),
    "nitrate_n": (0.0, 100.0),
    "alkalinity": (0.0, 3_000.0),
    # 60,000 uS/cm sits just above seawater (~50,000). Genuine brackish readings
    # from Kerala backwaters reach ~43,500 and must be kept; the 15,001,625 value
    # in the data is a keying error.
    "electrical_conductivity": (0.0, 60_000.0),
    "turbidity": (0.0, 1_000.0),
    "chloride": (0.0, 20_000.0),
    "iron": (0.0, 100.0),
}

# Columns that are 100% the '-' placeholder in every file and therefore carry
# no information (verified in the inspection phase).
DEAD_GEO_COLUMNS = ["block", "village", "river", "basin", "tributary",
                    "subtributary", "sub_sub_tributary", "local_river"]


@dataclass
class CleaningLog:
    """Accumulates a row-counted audit trail of every cleaning action."""

    entries: list[dict] = field(default_factory=list)

    def add(self, step: str, action: str, rows: int, detail: str = "") -> None:
        self.entries.append(
            {"step": step, "action": action, "rows_affected": int(rows), "detail": detail}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)


def load_raw(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate all chemical and physical CPCB files.

    State code is taken from the filename (e.g. ..._cpcb_mh_2021_2025.csv -> MH)
    and kept as a column so provenance survives concatenation.
    """
    chem_frames, phys_frames = [], []
    for path in sorted(glob.glob(os.path.join(raw_dir, "swq_*.csv"))):
        code = os.path.basename(path).split("_")[5].upper()
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df["state_code"] = code
        (chem_frames if "chemical" in path else phys_frames).append(df)
    return pd.concat(chem_frames, ignore_index=True), pd.concat(phys_frames, ignore_index=True)


def _clean_frame(df: pd.DataFrame, kind: str, log: CleaningLog) -> pd.DataFrame:
    """Common cleaning applied to both chemical and physical frames."""
    step = f"{kind}:load"
    log.add(step, "rows loaded", len(df))

    # 1. '-' placeholder -> NaN, then empty strings -> NaN
    before = df.replace("", np.nan).isna().sum().sum()
    df = df.replace("", np.nan).replace({"-": np.nan})
    log.add(step, "placeholder '-' and '' converted to NaN",
            df.isna().sum().sum() - before,
            "CPCB uses '-' for missing; left as text it defeats numeric parsing")

    # 2. standardise column names
    mapping = {**META_MAP, **(CHEM_MAP if kind == "chemical" else PHYS_MAP)}
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    # 3. types
    df["observed_at"] = pd.to_datetime(df["observed_at"], format="%d-%m-%Y %H:%M", errors="coerce")
    bad_dates = df["observed_at"].isna().sum()
    if bad_dates:
        log.add(step, "unparseable timestamps dropped", bad_dates)
        df = df.dropna(subset=["observed_at"])

    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [c for c in df.columns
                    if c not in META_MAP.values() and c not in ("state_code",)]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. text normalisation for categorical fields
    for col in ("state", "district", "tehsil", "station"):
        if col in df.columns:
            df[col] = df[col].str.strip()

    # 5. drop columns that are entirely empty (no information at all)
    dead = [c for c in numeric_cols if df[c].notna().sum() == 0]
    if dead:
        df = df.drop(columns=dead)
        log.add(step, "all-empty analyte columns dropped", 0,
                f"{len(dead)} columns: {', '.join(dead)}")

    # 6. drop dead geography columns (100% placeholder)
    present_dead = [c for c in DEAD_GEO_COLUMNS if c in df.columns]
    if present_dead:
        df = df.drop(columns=present_dead)
        log.add(step, "placeholder-only geography columns dropped", 0,
                ", ".join(present_dead))

    # 7. exact duplicates
    dup = df.duplicated().sum()
    if dup:
        df = df.drop_duplicates()
        log.add(step, "exact duplicate rows removed", dup)

    # 8. implausible values -> NaN (flagged, not deleted)
    for col, (lo, hi) in IMPLAUSIBLE_BOUNDS.items():
        if col in df.columns:
            mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            n = int(mask.sum())
            if n:
                df.loc[mask, col] = np.nan
                log.add(step, f"implausible {col} set to NaN", n, f"outside [{lo}, {hi}]")

    df["source_type"] = kind
    return df


def aggregate_to_station_month(
    chem: pd.DataFrame, phys: pd.DataFrame, log: CleaningLog
) -> pd.DataFrame:
    """Collapse repeated station observations to one row per station-month.

    Grain rationale: sampling is roughly monthly per station, but some stations
    are sampled twice in a month. Averaging numeric analytes to a station-month
    gives one comparable row per station per period, which matches the monthly
    rainfall and the monthly risk label.
    """
    # Group key is the station and the month ONLY. Some stations carry two
    # slightly different coordinate/district records for the same date (the
    # 'conflicting duplicate' case), and keying on latitude/longitude would
    # split those into separate rows instead of averaging them.
    key = ["station", "year_month"]
    out = []
    for df, kind in ((chem, "chemical"), (phys, "physical")):
        df = df.copy()
        df["year_month"] = df["observed_at"].dt.to_period("M").astype(str)

        # station attributes: first non-null value wins, coordinates averaged
        attrs = df.groupby(key, dropna=False).agg(
            state=("state", "first"),
            district=("district", "first"),
            state_code=("state_code", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )

        num_cols = [c for c in df.columns
                    if df[c].dtype.kind in "fiu"
                    and c not in ("sl_no", "state_lgd", "district_lgd",
                                  "latitude", "longitude")]
        meas = df.groupby(key, dropna=False).agg(
            {**{c: "mean" for c in num_cols}, "observed_at": ["count", "min", "max"]}
        )
        # flatten the MultiIndex: analytes keep their bare name, timestamp
        # statistics become explicit named columns
        meas.columns = [
            f"n_samples_{kind}" if (a == "observed_at" and b == "count")
            else f"first_sample_{kind}" if (a == "observed_at" and b == "min")
            else f"last_sample_{kind}" if (a == "observed_at" and b == "max")
            else a
            for a, b in meas.columns
        ]
        out.append(attrs.join(meas).reset_index())

    phys_only = out[1].drop(
        columns=[c for c in ("state", "district", "state_code", "latitude", "longitude")
                 if c in out[1].columns]
    )
    merged = out[0].merge(phys_only, on=key, how="left")
    assert not merged.duplicated(subset=key).any(), "station-month grain is not unique"
    log.add("aggregate", "station-month rows created", len(merged),
            f"chemical grain {len(out[0])}, physical grain {len(out[1])}")
    return merged


def run_pipeline(raw_dir: str, interim_dir: str, processed_dir: str,
                 reports_dir: str) -> pd.DataFrame:
    """Execute cleaning end-to-end and persist outputs."""
    log = CleaningLog()
    chem_raw, phys_raw = load_raw(raw_dir)
    chem = _clean_frame(chem_raw, "chemical", log)
    phys = _clean_frame(phys_raw, "physical", log)

    # duplicate station-date pairs need adjudication before aggregation
    for df, kind in ((chem, "chemical"), (phys, "physical")):
        dups = df.duplicated(subset=["station", "observed_at"], keep=False).sum()
        log.add("dedupe", f"repeated {kind} station-date pairs found", dups,
                "averaged during station-month aggregation, not deleted")

    sm = aggregate_to_station_month(chem, phys, log)

    # date features
    sm["year"] = sm["year_month"].str[:4].astype(int)
    sm["month"] = sm["year_month"].str[5:7].astype(int)
    sm["month_sin"] = np.sin(2 * np.pi * sm["month"] / 12)
    sm["month_cos"] = np.cos(2 * np.pi * sm["month"] / 12)

    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    chem.to_csv(os.path.join(interim_dir, "chemical_clean.csv"), index=False)
    phys.to_csv(os.path.join(interim_dir, "physical_clean.csv"), index=False)
    sm.to_csv(os.path.join(processed_dir, "station_month_clean.csv"), index=False)
    log.to_frame().to_csv(os.path.join(reports_dir, "cleaning_log.csv"), index=False)

    print(f"chemical rows (cleaned): {len(chem):,}")
    print(f"physical rows (cleaned): {len(phys):,}")
    print(f"station-month rows:      {len(sm):,}")
    print(f"cleaning log entries:    {len(log.entries)}")
    return sm


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_pipeline(
        os.path.join(base, "data", "raw"),
        os.path.join(base, "data", "interim"),
        os.path.join(base, "data", "processed"),
        os.path.join(base, "reports"),
    )
