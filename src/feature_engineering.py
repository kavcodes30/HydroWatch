"""
Phase 3 & 4 - Feature engineering and target construction.

Two rules govern everything here:

RULE 1 - NO LOOK-AHEAD. Every temporal feature is built from months *before*
the prediction month (shift(1) then expanding/rolling). Station climatology is
an expanding mean of prior months, never the full-year mean, because with a
single year of data a full-year mean would embed future information.

RULE 2 - PREDICTORS AND TARGET ARE DISJOINT. The risk label is anchored on
dissolved oxygen with ammonia/nitrate escalation. Those three analytes are
therefore never used as predictors. This is what stops the model from simply
reciting its own label.

TARGET DEFINITION (approved design)
-----------------------------------
Base tier from station-month mean dissolved oxygen against CPCB surface-water
criteria:
    DO >= 6.0 mg/L   -> Low
    4.0 <= DO < 6.0  -> Moderate
    DO < 4.0         -> High
Escalation: one level up (never down) where ammonia-N > 1.0 mgN/L or
nitrate-N > 10.2 mgN/L (the N-equivalent of the BIS 45 mg/L NO3 limit). Both
are established indicators of sewage / organic contamination.

This is a WATER-QUALITY STRESS proxy, not laboratory-confirmed disease risk.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

GRID = 0.5

# Analytes used to build the label. Excluded from predictors.
TARGET_COMPONENTS = ["dissolved_oxygen", "ammonia_n", "nitrate_n"]

# NOTE ON EXCLUDED PARAMETERS
# turbidity, tds, nitrate_n, chloride, alkalinity and iron have ZERO overlap with
# dissolved_oxygen in this export: they are recorded at a disjoint set of
# stations (bore wells, ponds, municipal points) where DO is never measured.
# They therefore cannot predict a DO-anchored label and are excluded. See
# reports/data_quality_report.md for the full co-occurrence table.
NUMERIC_FEATURES = [
    "ph",
    "electrical_conductivity",
    "rainfall_monthly_mm",
    "rainfall_3m_mm",
    "rainfall_anomaly_mm",
    "t2m_c",
    "month_sin",
    "month_cos",
    "latitude",
    "longitude",
    "n_samples_chemical",
    "n_samples_physical",
    "months_since_first",
    "lag1_ph",
    "lag1_ec",
    "delta_ph",
    "delta_ec",
    "rolling3_ec",
    "has_ec",
]

CATEGORICAL_FEATURES = ["state", "season", "monsoon_flag"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# CPCB surface-water dissolved-oxygen criteria (mg/L)
DO_LOW_THRESHOLD = 6.0
DO_HIGH_THRESHOLD = 4.0
# BIS IS 10500 / CPCB contamination indicators
AMMONIA_ESCALATION = 1.0        # mgN/L
NITRATE_ESCALATION = 10.2       # mgN/L as N (== 45 mg/L as NO3)


def attach_environmental(
    station_month: pd.DataFrame, env_daily: pd.DataFrame
) -> pd.DataFrame:
    """Attach monthly rainfall totals and mean air temperature per station.

    Daily reanalysis is aggregated to grid-cell-months, then each station is
    mapped to its 0.5-degree cell. Rainfall is modelled (MERRA-2), not gauged.
    """
    env = env_daily.copy()
    env["year_month"] = env["date"].dt.to_period("M").astype(str)
    monthly = env.groupby(["grid_lat", "grid_lon", "year_month"]).agg(
        rainfall_monthly_mm=("rainfall_mm", "sum"),
        rain_days=("rainfall_mm", lambda s: (s > 1.0).sum()),
        t2m_c=("t2m_c", "mean"),
    ).reset_index()

    sm = station_month.copy()
    sm["grid_lat"] = (sm["latitude"] / GRID).round() * GRID
    sm["grid_lon"] = (sm["longitude"] / GRID).round() * GRID
    sm = sm.merge(
        monthly,
        left_on=["grid_lat", "grid_lon", "year_month"],
        right_on=["grid_lat", "grid_lon", "year_month"],
        how="left",
    )
    return sm.drop(columns=["grid_lat", "grid_lon"])


def add_seasonal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Indian meteorological seasons and the monsoon flag."""
    df = df.copy()
    month = df["month"]

    def season(m: int) -> str:
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "pre_monsoon"
        if m in (6, 7, 8, 9):
            return "monsoon"
        return "post_monsoon"

    df["season"] = month.map(season)
    df["monsoon_flag"] = month.isin([6, 7, 8, 9]).astype(int)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal (prior-months-only) lag, delta, rolling and anomaly features."""
    df = df.sort_values(["station", "year", "month"]).copy()
    g = df.groupby("station", sort=False)

    # lags: previous observed month's value for the same station
    for col, name in (
        ("ph", "lag1_ph"),
        ("electrical_conductivity", "lag1_ec"),
    ):
        df[name] = g[col].shift(1)

    # change since previous month
    df["delta_ph"] = df["ph"] - df["lag1_ph"]
    df["delta_ec"] = df["electrical_conductivity"] - df["lag1_ec"]

    # prior-3-month mean conductivity (shift first, so current month excluded)
    df["rolling3_ec"] = g["electrical_conductivity"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    # 3-month rainfall total, current + previous two months.
    # Rainfall is environmental input known in real time, so including the
    # current month is legitimate (unlike a lab result reported later).
    df["rainfall_3m_mm"] = g["rainfall_monthly_mm"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )

    # deviation from the station's own prior-month climatology (causal)
    df["rain_climatology"] = g["rainfall_monthly_mm"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["rainfall_anomaly_mm"] = df["rainfall_monthly_mm"] - df["rain_climatology"]

    # station history depth
    period = df["year"] * 12 + df["month"]
    df["months_since_first"] = period - g["year_month"].transform(
        lambda s: (df.loc[s.index, "year"] * 12 + df.loc[s.index, "month"]).min()
    )

    # missingness indicator: whether a conductivity reading exists is itself
    # informative, since EC is measured at only ~16% of labelled station-months
    df["has_ec"] = df["electrical_conductivity"].notna().astype(int)
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the three-class risk label (documented thresholds only).

    Returns the frame with `risk_base` (DO tier), `escalated` flag and the final
    `risk_label`. Rows without dissolved oxygen cannot be labelled and are
    dropped, because inventing a label would be indefensible.
    """
    df = df.copy()
    do = df["dissolved_oxygen"]

    df["risk_base"] = np.select(
        [do < DO_HIGH_THRESHOLD, do < DO_LOW_THRESHOLD],
        ["High", "Moderate"],
        default="Low",
    )
    df.loc[do.isna(), "risk_base"] = np.nan

    escalation = (
        (df["ammonia_n"] > AMMONIA_ESCALATION)
        | (df["nitrate_n"] > NITRATE_ESCALATION)
    )
    df["escalated"] = (escalation & df["risk_base"].notna()).astype(int)

    # one level up, never down: Low->Moderate, Moderate->High, High->High
    order = {"Low": 0, "Moderate": 1, "High": 2}
    inv = {v: k for k, v in order.items()}
    base_num = df["risk_base"].map(order)
    df["risk_label"] = (base_num + df["escalated"]).clip(upper=2).map(inv)

    df = df.dropna(subset=["risk_label"]).copy()
    df["risk_label"] = pd.Categorical(df["risk_label"], ["Low", "Moderate", "High"], ordered=True)
    return df


def build_modelling_frame(
    station_month: pd.DataFrame, env_daily: pd.DataFrame
) -> pd.DataFrame:
    """Full feature + target assembly, returning the ML-ready frame."""
    df = attach_environmental(station_month, env_daily)
    df = add_seasonal_flags(df)
    df = add_temporal_features(df)
    df = build_target(df)
    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sm = pd.read_csv(os.path.join(base, "data", "processed", "station_month_clean.csv"))
    env = pd.read_csv(os.path.join(base, "data", "raw", "nasa_power_daily_2021.csv"),
                      parse_dates=["date"])
    frame = build_modelling_frame(sm, env)
    out = os.path.join(base, "data", "processed", "modelling_frame.csv")
    frame.to_csv(out, index=False)
    print(f"modelling rows: {len(frame):,}")
    print(frame["risk_label"].value_counts().to_string())
    print(f"features: {len(ALL_FEATURES)}")
    print(f"saved -> {out}")
