"""
Phase 5 - Exploratory data analysis.

Produces the figures and the numeric findings quoted in reports/eda_report.md.
Correlation is treated as association only; nothing here is causal evidence.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feature_engineering import ALL_FEATURES, NUMERIC_FEATURES
from train_model import CLASS_ORDER, load_frame

sns.set_theme(style="whitegrid", context="talk")
RISK_ORDER = CLASS_ORDER
RISK_COLOURS = {"Low": "#2E7D32", "Moderate": "#EF6C00", "High": "#C62828"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(frame: pd.DataFrame, figures_dir: str, reports_dir: str) -> dict:
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    findings = {}

    # 1. class distribution ------------------------------------------------
    counts = frame["risk_label"].value_counts().reindex(RISK_ORDER)
    findings["class_counts"] = counts.to_dict()
    findings["class_share"] = (counts / counts.sum()).round(4).to_dict()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=counts.index, y=counts.values,
                hue=counts.index, palette=RISK_COLOURS, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, f"{v:,}\n({v/counts.sum():.1%})", ha="center", fontsize=12)
    ax.set_title("Risk class distribution (7,687 station-months, 2021)")
    ax.set_ylabel("station-months")
    ax.set_ylim(0, counts.max() * 1.2)
    _save(fig, os.path.join(figures_dir, "01_class_distribution.png"))

    # 2. missing values ----------------------------------------------------
    miss = frame[ALL_FEATURES].isna().mean().sort_values(ascending=False)
    findings["feature_missing_share"] = miss.round(4).to_dict()

    fig, ax = plt.subplots(figsize=(8, 8))
    miss.plot(kind="barh", ax=ax, color="#1565C0")
    ax.set_xlabel("share missing")
    ax.set_title("Missingness by model feature")
    ax.invert_yaxis()
    _save(fig, os.path.join(figures_dir, "02_missingness.png"))

    # 3. key distributions -------------------------------------------------
    dist_cols = ["ph", "electrical_conductivity", "rainfall_monthly_mm",
                 "dissolved_oxygen", "ammonia_n"]
    fig, axes = plt.subplots(1, len(dist_cols), figsize=(22, 4.2))
    for ax, col in zip(axes, dist_cols):
        data = frame[col].dropna()
        if col == "electrical_conductivity":
            data = data[data < data.quantile(0.99)]
        ax.hist(data, bins=40, color="#1565C0", alpha=0.85)
        ax.set_title(col, fontsize=13)
        ax.set_yscale("log")
    _save(fig, os.path.join(figures_dir, "03_distributions.png"))

    # 4. correlation -------------------------------------------------------
    corr_cols = [c for c in NUMERIC_FEATURES if frame[c].notna().sum() > 500]
    corr_cols += ["dissolved_oxygen", "ammonia_n"]
    corr = frame[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.7}, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Correlation among model features and target components\n"
                 "(association only, not causation)")
    _save(fig, os.path.join(figures_dir, "04_correlation.png"))

    do_corr = corr["dissolved_oxygen"].drop("dissolved_oxygen").sort_values()
    findings["correlation_with_do"] = do_corr.round(3).to_dict()

    # 5. state-wise risk ---------------------------------------------------
    state_risk = pd.crosstab(frame["state"], frame["risk_label"], normalize="index")
    state_risk = state_risk.reindex(columns=RISK_ORDER).fillna(0)
    findings["state_high_share"] = state_risk["High"].round(4).to_dict()

    fig, ax = plt.subplots(figsize=(10, 5))
    state_risk.plot(kind="barh", stacked=True, ax=ax,
                    color=[RISK_COLOURS[c] for c in RISK_ORDER])
    ax.set_xlabel("share of station-months")
    ax.set_title("Risk composition by state")
    ax.legend(title="risk", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, os.path.join(figures_dir, "05_state_risk.png"))

    # 6. seasonal pattern --------------------------------------------------
    monthly = frame.groupby("month").agg(
        mean_do=("dissolved_oxygen", "mean"),
        mean_rain=("rainfall_monthly_mm", "mean"),
        high_share=("risk_label", lambda s: (s == "High").mean()),
    )
    findings["monthly"] = monthly.round(3).to_dict()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(monthly.index, monthly["mean_rain"], color="#90CAF9",
            label="mean monthly rainfall (mm)")
    ax1.set_xlabel("month (2021)")
    ax1.set_ylabel("rainfall (mm)", color="#1565C0")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["mean_do"], color="#C62828", marker="o",
             linewidth=2.5, label="mean dissolved oxygen")
    ax2.set_ylabel("dissolved oxygen (mg/L)", color="#C62828")
    ax1.set_title("Rainfall and dissolved oxygen by month")
    _save(fig, os.path.join(figures_dir, "06_seasonal.png"))

    # 7. rainfall vs risk --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(data=frame, x="risk_label", y="rainfall_monthly_mm",
                order=RISK_ORDER, hue="risk_label", palette=RISK_COLOURS,
                legend=False, ax=ax, showfliers=False)
    ax.set_title("Monthly rainfall by risk class")
    ax.set_ylabel("rainfall (mm)")
    _save(fig, os.path.join(figures_dir, "07_rainfall_risk.png"))

    # 8. pH vs risk --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.violinplot(data=frame, x="risk_label", y="ph", order=RISK_ORDER,
                   hue="risk_label", palette=RISK_COLOURS, legend=False,
                   ax=ax, cut=0)
    ax.axhline(6.5, ls="--", c="k", lw=1)
    ax.axhline(8.5, ls="--", c="k", lw=1)
    ax.set_title("pH by risk class (dashed = BIS acceptable band 6.5-8.5)")
    _save(fig, os.path.join(figures_dir, "08_ph_risk.png"))

    # 9. district coverage -------------------------------------------------
    per_district = frame.groupby("district")["station"].nunique().sort_values(ascending=False)
    findings["district_station_counts"] = {
        "n_districts": int(frame["district"].nunique()),
        "median_stations_per_district": float(per_district.median()),
        "max": int(per_district.max()),
        "min": int(per_district.min()),
    }

    pd.Series(findings["class_counts"]).to_csv(
        os.path.join(reports_dir, "eda_class_counts.csv"))
    print("EDA figures written to", figures_dir)
    print("class counts:", findings["class_counts"])
    print("state High share:", findings["state_high_share"])
    return findings


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame = load_frame(os.path.join(base, "data", "processed", "modelling_frame.csv"))
    run_eda(frame, os.path.join(base, "figures"), os.path.join(base, "reports"))
