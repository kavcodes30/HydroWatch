"""
Phase 9 - Early-warning component.

This is deliberately a SEPARATE layer from the classifier. The model outputs a
risk class and probabilities; this module decides whether that constitutes an
alert, attaches the parameter-level evidence, and states what a human should do
next.

HARD RULES BAKED INTO THIS MODULE
---------------------------------
* An alert never asserts an outbreak, contamination event or diagnosis. The
  wording is always "potential risk requiring investigation".
* Alert severity requires BOTH a statistical signal (model class/probability)
  AND, where data exists, a documented parameter exceedance. A statistically
  unusual value that is still within BIS limits is reported as "watch", not
  "alert".
* The synthetic symptom stream exists ONLY to exercise the dashboard end-to-end.
  It is labelled synthetic in every row it produces and is never used to train
  or evaluate the model.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from feature_engineering import ALL_FEATURES, TARGET_COMPONENTS
from train_model import CLASS_ORDER, load_frame

# ---- documented parameter limits -----------------------------------------
# BIS IS 10500:2012 drinking-water acceptable limits; CPCB surface-water
# criteria for dissolved oxygen. Used only for the "is this outside a published
# limit" test, never as a disease threshold.
PARAMETER_LIMITS = {
    "ph": {"low": 6.5, "high": 8.5, "unit": "pH units",
           "source": "BIS IS 10500 acceptable limit"},
    "electrical_conductivity": {"high": 1500.0, "unit": "uS/cm",
                                "source": "indicative salinity/ionic load limit"},
    "ammonia_n": {"high": 0.5, "unit": "mgN/L",
                  "source": "BIS IS 10500 acceptable limit"},
    "dissolved_oxygen": {"low": 5.0, "unit": "mg/L",
                         "source": "CPCB Class B surface-water criterion"},
}

ALERT_CONFIG = {
    "high_probability_threshold": 0.50,   # P(High) needed to raise an alert
    "moderate_probability_threshold": 0.40,
    "require_parameter_exceedance_for_alert": False,  # see docstring
}

DISCLAIMER = (
    "This is a decision-support indicator of potential water-quality stress "
    "consistent with contamination risk. It is not a diagnosis, not a "
    "confirmed outbreak, and not laboratory confirmation of bacterial or viral "
    "contamination. Verify with repeat sampling and public-health review."
)

RECOMMENDED_ACTIONS = [
    "Repeat water-quality sampling at the same station within 48 hours",
    "Forward the station and period to the district public-health officer for review",
    "Check for reported diarrhoeal illness clusters in the same district and week",
    "Inspect the water source and immediate catchment for visible contamination",
    "If the station supplies drinking water, advise precautionary treatment or an alternative source",
]


def parameter_breaches(row: pd.Series) -> list[dict]:
    """Documented limit exceedances for a single station-month."""
    breaches = []
    for param, spec in PARAMETER_LIMITS.items():
        if param not in row or pd.isna(row[param]):
            continue
        value = float(row[param])
        if "high" in spec and value > spec["high"]:
            breaches.append({
                "parameter": param, "value": value, "limit": spec["high"],
                "direction": "above", "unit": spec["unit"],
                "source": spec["source"],
            })
        if "low" in spec and value < spec["low"]:
            breaches.append({
                "parameter": param, "value": value, "limit": spec["low"],
                "direction": "below", "unit": spec["unit"],
                "source": spec["source"],
            })
    return breaches


def explain_prediction(model, prep, row_features: pd.DataFrame,
                       feature_names: list[str], top_n: int = 3) -> list[dict]:
    """Top SHAP contributions toward the High-Risk class for one prediction."""
    import shap

    Xt = prep.transform(row_features)
    explainer = shap.TreeExplainer(model.named_steps["model"])
    sv = explainer.shap_values(Xt)
    high_idx = CLASS_ORDER.index("High")
    if isinstance(sv, list):
        sv_high = sv[high_idx]
    elif np.ndim(sv) == 3:
        sv_high = sv[:, :, high_idx]
    else:
        sv_high = sv

    contrib = sv_high[0]
    order = np.argsort(np.abs(contrib))[::-1][:top_n]
    return [
        {"feature": feature_names[i], "contribution": float(contrib[i])}
        for i in order
    ]


def generate_synthetic_symptoms(frame: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Create a clearly-labelled SYNTHETIC aggregated symptom stream.

    Purpose: the synopsis requires a community-health pillar and a dashboard
    that displays symptom trends. No real health data was supplied, and the
    synopsis explicitly permits simulated data for the prototype. This stream is
    therefore generated, tagged `data_source='synthetic'`, and used only for the
    dashboard demonstration. It is NEVER used as a model feature or to compute
    any reported performance metric.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (state, district, month), grp in frame.groupby(["state", "district", "month"]):
        # small, deliberately weak association with the model's risk class so the
        # dashboard demo looks plausible; this is a visual aid, not evidence.
        high_share = (grp["risk_label"] == "High").mean()
        base = 12.0 + 40.0 * high_share
        reported = int(max(0, rng.poisson(base)))
        rows.append({
            "state": state,
            "district": district,
            "month": int(month),
            "symptom_reports": reported,
            "reporting_facilities": int(rng.integers(3, 25)),
            "data_source": "synthetic",
            "note": "SIMULATED for dashboard demonstration only - not real health data",
        })
    return pd.DataFrame(rows)


def evaluate_alerts(frame: pd.DataFrame, models_dir: str, reports_dir: str) -> pd.DataFrame:
    """Score the holdout period and emit alert records."""
    model = joblib.load(os.path.join(models_dir, "final_model.pkl"))
    prep = model.named_steps["prep"]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]

    hold = frame[frame["month"] > 9].copy()
    X = hold[ALL_FEATURES]
    proba = model.predict_proba(X)
    high_idx = CLASS_ORDER.index("High")
    pred = pd.Series(model.predict(X)).map(dict(enumerate(CLASS_ORDER))).values

    hold = hold.assign(
        predicted_risk=pred,
        p_high=proba[:, high_idx],
        p_moderate=proba[:, CLASS_ORDER.index("Moderate")],
    )

    records = []
    for idx, row in hold.iterrows():
        breaches = parameter_breaches(row)
        model_flag = (row["predicted_risk"] == "High"
                      or row["p_high"] >= ALERT_CONFIG["high_probability_threshold"])

        if not model_flag and not breaches:
            continue

        if model_flag and breaches:
            severity = "alert"
        elif model_flag:
            severity = "alert"          # model-driven, no measurement exceedance
        else:
            severity = "watch"          # measurement exceedance, model not concerned

        explanation = explain_prediction(
            model, prep, X.loc[[idx]], names, top_n=3
        )
        records.append({
            "station": row["station"],
            "district": row["district"],
            "state": row["state"],
            "period": row["year_month"],
            "predicted_risk": row["predicted_risk"],
            "p_high": round(float(row["p_high"]), 4),
            "p_moderate": round(float(row["p_moderate"]), 4),
            "severity": severity,
            "parameter_breaches": "; ".join(
                f"{b['parameter']} {b['value']:.2f} {b['direction']} {b['limit']} {b['unit']}"
                for b in breaches
            ) or "none recorded",
            "top_factors": "; ".join(
                f"{f['feature']} ({f['contribution']:+.3f})" for f in explanation
            ),
            "recommended_action": RECOMMENDED_ACTIONS[0],
            "disclaimer": DISCLAIMER,
        })

    alerts = pd.DataFrame(records).sort_values(
        ["severity", "p_high"], ascending=[True, False]
    )
    os.makedirs(reports_dir, exist_ok=True)
    alerts.to_csv(os.path.join(reports_dir, "alerts_holdout.csv"), index=False)

    symptoms = generate_synthetic_symptoms(frame)
    symptoms.to_csv(os.path.join(reports_dir, "synthetic_symptom_stream.csv"), index=False)

    summary = {
        "rows_scored": int(len(hold)),
        "alerts": int((alerts["severity"] == "alert").sum()),
        "watch": int((alerts["severity"] == "watch").sum()),
        "alert_rate": round(float((alerts["severity"] == "alert").mean()), 4)
                       if len(alerts) else 0.0,
        "config": ALERT_CONFIG,
        "parameter_limits": PARAMETER_LIMITS,
        "synthetic_symptom_rows": int(len(symptoms)),
        "disclaimer": DISCLAIMER,
    }
    with open(os.path.join(reports_dir, "alert_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    if len(alerts):
        print("\nExample alert:")
        print(alerts.iloc[0][["station", "district", "period", "predicted_risk",
                              "p_high", "parameter_breaches", "top_factors"]].to_string())
    return alerts


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame = load_frame(os.path.join(base, "data", "processed", "modelling_frame.csv"))
    evaluate_alerts(frame, os.path.join(base, "models"), os.path.join(base, "reports"))
