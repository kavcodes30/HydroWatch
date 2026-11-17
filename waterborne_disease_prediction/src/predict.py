"""
Phase 10 - Reusable prediction interface.

The saved pipeline already contains the fitted preprocessing (imputation,
scaling, one-hot encoding), so calling this function applies EXACTLY the same
transformations used in training. Nothing is re-derived by hand.

Usage
-----
    from predict import RiskPredictor

    predictor = RiskPredictor("models")
    result = predictor.predict_one({
        "state": "Maharashtra",
        "district": "Pune",
        "latitude": 18.52,
        "longitude": 73.85,
        "month": 7,
        "ph": 7.2,
        "electrical_conductivity": 640.0,
        "rainfall_monthly_mm": 210.0,
        "t2m_c": 27.5,
    })
    print(result["risk_label"], result["probabilities"])

Fields that are absent are passed as missing and imputed by the pipeline using
training-set medians. Temporal features (lag1_ph, rainfall_anomaly_mm, ...) need
history; if you do not supply them they are imputed, and the result is flagged
as "reduced_context" so a caller can tell the difference.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from feature_engineering import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from train_model import CLASS_ORDER

DISCLAIMER = (
    "Decision-support indicator of potential water-quality stress consistent "
    "with contamination risk. Not a diagnosis, not a confirmed outbreak, and "
    "not laboratory confirmation of contamination."
)

SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
}

HISTORY_FIELDS = ["lag1_ph", "lag1_ec", "rolling3_ec", "delta_ph", "delta_ec",
                  "rainfall_3m_mm", "rainfall_anomaly_mm", "months_since_first",
                  "n_samples_chemical", "n_samples_physical"]


class RiskPredictor:
    """Loads the trained artefacts and serves risk predictions."""

    def __init__(self, model_dir: str = "models", model_filename: str | None = None):
        # Prefer the regularised deployment model (less overfitting, higher
        # High-Risk recall); fall back to the CV-selected model if absent.
        if model_filename is None:
            deployment = os.path.join(model_dir, "deployment_model.pkl")
            model_filename = ("deployment_model.pkl" if os.path.exists(deployment)
                              else "final_model.pkl")
        self.model_filename = model_filename
        self.model = joblib.load(os.path.join(model_dir, model_filename))
        with open(os.path.join(model_dir, "model_metadata.json")) as fh:
            self.metadata = json.load(fh)
        self.high_threshold = 0.50

    # -- feature assembly ---------------------------------------------------
    def _prepare(self, records: list[dict]) -> tuple[pd.DataFrame, list[bool]]:
        rows, reduced = [], []
        for rec in records:
            month = int(rec.get("month", 1))
            row = {f: np.nan for f in NUMERIC_FEATURES}
            for f in NUMERIC_FEATURES:
                if f in rec and rec[f] is not None:
                    row[f] = rec[f]

            # derive cyclical month encoding and season if not supplied
            row["month_sin"] = np.sin(2 * np.pi * month / 12)
            row["month_cos"] = np.cos(2 * np.pi * month / 12)
            row.setdefault("latitude", rec.get("latitude", np.nan))
            row["latitude"] = rec.get("latitude", np.nan)
            row["longitude"] = rec.get("longitude", np.nan)
            row["n_samples_physical"] = rec.get("n_samples_physical", 0.0)
            row["n_samples_chemical"] = rec.get("n_samples_chemical", 1.0)
            row["has_ec"] = 0.0 if pd.isna(rec.get("electrical_conductivity")) else 1.0

            row["state"] = rec.get("state", "missing")
            row["season"] = rec.get("season") or SEASON_BY_MONTH.get(month, "missing")
            row["monsoon_flag"] = int(month in (6, 7, 8, 9))

            missing_history = [f for f in HISTORY_FIELDS if pd.isna(row.get(f))]
            reduced.append(len(missing_history) > 4)
            rows.append(row)

        return pd.DataFrame(rows, columns=ALL_FEATURES), reduced

    # -- prediction ---------------------------------------------------------
    def predict(self, records: list[dict], top_n: int = 3) -> list[dict]:
        X, reduced = self._prepare(records)
        proba = self.model.predict_proba(X)
        codes = self.model.predict(X)
        decode = dict(enumerate(CLASS_ORDER))

        import shap
        prep = self.model.named_steps["prep"]
        names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
        Xt = prep.transform(X)
        explainer = shap.TreeExplainer(self.model.named_steps["model"])
        sv = explainer.shap_values(Xt)
        high_idx = CLASS_ORDER.index("High")
        sv_high = sv[high_idx] if isinstance(sv, list) else (
            sv[:, :, high_idx] if np.ndim(sv) == 3 else sv)

        out = []
        for i, rec in enumerate(records):
            label = decode[int(codes[i])]
            probs = {c: round(float(proba[i, j]), 4) for j, c in enumerate(CLASS_ORDER)}
            contrib = sv_high[i]
            order = np.argsort(np.abs(contrib))[::-1][:top_n]

            if label == "High" or probs["High"] >= self.high_threshold:
                alert = "alert"
            elif label == "Moderate" or probs["Moderate"] >= 0.40:
                alert = "watch"
            else:
                alert = "normal"

            out.append({
                "station": rec.get("station", "unknown"),
                "district": rec.get("district"),
                "state": rec.get("state"),
                "period": rec.get("year_month"),
                "risk_label": label,
                "probabilities": probs,
                "alert_status": alert,
                "top_contributing_features": [
                    {"feature": names[j], "shap_contribution": round(float(contrib[j]), 4)}
                    for j in order
                ],
                "context_quality": "reduced_context" if reduced[i] else "full_context",
                "probability_caveat": (
                    "Probabilities are model scores, not calibrated real-world "
                    "probabilities; calibration was assessed but not guaranteed."
                ),
                "disclaimer": DISCLAIMER,
            })
        return out

    def predict_one(self, record: dict) -> dict:
        return self.predict([record])[0]


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    predictor = RiskPredictor(os.path.join(os.path.dirname(base), "models"))
    demo = [
        {"station": "DEMO-STATION-A", "district": "Pune", "state": "Maharashtra",
         "latitude": 18.52, "longitude": 73.85, "month": 7, "year_month": "2021-07",
         "ph": 7.1, "electrical_conductivity": 520.0,
         "rainfall_monthly_mm": 240.0, "t2m_c": 27.0},
        {"station": "DEMO-STATION-B", "district": "Patna", "state": "Bihar",
         "latitude": 25.6, "longitude": 85.1, "month": 8, "year_month": "2021-08",
         "ph": 8.9, "electrical_conductivity": 1450.0,
         "rainfall_monthly_mm": 310.0, "t2m_c": 29.5},
    ]
    for result in predictor.predict(demo):
        print(json.dumps(result, indent=2))
