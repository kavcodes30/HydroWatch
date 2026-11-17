"""
Spatial validation - leave-one-state-out.

Why this exists
---------------
The main holdout (Oct-Dec 2021) measures performance on new TIME PERIODS at
stations the model has already seen. Real deployment is the opposite: the model
meets new LOCATIONS. This module closes that gap by training on six states and
testing on the seventh, repeated for every state.

This is a deliberately harsh test. States differ enormously in base rate
(Punjab 24.3% High versus Madhya Pradesh 0.8%), so a model trained elsewhere has
no way to learn a region's prior. The results quantify how much of the headline
performance depends on location familiarity - which is exactly the question a
reviewer should ask.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score)

from feature_engineering import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from train_model import CLASS_ORDER, build_pipeline, load_frame, BalancedXGBClassifier
from sklearn.ensemble import RandomForestClassifier


def _selected_params(models_dir: str) -> dict:
    """Reuse the tuned hyperparameters of the selected model."""
    with open(os.path.join(models_dir, "model_metadata.json")) as fh:
        meta = json.load(fh)
    slug = meta["selected_model"].lower().replace(" ", "_")
    with open(os.path.join(models_dir, f"cv_{slug}.json")) as fh:
        return json.load(fh)["best_params"]


LOCATION_FEATURES = ["state", "latitude", "longitude"]


def leave_one_state_out(frame: pd.DataFrame, models_dir: str, reports_dir: str,
                        ablate_location: bool = False) -> dict:
    """Train on six states, test on the seventh, for every state.

    With ablate_location=True the location features (state, latitude, longitude)
    are removed, which tests whether the model's dependence on geography is what
    destroys spatial generalisation.
    """
    params = _selected_params(models_dir)
    features = ([f for f in ALL_FEATURES if f not in LOCATION_FEATURES]
                if ablate_location else ALL_FEATURES)
    tag = "no_location" if ablate_location else "with_location"
    # strip the pipeline prefix: params look like {'model__max_depth': ...}
    rf_kwargs = {k.replace("model__", ""): v for k, v in params.items()}

    class_index = {c: i for i, c in enumerate(CLASS_ORDER)}
    decode = dict(enumerate(CLASS_ORDER))
    y_all = frame["risk_label"].astype(str).map(class_index)

    rows, pooled_true, pooled_pred = [], [], []

    for state in sorted(frame["state"].unique()):
        train_mask = frame["state"] != state
        test_mask = frame["state"] == state

        X_tr, y_tr = frame.loc[train_mask, features], y_all[train_mask]
        X_te, y_te = frame.loc[test_mask, features], y_all[test_mask]

        num = [f for f in NUMERIC_FEATURES if f in features]
        cat = [f for f in CATEGORICAL_FEATURES if f in features]
        model = build_pipeline(RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42,
            n_jobs=-1, **rf_kwargs), numeric_features=num, categorical_features=cat)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        pooled_true.extend(y_te.tolist())
        pooled_pred.extend(pred.tolist())

        rows.append({
            "state": state,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "test_high_share": round(float((y_te == class_index["High"]).mean()), 4),
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "macro_f1": round(float(f1_score(y_te, pred, average="macro",
                                             labels=[0, 1, 2], zero_division=0)), 4),
            "high_recall": round(float(recall_score(
                y_te, pred, labels=[class_index["High"]],
                average="micro", zero_division=0)), 4),
        })
        print(f"  {state:18s} n={int(test_mask.sum()):5d}  "
              f"acc={rows[-1]['accuracy']:.3f}  macroF1={rows[-1]['macro_f1']:.3f}  "
              f"high_recall={rows[-1]['high_recall']:.3f}")

    detail = pd.DataFrame(rows)
    detail.to_csv(os.path.join(reports_dir, f"spatial_validation_{tag}.csv"), index=False)

    summary = {
        "scheme": "leave-one-state-out (train on 6 states, test on the 7th)",
        "features": "location ablated" if ablate_location else "all features",
        "n_features": len(features),
        "mean_accuracy": round(float(detail["accuracy"].mean()), 4),
        "mean_macro_f1": round(float(detail["macro_f1"].mean()), 4),
        "mean_high_recall": round(float(detail["high_recall"].mean()), 4),
        "pooled_accuracy": round(float(accuracy_score(pooled_true, pooled_pred)), 4),
        "pooled_macro_f1": round(float(f1_score(pooled_true, pooled_pred,
                                                average="macro")), 4),
        "worst_state_macro_f1": detail.loc[detail["macro_f1"].idxmin(), "state"],
        "best_state_macro_f1": detail.loc[detail["macro_f1"].idxmax(), "state"],
        "per_state": rows,
    }
    with open(os.path.join(reports_dir, f"spatial_validation_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[{tag}] Mean across states: acc {summary['mean_accuracy']:.4f}, "
          f"macro F1 {summary['mean_macro_f1']:.4f}, "
          f"high recall {summary['mean_high_recall']:.4f}")
    print(f"Pooled (all states): acc {summary['pooled_accuracy']:.4f}, "
          f"macro F1 {summary['pooled_macro_f1']:.4f}")
    return summary


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame = load_frame(os.path.join(base, "data", "processed", "modelling_frame.csv"))
    import sys
    ablate = "--ablate-location" in sys.argv
    leave_one_state_out(frame,
                        os.path.join(base, "models"),
                        os.path.join(base, "reports"),
                        ablate_location=ablate)
