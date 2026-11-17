"""
Phase 7 - Evaluation on the untouched holdout, plus a leakage audit.

The holdout (months 10-12) was never seen during model selection or tuning.
It is scored exactly once, here. Reported metrics include accuracy for
comparability with the project brief, but macro F1 and High-Risk recall are the
metrics that actually describe whether the system is useful, because 71% of
observations are Low Risk.
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from feature_engineering import ALL_FEATURES, TARGET_COMPONENTS
from train_model import CLASS_ORDER, load_frame

RISK_COLOURS = {"Low": "#2E7D32", "Moderate": "#EF6C00", "High": "#C62828"}


def evaluate_holdout(frame: pd.DataFrame, models_dir: str, reports_dir: str,
                     figures_dir: str) -> dict:
    model = joblib.load(os.path.join(models_dir, "final_model.pkl"))
    meta = json.load(open(os.path.join(models_dir, "model_metadata.json")))

    dev = frame[frame["month"] <= 9]
    hold = frame[frame["month"] > 9]

    X_dev, y_dev = dev[ALL_FEATURES], dev["risk_label"].astype(str)
    X_hold, y_hold = hold[ALL_FEATURES], hold["risk_label"].astype(str)

    # the estimator predicts integer codes; decode to class names once
    decode = dict(enumerate(CLASS_ORDER))

    def to_labels(preds):
        return pd.Series(preds).map(decode).to_numpy()

    # training accuracy on the data the model was fitted to (overfitting check)
    train_acc = accuracy_score(y_dev, to_labels(model.predict(X_dev)))

    y_pred = to_labels(model.predict(X_hold))

    proba = model.predict_proba(X_hold)

    acc = accuracy_score(y_hold, y_pred)
    macro_f1 = f1_score(y_hold, y_pred, average="macro")
    weighted_f1 = f1_score(y_hold, y_pred, average="weighted")

    per_class = {}
    for cls in CLASS_ORDER:
        per_class[cls] = {
            "precision": float(precision_score(y_hold, y_pred, labels=[cls],
                                               average="micro", zero_division=0)),
            "recall": float(recall_score(y_hold, y_pred, labels=[cls],
                                         average="micro", zero_division=0)),
            "f1": float(f1_score(y_hold, y_pred, labels=[cls],
                                 average="micro", zero_division=0)),
            "support": int((y_hold == cls).sum()),
        }

    cm = confusion_matrix(y_hold, y_pred, labels=CLASS_ORDER)

    # --- calibration check -------------------------------------------------
    # Probabilities are only meaningful as confidence if calibrated; report
    # Brier score per class and a reliability figure rather than asserting it.
    class_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    brier = {}
    for cls in CLASS_ORDER:
        y_bin = (y_hold == cls).astype(int)
        brier[cls] = float(brier_score_loss(y_bin, proba[:, class_idx[cls]]))

    # --- leakage audit -----------------------------------------------------
    overlap = set(X_dev.index) & set(X_hold.index)
    target_in_features = [c for c in TARGET_COMPONENTS if c in ALL_FEATURES]
    audit = {
        "target_components_in_feature_list": target_in_features,
        "row_overlap_train_test": len(overlap),
        "stations_in_both_splits": len(set(dev["station"]) & set(hold["station"])),
        "note": ("Time-based split shares stations across splits by design; "
                 "GroupKFold by station was used for model selection so that "
                 "within-development scores do not benefit from station overlap."),
        "duplicate_feature_rows_train_test": int(
            pd.merge(X_dev.assign(_y=y_dev), X_hold, on=ALL_FEATURES, how="inner").shape[0]
        ),
    }

    results = {
        "model": meta["selected_model"],
        "holdout_rows": int(len(hold)),
        "train_accuracy_on_dev": float(train_acc),
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "high_risk_precision": per_class["High"]["precision"],
        "high_risk_recall": per_class["High"]["recall"],
        "brier_scores": brier,
        "confusion_matrix": cm.tolist(),
        "leakage_audit": audit,
        "overfitting_gap": float(train_acc - acc),
    }

    # --- figures -----------------------------------------------------------
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Actual risk")
    ax.set_title(f"Holdout confusion matrix - {meta['selected_model']}\n"
                 f"Oct-Dec 2021, n={len(hold):,}")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # per-class metrics
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics_df = pd.DataFrame(per_class).T[["precision", "recall", "f1"]]
    metrics_df.plot(kind="bar", ax=ax,
                    color=["#1565C0", "#EF6C00", "#2E7D32"])
    ax.set_ylim(0, 1)
    ax.set_title("Per-class precision, recall and F1 (holdout)")
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "per_class_metrics.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(reports_dir, "evaluation_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"Holdout accuracy      : {acc:.4f}")
    print(f"Macro F1              : {macro_f1:.4f}")
    print(f"Weighted F1           : {weighted_f1:.4f}")
    print(f"High-Risk precision   : {per_class['High']['precision']:.4f}")
    print(f"High-Risk recall      : {per_class['High']['recall']:.4f}")
    print(f"Train accuracy (dev)  : {train_acc:.4f}")
    print(f"Overfitting gap       : {train_acc - acc:+.4f}")
    print("\n" + classification_report(y_hold, y_pred, labels=CLASS_ORDER, zero_division=0))
    print("Leakage audit:", json.dumps(audit, indent=2))
    return results


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame = load_frame(os.path.join(base, "data", "processed", "modelling_frame.csv"))
    evaluate_holdout(frame,
                     os.path.join(base, "models"),
                     os.path.join(base, "reports"),
                     os.path.join(base, "figures"))
