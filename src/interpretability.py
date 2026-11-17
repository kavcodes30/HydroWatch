"""
Phase 8 - Interpretability.

Three complementary views:
1. Impurity-based feature importance from the Random Forest (fast, biased toward
   high-cardinality features - reported with that caveat).
2. Permutation importance on the HOLDOUT, measured as the drop in macro F1 when
   a raw input column is shuffled. This is model-agnostic and less biased.
3. SHAP values (TreeExplainer) for the High-Risk class, giving per-prediction
   contributions that the dashboard can display.
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
import shap
from sklearn.inspection import permutation_importance

from feature_engineering import ALL_FEATURES
from train_model import CLASS_ORDER, load_frame


def run_interpretability(frame, models_dir, figures_dir, reports_dir) -> dict:
    model = joblib.load(os.path.join(models_dir, "final_model.pkl"))
    prep = model.named_steps["prep"]
    rf = model.named_steps["model"]

    hold = frame[frame["month"] > 9].copy()
    X = hold[ALL_FEATURES]
    # the estimator predicts integer codes, so scoring must use encoded labels
    y = hold["risk_label"].astype(str).map({c: i for i, c in enumerate(CLASS_ORDER)})

    os.makedirs(figures_dir, exist_ok=True)

    # ---- 1. impurity importance on transformed feature names -------------
    names = prep.get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    impurity = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    impurity.head(15)[::-1].plot(kind="barh", ax=ax, color="#1565C0")
    ax.set_title("Random Forest impurity importance (top 15)")
    ax.set_xlabel("mean decrease in impurity")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "09_feature_importance.png"), dpi=150)
    plt.close(fig)

    # ---- 2. permutation importance on raw columns, scored by macro F1 ----
    perm = permutation_importance(
        model, X, y, scoring="f1_macro", n_repeats=5,
        random_state=42, n_jobs=-1,
    )
    perm_df = (pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False))
    perm_df.to_csv(os.path.join(reports_dir, "permutation_importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    top = perm_df.head(15)[::-1]
    ax.barh(top["feature"], top["importance_mean"],
            xerr=top["importance_std"], color="#2E7D32")
    ax.set_title("Permutation importance on holdout (drop in macro F1)")
    ax.set_xlabel("mean decrease in macro F1 when shuffled")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "10_permutation_importance.png"), dpi=150)
    plt.close(fig)

    # ---- 3. SHAP for the High-Risk class --------------------------------
    Xt = prep.transform(X)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(Xt)
    high_idx = CLASS_ORDER.index("High")
    if isinstance(sv, list):
        sv_high = sv[high_idx]
    elif sv.ndim == 3:  # (n_samples, n_features, n_classes)
        sv_high = sv[:, :, high_idx]
    else:
        sv_high = sv

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_high, Xt, feature_names=names, show=False, max_display=15)
    plt.title("SHAP summary - contribution to High-Risk score")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "11_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    mean_abs = pd.Series(np.abs(sv_high).mean(axis=0), index=names).sort_values(ascending=False)
    mean_abs.to_csv(os.path.join(reports_dir, "shap_mean_abs_high_risk.csv"))

    summary = {
        "top_impurity_features": impurity.head(10).round(4).to_dict(),
        "top_permutation_features": perm_df.head(10)[
            ["feature", "importance_mean"]].round(4).to_dict("records"),
        "top_shap_features_high_risk": mean_abs.head(10).round(4).to_dict(),
    }
    with open(os.path.join(reports_dir, "interpretability_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Top permutation features (holdout, macro F1):")
    print(perm_df.head(8).to_string(index=False))
    return summary


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame = load_frame(os.path.join(base, "data", "processed", "modelling_frame.csv"))
    run_interpretability(frame,
                         os.path.join(base, "models"),
                         os.path.join(base, "figures"),
                         os.path.join(base, "reports"))
