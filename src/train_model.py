"""
Phase 6 - Model development, comparison and selection.

VALIDATION PROTOCOL (why it is built this way)
---------------------------------------------
* The year 2021 is split by TIME: months 1-9 are the development set, months
  10-12 are a holdout that is never touched during model selection or tuning.
* Within the development set, cross-validation is GROUPED BY STATION
  (GroupKFold). A station appearing in both train and validation folds would let
  the model memorise station identity and inflate scores.
* Hyperparameters are tuned on development-set CV only. The holdout is scored
  exactly once, at the end, by evaluate_model.py.
* Class imbalance is handled with balanced class weights. A SMOTE variant is
  also evaluated; SMOTE is applied strictly inside training folds via an
  imbalanced-learn Pipeline, never before splitting.
* Model selection uses MACRO F1, not accuracy, because 71% of rows are Low and
  accuracy alone rewards a model that ignores the High class.
"""

from __future__ import annotations

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from feature_engineering import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
CLASS_ORDER = ["Low", "Moderate", "High"]


class BalancedXGBClassifier(XGBClassifier):
    """XGBoost with balanced sample weights computed at fit time.

    XGBClassifier has no class_weight argument, so balanced weights are applied
    via sample_weight. Subclassing keeps the estimator fully cloneable, which
    RandomizedSearchCV requires.
    """

    def fit(self, X, y, **kwargs):
        weights = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=weights, **kwargs)


def make_preprocessor(numeric_features=None, categorical_features=None) -> ColumnTransformer:
    """Learned preprocessing, fitted on training data only.

    Median imputation for numerics (robust to the skewed water-quality
    distributions), most-frequent plus an explicit 'missing' category for
    categoricals, and scaling for the linear baseline. Trees do not need
    scaling, but applying it uniformly keeps one reproducible pipeline.
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, categorical_features),
    ])


def candidate_models() -> dict:
    """The four required models plus their search spaces."""
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "search": GridSearchCV,
            "params": {"model__C": [0.1, 1.0, 10.0]},
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "search": GridSearchCV,
            "params": {
                "model__max_depth": [4, 6, 8, None],
                "model__min_samples_leaf": [1, 5, 20],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "search": RandomizedSearchCV,
            "params": {
                "model__max_depth": [10, 20, None],
                "model__min_samples_leaf": [1, 5],
                "model__max_features": ["sqrt", 0.5],
            },
        },
        "XGBoost": {
            "estimator": BalancedXGBClassifier(
                n_estimators=200, learning_rate=0.1, subsample=0.9,
                colsample_bytree=0.9, random_state=RANDOM_STATE,
                eval_metric="mlogloss", n_jobs=-1,
            ),
            "search": RandomizedSearchCV,
            "params": {
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.05, 0.1],
            },
        },
    }


def build_pipeline(estimator, numeric_features=None, categorical_features=None) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(numeric_features, categorical_features)),
                     ("model", estimator)])


def load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # A station-month with no physical-file record has structurally zero
    # physical samples, not an unknown count: fill with 0 rather than impute.
    df["n_samples_physical"] = df["n_samples_physical"].fillna(0)
    return df


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "plus")


def develop_one(frame: pd.DataFrame, out_dir: str, name: str) -> dict:
    """Tune ONE model on the development period and cache the result.

    Split into one model per run because tuning all four in a single process
    exceeds the execution time budget; results are cached so the full comparison
    can be assembled afterwards by finalise().
    """
    dev = frame[frame["month"] <= 9].copy()
    X = dev[ALL_FEATURES]
    class_index = {c: i for i, c in enumerate(CLASS_ORDER)}
    y = dev["risk_label"].astype(str).map(class_index)
    groups = dev["station"]

    cv = GroupKFold(n_splits=5)
    spec = candidate_models()[name]
    pipe = build_pipeline(spec["estimator"])

    if spec["search"] is RandomizedSearchCV:
        search = RandomizedSearchCV(
            estimator=pipe, param_distributions=spec["params"], cv=cv,
            scoring="f1_macro", n_jobs=-1, refit=True,
            n_iter=6, random_state=RANDOM_STATE,
        )
    else:
        search = GridSearchCV(
            estimator=pipe, param_grid=spec["params"], cv=cv,
            scoring="f1_macro", n_jobs=-1, refit=True,
        )
    search.fit(X, y, groups=groups)

    idx = search.best_index_
    row = {
        "model": name,
        "cv_macro_f1_mean": float(search.cv_results_["mean_test_score"][idx]),
        "cv_macro_f1_std": float(search.cv_results_["std_test_score"][idx]),
        "best_params": {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
                        for k, v in search.best_params_.items()},
    }
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(search.best_estimator_, os.path.join(out_dir, f"est_{_slug(name)}.pkl"))
    with open(os.path.join(out_dir, f"cv_{_slug(name)}.json"), "w") as fh:
        json.dump(row, fh, indent=2)
    print(f"{name:24s} CV macro F1 = {row['cv_macro_f1_mean']:.4f} "
          f"(+/- {row['cv_macro_f1_std']:.4f})  params={row['best_params']}")
    return row


def develop_experiments(frame: pd.DataFrame, out_dir: str) -> None:
    """SMOTE variant and majority baseline, cached alongside the models."""
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import cross_val_score

    dev = frame[frame["month"] <= 9].copy()
    X = dev[ALL_FEATURES]
    class_index = {c: i for i, c in enumerate(CLASS_ORDER)}
    y = dev["risk_label"].astype(str).map(class_index)
    groups = dev["station"]
    cv = GroupKFold(n_splits=5)

    smote_pipe = ImbPipeline([
        ("prep", make_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("model", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    smote_scores = cross_val_score(smote_pipe, X, y, cv=cv, groups=groups,
                                   scoring="f1_macro", n_jobs=-1)

    dummy = build_pipeline(DummyClassifier(strategy="most_frequent"))
    dummy_scores = cross_val_score(dummy, X, y, cv=cv, groups=groups,
                                   scoring="f1_macro", n_jobs=-1)

    payload = {
        "smote": {"model": "Random Forest + SMOTE",
                  "cv_macro_f1_mean": float(smote_scores.mean()),
                  "cv_macro_f1_std": float(smote_scores.std()),
                  "best_params": {"note": "SMOTE applied inside training folds only"}},
        "baseline": {"majority_baseline_accuracy": float((y == y.mode()[0]).mean()),
                     "majority_baseline_macro_f1": float(dummy_scores.mean())},
    }
    with open(os.path.join(out_dir, "experiments.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"{'Random Forest + SMOTE':24s} CV macro F1 = {smote_scores.mean():.4f}")
    print(f"{'Majority baseline':24s} accuracy = {(y == y.mode()[0]).mean():.4f}, "
          f"macro F1 = {dummy_scores.mean():.4f}")


def finalise(frame: pd.DataFrame, out_dir: str) -> dict:
    """Assemble the comparison table and promote the best model."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "cv_*.json"))):
        rows.append(json.load(open(path)))
    with open(os.path.join(out_dir, "experiments.json")) as fh:
        exp = json.load(fh)
    rows.append(exp["smote"])

    comparison = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    comparison.to_csv(os.path.join(out_dir, "cv_comparison.csv"), index=False)
    print(comparison[["model", "cv_macro_f1_mean", "cv_macro_f1_std"]].to_string(index=False))

    supervised = [r for r in rows if not r["model"].endswith("+ SMOTE")]
    best = max(supervised, key=lambda r: r["cv_macro_f1_mean"])
    best_name = best["model"]
    print(f"\nSelected model (development CV macro F1): {best_name}")

    src_path = os.path.join(out_dir, f"est_{_slug(best_name)}.pkl")
    model = joblib.load(src_path)
    joblib.dump(model, os.path.join(out_dir, "final_model.pkl"))
    joblib.dump(model.named_steps["prep"],
                os.path.join(out_dir, "preprocessing_pipeline.pkl"))

    metadata = {
        "selected_model": best_name,
        "selected_cv_macro_f1": best["cv_macro_f1_mean"],
        "majority_baseline_accuracy": exp["baseline"]["majority_baseline_accuracy"],
        "majority_baseline_macro_f1": exp["baseline"]["majority_baseline_macro_f1"],
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_order": CLASS_ORDER,
        "random_state": RANDOM_STATE,
        "development_period": "2021-01 to 2021-09",
        "holdout_period": "2021-10 to 2021-12",
        "cv_scheme": "GroupKFold(5) grouped by station",
        "selection_metric": "macro F1 (development CV only)",
        "target_definition": {
            "base": "CPCB DO criteria: >=6 Low, 4-6 Moderate, <4 High",
            "escalation": "one level up if ammonia_n>1.0 or nitrate_n>10.2 mgN/L",
            "note": "water-quality stress proxy, not confirmed disease risk",
        },
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


if __name__ == "__main__":
    import argparse
    import glob

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base, "models")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="tune a single model and cache it")
    parser.add_argument("--experiments", action="store_true",
                        help="run SMOTE and baseline comparisons")
    parser.add_argument("--select", action="store_true",
                        help="assemble comparison and promote best model")
    args = parser.parse_args()

    frame = load_frame(os.path.join(base, "data", "processed", "modelling_frame.csv"))
    if args.select:
        finalise(frame, out_dir)
    elif args.experiments:
        develop_experiments(frame, out_dir)
    elif args.model:
        develop_one(frame, out_dir, args.model)
    else:
        print("use --model/--experiments/--select")
