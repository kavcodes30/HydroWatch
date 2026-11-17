#!/usr/bin/env bash
# Reproduce the whole pipeline end to end.
# Training is split per model because tuning all four in one process exceeds
# typical execution time limits; --select assembles the comparison afterwards.
set -euo pipefail

cd "$(dirname "$0")/src"

echo "== Phase 1-2: acquisition and cleaning =="
python data_acquisition.py
python data_preprocessing.py

echo "== Phase 3-4: features and target =="
python feature_engineering.py

echo "== Phase 5: EDA =="
python eda.py

echo "== Phase 6: model tuning and selection =="
python train_model.py --model "Logistic Regression"
python train_model.py --model "Decision Tree"
python train_model.py --model "Random Forest"
python train_model.py --model "XGBoost"
python train_model.py --experiments
python train_model.py --select

echo "== Phase 7: holdout evaluation and leakage audit =="
python evaluate_model.py

echo "== Phase 8: interpretability =="
python interpretability.py

echo "== Deployment variant (regularised + calibration comparison) =="
python deploy_model.py

echo "== Spatial validation (leave-one-state-out) =="
python spatial_validation.py
python spatial_validation.py --ablate-location

echo "== Phase 9: early warning =="
python early_warning.py

echo "== Phase 10: prediction smoke test =="
python predict.py

echo "== API + dashboard =="
echo "  cd app && uvicorn main:app --reload --port 8000"
echo "  then open http://127.0.0.1:8000"

echo "Done. See reports/ and figures/."
