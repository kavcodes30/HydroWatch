# Smart Community Health Monitoring and Early Warning System for Water-Borne Diseases

Machine-learning pipeline that classifies **potential community-level water-borne disease risk**
into Low / Moderate / High from ambient water-quality measurements and modelled environmental
data, with an interpretable early-warning layer.

> **What this system is not.** It is a decision-support and early-warning indicator. It does not
> diagnose individuals, does not confirm bacterial or viral contamination, and does not declare
> outbreaks. Every alert is a prompt for human investigation.

---

## Results at a glance

| Metric | Value |
|---|---|
| Holdout accuracy (Oct–Dec 2021) | **0.789** |
| Holdout macro F1 | **0.660** |
| High-Risk recall | **0.727** |
| Majority-class baseline accuracy | 0.707 |
| Majority-class baseline macro F1 | 0.276 |
| **Unseen-state macro F1 (leave-one-state-out)** | **0.315** |
| **Unseen-state High-Risk recall** | **0.057** |

Read the accuracy against the 0.707 baseline: the honest measure of skill is macro F1, which is
2.4× the baseline. But read that in turn against the last two rows: **the model does not transfer
to a state it has never seen.** The 0.660 figure is within-region, across-time performance at
known stations, not generalisable risk prediction. Full reasoning in
`reports/model_evaluation_report.md` and `reports/spatial_validation_report.md`.

The served model is a **regularised** Random Forest (depth 12) chosen for deployment: holdout
macro F1 0.638, High-Risk recall **0.855**, overfitting gap 0.132 (versus 0.199 for the
CV-selected model). Isotonic calibration was tested and rejected — it improved the Brier score
but halved High-Risk recall.

## Target definition (and why it is a proxy)

No health or surveillance data was supplied, so no supervised disease label exists. The label is
a documented **water-quality risk index**:

- Base tier from CPCB surface-water dissolved-oxygen criteria: ≥6 mg/L Low, 4–6 Moderate, <4 High.
- Escalated one level (never down) where ammonia-N > 1.0 mgN/L or nitrate-N > 10.2 mgN/L as N.

Dissolved oxygen, ammonia and nitrate are **excluded from the predictors**, so the model cannot
recite its own label. This is a water-quality stress proxy consistent with contamination risk —
not laboratory-confirmed disease risk.

## Project layout

```
waterborne_disease_prediction/
├── data/
│   ├── raw/            CPCB CSVs as supplied + NASA POWER daily reanalysis (cached)
│   ├── interim/        cleaned chemical and physical observations
│   └── processed/      station-month table and the final modelling frame
├── notebooks/          01-05, thin runnable wrappers around src/
├── src/
│   ├── data_acquisition.py   NASA POWER rainfall/temperature fetch (cached)
│   ├── data_preprocessing.py cleaning, validation, station-month aggregation
│   ├── feature_engineering.py causal features + target construction
│   ├── train_model.py        tuning, model comparison, selection
│   ├── evaluate_model.py     holdout evaluation + leakage audit
│   ├── interpretability.py   importance, permutation, SHAP
│   ├── early_warning.py      alerts, thresholds, synthetic symptom stream
│   ├── deploy_model.py       regularised variant + calibration comparison
│   ├── spatial_validation.py leave-one-state-out generalisation test
│   └── predict.py            reusable prediction interface
├── app/
│   ├── main.py         FastAPI service (predict, alerts, symptoms, model-info)
│   └── static/index.html   self-contained dashboard
├── models/             final_model.pkl, deployment_model.pkl, preprocessing_pipeline.pkl
├── reports/            data quality, EDA, evaluation, technical report, alerts
├── figures/            all generated charts
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

cd src
python data_acquisition.py       # fetch NASA POWER (skipped if cache exists)
python data_preprocessing.py     # clean + aggregate -> data/processed/
python feature_engineering.py    # features + target -> modelling_frame.csv

# training is split per model to stay inside execution time limits
python train_model.py --model "Logistic Regression"
python train_model.py --model "Decision Tree"
python train_model.py --model "Random Forest"
python train_model.py --model "XGBoost"
python train_model.py --experiments     # SMOTE + majority baseline
python train_model.py --select          # compare, promote best, write metadata

python evaluate_model.py         # holdout metrics + leakage audit
python eda.py                    # EDA figures
python interpretability.py       # importance, permutation, SHAP
python early_warning.py          # alerts + synthetic symptom stream
python deploy_model.py           # regularised deployment variant + calibration check
python spatial_validation.py     # leave-one-state-out generalisation test
python spatial_validation.py --ablate-location   # same, location features removed
python predict.py                # smoke-test the prediction API
```

Every script is deterministic (`random_state=42`) and writes its outputs to `reports/` or
`figures/`.

## Prediction API

```python
from predict import RiskPredictor

predictor = RiskPredictor("models")
result = predictor.predict_one({
    "station": "ST-001", "district": "Pune", "state": "Maharashtra",
    "latitude": 18.52, "longitude": 73.85, "month": 7,
    "ph": 7.2, "electrical_conductivity": 640.0,
    "rainfall_monthly_mm": 210.0, "t2m_c": 27.5,
})
# result["risk_label"], result["probabilities"], result["alert_status"],
# result["top_contributing_features"], result["disclaimer"]
```

The saved pipeline contains the fitted preprocessing, so the same imputation, scaling and
encoding used in training are applied automatically. Absent fields are imputed with training
medians and the response is tagged `reduced_context`.

## API and dashboard

```bash
cd app
uvicorn main:app --reload --port 8000
# open http://127.0.0.1:8000
```

| Endpoint | Purpose |
|---|---|
| `GET /` | dashboard (risk form, alerts table, symptom chart) |
| `GET /health` | service and model status |
| `GET /model-info` | metrics, target definition, spatial caveat |
| `POST /predict` | single record → risk, probabilities, SHAP factors, alert status |
| `POST /predict/batch` | up to 500 records |
| `GET /alerts` | filter by `state`, `severity`, `limit` |
| `GET /symptoms` | monthly synthetic symptom totals (labelled synthetic) |
| `GET /disclaimer` | standing disclaimers |

Interactive docs at `/docs`. Every risk-bearing response includes the disclaimer; the
dashboard shows the spatial-coverage warning permanently.

## Data caveats you must keep in mind

1. **One year only.** The `2021_2025` filenames are misleading: every observation is from 2021.
2. **Turbidity and TDS are unusable.** They never co-occur with dissolved oxygen; the 173
   turbidity stations measure no DO at any time. See `reports/data_quality_report.md` §4.
3. **Ambient river stations, not drinking-water points.** CPCB monitoring is surface water.
4. **Rainfall is modelled.** NASA POWER MERRA-2 reanalysis on a ~0.5° grid, not gauge data.
5. **Geography dominates the model.** Longitude, latitude and state outrank every chemical
   predictor by an order of magnitude. The model learns regional base rates, not contamination
   mechanisms.
6. **Probabilities are uncalibrated** model scores — calibration was tested and made High-Risk
   recall worse, so it was rejected.
7. **Do not apply the model outside its training states.** Leave-one-state-out macro F1 is 0.315
   and High-Risk recall 0.057 on unseen states.
8. **The symptom stream is synthetic**, generated only to exercise the dashboard, tagged
   `data_source='synthetic'`, and never used for training or evaluation.

## Reaching higher accuracy honestly

Multiple years of CPCB history, dissolved oxygen and turbidity co-measured at the same points,
drinking-water supply sampling locations, and district health outcomes. These are data problems,
not modelling problems — no algorithm change closes the gap.

## References

- WHO, *Guidelines for drinking-water quality*, 4th ed. incl. first–third addenda.
- BIS IS 10500:2012, *Drinking Water — Specification*.
- CPCB, surface-water quality criteria (designated best use classes).
- NASA POWER / MERRA-2 reanalysis API (`power.larc.nasa.gov`).
