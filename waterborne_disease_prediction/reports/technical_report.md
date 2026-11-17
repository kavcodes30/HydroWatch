# Technical Report

**Project:** Smart Community Health Monitoring and Early Warning System for Water-Borne Diseases
**Deliverable:** End-to-end machine-learning pipeline, evaluation, and early-warning layer
**Data vintage:** calendar year 2021

---

## 1. Executive summary

A three-class water-quality risk classifier was built on 8,674 CPCB chemical and 3,868 CPCB
physical observations from 1,096 stations across 7 Indian states, enriched with NASA POWER
reanalysis rainfall and temperature. The system classifies station-months as Low, Moderate or
High risk and drives an interpretable alerting layer.

On a time-based holdout (Oct–Dec 2021, never used for selection or tuning) the selected Random
Forest achieves **78.9% accuracy, 0.660 macro F1 and 0.727 High-Risk recall**. Against a 70.7%
majority-class baseline, macro F1 is 2.4× the baseline.

Four findings dominate the project and are stated plainly rather than buried:

1. **No health data exists** in any supplied file, so no supervised disease label could be
   built. The target is a documented water-quality risk index, explicitly a proxy.
2. **Turbidity and TDS — the synopsis's priority parameters — are unusable.** They never
   co-occur with dissolved oxygen; the 173 turbidity stations record no DO at any time.
3. **Geography, not chemistry, drives the model.** Longitude, latitude and state outrank every
   water-quality predictor by more than an order of magnitude.
4. **The model does not transfer to unseen regions.** Under leave-one-state-out validation it
   achieves macro F1 0.315 and High-Risk recall 0.057 — the headline 0.660 is within-region,
   across-time performance at known stations, not generalisable risk prediction.

The 90% accuracy aspiration is **not achievable honestly** with this data. Section 9 explains why
and what would change it.

## 2. Data sources

| Source | Content | Role | Provenance |
|---|---|---|---|
| CPCB chemical, 7 states | 8,674 rows, 52 cols, 33 analytes | label + predictors | measured, regulatory monitoring |
| CPCB physical, 7 states | 3,868 rows, 25 cols, 6 analytes | predictors | measured |
| NASA POWER (MERRA-2) | daily precipitation, 2 m temperature | environmental predictors | **modelled reanalysis** |

CPCB files were supplied by the project team. NASA POWER was fetched programmatically per 0.5°
grid cell (1,096 stations collapse to 255 cells) and cached to
`data/raw/nasa_power_daily_2021.csv`.

Two provenance caveats are carried throughout: CPCB stations are **ambient surface-water**
monitoring points (rivers, lakes, canals, ponds, bore wells), not household drinking-water supply
points; and POWER rainfall is gridded reanalysis, not gauge observation.

## 3. Preprocessing

Implemented in `src/data_preprocessing.py`, fully logged to `reports/cleaning_log.csv`.

1. `-` placeholder (69,395 chemical and 30,944 physical cells) converted to null — left as text
   it defeats numeric parsing and makes columns appear complete.
2. Column names standardised to snake_case; timestamps parsed from `dd-mm-yyyy HH:MM`.
3. 14 all-empty chemical columns and 4 all-empty physical columns dropped (including water
   temperature, colour, odour, total solids).
4. 8 placeholder-only geography columns dropped (`Village`, `Block`, `River`, `Basin`, …).
5. Exact duplicates removed (none found); 18 conflicting station-date pairs per family retained
   and averaged, not deleted.
6. Implausible values flagged and nulled against physical bounds (pH 3–12, DO 0–20 mg/L,
   TDS 0–10,000 mg/L, alkalinity 0–3,000, EC 0–60,000 µS/cm). The EC bound was set just above
   seawater so genuine Kerala backwater readings (~43,500 µS/cm) survive.
7. Aggregated to **station-month** grain (8,518 rows), keyed on station and month only — keying
   on coordinates split the conflicting duplicates into separate rows.

No imputation happens here. All imputation lives inside the scikit-learn pipeline, fitted on
training folds only, so test information never leaks backwards.

## 4. Feature engineering

Implemented in `src/feature_engineering.py`. **22 features**, all causal:

| Group | Features |
|---|---|
| Water quality | ph, electrical_conductivity, has_ec |
| Environmental | rainfall_monthly_mm, rainfall_3m_mm, rainfall_anomaly_mm, t2m_c |
| Temporal | month_sin, month_cos, season, monsoon_flag |
| Location | latitude, longitude, state |
| History | lag1_ph, lag1_ec, rolling3_ec, delta_ph, delta_ec, months_since_first, n_samples_chemical, n_samples_physical |

**No look-ahead.** Every temporal feature uses `shift(1)` before rolling or expanding
aggregation. Station climatology for the rainfall-anomaly feature is an expanding mean of
*prior* months, never a full-year mean — with a single year of data a full-year mean would embed
future information. The 3-month rainfall total does include the current month, which is
legitimate because rainfall is known in real time (unlike a lab result reported later).

## 5. Target construction

| Element | Definition | Source |
|---|---|---|
| Base tier | DO ≥ 6.0 → Low; 4.0–6.0 → Moderate; < 4.0 → High | CPCB surface-water criteria |
| Escalation | one level up (never down) if ammonia-N > 1.0 mgN/L or nitrate-N > 10.2 mgN/L as N | BIS IS 10500 (45 mg/L as NO3 = 10.2 as N) |

Resulting distribution over 7,687 labelled station-months: **Low 5,488 (71.4%), Moderate 1,557
(20.3%), High 642 (8.4%)**. 201 rows were escalated.

**Leakage control.** Dissolved oxygen, ammonia and nitrate are the label components and are
excluded from the feature list — verified programmatically in notebook 04 and in the evaluation
leakage audit. A circular design (predicting an index from its own inputs) would have produced
97–99% accuracy and zero scientific value. The deliberate consequence is a lower, honest score.

**What the label is not.** It is a water-quality stress indicator consistent with contamination
risk. It is not a disease outcome, not a pathogen measurement, and not an outbreak label.

Nitrate escalation is defined but never fires, because nitrate and DO never co-occur (§6.2 of
the data quality report). Ammonia escalation does fire (201 rows).

## 6. Algorithms compared

| Model | Role | CV macro F1 | ± std |
|---|---|---|---|
| Logistic Regression | linear baseline | 0.5150 | 0.0251 |
| Decision Tree | interpretable non-linear | 0.6414 | 0.0244 |
| Random Forest | **selected** | **0.7066** | 0.0236 |
| XGBoost | boosted ensemble | 0.6959 | 0.0153 |
| Random Forest + SMOTE | imbalance experiment | 0.6918 | 0.0260 |
| Majority baseline | reference floor | 0.2761 | — |

Tuning: `GridSearchCV` for Logistic Regression and Decision Tree, `RandomizedSearchCV`
(6 iterations) for Random Forest and XGBoost. Selection criterion was **macro F1 on development
CV**, declared before any holdout scoring.

Random Forest and XGBoost are within one standard deviation; the choice follows the declared
criterion, not holdout results. SMOTE did not help and slightly increased fold variance —
expected when the minority class is not separable by the available features.

## 7. Validation strategy

| Element | Choice | Rationale |
|---|---|---|
| Development set | Jan–Sep 2021 (5,656 rows) | tuning and selection only |
| Holdout | Oct–Dec 2021 (2,031 rows) | scored once, at the end |
| CV scheme | GroupKFold(5) by station | prevents station memorisation across folds |
| Imbalance | balanced class weights; SMOTE inside folds only | resampling never precedes splitting |
| Primary metric | macro F1 | accuracy is inflated by the 71% majority class |
| Safety metric | High-Risk recall | a missed high-risk community is worse than a false alarm |

**Honest limitation:** the time split shares 727 stations between train and test, which is
unavoidable when splitting within a single year. The holdout therefore measures performance on
new *periods* at known stations, not at unseen stations. A leave-states-out spatial holdout is
recommended as the realistic deployment test (see §10).

## 8. Results

Holdout (n = 2,031):

| Metric | Value |
|---|---|
| Accuracy | 0.7888 |
| Macro F1 | 0.6599 |
| Weighted F1 | 0.7984 |
| High-Risk precision | 0.4188 |
| High-Risk recall | **0.7273** |
| Training accuracy (dev) | 0.9878 |
| Overfitting gap | +0.1990 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.923 | 0.853 | 0.887 | 1,490 |
| Moderate | 0.542 | 0.582 | 0.562 | 431 |
| High | 0.419 | 0.727 | 0.532 | 110 |

Confusion matrix: only **8 of 110** High-Risk cases fell all the way to Low. The dominant error
is Moderate↔High confusion, which for early warning is acceptable — both trigger human review.

**Overfitting** is real and disclosed: 0.988 train versus 0.789 holdout. The selected forest is
unlimited-depth. It was not swapped for a regularised variant after seeing holdout numbers,
because that would compromise the protocol; a depth-limited forest is recommended for
deployment.

**Calibration** was assessed, not assumed: Brier scores are 0.126 (Low), 0.132 (Moderate) and
0.039 (High). No isotonic calibration was fitted, so the interface labels every probability as
an uncalibrated score.

## 9. Spatial validation — the binding constraint

The main holdout splits by time and therefore shares 727 stations between train and test. A
leave-one-state-out test (train on six states, test on the seventh, repeated seven times) asks
the deployment question instead:

| Held-out state | Accuracy | Macro F1 | High-Risk recall |
|---|---|---|---|
| Gujarat | 0.614 | 0.449 | 0.339 |
| Jammu & Kashmir | 0.718 | 0.279 | 0.000 |
| Kerala | 0.800 | 0.296 | 0.000 |
| Madhya Pradesh | 0.909 | 0.364 | 0.059 |
| Maharashtra | 0.491 | 0.271 | 0.000 |
| Punjab | 0.567 | 0.241 | 0.000 |
| West Bengal | 0.754 | 0.303 | 0.000 |
| **Mean** | **0.693** | **0.315** | **0.057** |

High-Risk recall collapses to near zero in six of seven states. Madhya Pradesh illustrates why
accuracy alone is untrustworthy: 0.909 accuracy while detecting 5.9% of its high-risk cases,
because the state is 94% Low.

Ablating the location features (`state`, `latitude`, `longitude`) does **not** rescue transfer —
macro F1 stays flat (0.310) and High-Risk recall falls further (0.029). The limitation is
therefore not "geography broke the model"; it is that no configuration of these features carries
signal that transfers between regions.

**Deployment rule derived from this:** the model may only be applied to states present in
training. A new state requires retraining, ideally with a hierarchical model that pools national
signal while allowing region-specific base rates.

## 10. Deployment model variant

Two properties matter for a dashboard showing confidence: a smaller overfitting gap, and
probabilities that mean something. Both were tested on the holdout, and this is disclosed as a
post-hoc comparison rather than a re-selection:

| Configuration | Accuracy | Macro F1 | High-Risk recall | Overfit gap | Brier (High) |
|---|---|---|---|---|---|
| Selected model (unlimited depth) | 0.789 | 0.660 | 0.727 | 0.199 | 0.0386 |
| **Regularised RF (depth 12) — shipped** | 0.763 | 0.638 | **0.855** | **0.132** | 0.0485 |
| Regularised RF + isotonic — rejected | 0.781 | 0.555 | 0.473 | 0.097 | 0.0389 |

Depth-limiting the forest raises High-Risk recall from 0.727 to 0.855 and cuts the overfitting
gap from 0.199 to 0.132, at a cost of 0.022 macro F1. That is the right trade for early warning,
so the regularised forest is the model the API serves.

**Isotonic calibration was tested and rejected.** It improved the Brier score for High
(0.0485 → 0.0389) but halved High-Risk recall (0.855 → 0.473), because reshaping probabilities
moves borderline cases out of the High class. For a system whose costliest error is a missed
high-risk community, that is a bad trade. Probabilities remain labelled as uncalibrated scores.

## 11. Why 90% accuracy was not reached

| Cause | Evidence |
|---|---|
| Class imbalance | 71.4% Low; baseline accuracy 70.7% |
| Weak predictors | max correlation with DO is 0.19 (longitude) |
| Priority parameters unusable | turbidity/TDS: 0% overlap with DO |
| Geography dominates | longitude 0.178 vs pH 0.008 permutation importance |
| One year of data | no trend, no multi-year climatology |
| Weak monsoon signal | mean DO varies only 6.28–6.77 mg/L all year |
| Sparse sampling | ~11 monthly observations per station |

Every one of these is a data limitation, not an algorithmic one. No model choice closes the gap.

## 12. Limitations

1. **Proxy target, not disease risk.** The label is water-quality stress; it cannot be described
   as predicting water-borne disease.
2. **No health data.** The community-health pillar of the synopsis is represented only by a
   clearly-labelled synthetic stream for dashboard demonstration.
3. **Surface-water stations, not drinking-water points.** Ecological mismatch with the disease
   pathway, which runs through drinking water.
4. **Turbidity and TDS excluded** (0% co-occurrence with DO).
5. **Water temperature absent entirely.**
6. **Single year** — the largest constraint on validation depth.
7. **Modelled rainfall**, not gauge data.
8. **Geography-driven performance** — the model learns regional base rates, not mechanisms.
9. **Station overlap across the time split** — holdout is periods, not new locations.
10. **No spatial generalisation** — leave-one-state-out macro F1 0.315, High-Risk recall 0.057.
11. **Uncalibrated probabilities** — isotonic calibration reduced High-Risk recall and was
    rejected, so scores are uncalibrated by deliberate choice.

## 13. Future improvements

| Priority | Action | Expected effect |
|---|---|---|
| 1 | Obtain multi-year CPCB exports (2019–2024) | largest single gain; enables real trend and climatology features |
| 2 | Request DO + turbidity co-measured at the same stations, or instrument with the ESP32 rig | restores the synopsis's priority parameters |
| 3 | Add IDSP/IHIP district health outcomes | converts the proxy into a genuine disease-risk target |
| 4 | Depth-limit the forest (10–15) for deployment | shrinks the 20-point overfitting gap |
| 5 | Add a leave-states-out spatial holdout | measures realistic unseen-location performance |
| 6 | Isotonic probability calibration | makes dashboard confidence meaningful |
| 7 | IMD gauge rainfall instead of reanalysis | removes a modelled-input dependency |

## 14. Deliverables

- `src/` — 10 executable modules (acquisition, preprocessing, features, training, evaluation,
  interpretability, early warning, deployment variant, spatial validation, prediction)
- `app/` — FastAPI service and self-contained dashboard (8 endpoints, `/docs` enabled)
- `models/` — `final_model.pkl`, `deployment_model.pkl`, `preprocessing_pipeline.pkl`,
  `model_metadata.json`
- `reports/` — data quality, EDA, model evaluation, spatial validation, this report, cleaning
  log, alerts, audit outputs
- `figures/` — 12 charts including confusion matrix, permutation importance, SHAP summary and
  spatial-validation comparison
- `notebooks/` — 5 notebooks, all verified to execute
- `data/` — raw, interim and processed datasets
- `requirements.txt`, `README.md`

## 15. Reproducibility

All scripts are deterministic (`random_state=42`), run in the order given in the README, and
write every output to `reports/` or `figures/`. Training is split per model
(`--model`, `--experiments`, `--select`) to fit execution time limits; the full comparison is
assembled by `--select`. The raw NASA POWER cache is committed so the environmental step does
not require network access on re-run.
