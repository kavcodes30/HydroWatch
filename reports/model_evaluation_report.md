# Model Evaluation Report

**Model:** Random Forest (selected on development-set cross-validated macro F1)
**Target:** three-class water-quality risk (Low / Moderate / High)
**Holdout:** October–December 2021, scored once, never used for selection or tuning

> **Read this alongside `reports/spatial_validation_report.md`.** The results below measure
> performance on new *time periods* at stations already seen in training. On states never seen
> in training, leave-one-state-out validation gives macro F1 0.315 and High-Risk recall 0.057.
> The numbers here are a correct within-region measurement, not evidence of generalisable
> risk prediction.

---

## 1. Headline results

| Metric | Value |
|---|---|
| Holdout accuracy | **0.7888** |
| Holdout macro F1 | **0.6599** |
| Holdout weighted F1 | **0.7984** |
| High-Risk precision | 0.4188 |
| High-Risk recall | **0.7273** |
| Training accuracy (development set) | 0.9878 |
| Overfitting gap (train − holdout) | **+0.1990** |
| Majority-class baseline accuracy | 0.7069 |
| Majority-class baseline macro F1 | 0.2761 |
| Holdout rows | 2,031 |

The 78.9% accuracy is only 8.2 points above the 70.7% a model gets by labelling everything
"Low". **Macro F1 is the honest headline**: 0.66 against a 0.28 baseline. High-Risk recall of
0.73 means roughly three in four genuinely high-risk station-months are flagged — the property
that matters most for an early-warning system, where a missed high-risk community is worse than
a false alarm.

## 2. Model comparison (development-set CV, GroupKFold by station)

| Model | CV macro F1 | ± std | Selected |
|---|---|---|---|
| Random Forest | **0.7066** | 0.0236 | yes |
| XGBoost | 0.6959 | 0.0153 | |
| Random Forest + SMOTE | 0.6918 | 0.0260 | |
| Decision Tree | 0.6414 | 0.0244 | |
| Logistic Regression | 0.5150 | 0.0251 | |
| Majority baseline | 0.2761 | — | |

Random Forest and XGBoost are within one standard deviation of each other; Random Forest was
chosen on the pre-declared criterion (highest CV macro F1), not on holdout performance.
**SMOTE did not help** (0.6918 vs 0.7066) and made folds slightly less stable, which is a
common outcome when class imbalance is moderate and the minority class is not separable —
resampling cannot manufacture signal that the features do not carry.

## 3. Per-class performance (holdout)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.923 | 0.853 | 0.887 | 1,490 |
| Moderate | 0.542 | 0.582 | 0.562 | 431 |
| High | 0.419 | 0.727 | 0.532 | 110 |

Confusion matrix (rows = actual, columns = predicted):

|  | Low | Moderate | High |
|---|---|---|---|
| **Low** | 1,271 | 190 | 29 |
| **Moderate** | 98 | 251 | 82 |
| **High** | 8 | 22 | 80 |

Only 8 of 110 High-Risk cases were predicted as Low — the most consequential error type is
rare. The dominant error is Moderate↔High confusion (82 and 22 cases), which for an
early-warning system is a tolerable trade: both outcomes trigger human follow-up.

## 4. Overfitting

Training accuracy 0.9878 versus holdout 0.7888 is a 20-point gap, and it is reported rather
than smoothed away. The selected Random Forest has `max_depth=None` and
`min_samples_leaf=1`, so it memorises the development set. Two points matter:

1. Model **selection** was made on development-set CV, where the gap does not influence the
   choice between candidates — XGBoost, the more regularised option, was also evaluated and
   scored lower.
2. For deployment, a depth-limited variant is preferable if stable probabilities matter more
   than peak CV score. This is flagged in the recommendations rather than applied silently,
   because changing the model after seeing holdout numbers would compromise the protocol.

## 5. Probability calibration

Probabilities are reported as model scores, not calibrated real-world risks. Brier scores on
the holdout: Low 0.126, Moderate 0.132, High 0.039. The High class scores well because it is
rare; the Low and Moderate scores reflect genuine uncertainty. No isotonic or Platt calibration
was fitted, so the interface labels every probability with a caveat.

## 6. Leakage audit

| Check | Result |
|---|---|
| Target components present in feature list | **none** (DO, ammonia, nitrate excluded) |
| Row overlap between train and test | 0 |
| Duplicate feature rows across splits | 0 |
| Stations present in both splits | 727 (by design, time split) |

The time-based split shares stations across train and test, which is unavoidable when splitting
within one year. The mitigation is that **model selection used GroupKFold by station**, so the
development scores never benefited from the same station appearing in both folds. The holdout
therefore measures performance on new time periods at known stations, not at unseen stations —
a distinction stated plainly rather than glossed.

The label was built from dissolved oxygen, ammonia and nitrate; the predictors contain none of
those three. This is the single most important safeguard in the project, and it is why the
accuracy is 79% rather than the 97–99% a circular design would have produced.

## 7. Interpretability — the uncomfortable finding

Permutation importance on the holdout (drop in macro F1 when a raw column is shuffled):

| Feature | Mean decrease in macro F1 |
|---|---|
| longitude | 0.1784 |
| latitude | 0.1271 |
| state | 0.1155 |
| ph | 0.0075 |
| n_samples_physical | 0.0071 |
| lag1_ph | 0.0071 |
| n_samples_chemical | 0.0065 |
| electrical_conductivity | 0.0058 |

SHAP for the High-Risk class agrees: longitude 0.114, latitude 0.063, state (Madhya Pradesh)
0.046, then pH at 0.020.

**The model's skill comes overwhelmingly from where a station is, not from what was measured
there.** Geography outranks every water-quality variable by more than an order of magnitude.
This is an honest and important result: the features available after excluding the label
components do not carry much water-quality signal, so the model effectively learns regional
base rates (Punjab 24% High versus Madhya Pradesh 0.8%).

Implication for the project's claims: the system can rank regions and flag unusual
station-months, but it is **not** demonstrating that rainfall or conductivity drive risk. Any
statement to that effect would be unsupported.

## 8. Early-warning layer (holdout, 2,031 station-months)

| Output | Count |
|---|---|
| Alerts (model High or P(High) ≥ 0.50) | 191 |
| Watch (parameter exceedance, model not concerned) | 231 |
| Synthetic symptom rows (dashboard demo only) | 1,536 |

Every alert carries the parameter exceedances found (against BIS IS 10500 / CPCB limits), the
top three SHAP contributions, a recommended action, and the standing disclaimer that the alert
is a potential-risk indicator requiring investigation — not an outbreak declaration.

## 9. Why 90% accuracy was not reached

| Reason | Evidence |
|---|---|
| Class imbalance | 71.4% of rows are Low; baseline accuracy alone is 70.7% |
| Predictors are weak by construction | label components excluded to prevent leakage |
| Turbidity/TDS unusable | 0% overlap with dissolved oxygen |
| Weak seasonal signal | mean DO varies only 6.28–6.77 mg/L across the year |
| One year of data | no multi-year baselines, no trend features |
| Coarse temporal resolution | ~11 monthly observations per station |
| Geography dominates | longitude/latitude outrank all chemistry |

Reaching 90% honestly would require: multiple years of history, DO and turbidity measured at
the same points, drinking-water (not river) sampling locations, and health outcome data.

## 10. Recommendations

1. **Depth-limit the forest** (`max_depth` 10–15) for deployment to shrink the overfitting gap.
2. **Obtain multi-year CPCB exports** — the largest single gain available, enabling genuine
   trend and climatology features.
3. **Request DO and turbidity at the same stations** from CPCB, or instrument a small number of
   stations with the ESP32 rig to generate co-measured data.
4. **Add health outcomes** (IDSP/IHIP district data) to move from a water-quality proxy to a
   genuine disease-risk target.
5. **Calibrate probabilities** (isotonic) before displaying confidence in the dashboard.
6. **Add a spatial holdout** (leave-states-out) to measure performance at unseen locations,
   which is the realistic deployment scenario.
