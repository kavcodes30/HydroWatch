# Spatial Validation Report

**Scheme:** leave-one-state-out cross-validation
**Question answered:** does the model work at a location it has never seen?

---

## 1. Why this test was needed

The main evaluation splits by **time** (train Jan–Sep, test Oct–Dec) and therefore shares
727 stations between train and test. It answers: *can the model predict a new period at a known
station?* Real deployment asks something harder: *can it work in a district or state it has
never seen?*

This report answers the harder question, and the answer is uncomfortable.

## 2. Protocol

- For each of the 7 states: train on the other 6 states, test on the held-out state.
- Same tuned Random Forest hyperparameters as the selected model, so the split is the only
  thing that changes.
- Reported per state: accuracy, macro F1, High-Risk recall, and the state's own High-Risk share
  (to show how far its base rate differs from the training pool).

## 3. Results

| Held-out state | Test rows | State High share | Accuracy | Macro F1 | High-Risk recall |
|---|---|---|---|---|---|
| Gujarat | 878 | 12.4% | 0.614 | **0.449** | **0.339** |
| Jammu & Kashmir | 543 | 10.5% | 0.718 | 0.279 | 0.000 |
| Kerala | 1,116 | 8.0% | 0.800 | 0.296 | 0.000 |
| Madhya Pradesh | 2,036 | 0.8% | 0.909 | 0.364 | 0.059 |
| Maharashtra | 1,848 | 9.5% | 0.491 | 0.271 | 0.000 |
| Punjab | 563 | 24.3% | 0.567 | 0.241 | 0.000 |
| West Bengal | 703 | 8.1% | 0.754 | 0.303 | 0.000 |

**Averages:** accuracy 0.693, macro F1 **0.315**, High-Risk recall **0.057**.
Pooled across all held-out rows: accuracy 0.706, macro F1 0.355.

## 4. What the numbers mean

**High-Risk recall collapses to near zero.** In six of seven states the model fails to identify
a single High-Risk station-month when that state was absent from training. The early-warning
capability that makes the system worth building does not transfer across state boundaries.

**Accuracy becomes meaningless.** Madhya Pradesh scores 0.909 accuracy while detecting 5.9% of
its High-Risk cases — because the state is 94% Low and the model simply predicts Low. This is the
clearest possible demonstration of why accuracy was demoted as the headline metric.

**Macro F1 falls from 0.660 to 0.315** — less than half the within-region figure, and barely
above the 0.276 majority baseline measured earlier. The pooled 0.355 is only marginally better.

**Gujarat is the sole exception** (macro F1 0.449, High-Risk recall 0.339). It is the state whose
river systems and industrial loadings most resemble the training pool, which is itself evidence
that the model is matching regional character rather than learning a mechanism.

## 5. Is geography the cause? (ablation)

If location features were responsible for the poor transfer, removing them should help. It does
not:

| Configuration | Mean accuracy | Mean macro F1 | Mean High-Risk recall |
|---|---|---|---|
| All 22 features | 0.693 | 0.3146 | 0.0569 |
| Location features removed (19 features) | 0.624 | 0.3102 | 0.0289 |

Dropping `state`, `latitude` and `longitude` makes accuracy and High-Risk recall **worse** while
macro F1 stays flat. The conclusion is not "geography broke it" — it is that **no configuration
of these features carries signal that transfers between regions**. Location inflates in-sample
performance, but removing it does not reveal hidden chemistry-driven skill underneath.

## 6. Revised claim about the headline result

The earlier result — 0.789 accuracy, 0.660 macro F1, 0.727 High-Risk recall on the Oct–Dec
holdout — remains a correct measurement of **within-region, across-time** performance at known
stations. It is **not** evidence of generalisable risk prediction, and the system must not be
presented as deployable to an unseen state.

Stated precisely, what the model does: given a station already in the training pool, it predicts
the next quarter's water-quality risk class reasonably well, largely from that station's
regional character and recent history.

## 7. Deployment implications

1. **Restrict application to states present in training.** For a new state, retrain including it.
2. **A hierarchical or per-region model** would be the principled fix — pooling national signal
   while allowing region-specific base rates.
3. **Report both numbers in any write-up.** Quoting 0.789 accuracy without the 0.315 spatial
   macro F1 would overstate the system.
4. **Treat the early-warning layer as regional**, not national. Its alert volume and precision
   are only characterised for regions the model has seen.

## 8. Why this strengthens rather than weakens the project

Finding that a model does not transfer, and proving it with a leave-one-state-out test plus an
ablation, is a stronger academic contribution than reporting an inflated within-region score. It
identifies the binding constraint precisely: not the algorithm, not the hyperparameters, but the
absence of co-measured, health-linked, multi-year data. That is a concrete, defensible
recommendation for the next phase of work.

Figure: `figures/12_spatial_validation.png`
Data: `reports/spatial_validation_with_location.csv`, `reports/spatial_validation_no_location.csv`
