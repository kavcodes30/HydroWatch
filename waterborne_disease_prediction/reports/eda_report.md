# Exploratory Data Analysis Report

**Analysed set:** 7,687 labelled station-months, 7 states, 145 districts, calendar year 2021
**Figures:** `figures/01_*.png` … `figures/08_*.png`

All statements below are associations. Nothing here establishes causation, and the data cannot
support causal claims: it is observational, single-year and sparsely sampled.

---

## 1. Class distribution

| Class | Station-months | Share |
|---|---|---|
| Low | 5,488 | 71.4% |
| Moderate | 1,557 | 20.3% |
| High | 642 | 8.4% |

Moderately imbalanced. The consequence is direct: a classifier that always predicts Low scores
**70.7% accuracy** while being useless. Every accuracy figure in this project must be read
against that floor, which is why macro F1 drives model selection.

## 2. Missingness

Missingness is structural, not random, and this drives the whole feature design.

| Feature | Share missing |
|---|---|
| electrical_conductivity | 84.1% |
| lag1_ec / rolling3_ec / delta_ec | ~85.7% |
| n_samples_physical | 59.6% |
| lag1_ph / delta_ph | ~9.9% |
| rainfall_anomaly_mm | 9.7% |
| ph | 0.04% |
| rainfall, t2m, month, season, state, coordinates | 0% |

Conductivity is missing for every Maharashtra station (0% coverage) and only 4.3% of Punjab
stations, versus 18–34% elsewhere. Imputing the median and adding a `has_ec` indicator is the
only defensible treatment; the pipeline does exactly that, fitted on training folds.

## 3. Distributions

- **pH** is tightly clustered around 7.5–8.0 with a left tail to ~6.0; a handful of readings
  sit outside the BIS acceptable band of 6.5–8.5.
- **Dissolved oxygen** is centred near 6.4 mg/L with a long left tail to below 2 mg/L — that
  tail is where the High-Risk class lives.
- **Conductivity** spans three orders of magnitude (13 to ~43,500 µS/cm), with the high end
  driven by Kerala backwater stations. Log-scale reading is essential.
- **Rainfall** is strongly seasonal (see §6) and zero-inflated in winter months.
- **Ammonia** is available for only 14% of rows and is right-skewed.

## 4. Correlation with dissolved oxygen (the label anchor)

| Feature | r with DO |
|---|---|
| longitude | 0.193 |
| electrical_conductivity | −0.133 |
| t2m_c | −0.133 |
| ph | 0.123 |
| rainfall_monthly_mm | −0.104 |
| rainfall_3m_mm | −0.091 |
| rainfall_anomaly_mm | −0.083 |
| latitude | 0.088 |

**No feature correlates meaningfully with the label.** The strongest linear association is 0.19.
This is the single most important EDA finding: it explains why the model's macro F1 plateaus
near 0.66 and why geography dominates the importance rankings. The available predictors simply
do not carry much information about dissolved oxygen.

The negative rainfall–DO correlation is directionally consistent with the expectation that
monsoon runoff and organic load depress oxygen levels, but at r ≈ −0.10 it is far too weak to
support any claim, and it is reported as a weak association only.

## 5. State-wise patterns

| State | Share High | Share Moderate | Share Low |
|---|---|---|---|
| Punjab | 24.3% | 16.5% | 59.3% |
| Gujarat | 12.4% | 22.7% | 64.9% |
| Jammu & Kashmir | 10.5% | 18.2% | 71.3% |
| Maharashtra | 9.5% | 42.4% | 48.1% |
| West Bengal | 8.1% | 14.7% | 77.2% |
| Kerala | 8.0% | 11.9% | 80.1% |
| Madhya Pradesh | 0.8% | 5.0% | 94.2% |

State-level variation is large — Punjab is 30 times more likely to be High-Risk than Madhya
Pradesh. This is a real property of the data (different rivers, different industrial and
agricultural loads, different monitoring point types), and it is also why the model leans on
location. Maharashtra is unusual in having more Moderate than High, consistent with widespread
moderate oxygen depletion rather than localised severe events.

## 6. Seasonal and rainfall patterns

| Month | Mean DO (mg/L) | Mean rainfall (mm) | Mean T2M (°C) | Share High |
|---|---|---|---|---|
| Jan | 6.72 | 28.6 | 20.4 | 8.0% |
| Feb | 6.66 | 8.3 | 22.6 | 10.0% |
| Mar | 6.51 | 18.5 | 27.7 | 8.0% |
| Apr | 6.47 | 31.5 | 30.2 | 11.0% |
| May | 6.28 | 190.1 | 30.5 | 10.0% |
| Jun | 6.48 | 263.0 | 27.7 | 9.0% |
| Jul | 6.35 | 385.2 | 26.8 | 11.0% |
| Aug | 6.41 | 220.7 | 25.6 | 8.0% |
| Sep | 6.41 | 309.5 | 25.3 | 8.0% |
| Oct | 6.59 | 137.4 | 23.4 | 6.0% |
| Nov | 6.77 | 67.5 | 20.4 | 6.0% |
| Dec | 6.76 | 34.2 | 19.0 | 5.0% |

Rainfall varies by a factor of ~45 between July (385 mm) and February (8 mm), but mean DO moves
only between 6.28 and 6.77 mg/L, and the High-Risk share stays within 5–11% all year. The
monsoon signal in this dataset is far weaker than the literature on tropical surface water would
suggest — plausibly because sampling is monthly and the export captures the season's average
rather than event-driven oxygen crashes.

By season: High-Risk share is 9.3% in monsoon, 9.8% pre-monsoon, 7.6% winter, 5.8%
post-monsoon. Differences exist but are small.

## 7. Data-quality flags surfaced by EDA

- `figures/04_correlation.png` shows near-zero structure among predictors, indicating no
  multicollinearity problem but also no strong joint signal.
- Extreme conductivity and alkalinity values (flagged and nulled in cleaning) were visible as
  impossible outliers — e.g. a TDS of 960,998 mg/L, which exceeds seawater by three orders of
  magnitude.
- The Moderate class is the least separable: it sits between a tight Low cluster and a sparse
  High tail, which is where most confusion occurs.

## 8. What EDA changed in the design

1. **Turbidity and TDS were dropped** after co-occurrence analysis showed zero overlap with DO.
2. **Accuracy was demoted** as the primary metric, given the 70.7% majority baseline.
3. **Geography was retained as a feature** despite being uninformative about water quality,
   because it encodes real regional base rates — but its dominance is disclosed rather than
   presented as insight into contamination.
4. **Conductivity was kept with a missingness indicator** rather than dropped, since it is the
   only substantial non-pH chemical predictor that overlaps the label.
