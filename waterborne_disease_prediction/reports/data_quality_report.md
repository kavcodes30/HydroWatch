# Data Quality Report

**Project:** Smart Community Health Monitoring and Early Warning System for Water-Borne Diseases
**Phase covered:** Phase 1 (dataset inspection and source validation) and Phase 2 (cleaning)
**Data vintage:** all observations fall in calendar year 2021

---

## 1. Sources and original purpose

Fourteen CSV files were supplied, all from the Central Pollution Control Board (CPCB) ambient
water-quality monitoring programme, exported per state in two parameter families:

- `swq_chemical_parameter_manual_cpcb_<state>_2021_2025.csv` — 7 states, 52 columns
- `swq_physical_parameter_manual_cpcb_<state>_2021_2025.csv` — 7 states, 25 columns

The files were produced for regulatory surface-water monitoring. They were **not** collected to
study water-borne disease, and the monitoring points are ambient river, lake, canal, pond and
bore-well stations — not household drinking-water supply points. This distinction is carried
through every downstream conclusion.

An environmental dataset was additionally acquired from NASA POWER (MERRA-2 reanalysis) for
daily precipitation and 2 m air temperature at each station's coordinates, because the CPCB
files contain no rainfall or ambient temperature. This is **modelled reanalysis, not gauge
observation**, and is labelled as such throughout.

## 2. Dataset inventory

| File | Type | State | Rows | Cols | Stations | Districts | Start | End | Exact dups |
|---|---|---|---|---|---|---|---|---|---|
| swq_chemical_..._gj_... | chemical | Gujarat | 1,127 | 52 | 168 | 21 | 2021-01-01 | 2021-12-29 | 0 |
| swq_chemical_..._jk_... | chemical | Jammu & Kashmir | 555 | 52 | 100 | 9 | 2021-01-14 | 2021-12-29 | 0 |
| swq_chemical_..._kl_... | chemical | Kerala | 1,189 | 52 | 133 | 14 | 2021-01-01 | 2021-12-28 | 0 |
| swq_chemical_..._mh_... | chemical | Maharashtra | 1,942 | 52 | 225 | 31 | 2021-01-01 | 2021-12-28 | 0 |
| swq_chemical_..._mp_... | chemical | Madhya Pradesh | 2,158 | 52 | 240 | 39 | 2021-01-01 | 2021-12-31 | 0 |
| swq_chemical_..._pb_... | chemical | Punjab | 715 | 52 | 98 | 12 | 2021-01-05 | 2021-12-23 | 0 |
| swq_chemical_..._wb_... | chemical | West Bengal | 988 | 52 | 132 | 19 | 2021-01-04 | 2021-12-30 | 0 |
| swq_physical_..._gj_... | physical | Gujarat | 621 | 25 | 167 | 21 | 2021-01-01 | 2021-12-10 | 0 |
| swq_physical_..._jk_... | physical | Jammu & Kashmir | 256 | 25 | 69 | 6 | 2021-01-14 | 2021-12-14 | 0 |
| swq_physical_..._kl_... | physical | Kerala | 535 | 25 | 132 | 14 | 2021-01-01 | 2021-12-16 | 0 |
| swq_physical_..._mh_... | physical | Maharashtra | 828 | 25 | 201 | 29 | 2021-01-01 | 2021-12-10 | 0 |
| swq_physical_..._mp_... | physical | Madhya Pradesh | 897 | 25 | 237 | 39 | 2021-01-01 | 2021-12-31 | 0 |
| swq_physical_..._pb_... | physical | Punjab | 261 | 25 | 98 | 12 | 2021-01-05 | 2021-12-13 | 0 |
| swq_physical_..._wb_... | physical | West Bengal | 470 | 25 | 130 | 19 | 2021-01-04 | 2021-12-30 | 0 |

Totals: **8,674 chemical rows + 3,868 physical rows = 12,542 observations**, covering
**1,096 unique chemical monitoring stations** and **145 districts across 7 states**.

### Column groups

Both families share 19 identifier columns: `SlNo`, `Station`, `Agency`, `State LGD Code`,
`State`, `District LGD Code`, `District`, `Tehsil`, `Block`, `Village`, `River`, `Basin`,
`Tributary`, `Subtributary`, `SubSubtributary`, `Local River`, `Latitude`, `Longitude`,
`Data Acquisition Time`.

Chemical files add 33 analytes (pH, dissolved oxygen, TDS, ammonia-N, nitrate-N, chloride,
alkalinity, hardness series, metals, etc.). Physical files add 6: electrical conductivity,
colour, odour, turbidity, temperature, total solids.

## 3. Finding 1 — the files cover one year, not five

Despite the `2021_2025` filenames, **every timestamp in every file falls within 2021**. There is
no multi-year history. Consequences:

- no year-over-year baseline or "previous monsoon" comparison;
- the time-based holdout can only split within a single year;
- the time split is confounded with season (the holdout is post-monsoon while training spans
  the monsoon), so the time-split score is a conservative estimate.

## 4. Finding 2 — two disjoint monitoring regimes (the decisive data-quality issue)

Analytes were checked for co-occurrence with dissolved oxygen, the anchor of the risk label:

| Analyte | Non-null station-months | Also has dissolved oxygen | Overlap |
|---|---|---|---|
| ph | 8,438 | 7,684 | 91.1% |
| ammonia_n | 1,096 | 1,076 | 98.2% |
| electrical_conductivity | 1,783 | 1,220 | 68.4% |
| turbidity | 258 | 0 | **0.0%** |
| tds | 549 | 0 | **0.0%** |
| nitrate_n | 292 | 0 | **0.0%** |
| chloride | 546 | 0 | **0.0%** |
| alkalinity | 501 | 0 | **0.0%** |
| iron | 228 | 0 | **0.0%** |

Turbidity, TDS, nitrate, chloride, alkalinity and iron never co-occur with dissolved oxygen.
The station sets are disjoint: the 173 stations with turbidity readings have **no dissolved
oxygen reading at any time in 2021**. The export therefore contains two separate sampling
regimes — surface-water stations where DO and pH are measured, and bore-well/pond/municipal
points where turbidity, TDS and the ion balance are measured.

**Impact.** Turbidity and TDS are the parameters the project synopsis prioritises most, and they
cannot be used to predict a dissolved-oxygen-anchored label. They are excluded from the model.
The nitrate escalation rule is defined but never fires, because nitrate and DO never overlap.
This is recorded as the principal data limitation of the project.

## 5. Missingness

- 69,395 chemical cells and 30,944 physical cells were the placeholder `-` (CPCB's
  missing-value marker) and were converted to true nulls. Left as text, `-` makes a column
  appear complete.
- Fourteen chemical analyte columns are **100% empty** and were dropped: boron, carbonate,
  calcium, fluoride, bicarbonate, potassium, magnesium, sodium, nitrite+nitrate, sulphate,
  total phosphorus, manganese, nitrate (as NO3), copper.
- Four physical columns are 100% empty and were dropped: colour, odour, water temperature,
  total solids. **Water temperature is entirely absent from the dataset.**
- Retained analytes are sparse outside pH and DO: TDS 6.4% complete, ammonia 12.6%, nitrate
  3.4%, iron 2.7%, chloride 6.4%, alkalinity 5.9%.

## 6. Duplicates and conflicting records

- No exact duplicate rows in any file.
- 18 repeated (station, date) pairs in chemical and 18 in physical, concentrated in six Punjab
  stations. These are **conflicting duplicates**: the same station and date carry different
  values (e.g. DERA BASSI, 2021-04-08, EC = 2,721 and 1,221 µS/cm) and slightly different
  coordinates. They were averaged at station-month grain rather than deleted, and the conflict
  is logged.
- Grouping on station and month alone was necessary: keying on latitude/longitude split these
  conflicts into separate rows and produced 27 duplicate station-months in an early build.

## 7. Implausible values

Values outside generous physical bounds were flagged and set to null (never silently deleted):

| Parameter | Bound applied | Flagged |
|---|---|---|
| pH | 3.0 – 12.0 | 1 (a reading of 2.00 in Madhya Pradesh) |
| dissolved oxygen | 0 – 20 mg/L | 1 (28.0 mg/L in Maharashtra) |
| TDS | 0 – 10,000 mg/L | 1 (960,998 mg/L in Madhya Pradesh) |
| alkalinity | 0 – 3,000 mg/L as CaCO3 | 5 (incl. 590,620 and 44,383) |
| electrical conductivity | 0 – 60,000 µS/cm | 1 (15,001,625 µS/cm) |

The electrical-conductivity bound was deliberately set just above seawater (~50,000 µS/cm) so
that genuine brackish readings from Kerala backwaters (up to ~43,500 µS/cm) survive. Deleting
those would have discarded real estuarine measurements.

## 8. Geography and units

- `Village`, `Block`, `River`, `Basin`, `Tributary`, `Subtributary`, `SubSubtributary` and
  `Local River` are **100% the placeholder `-`** in every file and were dropped. Village/ward
  level analysis is not possible; district, station name and coordinates are what exist.
- Latitude and longitude are complete (100%) for all 1,096 stations.
- State and district names are internally consistent within each file (145 districts, no
  case-duplicates after upper-casing checks).
- Units follow CPCB conventions: mg/L for most analytes, mgN/L for nitrogen species,
  mg/L as CaCO3 for hardness and alkalinity, µS/cm for conductivity, NTU for turbidity.
  Nitrogen is reported as N, not as NO3 — the BIS nitrate limit of 45 mg/L as NO3 corresponds
  to 10.2 mg/L as N, and this conversion is applied explicitly in the target rule.

## 9. Integration validity

The chemical and physical files **can** be joined, and this was verified rather than assumed:
station names match exactly and observation dates align, giving (station, date) match counts of
621/621 (Gujarat), 256/256 (Jammu & Kashmir), 535/535 (Kerala), 828/828 (Maharashtra),
896/897 (Madhya Pradesh), 252/261 (Punjab) and 470/470 (West Bengal).

Integration is therefore valid at **station-month** grain, which is the modelling grain.
Two cautions are recorded:

1. Aggregating stations to district-months changes the unit of analysis from "a monitoring
   station" to "a district mean". District values are means of their stations, not
   district-wide measurements, and are used only for reporting, never as model rows.
2. NASA POWER values are reanalysis on a ~0.5° grid, so several stations share a grid cell
   (1,096 stations collapse to 255 cells). Rainfall is therefore a regional, modelled
   quantity, not a station-level gauge reading.

## 10. Cleaning decisions and their effect

| Step | Action | Rows affected |
|---|---|---|
| Load | chemical rows loaded | 8,674 |
| Load | `-`/empty placeholders → null | 69,395 cells |
| Load | 14 all-empty analyte columns dropped | 0 (columns) |
| Load | 8 placeholder-only geography columns dropped | 0 (columns) |
| Load | implausible pH → null | 1 |
| Load | implausible DO → null | 1 |
| Load | implausible TDS → null | 1 |
| Load | implausible alkalinity → null | 5 |
| Load | physical rows loaded | 3,868 |
| Load | physical placeholders → null | 30,944 cells |
| Load | 4 all-empty physical columns dropped | 0 (columns) |
| Load | implausible EC → null | 1 |
| Dedupe | repeated chemical station-date pairs found | 18 |
| Dedupe | repeated physical station-date pairs found | 18 |
| Aggregate | station-month rows created | 8,518 |

Final modelling set: **7,687 station-months** with a usable dissolved-oxygen label, 0 duplicate
station-months.

## 11. What is missing and what it would take

| Requirement | Status | What is needed |
|---|---|---|
| Aggregated community health / symptom data | **Absent** | IDSP/IHIP or district health office export |
| Turbidity, TDS for modelling | Present but unusable with DO | Sampling the same points for DO and turbidity |
| Water temperature | **Absent** | ESP32 sensor rig or lab records |
| Multi-year history | Absent | 3+ years of the same export |
| Village/ward geography | Absent | Different data source |
| Rainfall | Substituted | IMD gauge data would be preferable to reanalysis |
