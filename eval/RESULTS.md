# peakweek evaluation results

Every run is listed, including validation runs. Raw records: `eval/runs.jsonl` (one JSON line per run:
config, n, metrics, seconds, max RSS) and per-row probabilities in `eval/preds/<split>_<name>.csv`;
`python eval/summarize.py --split val|test` recomputes every table below from those files.

## Setup

- Task: for one observation (species, place, date), predict P(green), P(colored), P(bare) for its iNaturalist
  Leaves annotation. Data: `data/features.csv` (see `data/README.md`).
- Splits by year. **Validation**: train 2018-2023, evaluate 2024 (used to choose features, context strategy,
  n_estimators and baseline hyperparameters). **Test**: train 2018-2024, evaluate 2025, run once with the
  configuration locked in `eval/locked.json`.
- TabPFN: v2 classifier open weights only (`ModelVersion.V2`), CPU, run on a 4-core box under a hard 800 MB
  memory cap (`systemd-run -p MemoryMax=800M -p MemorySwapMax=0`), one job at a time, prediction in chunks
  of <= 200 rows, context <= 1000 rows.
- Metrics: multiclass log loss (primary), multiclass Brier (sum over classes, range 0-2), accuracy;
  per-species log loss; paired bootstrap (2000 resamples) of the per-row loss difference, resampling
  observations and, separately, grid-cell x ISO-week clusters.

## TabPFN speed and memory on the CPU box (synthetic data, peakweek shape)

`eval/bench_tabpfn.py` (13 features incl. 1 categorical, 3 classes), Dell 4-core CPU, 800 MB cap, one process per
configuration (`eval/bench.jsonl`). Cost is dominated by a fixed per-call pass over the context: predicting 1 row
costs about as much as predicting 200.

| context rows | n_estimators | predict 1 row (s) | 15 rows | 120 rows | 200 rows | max RSS (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 4 | 15.5 | 15.3 | 16.8 | 17.9 | 674 |
| 1000 | 8 | 30.2 | 30.3 | 33.6 | 35.9 | 675 |
| 500 | 4 | 6.4 | 6.7 | 7.7 | 8.6 | 579 |

(import of tabpfn: 3-5 s; `fit` itself < 1 s because TabPFN only stores the context.)

## Validation (train 2018-2023, n_train 2343; evaluate 2024, n 2582)

All candidates, sorted by log loss. Host `001` = the Mac (sklearn baselines), `ubuntu` = the Dell (TabPFN).
TabPFN names: `tabpfn_<feature set>_<context strategy>_c<context rows>_e<n_estimators>_s<contexts averaged>`.
Baseline grids: logistic regression C in {0.1, 1, 10}; HGB g0-g7 = learning_rate {0.03, 0.1} x max_leaf_nodes
{7, 15} x max_iter {50, 150} (min_samples_leaf 40, l2 1.0), g8 = early stopping; `*_full_no_prcp_*` = the same
grids without the precipitation feature. Feature sets: `full` = 13 features (species, lat, lon, elevation, day of
year, day length, chill degree days < 20 C since Sep 1, frost nights < 0 C and < 5 C, 7-day mean Tmin, 14-day mean
Tmean, 30-day precipitation, season temperature anomaly); `calendar` = species, lat, lon, elevation, day of year;
`core` = calendar + chill degree days, frost < 5 C, Tmin7, anomaly; `full_no_anom` = full without the anomaly.

| model | n | log loss | Brier | accuracy | s / 100 rows | max RSS MB | host |
|---|---:|---:|---:|---:|---:|---:|---|
| logreg_all_C1.0 | 2582 | 0.4305 | 0.2515 | 0.827 | 0.00 | 184.2 | 001 |
| tabpfn_full_stratified_c1000_e4_s3 | 2582 | 0.4307 | 0.2522 | 0.825 | 27.27 | 679.6 | ubuntu |
| logreg_all_C10.0 | 2582 | 0.4330 | 0.2526 | 0.824 | 0.00 | 184.2 | 001 |
| tabpfn_full_stratified_c1000_e4_s1 | 2582 | 0.4359 | 0.2548 | 0.826 | 9.74 | 673.0 | ubuntu |
| tabpfn_full_no_anom_stratified_c1000_e4_s1 | 2582 | 0.4389 | 0.2571 | 0.829 | 8.77 | 665.5 | ubuntu |
| tabpfn_full_stratified_c1000_e8_s1 | 2582 | 0.4401 | 0.2583 | 0.823 | 18.15 | 676.2 | ubuntu |
| logreg_full_no_prcp_C1.0 | 2582 | 0.4422 | 0.2575 | 0.824 | 0.00 | 182.3 | 001 |
| tabpfn_calendar_stratified_c1000_e4_s1 | 2582 | 0.4427 | 0.2596 | 0.825 | 3.87 | 554.4 | ubuntu |
| logreg_full_no_prcp_C10.0 | 2582 | 0.4461 | 0.2580 | 0.822 | 0.00 | 182.3 | 001 |
| tabpfn_core_stratified_c1000_e4_s1 | 2582 | 0.4465 | 0.2595 | 0.820 | 6.89 | 613.7 | ubuntu |
| logreg_all_C0.1 | 2582 | 0.4475 | 0.2617 | 0.825 | 0.00 | 184.2 | 001 |
| tabpfn_full_local_c1000_e4_s1_groups80 | 290 | 0.4548 | 0.2662 | 0.814 | 313.99 | 623.6 | ubuntu |
| logreg_full_no_prcp_C0.1 | 2582 | 0.4564 | 0.2669 | 0.822 | 0.00 | 182.3 | 001 |
| tabpfn_full_stratified_c1000_e4_s1_groups80 | 290 | 0.4570 | 0.2754 | 0.810 | 13.07 | 673.8 | ubuntu |
| tabpfn_full_per_species_smoothed_c1000_e4_s1 | 2582 | 0.4612 | 0.2643 | 0.813 | 4.23 | 647.8 | ubuntu |
| logreg_calendar_C10.0 | 2582 | 0.4623 | 0.2708 | 0.805 | 0.00 | 184.2 | 001 |
| logreg_calendar_C1.0 | 2582 | 0.4645 | 0.2730 | 0.801 | 0.00 | 184.2 | 001 |
| hgb_g4 | 2582 | 0.4754 | 0.2824 | 0.796 | 0.01 | 187.1 | 001 |
| hgb_g1 | 2582 | 0.4792 | 0.2854 | 0.796 | 0.03 | 187.1 | 001 |
| hgb_full_no_prcp_g1 | 2582 | 0.4820 | 0.2841 | 0.801 | 0.09 | 184.1 | 001 |
| hgb_g3 | 2582 | 0.4833 | 0.2861 | 0.802 | 0.05 | 187.1 | 001 |
| hgb_full_no_prcp_g4 | 2582 | 0.4857 | 0.2851 | 0.801 | 0.03 | 184.1 | 001 |
| logreg_calendar_C0.1 | 2582 | 0.4881 | 0.2898 | 0.792 | 0.00 | 184.0 | 001 |
| hgb_g2 | 2582 | 0.4885 | 0.2886 | 0.794 | 0.02 | 187.1 | 001 |
| hgb_g0 | 2582 | 0.4896 | 0.2861 | 0.797 | 0.01 | 186.3 | 001 |
| hgb_full_no_prcp_g3 | 2582 | 0.4909 | 0.2866 | 0.801 | 0.23 | 184.1 | 001 |
| hgb_g6 | 2582 | 0.4947 | 0.2903 | 0.804 | 0.01 | 187.1 | 001 |
| hgb_full_no_prcp_g6 | 2582 | 0.4962 | 0.2884 | 0.802 | 0.02 | 184.1 | 001 |
| hgb_full_no_prcp_g0 | 2582 | 0.4967 | 0.2888 | 0.797 | 0.02 | 184.1 | 001 |
| hgb_full_no_prcp_g2 | 2582 | 0.4998 | 0.2929 | 0.794 | 0.10 | 184.1 | 001 |
| hgb_g8 | 2582 | 0.5033 | 0.2959 | 0.796 | 0.01 | 187.1 | 001 |
| hgb_full_no_prcp_g8 | 2582 | 0.5048 | 0.2952 | 0.792 | 0.04 | 184.1 | 001 |
| hgb_g5 | 2582 | 0.5068 | 0.2905 | 0.801 | 0.03 | 187.1 | 001 |
| hgb_full_no_prcp_g5 | 2582 | 0.5235 | 0.2948 | 0.799 | 0.11 | 184.1 | 001 |
| climatology | 2582 | 0.5608 | 0.3201 | 0.779 | 0.00 | 182.4 | 001 |
| hgb_full_no_prcp_g7 | 2582 | 0.5651 | 0.3002 | 0.801 | 0.06 | 184.1 | 001 |
| tabpfn_calendar_per_species_c1000_e4_s1 | 2582 | 0.5709 | 0.2609 | 0.825 | 1.82 | 545.4 | ubuntu |
| hgb_g7 | 2582 | 0.5735 | 0.3125 | 0.796 | 0.04 | 187.1 | 001 |
| tabpfn_full_per_species_c1000_e4_s1 | 2582 | 0.5766 | 0.2644 | 0.813 | 4.39 | 647.4 | ubuntu |
| hgb_calendar_default | 2582 | 0.6692 | 0.3074 | 0.802 | 0.04 | 187.1 | 001 |
| hgb_default | 2582 | 0.7339 | 0.3327 | 0.789 | 0.09 | 187.1 | 001 |

Notes on the TabPFN rows:
- `per_species` (one context per species, all its training rows up to 1000) gives near-zero probability to a class
  that a species' context lacks, which explodes log loss (0.577); add-one mass for missing classes
  (`_smoothed`, 0.461) fixes the blow-up but it is still clearly worse than one shared stratified context.
- `local` (k nearest training rows in lat / lon / week, per species) was too slow to run on all 2582 rows
  (314 s per 100 rows: one context per (cell, week) group), so it was compared with stratified on the same
  random 80 (cell, week) groups (290 rows), below. It is not significantly better (-0.002 log loss,
  CI [-0.028, +0.022]) and is 24x slower.

### Paired comparisons used for the decision

TabPFN averaging 3 stratified contexts (s3) vs 1 (s1), all 2582 rows:

Paired bootstrap, tabpfn_full_stratified_c1000_e4_s3 minus tabpfn_full_stratified_c1000_e4_s1 (negative = tabpfn_full_stratified_c1000_e4_s3 better), 2000 resamples:

| metric | resampling unit | units | mean diff | 95% CI | P(diff<0) |
|---|---|---:|---:|---|---:|
| log_loss | rows | 2582 | -0.0051 | [-0.0109, +0.0001] | 0.970 |
| log_loss | cell_x_week | 604 | -0.0051 | [-0.0112, +0.0007] | 0.959 |
| brier | rows | 2582 | -0.0026 | [-0.0056, +0.0006] | 0.950 |
| brier | cell_x_week | 604 | -0.0026 | [-0.0058, +0.0007] | 0.939 |

`local` vs `stratified` on the 80-group subset (290 rows):

| model | n | log loss | Brier | accuracy | s / 100 rows | max RSS MB | host |
|---|---:|---:|---:|---:|---:|---:|---|
| tabpfn_full_local_c1000_e4_s1_groups80 | 290 | 0.4548 | 0.2662 | 0.814 | 313.99 | 623.6 | ubuntu |
| tabpfn_full_stratified_c1000_e4_s1_groups80 | 290 | 0.4570 | 0.2754 | 0.810 | 13.07 | 673.8 | ubuntu |
| logreg_all_C1.0 | 290 | 0.4148 | 0.2524 | 0.807 | 0.00 | 184.2 | 001 |

Paired bootstrap, tabpfn_full_local_c1000_e4_s1_groups80 minus tabpfn_full_stratified_c1000_e4_s1_groups80 (negative = tabpfn_full_local_c1000_e4_s1_groups80 better), 2000 resamples:

| metric | resampling unit | units | mean diff | 95% CI | P(diff<0) |
|---|---|---:|---:|---|---:|
| log_loss | rows | 290 | -0.0022 | [-0.0275, +0.0217] | 0.572 |
| log_loss | cell_x_week | 80 | -0.0022 | [-0.0314, +0.0326] | 0.560 |
| brier | rows | 290 | -0.0092 | [-0.0242, +0.0055] | 0.888 |
| brier | cell_x_week | 80 | -0.0092 | [-0.0263, +0.0084] | 0.854 |

TabPFN (locked config) vs the best baseline on validation:

Paired bootstrap, tabpfn_full_stratified_c1000_e4_s3 minus logreg_all_C1.0 (negative = tabpfn_full_stratified_c1000_e4_s3 better), 2000 resamples:

| metric | resampling unit | units | mean diff | 95% CI | P(diff<0) |
|---|---|---:|---:|---|---:|
| log_loss | rows | 2582 | +0.0002 | [-0.0084, +0.0093] | 0.460 |
| log_loss | cell_x_week | 604 | +0.0002 | [-0.0100, +0.0101] | 0.475 |
| brier | rows | 2582 | +0.0007 | [-0.0040, +0.0057] | 0.362 |
| brier | cell_x_week | 604 | +0.0007 | [-0.0055, +0.0065] | 0.391 |

## Locked configuration (written to `eval/locked.json` at 2026-10-06 14:33 EDT, before any 2025 run)

Rule, written down while the last validation batch was still running (before the s3, e8, no-anomaly and local
results were in): lowest validation log loss, ties broken by speed.

- **TabPFN**: `full` features, one shared **stratified** context of **1000** rows (species x label strata, floor
  8 rows each, deterministic per seed), **n_estimators 4**, **3 contexts averaged** (seeds 0, 1, 2), CPU, v2
  weights. Validation log loss 0.4307 (best TabPFN; s1 0.4359, e8 0.4401, no anomaly 0.4389, core 0.4465,
  calendar 0.4427). s3 beats s1 with P = 0.97 (row bootstrap) so it is not a tie under the stated rule. Cost: three
  context passes per forecast instead of one (about 3x the time, see the forecast timings below).
- **Baselines**: logistic regression on calendar features C = 10 (0.4623), logistic regression on all features
  C = 1 (0.4305), HGB g4 = learning_rate 0.1, max_leaf_nodes 7, max_iter 50, min_samples_leaf 40, l2 1.0
  (0.4754), climatology (0.5608). **Best baseline: logreg_all** (C = 1).
- Ablation for the test: the same TabPFN configuration on the `calendar` features only.
- On validation the locked TabPFN and logreg_all are indistinguishable (difference +0.0002, 95% CI
  [-0.0084, +0.0093]).

## Test: 2025, run once (train 2018-2024, n_train 4925; evaluate 2025, n 1725)

Run at 2026-10-06 14:34 (baselines, Mac) and 14:35-14:46 EDT (TabPFN, Dell) with `eval/locked.json` unchanged.
2025 labels: colored 862, green 789, bare 74. `tabpfn_calendar` is the ablation (same TabPFN configuration,
features species + lat + lon + elevation + day of year only). Names without a suffix are the locked
configurations (`logreg_all` = C 1, `logreg_calendar` = C 10, `hgb` = g4).

| model | n | log loss | Brier | accuracy | s / 100 rows | max RSS MB | host |
|---|---:|---:|---:|---:|---:|---:|---|
| tabpfn | 1725 | 0.4979 | 0.2850 | 0.805 | 27.61 | 679.3 | ubuntu |
| tabpfn_calendar | 1725 | 0.5163 | 0.2906 | 0.800 | 12.08 | 679.3 | ubuntu |
| logreg_all | 1725 | 0.5186 | 0.2889 | 0.805 | 0.00 | 183.0 | 001 |
| hgb | 1725 | 0.5048 | 0.2910 | 0.802 | 0.01 | 184.7 | 001 |
| logreg_calendar | 1725 | 0.5517 | 0.3067 | 0.795 | 0.00 | 183.0 | 001 |
| climatology | 1725 | 0.5815 | 0.3453 | 0.752 | 0.00 | 181.1 | 001 |

Per-species log loss:

| species | n | tabpfn | tabpfn_calendar | logreg_all | hgb | logreg_calendar | climatology |
|---|---:|---:|---:|---:|---:|---:|---:|
| red maple | 470 | 0.523 | 0.531 | 0.534 | 0.552 | 0.569 | 0.657 |
| sugar maple | 174 | 0.456 | 0.470 | 0.486 | 0.512 | 0.471 | 0.616 |
| sweetgum | 61 | 0.595 | 0.646 | 0.614 | 0.550 | 0.628 | 0.720 |
| northern red oak | 164 | 0.389 | 0.413 | 0.369 | 0.376 | 0.402 | 0.494 |
| American beech | 184 | 0.553 | 0.569 | 0.641 | 0.586 | 0.700 | 0.607 |
| black gum | 45 | 0.719 | 0.807 | 0.644 | 0.656 | 0.837 | 0.769 |
| sassafras | 128 | 0.779 | 0.859 | 0.896 | 0.713 | 0.934 | 0.728 |
| Norway maple | 499 | 0.401 | 0.403 | 0.400 | 0.398 | 0.425 | 0.446 |

**Against the pre-registered best baseline (logreg_all):**

Paired bootstrap, tabpfn minus logreg_all (negative = tabpfn better), 2000 resamples:

| metric | resampling unit | units | mean diff | 95% CI | P(diff<0) |
|---|---|---:|---:|---|---:|
| log_loss | rows | 1725 | -0.0207 | [-0.0351, -0.0073] | 1.000 |
| log_loss | cell_x_week | 504 | -0.0207 | [-0.0407, -0.0032] | 0.994 |
| brier | rows | 1725 | -0.0039 | [-0.0097, +0.0018] | 0.900 |
| brier | cell_x_week | 504 | -0.0039 | [-0.0125, +0.0047] | 0.802 |

Against HGB, which has the lowest 2025 log loss among the baselines (a post-hoc comparison, reported for honesty):

Paired bootstrap, tabpfn minus hgb (negative = tabpfn better), 2000 resamples:

| metric | resampling unit | units | mean diff | 95% CI | P(diff<0) |
|---|---|---:|---:|---|---:|
| log_loss | rows | 1725 | -0.0068 | [-0.0201, +0.0068] | 0.830 |
| log_loss | cell_x_week | 504 | -0.0068 | [-0.0278, +0.0122] | 0.731 |
| brier | rows | 1725 | -0.0059 | [-0.0132, +0.0013] | 0.939 |
| brier | cell_x_week | 504 | -0.0059 | [-0.0189, +0.0057] | 0.819 |

Ablation, full features minus calendar-only features (both TabPFN, same configuration):

Paired bootstrap, tabpfn minus tabpfn_calendar (negative = tabpfn better), 2000 resamples:

| metric | resampling unit | units | mean diff | 95% CI | P(diff<0) |
|---|---|---:|---:|---|---:|
| log_loss | rows | 1725 | -0.0184 | [-0.0256, -0.0111] | 1.000 |
| log_loss | cell_x_week | 504 | -0.0184 | [-0.0282, -0.0082] | 1.000 |
| brier | rows | 1725 | -0.0056 | [-0.0102, -0.0011] | 0.993 |
| brier | cell_x_week | 504 | -0.0056 | [-0.0117, +0.0010] | 0.959 |

Reading: on 2025 the locked TabPFN has the lowest log loss and Brier of all models. It beats the pre-registered
best baseline on log loss (-0.021, CI excludes 0 for both resampling units) but not clearly on Brier. Against
HGB the difference is not significant. The weather features help TabPFN (-0.018 log loss vs the calendar-only
ablation, CI excludes 0). All models are worse on 2025 than on 2024 (e.g. logreg_all 0.431 -> 0.519), so year to
year shift is larger than the gaps between the good models. Per species, TabPFN is not best everywhere:
logistic regression or HGB are better for northern red oak, black gum, sweetgum and sassafras (n 45-164;
on sassafras even climatology is better), Norway maple is a tie, and TabPFN is best for red maple, sugar maple
and American beech.
Accuracy is about 0.80 for every model except climatology; the gains are in calibration, not in the top class.

## Live check: 2026 observations from the last 10 days (`eval/live_check.py`)

Run 2026-10-06 (Dell, 14:50 EDT). Observations of the 8 species observed 2026-09-26..2026-10-05 and annotated by
then, with the training data's quality, coordinate and annotation filters (the license filter is not applied:
these rows are only scored, never used for training): **n = 99** (green 77, colored 22, bare 0). Features are built exactly as
in the app (ERA5 archive Sep 1 to yesterday, forecast API for the rest; all-years climatology; point
elevation). Every model is trained on all of 2018-2025 (6650 rows) with the locked settings.

| model | n | log loss | Brier | accuracy |
|---|---:|---:|---:|---:|
| logreg_all | 99 | 0.4360 | 0.2775 | 0.798 |
| tabpfn (locked) | 99 | 0.4539 | 0.2877 | 0.788 |
| hgb | 99 | 0.4653 | 0.3051 | 0.778 |
| logreg_calendar | 99 | 0.4917 | 0.3222 | 0.798 |
| climatology | 99 | 0.5826 | 0.3718 | 0.727 |

TabPFN minus logreg_all, log loss: +0.018, 95% row-bootstrap CI [-0.011, +0.046] (P(TabPFN better) = 0.12).
Too few rows to rank the models; it is a sanity check that the app pipeline gives sensible probabilities on
fresh data (TabPFN log loss 0.454 here vs 0.498 on the 2025 test). To re-run on a later date: run
`eval/live_check.py --stage prepare --today YYYY-MM-DD` on a machine with network access (fetches observations
and weather into `.cache/` and `eval/live/`), then `eval/live_check.py --stage predict --today YYYY-MM-DD` on the
machine that runs TabPFN (it can run with `PEAKWEEK_OFFLINE=1` once the caches are copied over). Row-level files
(`eval/live/rows_*.csv`, `preds_*.csv`) are not committed: they include observations without an open license.

iNaturalist annotations accrue for weeks, so later runs will have more rows.

## Forecast timing and memory (`scripts/make_examples.py`, Dell, 2026-10-06)

`forecast()` with the locked configuration, weather already cached (the Dell runs with `PEAKWEEK_OFFLINE=1`),
one process for the three places, 800 MB cgroup cap:

| place | wall time (s) | model time (s) | max RSS so far (MB) |
|---|---:|---:|---:|
| New Brunswick, NJ (40.4862, -74.4518) | 56.9 | 56.6 | 665 |
| Burlington, VT (44.4759, -73.2121) | 51.0 | 50.8 | 666 |
| Pittsburgh, PA (40.4406, -79.9959) | 51.0 | 50.8 | 666 |

Cost is three TabPFN context passes (3 averaged contexts x 120 query rows, ~17 s each); the stratified
context does not depend on the location. The first call is ~6 s slower (tabpfn import and weight loading).
systemd reported a cgroup memory peak of 254 MB, below the process RSS (probably because shared libraries
were already in the page cache and charged elsewhere); the RSS column is the conservative number.
A cold forecast (new grid cell or new day) also costs Open-Meteo weighted calls: one forecast-API call
(108 days x 3 variables = weight 7.7), one archive call (Sep 1 to yesterday; weight 2.5 on 2026-10-06, growing
by 1 every 14 days) and one elevation call (weight 1); everything is cached by date.

## App weather vs training weather (`eval/weather_shift.py`)

Training features come from the ERA5 archive. For Sep 2026, at the five example and training weather points,
the forecast API's `past_days` values differed from ERA5 by (forecast minus ERA5, mean over points):
Tmean -0.05 C, Tmin -0.44 C, precipitation -2.3 mm/day (mean absolute difference 3.3 mm/day). The forecast
API also returned values for only the last ~50 of the 92 requested past days. So `forecast()` takes Sep 1 to
yesterday from the archive and only today onward from the forecast API (`weather.app_daily`).

## Caveats

- Labels are iNaturalist annotations: what people chose to photograph and label. Bare trees are rare (347 of
  6650 rows; 0 of 99 in the live check), so P(bare) is small even late in the window and means "an annotated
  photo of this species would show no live leaves", not "the trees are bare".
- Observations with conflicting leaf annotations (for example green and colored on the same tree, common in
  transition) were dropped, which removes some of the most ambiguous cases from both training and test.
- Data volume rises steeply over time (2018: 53 rows, 2023: 752, 2024: 2582, 2025: 1725), so validation trained
  on only 2343 rows and the earlier years carry little weight.
- Weather is per 1-degree grid cell (weather at the centroid of that cell's training observations), not per point.
- Training elevations (and the live check's) are SRTM 90 m from OpenTopoData; `forecast()` uses Open-Meteo's
  elevation API (Copernicus GLO-90 DEM) for the query point. The two DEMs were not compared systematically.
- Accuracy is about 0.80 for every decent model; the differences are in calibrated probabilities.
