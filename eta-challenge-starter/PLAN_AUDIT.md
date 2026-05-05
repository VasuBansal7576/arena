# Plan Audit

This file cross-checks the final ETA submission against the plan we agreed to
before building. It is intentionally blunt: implemented means the shipped code
uses it, partial means the idea exists but is narrower than the original plan,
and changed means the AutoResearch loop found a better variant than the plan.

## Data & Cleaning

| Planned item | Status | Evidence / note |
|---|---|---|
| Pull raw TLC data and extract `trip_distance`, `RatecodeID` | Implemented | `data/download_data.py` retains `trip_distance` and `ratecode_id` for training priors. |
| Drop passenger count from model features | Implemented | `passenger_count` is only used as a cleaning filter; it is not in `FEATURE_NAMES`. |
| Winsorize target at p99.5 per route class | Changed | Implemented and tested, but AutoResearch found `target_cap_quantile=1.0` scored better, so final model does not cap the target. |
| Flag/separate same-zone trips | Implemented | Same-zone route class plus dedicated same-zone model path. |

## Feature Stack

| Planned item | Status | Evidence / note |
|---|---|---|
| Cyclical time encodings | Implemented | Hour, day-of-week, day-of-year, and quarter-hour sin/cos in `features.py`. |
| Holiday/event flags | Implemented | Federal holidays, holiday eves, December period, NYE/New Year. |
| Observed median distance per zone pair | Implemented | Built from `trip_distance` into shrinked route priors. |
| Historical trip density per 15-minute window | Implemented | Pickup/dropoff density by zone, DOW, quarter-hour. |
| Ratecode-derived priors only | Implemented | Pair-level JFK/Newark/negotiated probabilities; no direct inference-time rate code. |
| CBD and airport flags | Implemented | Static structural zone flags in `features.py`. |
| Zone adjacency neighbor aggregates | Partial | Implemented as nearest-centroid neighbor aggregates from the TLC shapefile, not polygon-border adjacency. |
| Speed regime clusters | Implemented | KMeans on zone-pair hourly speed profiles, used for cluster-hour prior and shrinkage. |
| Empirical Bayes shrinkage toward cluster mean | Implemented | Pair priors shrink toward cluster/route-hour parent estimates. |

## Modeling

| Planned item | Status | Evidence / note |
|---|---|---|
| LightGBM quantile alpha=0.5 | Changed | Used scikit-learn HistGradientBoosting. Quantile/absolute-error were tested, but squared-error won Dev MAE after ablation. |
| Recency-weighted samples | Implemented | Final promoted model uses half-life 90 days, floor 0.30. |
| Separate same-zone model | Implemented | Kept; disabling it was slightly worse in the research log. |
| Ablation table | Implemented | `research_log.csv` records named experiments, Dev MAE, promotion status, and args. |

## Diagnostics

| Planned item | Status | Evidence / note |
|---|---|---|
| Stratified MAE table | Implemented | `metrics.json` and README include segment MAE. |
| Residual analysis | Implemented | `metrics.json` records top residual groups by hour, DOW, route class, pickup zone, and dropoff zone. |

## AutoResearch Loop

| Planned item | Status | Evidence / note |
|---|---|---|
| Metric-driven experiment loop | Implemented after audit | `autoresearch.py` runs named experiments, parses `dev_mae`, writes `research_log.csv`, and promotes only better artifacts. |
| One experiment per meaningful change | Partial | Experiments are logged in a ledger. I did not create one git commit per experiment because several runs were intentionally grouped, but the ledger preserves the trajectory. |
| Keep only if better | Implemented | Promoted only when full Dev and the late time-holdout improved after the holdout gate was added. Rejected target encoding, variance features, and median residual tables despite plausible hypotheses. |

Final promoted experiment: `fine_affine_calibration_route_pruned_1m_500_lr04`.

Follow-up AutoResearch pass on 2026-05-05 tested 60-day and 120-day recency
half-lives against the final setup. Both were worse than the promoted 90-day
model, so no artifact was promoted.

A second AutoResearch pass exposed sample seed and tree hyperparameters to the
runner. The slower 500-iteration, 0.04-learning-rate model improved full Dev MAE
from 250.8 to 250.3 and was promoted.

Route-class specialist models were implemented and tested. They improved full
Dev MAE to 250.1, but worsened the later time-holdout from 259.10 to 259.18, so
the holdout gate rejected them for the final shipped model.

A pruned specialist pass kept only route classes whose specialist blend also
improved the later time-holdout. That retained the Manhattan-to/from-outer
specialist at a 0.2 blend, improved full Dev MAE to 250.27, and improved the
holdout from 259.10 to 259.07, so it was promoted.

Target-encoding and duration-variance follow-ups were tested after that. Both
worsened the late time-holdout, so they remain documented as rejected
experiments instead of being blended into the shipped model.

The final AutoResearch pass switched from feature stacking to metric-aware
post-model calibration. Median residual tables selected zero weight everywhere,
but affine route/hour/day/dropoff calibration improved full Dev MAE from 250.27
to 245.50 and the late time-holdout from 259.07 to 253.72, so it was promoted.

A final fine-calibration pass added higher-resolution route-hour, day-hour,
dropoff-hour, airport-hour, and route-dropoff interactions. It improved full Dev
MAE again to 244.13 and the late time-holdout to 252.74. This is the most
overfit-sensitive layer, so it is documented explicitly rather than hidden as a
generic model gain.
