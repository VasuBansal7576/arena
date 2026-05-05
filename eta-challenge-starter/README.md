# Gobblecube ETA Challenge Submission

This repo builds a Dockerized ETA predictor for the Gobblecube ride-hailing
take-home. The grader calls:

```python
predict({
    "pickup_zone": int,
    "dropoff_zone": int,
    "requested_at": "YYYY-MM-DDTHH:MM:SS",
    "passenger_count": int,
}) -> float  # duration_seconds
```

No network or external data files are used at inference time. Everything needed
by `predict.py` is stored in `model.pkl`.

## Final Dev Score

- Full local Dev MAE from `train.py`: **244.1 seconds**
- Late time-holdout MAE (`requested_at >= 2023-12-25`): **252.7 seconds**
- Docker image size: **992MB**

Starter reference from the challenge README: naive GBT baseline is about
`351s` Dev / `367s` Eval, and a simple zone-pair lookup is about `300s` Dev.

## Approach

The core model treats ETA as a structured tabular problem rather than a raw
zone-id regression. I build full-year 2023 priors for route duration, observed
distance, speed, demand density, fare-regime likelihood, and route class. I
started with quantile loss because Gobblecube scores MAE, but the ablation loop
found that a squared-error `HistGradientBoostingRegressor` on the cleaned
feature stack scored better on Dev. The shipped model uses the measured winner,
not the initial theory.

Main pieces:

- **Measured objective choice:** quantile, absolute-error, and squared-error
  losses were all tested; squared-error won on Dev for the final feature stack.
- **Empirical Bayes route priors:** zone-pair and zone-pair-hour medians shrink
  toward speed-regime cluster means, not the global average.
- **Observed distance/speed:** `trip_distance` and `RatecodeID` are used only
  during training to build compact priors; `predict()` never requires them.
- **Cyclical time features:** hour, day-of-week, day-of-year, and 15-minute
  bucket encoded with sin/cos, plus weekend/rush/late-night flags.
- **Calendar structure:** federal holidays, holiday eves, December holiday
  period, and NYE/New Year period.
- **NYC route structure:** airport flags, Manhattan/CBD flags, route-class
  flags, and a separate same-zone model.
- **Demand proxy:** historical pickup/dropoff trip density by zone, day, and
  15-minute bucket.
- **Geographic borrowing:** nearest-zone aggregate features from the TLC taxi
  zone shapefile.
- **Recency weighting:** late-2023 rows receive more weight than early-2023
  rows, while older rows still help rare routes.
- **MAE-aware affine calibration:** after the model path is selected, small
  route/hour/day/dropoff and higher-resolution interaction adjustments are
  fitted on the pre-holdout Dev slice and kept only when they also improve the
  later time-holdout.
- **Target cleaning tested, not assumed:** p99.5 route-class winsorization was
  implemented and tested; the final model keeps the raw cleaned target because
  the no-cap experiment won Dev MAE.
- **Passenger count skepticism:** passenger count is retained for cleaning but
  deliberately excluded from model features because it is driver-entered and
  default-heavy.

The final prediction uses the squared-error model, a lightly blended same-zone
path, a pruned Manhattan-to/from-outer specialist, and the metric-aware affine
calibration rules selected by the AutoResearch loop.

## Ablations

Measured on full local Dev inside `train.py`:

| Model / prior | MAE |
|---|---:|
| Pair median prior | 299.9s |
| Distance / speed physics prior | 300.4s |
| Pair-hour shrinkage prior | 273.8s |
| Initial 3M-row quantile blend | 253.8s |
| 1M-row quantile control | 252.1s |
| 1M-row squared-error | 251.5s |
| 1M-row squared-error, no target cap | 251.2s |
| 1M-row squared-error, no target cap, 340 iters | 250.8s |
| Same final setup, 60-day recency half-life | 251.9s |
| Same final setup, 120-day recency half-life | 252.1s |
| 1M-row squared-error, no cap, 420 iters, lr 0.045 | 250.3s |
| 1M-row squared-error, no cap, 500 iters, lr 0.04 | 250.3s |
| Route-class specialists, unpruned | 250.1s full Dev, rejected by time-holdout |
| Pruned route-class specialist | 250.3s (`250.27s`) |
| Target-encoded pair means | 250.3s full Dev, rejected by time-holdout |
| Duration variance features | 250.8s, rejected |
| Affine-calibrated route model | 245.5s |
| Final fine affine-calibrated route model | **244.1s** |

The AutoResearch-inspired loop is described in `program.md` and implemented as
the metric-gated harness in `autoresearch.py`, with results in
`research_log.csv` and per-run JSON files in `research_runs/`. A larger 2M-row
variant and a 6M-row earlier run both scored worse than the final 1M-row model,
so the smaller model is intentional.

## AutoResearch Loop

I used an Andrej Karpathy-inspired AutoResearch loop: keep a written research
program, propose one measurable change at a time, run it under a fixed local
budget, and promote only when the metric improves. In this repo,
`program.md` is the research brief, `autoresearch_agent.py` is the
program-driven loop, and `autoresearch.py` is the deterministic promotion gate.

The important part was not asking an LLM for more random features. The useful
shift was changing the search space when the obvious path stalled:

| AutoResearch step | Decision | Result / reason |
|---|---|---|
| Objective search: quantile, absolute-error, squared-error | Passed | Squared-error beat the seemingly MAE-aligned losses on this feature stack. |
| Target-cap search | Passed | Removing p99.5 target capping improved Dev MAE. |
| Longer 1M-row learner, lr 0.04 | Passed | Reduced underfit without needing a larger/slower training sample. |
| Route-class specialists | Passed after pruning | Unpruned specialists overfit the time split; holdout pruning kept only the useful Manhattan-to/from-outer path. |
| Target encoding and variance features | Rejected | Plausible feature ideas, but they worsened the late time-holdout. |
| Median residual calibration | Rejected | Every correction alpha selected zero, suggesting the model was already close to additive median calibration. |
| Fine affine calibration | Passed | The biggest late gain came from metric-aware route/hour/day/dropoff calibration, improving full Dev to 244.1s and the late holdout to 252.7s. |

The tradeoff is honesty versus raw score chasing. The final affine calibration
layer is deliberately transparent and small enough to inspect, but it is also
the most Dev-sensitive part of the system. I kept it because it improved both
the tuning slice and the later time-holdout; the next improvement would need a
stronger nested time split before I would trust more calibration rules.

## Diagnostics

Segmented MAE from the selected model:

| Segment | MAE |
|---|---:|
| Overall | 244.1s |
| Same-zone | 209.0s |
| Manhattan internal | 211.8s |
| Airport route | 402.6s |
| Manhattan to/from outer borough | 398.4s |
| Outer-to-outer | 534.9s |
| Rush hour | 257.3s |
| Late night | 190.5s |

Residual analysis shows remaining error is concentrated in afternoon peak
hours, airport routes, outer-borough routes, and dropoffs into zone `265`
("Unknown"). The next useful work would target those segments rather than
blindly adding more global features.

## What Did Not Work

- Training on a larger 6M-row weighted sample worsened Dev MAE, likely because
  the tree started fitting older/noisier regimes instead of the cleaner
  recency-weighted signal.
- The metric argument for quantile/absolute-error loss was directionally
  sensible but empirically wrong here. Squared-error scored better once the
  target, priors, and features were in place.
- p99.5 target winsorization also lost to training on the raw cleaned target.
- Recency half-life sweeps at 60 and 120 days both lost to the final 90-day
  setting, so the current weighting is not just a default left untouched.
- Unpruned route-class specialist models improved full Dev to 250.1s but
  worsened the later time-holdout, so I rejected that headline number. A pruned
  version kept only the Manhattan-to/from-outer specialist, improving both full
  Dev and the later holdout, and that is the shipped model.
- Target-encoded pair means and duration variance features were also tested.
  Both made temporal robustness worse, so they are kept in the research trail
  but disabled for the shipped model.
- Median residual tables were a sensible first calibration attempt, but every
  alpha was selected as zero. The successful version was affine segment
  calibration, which can correct scale and offset together.
- Fine affine calibration improved further, but it is the most Dev-sensitive
  piece of the submission. I kept it because it also improved the later
  time-holdout, not only the tuning slice.
- A pure distance/speed physics prior was not enough by itself. Observed
  distance is useful as a feature, but route duration still needs historical
  priors and a flexible model.
- Treating all same-zone trips like normal route pairs produced unstable
  distance/speed behavior, so same-zone examples use a dedicated model path.

## AI Tooling

I used Codex as an implementation partner for repo reading, feature-pipeline
construction, verification, and the ablation loop. The useful loop was inspired
by Karpathy's AutoResearch pattern: write the research brief in `program.md`,
propose one modeling change, encode it in `train.py` or feature code, run
`autoresearch_agent.py`, compare Dev MAE, and promote only if the metric
improved. `autoresearch_agent.py` is the program-driven loop; `autoresearch.py`
is the lower-level experiment runner and promotion gate.
The loop overturned two plausible assumptions: quantile loss and winsorization
both sounded right, but squared-error with no target cap scored better. The
largest late-stage gain came from changing the search space from feature
stacking to MAE-aware calibration.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Downloads public 2023 TLC yellow taxi parquet files and builds train/dev.
python data/download_data.py

# Reproduce the final promoted model.pkl and metrics.json.
python train.py \
  --experiment-name fine_affine_calibration_route_pruned_1m_500_lr04 \
  --sample-n 1000000 \
  --max-iter 500 \
  --learning-rate 0.04 \
  --loss squared_error \
  --target-cap-quantile 1.0 \
  --disable-feature-group target_encoding \
  --disable-feature-group variance \
  --route-class-models \
  --route-class-holdout-prune \
  --route-class-max-iter 260 \
  --route-class-sample-n 400000 \
  --affine-calibration \
  --fine-affine-calibration \
  --calibration-holdout-prune

# Optional: rerun the recorded ablation loop.
python autoresearch.py --promote

# Local contract and scoring checks.
python -m pytest tests/
python grade.py

# Docker grader pathway.
docker build -t gobblecube-eta-submission .
docker run --rm --network=none -v $(pwd)/data:/work \
  gobblecube-eta-submission /work/dev.parquet /work/preds.csv
```

## External Data

Only public NYC TLC data is used:

- 2023 NYC Yellow Taxi trip records, downloaded by `data/download_data.py`
- NYC TLC taxi zone lookup CSV
- NYC TLC taxi zone shapefile for nearest-zone geographic borrowing

No external API calls are made at inference time.

Total active build time: about one focused work session.

*Submit your repo URL and LinkedIn profile to agentic-hiring@gobblecube.ai. Questions welcome at the same address.*
