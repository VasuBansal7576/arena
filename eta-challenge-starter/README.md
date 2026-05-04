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

- `python grade.py` 50k Dev sample MAE: **257.4 seconds**
- Full local Dev MAE from `train.py`: **253.8 seconds**
- Docker 50k run: **257.4 seconds**, **439s wall time**, image size **966MB**

Starter reference from the challenge README: naive GBT baseline is about
`351s` Dev / `367s` Eval, and a simple zone-pair lookup is about `300s` Dev.

## Approach

The core model treats ETA as a structured tabular problem rather than a raw
zone-id regression. I build full-year 2023 priors for route duration, observed
distance, speed, demand density, fare-regime likelihood, and route class, then
train a scikit-learn `HistGradientBoostingRegressor` with quantile loss
(`quantile=0.5`) because Gobblecube scores MAE and the conditional median is
the right target for absolute error.

Main pieces:

- **MAE-aligned objective:** quantile/median regression instead of default MSE.
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
- **Training target winsorization:** duration is capped at p99.5 per route
  class during training to reduce meter-left-running style label noise.
- **Passenger count skepticism:** passenger count is retained for cleaning but
  deliberately excluded from model features because it is driver-entered and
  default-heavy.

The final prediction is a small blend of the quantile model and the
zone-pair-hour prior. The blend weight is selected on Dev; the current model
uses `0.90 * model + 0.10 * prior`.

## Ablations

Measured on full local Dev inside `train.py`:

| Model / prior | MAE |
|---|---:|
| Pair median prior | 299.9s |
| Distance / speed physics prior | 300.4s |
| Pair-hour shrinkage prior | 273.8s |
| Quantile model only | 254.0s |
| Final blend | **253.8s** |

One larger training run with 6M weighted rows and 520 iterations scored
`255.1s`, worse than the selected 3M-row / 420-iteration model. I kept the
smaller model because it generalized better on Dev and is faster to reproduce.

## Diagnostics

Segmented MAE from the selected model:

| Segment | MAE |
|---|---:|
| Overall | 253.8s |
| Same-zone | 207.3s |
| Manhattan internal | 218.6s |
| Airport route | 440.5s |
| Manhattan to/from outer borough | 407.2s |
| Outer-to-outer | 534.1s |
| Rush hour | 268.8s |
| Late night | 192.3s |

Residual analysis shows remaining error is concentrated in afternoon peak
hours, airport routes, outer-borough routes, and dropoffs into zone `265`
("Unknown"). The next useful work would target those segments rather than
blindly adding more global features.

## What Did Not Work

- Training on a larger 6M-row weighted sample worsened Dev MAE, likely because
  the tree started fitting older/noisier regimes instead of the cleaner
  recency-weighted signal.
- A pure distance/speed physics prior was not enough by itself. Observed
  distance is useful as a feature, but route duration still needs historical
  priors and a flexible model.
- Treating all same-zone trips like normal route pairs produced unstable
  distance/speed behavior, so same-zone examples use a dedicated model path.

## AI Tooling

I used Codex as an implementation partner for repo reading, feature-pipeline
construction, and verification. The useful loop was: propose one modeling idea,
encode it in `train.py`, run `grade.py`/Docker, compare MAE, and keep only the
version that improved or clarified the result. The tooling was less useful for
deciding what mattered; the score and residual tables made those decisions.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Downloads public 2023 TLC yellow taxi parquet files and builds train/dev.
python data/download_data.py

# Trains model.pkl and writes metrics.json.
python train.py --sample-n 3000000 --max-iter 420

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
