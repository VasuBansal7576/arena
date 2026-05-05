# Gobblecube ETA Challenge Submission

This is my submission for the **Gobblecube AI Builder Take-Home**. I chose the
**ETA Challenge**: predict NYC yellow taxi trip duration in seconds from the
grader request fields.

The working submission is in [`eta-challenge-starter/`](./eta-challenge-starter/).

## Result

| Metric | Score |
|---|---:|
| Full local Dev MAE from `train.py` | **244.1s** |
| Late time-holdout MAE, `requested_at >= 2023-12-25` | **252.7s** |
| Challenge baseline reference | **~367s Eval MAE** |

The Docker image builds, runs with `--network=none`, and writes one prediction
per input row through the required `predict.py` interface.

## What Is Included

- [`eta-challenge-starter/predict.py`](./eta-challenge-starter/predict.py):
  grader entry point, `predict(request: dict) -> float`
- [`eta-challenge-starter/Dockerfile`](./eta-challenge-starter/Dockerfile):
  Dockerized inference path
- [`eta-challenge-starter/model.pkl`](./eta-challenge-starter/model.pkl):
  trained model bundle and offline artifacts
- [`eta-challenge-starter/README.md`](./eta-challenge-starter/README.md):
  full modeling write-up, ablations, diagnostics, and reproduction steps
- [`eta-challenge-starter/AGENTS.md`](./eta-challenge-starter/AGENTS.md):
  agent/submission notes
- [`eta-challenge-starter/program.md`](./eta-challenge-starter/program.md):
  AutoResearch program brief

## Approach

The final model is a CPU-friendly tabular system, not a generic zone-id
regressor. It uses:

- offline 2023 NYC taxi route priors
- cyclical time and calendar features
- airport, Manhattan/CBD, and route-class structure
- demand-density proxies
- nearest-zone geographic borrowing
- recency-weighted training
- a dedicated same-zone path
- pruned route-class specialist blending
- MAE-aware affine calibration

The strongest late-stage improvement did **not** come from adding more generic
features. It came from looking directly at the MAE objective and calibrating
the prediction surface by stable route/hour/day/dropoff pockets.

## AutoResearch

I used an Andrej Karpathy-inspired AutoResearch loop: keep a short research
program, run one measurable experiment at a time, and promote only changes that
improve the metric under a fixed local budget. In this repo:

- [`program.md`](./eta-challenge-starter/program.md) is the research brief.
- [`autoresearch_agent.py`](./eta-challenge-starter/autoresearch_agent.py)
  reads that brief and selects queued experiments.
- [`autoresearch.py`](./eta-challenge-starter/autoresearch.py) is the
  deterministic runner and promotion gate.
- [`research_log.csv`](./eta-challenge-starter/research_log.csv) and
  [`research_runs/`](./eta-challenge-starter/research_runs/) preserve the
  pass/fail trail.

What passed:

- squared-error objective beat quantile and absolute-error variants
- removing target capping improved MAE
- a slower 1M-row learner reduced underfit
- route-class specialists helped only after holdout pruning
- fine affine calibration moved the result from `250.3s` to `244.1s`

What failed:

- larger weighted samples
- target encoding
- duration variance features
- median residual correction tables
- unpruned route-class specialists

The important lesson was that AutoResearch is only useful when the search space
changes. Once feature stacking stalled around `250s`, the winning move was
metric-aware calibration. The tradeoff is that this layer is more
Dev-sensitive, so I kept it explicit, small, and gated by a later time-holdout.

## Quick Verify

```bash
cd eta-challenge-starter
python -m pytest tests/
python grade.py

docker build -t gobblecube-eta-submission .
docker run --rm --network=none -v $(pwd)/data:/work \
  gobblecube-eta-submission /work/dev.parquet /work/preds.csv
```

## Notes

Only the ETA challenge is submitted. The unselected
[`crossing-challenge-starter/`](./crossing-challenge-starter/) remains from the
original fork, but its tracked starter data/model artifacts were removed from
this submission branch.
