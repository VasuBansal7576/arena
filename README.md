# Gobblecube ETA Challenge Submission

This repo is my submission for the **Gobblecube AI Builder Take-Home**. I chose
the **ETA Challenge**: predict NYC taxi trip duration in seconds from the
grader request fields.

The submission lives in [`eta-challenge-starter/`](./eta-challenge-starter/).

## Result

- Full local Dev from `train.py`: **244.1s MAE**
- Late time-holdout MAE (`requested_at >= 2023-12-25`): **252.7s**
- Challenge baseline reference: about **367s Eval MAE**
- Docker image: builds and runs with no network access at inference time

## Submission Files

- [`eta-challenge-starter/predict.py`](./eta-challenge-starter/predict.py):
  grader entry point, `predict(request: dict) -> float`
- [`eta-challenge-starter/Dockerfile`](./eta-challenge-starter/Dockerfile):
  Dockerized grader path
- [`eta-challenge-starter/model.pkl`](./eta-challenge-starter/model.pkl):
  trained model bundle
- [`eta-challenge-starter/README.md`](./eta-challenge-starter/README.md):
  modeling approach, ablations, diagnostics, and reproduction steps
- [`eta-challenge-starter/AGENTS.md`](./eta-challenge-starter/AGENTS.md):
  agent constraints and submission notes
- [`eta-challenge-starter/program.md`](./eta-challenge-starter/program.md):
  AutoResearch-inspired experiment brief

## Quick Verify

```bash
cd eta-challenge-starter
python -m pytest tests/
python grade.py

docker build -t gobblecube-eta-submission .
docker run --rm --network=none -v $(pwd)/data:/work \
  gobblecube-eta-submission /work/dev.parquet /work/preds.csv
```

## What I Tried

The final model uses offline NYC taxi priors, cyclical time features,
route-class structure, demand-density proxies, geographic borrowing, recency
weighting, a dedicated same-zone path, and a metric-aware affine calibration
layer. The ablation loop overturned two reasonable assumptions:
quantile/absolute loss and target capping both sounded metric-aligned, but
squared-error with no target cap scored better on Dev. The final improvement
came from thinking directly about MAE calibration rather than adding more
generic features.

The research loop was inspired by Andrej Karpathy's AutoResearch framing:
write a short research program, run one measurable experiment at a time, and
promote only what improves the metric. That loop passed objective search,
route-class pruning, and fine affine calibration; it rejected target encoding,
duration variance, and median residual tables when they failed the
time-holdout.

The unselected `crossing-challenge-starter/` is the original alternate starter
from the take-home. This submission is the ETA challenge only.

Submit your repo URL and LinkedIn profile to agentic-hiring@gobblecube.ai.
