# AutoResearch Program

This is the human-written research brief for the ETA challenge. It adapts the
core idea from Karpathy-style AutoResearch to a tabular Gobblecube submission:
make one measurable modeling change, train under a bounded local budget, compare
Dev MAE, keep only real improvements, and leave a readable trail.

## Goal

Minimize Dev MAE for `predict(request: dict) -> float` while preserving the
grader contract:

- no external API calls at inference time
- no Eval-set training
- no inference dependency on fields missing from the request
- Docker image stays under the challenge limit
- `predict.py` remains deterministic and row-safe

## Loop

1. Read the latest `metrics.json`, `research_log.csv`, and residual diagnostics.
2. Pick one hypothesis tied to a current error segment.
3. Edit `train.py`, `features.py`, or experiment flags for that hypothesis only.
4. Run a short validation experiment through `autoresearch.py --promote`.
5. Promote only if full Dev MAE beats the current `metrics.json`.
6. If promoted, update `README.md`, `PLAN_AUDIT.md`, and this file if the
   research direction changes.
7. Commit the winning change separately from losing attempts when preparing the
   public repo.

Use `python autoresearch_agent.py` for the program-driven loop. It reads this
brief, chooses the next queued code-backed experiment, invokes
`autoresearch.py --promote`, and appends a keep/reject note to
`research_notes.md`.

## Current Best

`fine_affine_calibration_route_pruned_1m_500_lr04`

- full Dev MAE: `244.13189891018882`
- late time-holdout MAE: `252.74299270394238`
- loss: `squared_error`
- sample: `1,000,000` recency-weighted rows
- target cap: disabled with `--target-cap-quantile 1.0`
- recency half-life: `90` days; follow-up `60` and `120` day sweeps were worse
- learner: `500` iterations, learning rate `0.04`
- route-class specialist: only Manhattan-to/from-outer kept a nonzero blend
  after the holdout gate
- calibration: 116 small route/hour/day/dropoff and interaction affine rules,
  fitted on the pre-holdout Dev slice and kept only when they improved the
  later holdout

## Next Experiments

Prioritize the largest residual pockets instead of adding generic features:

- reduce calibration overfit risk with stronger nested time splits
- airport route specialization: airport MAE remains much worse than overall MAE
- outer-borough sparse-route borrowing: outer-to-outer is the hardest route
  class
- unknown dropoff zone `265`: high error and enough rows to justify a targeted
  fallback
- second-order recency work: route-class-specific recency weighting rather than
  a single global half-life
- route-class specialist models only if they beat the later time-holdout, not
  just full Dev

## Keep / Reject Rules

Keep an experiment only when:

- `dev_mae` improves over `metrics.json`
- contract tests still pass
- Docker grader mode still writes one prediction per input row
- the improvement can be explained in the README without hand-waving

Reject an experiment when:

- it improves one segment but worsens full Dev MAE
- it depends on unavailable inference fields
- it adds network/runtime dependencies
- it hides the actual model behavior from reviewers
