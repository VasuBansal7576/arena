#!/usr/bin/env python
"""Small metric-driven experiment loop for the ETA submission.

This is the practical version of the AutoResearch idea used for this take-home:
run one named modeling change at a time, measure Dev MAE, append a ledger row,
and only promote an artifact when it beats the current best metric.

The script intentionally avoids any inference-time APIs. It shells out to
train.py with explicit flags so every experiment is reproducible from the git
log and research ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
RESEARCH_DIR = ROOT / "research_runs"
LEDGER_PATH = ROOT / "research_log.csv"
BEST_MODEL_PATH = ROOT / "model.pkl"
BEST_METRICS_PATH = ROOT / "metrics.json"


@dataclass
class Experiment:
    name: str
    args: list[str] = field(default_factory=list)


DEFAULT_EXPERIMENTS = [
    Experiment("baseline_1m", ["--sample-n", "1000000", "--max-iter", "260"]),
    Experiment("no_density_1m", ["--sample-n", "1000000", "--max-iter", "260", "--disable-feature-group", "density"]),
    Experiment("no_ratecode_priors_1m", ["--sample-n", "1000000", "--max-iter", "260", "--disable-feature-group", "ratecode_priors"]),
    Experiment("no_neighbor_1m", ["--sample-n", "1000000", "--max-iter", "260", "--disable-feature-group", "neighbor"]),
    Experiment("no_recency_1m", ["--sample-n", "1000000", "--max-iter", "260", "--recency-half-life-days", "0"]),
    Experiment("squared_error_1m", ["--sample-n", "1000000", "--max-iter", "260", "--loss", "squared_error"]),
    Experiment(
        "squared_error_no_recency_1m",
        ["--sample-n", "1000000", "--max-iter", "260", "--loss", "squared_error", "--recency-half-life-days", "0"],
    ),
    Experiment(
        "squared_error_no_density_1m",
        ["--sample-n", "1000000", "--max-iter", "260", "--loss", "squared_error", "--disable-feature-group", "density"],
    ),
    Experiment(
        "squared_error_no_cap_1m",
        ["--sample-n", "1000000", "--max-iter", "260", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment(
        "squared_error_no_cap_1m_180",
        ["--sample-n", "1000000", "--max-iter", "180", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment(
        "squared_error_no_cap_1m_340",
        ["--sample-n", "1000000", "--max-iter", "340", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment(
        "squared_error_no_cap_1m_340_hl60",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "60",
        ],
    ),
    Experiment(
        "squared_error_no_cap_1m_340_hl120",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "120",
        ],
    ),
    Experiment(
        "squared_error_no_cap_1m_340_seed7",
        [
            "--sample-n",
            "1000000",
            "--sample-seed",
            "7",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
        ],
    ),
    Experiment(
        "squared_error_no_cap_1m_420_lr045",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "420",
            "--learning-rate",
            "0.045",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
        ],
    ),
    Experiment(
        "squared_error_no_cap_1m_420_leaf95",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "420",
            "--max-leaf-nodes",
            "95",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
        ],
    ),
    Experiment(
        "squared_error_no_cap_1m_500_lr04",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "500",
            "--learning-rate",
            "0.04",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
        ],
    ),
    Experiment(
        "squared_error_no_cap_1500k_420_lr045",
        [
            "--sample-n",
            "1500000",
            "--max-iter",
            "420",
            "--learning-rate",
            "0.045",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
        ],
    ),
    Experiment(
        "route_class_specialists_1m_500_lr04",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "500",
            "--learning-rate",
            "0.04",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--disable-feature-group",
            "target_encoding",
            "--disable-feature-group",
            "variance",
            "--route-class-models",
            "--route-class-max-iter",
            "260",
            "--route-class-sample-n",
            "400000",
        ],
    ),
    Experiment(
        "route_class_specialists_pruned_1m_500_lr04",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "500",
            "--learning-rate",
            "0.04",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--disable-feature-group",
            "target_encoding",
            "--disable-feature-group",
            "variance",
            "--route-class-models",
            "--route-class-holdout-prune",
            "--route-class-max-iter",
            "260",
            "--route-class-sample-n",
            "400000",
        ],
    ),
    Experiment(
        "target_encoding_1m_500_lr04",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "500",
            "--learning-rate",
            "0.04",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--disable-feature-group",
            "variance",
        ],
    ),
    Experiment(
        "variance_route_class_pruned_1m_500_lr04",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "500",
            "--learning-rate",
            "0.04",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--disable-feature-group",
            "target_encoding",
            "--route-class-models",
            "--route-class-holdout-prune",
            "--route-class-max-iter",
            "260",
            "--route-class-sample-n",
            "400000",
        ],
    ),
    Experiment(
        "residual_calibration_route_pruned_1m_500_lr04",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "500",
            "--learning-rate",
            "0.04",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--disable-feature-group",
            "target_encoding",
            "--disable-feature-group",
            "variance",
            "--route-class-models",
            "--route-class-holdout-prune",
            "--route-class-max-iter",
            "260",
            "--route-class-sample-n",
            "400000",
            "--residual-calibration",
            "--calibration-holdout-prune",
        ],
    ),
    Experiment(
        "affine_calibration_route_pruned_1m_500_lr04",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "500",
            "--learning-rate",
            "0.04",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--disable-feature-group",
            "target_encoding",
            "--disable-feature-group",
            "variance",
            "--route-class-models",
            "--route-class-holdout-prune",
            "--route-class-max-iter",
            "260",
            "--route-class-sample-n",
            "400000",
            "--affine-calibration",
            "--calibration-holdout-prune",
        ],
    ),
    Experiment(
        "fine_affine_calibration_route_pruned_1m_500_lr04",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "500",
            "--learning-rate",
            "0.04",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--disable-feature-group",
            "target_encoding",
            "--disable-feature-group",
            "variance",
            "--route-class-models",
            "--route-class-holdout-prune",
            "--route-class-max-iter",
            "260",
            "--route-class-sample-n",
            "400000",
            "--affine-calibration",
            "--fine-affine-calibration",
            "--calibration-holdout-prune",
        ],
    ),
    Experiment(
        "squared_error_no_cap_2m_260",
        ["--sample-n", "2000000", "--max-iter", "260", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment("absolute_error_1m", ["--sample-n", "1000000", "--max-iter", "260", "--loss", "absolute_error"]),
    Experiment("no_same_zone_model_1m", ["--sample-n", "1000000", "--max-iter", "260", "--no-same-zone-model"]),
]


def load_best_score() -> float:
    if not BEST_METRICS_PATH.exists():
        return float("inf")
    with open(BEST_METRICS_PATH) as f:
        return float(json.load(f)["dev_mae"])


def load_best_holdout() -> float | None:
    if not BEST_METRICS_PATH.exists():
        return None
    with open(BEST_METRICS_PATH) as f:
        metrics = json.load(f)
    holdout = metrics.get("time_holdout") or {}
    if "holdout_mae" in holdout:
        return float(holdout["holdout_mae"])
    route_holdout = metrics.get("route_class_holdout") or {}
    if "specialist_holdout_mae" in route_holdout:
        return float(route_holdout["specialist_holdout_mae"])
    return None


def append_ledger(row: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LEDGER_PATH.exists()
    with open(LEDGER_PATH, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "status",
                "dev_mae",
                "best_before",
                "promoted",
                "elapsed_seconds",
                "metrics_path",
                "args",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_experiments() -> set[str]:
    if not LEDGER_PATH.exists():
        return set()
    with open(LEDGER_PATH, newline="") as f:
        return {
            row["experiment"]
            for row in csv.DictReader(f)
            if row.get("status") == "ok"
        }


def run_experiment(exp: Experiment, promote: bool) -> dict:
    RESEARCH_DIR.mkdir(exist_ok=True)
    model_path = RESEARCH_DIR / f"{exp.name}.pkl"
    metrics_path = RESEARCH_DIR / f"{exp.name}.json"
    best_before = load_best_score()
    holdout_before = load_best_holdout()
    cmd = [
        sys.executable,
        "train.py",
        "--experiment-name",
        exp.name,
        "--model-path",
        str(model_path),
        "--metrics-path",
        str(metrics_path),
        *exp.args,
    ]

    print("\n==>", " ".join(cmd))
    t0 = time.time()
    status = "ok"
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        with open(metrics_path) as f:
            metrics = json.load(f)
        dev_mae = float(metrics["dev_mae"])
        holdout = metrics.get("route_class_holdout") or {}
        holdout_ok = True
        if holdout:
            base_holdout = float(holdout.get("base_holdout_mae", float("inf")))
            specialist_holdout = float(holdout.get("specialist_holdout_mae", float("inf")))
            holdout_ok = specialist_holdout <= base_holdout
        time_holdout = metrics.get("time_holdout") or {}
        if holdout_before is not None and time_holdout:
            holdout_ok = holdout_ok and float(time_holdout["holdout_mae"]) <= holdout_before
        promoted = bool(promote and dev_mae < best_before and holdout_ok)
        if promoted:
            shutil.copy2(model_path, BEST_MODEL_PATH)
            shutil.copy2(metrics_path, BEST_METRICS_PATH)
            print(f"PROMOTED {exp.name}: {dev_mae:.3f} < {best_before:.3f}")
        elif promote and dev_mae < best_before and not holdout_ok:
            print(f"rejected by holdout gate: {dev_mae:.3f} beats full Dev but worsens time holdout")
        else:
            print(f"kept current best: {dev_mae:.3f} vs {best_before:.3f}")
    except Exception:
        status = "failed"
        dev_mae = float("nan")
        promoted = False
        raise
    finally:
        elapsed = round(time.time() - t0, 1)
        append_ledger(
            {
                "experiment": exp.name,
                "status": status,
                "dev_mae": dev_mae,
                "best_before": best_before,
                "promoted": promoted,
                "elapsed_seconds": elapsed,
                "metrics_path": metrics_path,
                "args": " ".join(exp.args),
            }
        )
    return {"name": exp.name, "dev_mae": dev_mae, "promoted": promoted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="copy a better model/metrics into the submission")
    parser.add_argument(
        "--only",
        action="append",
        help="run only experiments whose names match this value; may be repeated",
    )
    parser.add_argument("--rerun", action="store_true", help="rerun experiments already marked ok in the ledger")
    args = parser.parse_args()

    experiments = DEFAULT_EXPERIMENTS
    if args.only:
        names = set(args.only)
        experiments = [exp for exp in experiments if exp.name in names]
        missing = names - {exp.name for exp in experiments}
        if missing:
            raise SystemExit(f"Unknown experiments: {sorted(missing)}")
    if not args.rerun:
        done = completed_experiments()
        experiments = [exp for exp in experiments if exp.name not in done]
        if not experiments:
            print("No new experiments to run. Use --rerun to repeat completed experiments.")
            return

    results = [run_experiment(exp, promote=args.promote) for exp in experiments]
    print("\nSummary")
    for result in results:
        print(f"  {result['name']}: {result['dev_mae']:.3f} promoted={result['promoted']}")
    print(f"\nLedger: {LEDGER_PATH}")


if __name__ == "__main__":
    main()
