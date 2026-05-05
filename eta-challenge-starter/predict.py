"""Submission interface for the Gobblecube ETA Challenge.

The grader imports this module and calls predict(request) row-by-row. All
state needed at inference lives in model.pkl; no network or data files are
required inside the container.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from features import FEATURE_NAMES, build_single_features

MODEL_PATH = Path(__file__).parent / "model.pkl"

_BUNDLE = None


def _load_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        with open(MODEL_PATH, "rb") as f:
            _BUNDLE = pickle.load(f)
    return _BUNDLE


def _predict_model(model, x: np.ndarray) -> float:
    n_features = getattr(model, "n_features_in_", x.shape[1])
    return float(model.predict(x[:, :n_features])[0])


def _feature_vector_for_bundle(request: dict, artifacts: dict) -> np.ndarray:
    full = build_single_features(request, artifacts)
    saved_names = artifacts.get("feature_names", FEATURE_NAMES)
    if list(saved_names) == FEATURE_NAMES:
        return full.reshape(1, -1)
    current_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    aligned = np.asarray([full[current_index[name]] for name in saved_names], dtype=np.float32)
    return aligned.reshape(1, -1)


def _predict_with_bundle(request: dict, bundle: dict) -> float:
    artifacts = bundle["artifacts"]
    x = _feature_vector_for_bundle(request, artifacts)

    model_pred = _predict_model(bundle["model"], x)
    prior_idx = artifacts["feature_index"]["pair_hour_prior_duration"]
    prior_pred = float(x[0, prior_idx])
    blend = float(bundle.get("blend_weight", 1.0))
    pred = blend * model_pred + (1.0 - blend) * prior_pred

    same_idx = artifacts["feature_index"]["same_zone"]
    if x[0, same_idx] > 0.5 and bundle.get("same_zone_model") is not None:
        same_pred = _predict_model(bundle["same_zone_model"], x)
        same_blend = float(bundle.get("same_zone_blend_weight", 1.0))
        pred = same_blend * same_pred + (1.0 - same_blend) * prior_pred
    else:
        class_idx = artifacts["feature_index"]["route_class"]
        route_class = int(round(float(x[0, class_idx])))
        route_models = bundle.get("route_class_models") or {}
        route_model = route_models.get(route_class)
        if route_model is not None:
            class_pred = _predict_model(route_model, x)
            class_blends = bundle.get("route_class_blend_weights") or {}
            class_blend = float(class_blends.get(route_class, class_blends.get(str(route_class), 0.0)))
            pred = class_blend * class_pred + (1.0 - class_blend) * pred

    calibration = bundle.get("residual_calibration") or {}
    alphas = calibration.get("alphas") or {}
    if calibration:
        route_class_idx = artifacts["feature_index"]["route_class"]
        route_class = int(round(float(x[0, route_class_idx])))
        requested_at = request.get("requested_at")
        from datetime import datetime

        ts = datetime.fromisoformat(str(requested_at))
        hour = ts.hour
        dow = ts.weekday()
        pickup_zone = int(round(float(x[0, artifacts["feature_index"]["pickup_zone"]])))
        dropoff_zone = int(round(float(x[0, artifacts["feature_index"]["dropoff_zone"]])))
        correction = 0.0
        if "global" in calibration:
            correction += float(alphas.get("global", 0.0)) * float(calibration["global"])
        table = calibration.get("route_class")
        if table is not None:
            correction += float(alphas.get("route_class", 0.0)) * float(table[route_class])
        table = calibration.get("route_hour")
        if table is not None:
            correction += float(alphas.get("route_hour", 0.0)) * float(table[route_class, hour])
        table = calibration.get("dow_hour")
        if table is not None:
            correction += float(alphas.get("dow_hour", 0.0)) * float(table[dow, hour])
        table = calibration.get("dropoff_zone")
        if table is not None:
            correction += float(alphas.get("dropoff_zone", 0.0)) * float(table[dropoff_zone])
        table = calibration.get("pickup_zone")
        if table is not None:
            correction += float(alphas.get("pickup_zone", 0.0)) * float(table[pickup_zone])
        pred += correction

    rules = bundle.get("affine_calibration") or []
    if rules:
        from datetime import datetime

        ts = datetime.fromisoformat(str(request.get("requested_at")))
        route_class_idx = artifacts["feature_index"]["route_class"]
        values = {
            "route_class": int(round(float(x[0, route_class_idx]))),
            "hour": ts.hour,
            "dow": ts.weekday(),
            "pickup_zone": int(round(float(x[0, artifacts["feature_index"]["pickup_zone"]]))),
            "dropoff_zone": int(round(float(x[0, artifacts["feature_index"]["dropoff_zone"]]))),
        }
        values["route_hour"] = values["route_class"] * 24 + values["hour"]
        values["dow_hour"] = values["dow"] * 24 + values["hour"]
        values["dropoff_hour"] = values["dropoff_zone"] * 24 + values["hour"]
        values["airport_hour"] = int(values["route_class"] == 1) * 24 + values["hour"]
        values["route_dropoff"] = values["route_class"] * 266 + values["dropoff_zone"]
        for rule in rules:
            if values.get(rule["feature"]) == int(rule["value"]):
                pred = float(np.clip(pred * float(rule["scale"]) + float(rule["offset"]), 30.0, 3.0 * 3600.0))

    if not np.isfinite(pred):
        pred = float(artifacts["global_median_duration"])
    return float(max(30.0, min(pred, 3.0 * 3600.0)))


def predict(request: dict) -> float:
    """Predict trip duration in seconds."""
    return _predict_with_bundle(request, _load_bundle())
