"""Train-fold-only searches, validation selection and one final test evaluation."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.models.deep import NeuralModel, scale_sequences, transform_sequences
from src.models.preprocessing import feature_columns, preprocessor
from src.utils.common import read_json, write_json

LOG = logging.getLogger(__name__)


def classification(
    y: np.ndarray, p: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute every PRD classification metric with a fixed threshold."""
    predicted = p >= threshold
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def regression(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE and R-squared in original target units."""
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": root_mean_squared_error(y, p),
        "r2": r2_score(y, p),
    }


def threshold_for(y: np.ndarray, p: np.ndarray, cfg: dict[str, Any]) -> float:
    """Choose the highest-F1 validation threshold meeting the recall target."""
    choices = [(t, classification(y, p, t)) for t in cfg["models"]["threshold_grid"]]
    eligible = [
        (t, m) for t, m in choices if m["recall"] >= cfg["models"]["recall_target"]
    ]
    return max(
        eligible or choices, key=lambda pair: (pair[1]["f1"], pair[1]["recall"])
    )[0]


def fit_classical(
    name: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    columns: list[str],
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Fit preprocessing inside this partition; early-stop XGBoost internally."""
    settings = dict(cfg["models"][name])
    settings.update(params or {})
    scaled = name in ["logistic", "knn"]
    prep = preprocessor(columns, scaled)
    builders = {
        "logistic": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        "knn": KNeighborsClassifier,
        "clv": RandomForestRegressor,
    }
    if name != "knn":
        settings["random_state"] = cfg["seed"]
    if name == "xgboost":
        a, b = train_test_split(
            np.arange(len(y)),
            test_size=cfg["models"]["inner_validation_fraction"],
            stratify=y,
            random_state=cfg["seed"],
        )
        probe = preprocessor(columns, scaled=False)
        xa = probe.fit_transform(frame.iloc[a][columns])
        xb = probe.transform(frame.iloc[b][columns])
        settings["scale_pos_weight"] = float(
            (y[a] == 0).sum() / max((y[a] == 1).sum(), 1)
        )
        early = XGBClassifier(**settings).fit(
            xa, y[a], eval_set=[(xb, y[b])], verbose=False
        )
        settings["n_estimators"] = early.best_iteration + 1
        settings.pop("early_stopping_rounds")
        settings["scale_pos_weight"] = float((y == 0).sum() / max((y == 1).sum(), 1))
    pipeline = Pipeline([("preprocess", prep), ("model", builders[name](**settings))])
    pipeline.fit(frame[columns], y)
    return pipeline


def run_training(cfg: dict[str, Any]) -> None:
    """Train all required models, persist logs, then freeze the final evaluation."""
    output = Path(cfg["paths"]["artifacts"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        saved = read_json(output / "selection.json")["config"]
        if any(
            saved[key] != cfg[key] for key in ["features", "models", "split", "seed"]
        ):
            raise ValueError(
                "Training settings differ from the frozen run; use a new artifact directory"
            )
        LOG.info("Reusing frozen evaluation and artifacts; test set is not re-opened")
        return
    data = pd.read_parquet(Path(cfg["paths"]["processed"]) / "customers.parquet")
    sequences = np.load(Path(cfg["paths"]["processed"]) / "sequences.npy")
    train = data[data.split == "train"].copy()
    val = data[data.split == "validation"].copy()
    columns = feature_columns(data)
    y = train.churn.to_numpy()
    folds = list(
        StratifiedKFold(
            n_splits=cfg["split"]["folds"], shuffle=True, random_state=cfg["seed"]
        ).split(train, y)
    )
    logs: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    thresholds: dict[str, float] = {}

    def record(
        name: str,
        phase: str,
        fold: int | None,
        params: dict[str, Any],
        seconds: float,
        metrics: dict[str, Any],
    ) -> None:
        logs.append(
            {
                "model": name,
                "phase": phase,
                "fold": fold,
                "parameters": json.dumps(params, sort_keys=True),
                "training_seconds": seconds,
                **metrics,
            }
        )
        pd.DataFrame(logs).to_csv(output / "experiment_log.csv", index=False)

    for name in [
        "logistic",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "knn",
        "clv",
    ]:
        LOG.info("Training %s with five-fold training-only validation", name)
        target = train.clv_90d.to_numpy() if name == "clv" else y
        candidates = cfg["models"]["searches"].get(name, [{}])
        candidate_scores = []
        for params in candidates:
            scores = []
            for fold, (a, b) in enumerate(folds, start=1):
                start = time.perf_counter()
                model = fit_classical(
                    name, train.iloc[a], target[a], columns, cfg, params
                )
                seconds = time.perf_counter() - start
                if name == "clv":
                    metrics = regression(
                        target[b], model.predict(train.iloc[b][columns])
                    )
                    scores.append(-metrics["rmse"])
                else:
                    metrics = classification(
                        target[b], model.predict_proba(train.iloc[b][columns])[:, 1]
                    )
                    scores.append(metrics["roc_auc"])
                record(
                    name,
                    "cross_validation",
                    fold,
                    {**cfg["models"][name], **params},
                    seconds,
                    metrics,
                )
            candidate_scores.append(float(np.mean(scores)))
        best = candidates[int(np.argmax(candidate_scores))]
        start = time.perf_counter()
        model = fit_classical(name, train, target, columns, cfg, best)
        seconds = time.perf_counter() - start
        fitted[name] = model
        if name == "clv":
            metrics = regression(val.clv_90d.to_numpy(), model.predict(val[columns]))
        else:
            p = model.predict_proba(val[columns])[:, 1]
            thresholds[name] = threshold_for(val.churn.to_numpy(), p, cfg)
            metrics = classification(val.churn.to_numpy(), p, thresholds[name])
            validation[name] = metrics
        record(
            name,
            "validation",
            None,
            model.named_steps["model"].get_params(),
            seconds,
            metrics,
        )
        joblib.dump(model, output / f"{name}.joblib")
    category_names = [
        *cfg["features"]["category_patterns"],
        cfg["features"]["fallback_category"],
    ]
    timing_target = np.column_stack(
        [
            np.log1p(data.next_purchase_days_capped)
            / np.log1p(cfg["features"]["horizon_days"]),
            data.next_category_proxy.map(
                {name: i for i, name in enumerate(category_names)}
            ).fillna(-1),
        ]
    )
    for name in ["ann", "lstm", "autoencoder"]:
        LOG.info("Training %s with validation early stopping", name)
        for fold, (a, b) in enumerate(folds + [(np.arange(len(train)), None)], start=1):
            if name == "autoencoder" and b is not None:
                continue  # Unsupervised; no claim of supervised CV for fraud detection.
            outer_a = train.iloc[a]
            outer_b = train.iloc[b] if b is not None else val
            if b is not None:
                inner_a, inner_b = train_test_split(
                    np.arange(len(outer_a)),
                    test_size=cfg["models"]["inner_validation_fraction"],
                    stratify=outer_a.churn,
                    random_state=cfg["seed"],
                )
                fit_rows, stop_rows = outer_a.iloc[inner_a], outer_a.iloc[inner_b]
            else:
                fit_rows, stop_rows = outer_a, outer_b
            prep = preprocessor(columns, scaled=True)
            start = time.perf_counter()
            if name == "lstm":
                (xa, xv, xb), scaler = scale_sequences(
                    sequences[fit_rows.index],
                    sequences[stop_rows.index],
                    sequences[outer_b.index],
                )
                ya, yv = timing_target[fit_rows.index], timing_target[stop_rows.index]
                transformation = scaler
            else:
                xa = prep.fit_transform(fit_rows[columns])
                xv = prep.transform(stop_rows[columns])
                xb = prep.transform(outer_b[columns])
                ya, yv = fit_rows.churn.to_numpy(), stop_rows.churn.to_numpy()
                transformation = prep
            model = NeuralModel(
                name,
                cfg["models"]["dl"],
                cfg["seed"],
                cfg["features"]["horizon_days"],
                len(category_names),
            ).fit(xa, ya, xv, yv)
            seconds = time.perf_counter() - start
            if name == "ann":
                p = model.predict_proba(xb)[:, 1]
                if b is None:
                    thresholds[name] = threshold_for(outer_b.churn.to_numpy(), p, cfg)
                metrics = classification(
                    outer_b.churn.to_numpy(), p, thresholds.get(name, 0.5)
                )
            elif name == "lstm":
                metrics = regression(
                    outer_b.next_purchase_days_capped.to_numpy(), model.predict(xb)
                )
                labels = timing_target[outer_b.index, 1]
                known = labels >= 0
                metrics["category_accuracy_conditional_purchase"] = float(
                    (model.raw(xb)[known, 1:].argmax(axis=1) == labels[known]).mean()
                )
            else:
                errors = model.predict(xb)
                threshold = float(
                    np.quantile(errors, cfg["models"]["dl"]["anomaly_quantile"])
                )
                metrics = {
                    "mean_reconstruction_error": float(errors.mean()),
                    "threshold": threshold,
                    "validation_flag_rate": float((errors > threshold).mean()),
                }
            record(
                name,
                "cross_validation" if b is not None else "validation",
                fold if b is not None else None,
                cfg["models"]["dl"],
                seconds,
                metrics,
            )
            if b is None:
                bundle = {"model": model, "transform": transformation}
                if name == "autoencoder":
                    bundle["threshold"] = threshold
                fitted[name] = bundle
                joblib.dump(bundle, output / f"{name}.joblib")
                torch.save(model.network.state_dict(), output / f"{name}.pt")
                pd.DataFrame(model.history).to_csv(
                    output / f"{name}_loss.csv", index=False
                )
                if name == "ann":
                    validation[name] = metrics
    # Select by validation AUC, then recall. No test results guide selection.
    selected = max(
        validation,
        key=lambda name: (validation[name]["roc_auc"], validation[name]["recall"]),
    )
    write_json(
        output / "selection.json",
        {
            "model": selected,
            "threshold": thresholds[selected],
            "thresholds": thresholds,
            "validation": validation,
            "feature_columns": columns,
            "seed": cfg["seed"],
            "category_names": category_names,
            "selection_rule": "Highest validation ROC-AUC, then recall; threshold chosen on validation only",
            "config": cfg,
        },
    )
    LOG.info("Selection frozen: %s. Starting one final held-out evaluation", selected)
    test = data[data.split == "test"]
    evaluation: dict[str, Any] = {
        "selected_model": selected,
        "test_customers": len(test),
        "churn": {},
        "regression": {},
        "anomaly": {},
    }
    test_predictions = test[
        [
            "customer_id",
            "churn",
            "clv_90d",
            "next_purchase_days_capped",
            "next_category_proxy",
        ]
    ].copy()
    for name, model in fitted.items():
        if name in ["ann", "lstm", "autoencoder"]:
            xb = (
                transform_sequences(sequences[test.index], model["transform"])
                if name == "lstm"
                else model["transform"].transform(test[columns])
            )
            estimator = model["model"]
        else:
            xb, estimator = test[columns], model
        if name in thresholds:
            p = estimator.predict_proba(xb)[:, 1]
            metrics = classification(test.churn.to_numpy(), p, thresholds[name])
            evaluation["churn"][name] = {**metrics, "threshold": thresholds[name]}
            test_predictions[f"{name}_probability"] = p
        elif name in ["clv", "lstm"]:
            target = test.clv_90d if name == "clv" else test.next_purchase_days_capped
            p = estimator.predict(xb)
            metrics = regression(target.to_numpy(), p)
            if name == "lstm":
                labels = (
                    test.next_category_proxy.map(
                        {v: i for i, v in enumerate(category_names)}
                    )
                    .fillna(-1)
                    .to_numpy()
                )
                known = labels >= 0
                metrics["category_accuracy_conditional_purchase"] = float(
                    (
                        estimator.raw(xb)[known, 1:].argmax(axis=1) == labels[known]
                    ).mean()
                )
            evaluation["regression"][name] = metrics
            test_predictions[f"{name}_prediction"] = p
        else:
            errors = estimator.predict(xb)
            metrics = {
                "mean_reconstruction_error": float(errors.mean()),
                "threshold": model["threshold"],
                "test_flag_rate": float((errors > model["threshold"]).mean()),
                "test_flagged": int((errors > model["threshold"]).sum()),
            }
            evaluation["anomaly"] = metrics
            test_predictions["anomaly_error"] = errors
        record(name, "test", None, {}, 0, metrics)
    test_predictions.to_csv(output / "test_predictions.csv", index=False)
    write_json(output / "evaluation.json", evaluation)
    LOG.info("Final evaluation saved; test set is now frozen")


if __name__ == "__main__":
    from src.utils.common import config, setup_logging

    setup_logging()
    run_training(config())
