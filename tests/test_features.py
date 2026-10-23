"""Source-backed tests; no invented customers or reported model metrics."""

from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

from src.data.pipeline import clean, download, validate
from src.features.build import build_features, build_targets, make_sequences
from src.models.deep import NeuralModel, scale_sequences
from src.models.predict import Predictor
from src.models.preprocessing import FrequencyEncoder, feature_columns
from src.models.train import classification, run_training
from src.utils.common import config, digest, read_json


@pytest.fixture(scope="module")
def cfg() -> dict:
    """Use the actual project configuration."""
    return config()


@pytest.fixture(scope="module")
def transactions(cfg: dict) -> pd.DataFrame:
    """Use genuine records for customers with returns and repeat purchases."""
    frame = pd.read_parquet(Path(cfg["paths"]["interim"]) / "transactions.parquet")
    ids = frame[frame.is_return & frame.customer_id.notna()].customer_id.unique()[:25]
    return frame[frame.customer_id.isin(ids)].copy()


def test_original_workbook_preserved(cfg: dict) -> None:
    """Confirm the published source bytes and idempotent download behavior."""
    path = download(cfg)
    assert (
        digest(path)
        == "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
    )
    assert (
        read_json(Path(cfg["paths"]["interim"]) / "validation_audit.json")["raw_rows"]
        == 1067371
    )


def test_future_changes_cannot_change_features(
    transactions: pd.DataFrame, cfg: dict
) -> None:
    """Alter only future values and require identical pre-cutoff inputs."""
    cutoff = pd.Timestamp(cfg["features"]["cutoff"])
    expected, expected_orders = build_features(transactions, cutoff, cfg)
    changed = transactions.copy()
    future = changed.invoice_date >= cutoff
    changed.loc[future, "revenue"] *= 100
    changed.loc[future, "price"] *= 100
    changed.loc[future, "description"] = "CHRISTMAS MUG"
    actual, actual_orders = build_features(changed, cutoff, cfg)
    assert_frame_equal(expected, actual)
    assert_frame_equal(expected_orders, actual_orders)
    assert (actual_orders.invoice_date < cutoff).all()


def test_cutoff_equality_is_excluded(transactions: pd.DataFrame, cfg: dict) -> None:
    """Choose a genuine event timestamp and exclude its rows from features."""
    timestamp = transactions.invoice_date.sort_values().iloc[len(transactions) // 2]
    expected, _ = build_features(
        transactions[transactions.invoice_date < timestamp], timestamp, cfg
    )
    actual, _ = build_features(transactions, timestamp, cfg)
    assert_frame_equal(expected, actual)


def test_returns_and_distinct_purchases_survive(
    transactions: pd.DataFrame, cfg: dict
) -> None:
    """Only exact duplicates are removed; returns remain negative revenue."""
    source_columns = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]
    source = transactions[source_columns]
    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    cleaned, audit = clean(duplicate, cfg)
    assert len(cleaned) == len(source.drop_duplicates())
    assert audit["exact_duplicates_removed"] >= 1
    assert (
        cleaned.is_return.sum()
        == source[
            source.quantity.lt(0) | source.invoice.str.upper().str.startswith("C")
        ]
        .drop_duplicates()
        .shape[0]
    )
    assert (cleaned.loc[cleaned.is_return, "revenue"] <= 0).all()
    assert (
        cleaned.loc[cleaned.bulk_purchase, "quantity"]
        .ge(cfg["features"]["bulk_quantity"])
        .all()
    )


def test_validation_fails_on_bad_schema_dates_and_nulls(
    transactions: pd.DataFrame, cfg: dict
) -> None:
    """Each invalid-input case must fail loudly before feature building."""
    for frame in [
        transactions.drop(columns="price"),
        transactions.assign(invoice_date=pd.Timestamp("2020-01-01")),
        transactions.assign(price=np.inf),
        transactions.assign(customer_id=pd.NA),
    ]:
        with pytest.raises(ValueError):
            validate(frame, cfg)


def test_targets_require_complete_window(transactions: pd.DataFrame, cfg: dict) -> None:
    """Reject right-censored data rather than label absent future rows as churn."""
    cutoff = pd.Timestamp(cfg["features"]["cutoff"])
    with pytest.raises(ValueError, match="incomplete"):
        build_targets(
            transactions[transactions.invoice_date < cutoff],
            transactions.customer_id.drop_duplicates(),
            cutoff,
            cfg,
        )


def test_real_targets_and_split_integrity(cfg: dict) -> None:
    """Check source-derived labels and unique customer partitions."""
    data = pd.read_parquet(Path(cfg["paths"]["processed"]) / "customers.parquet")
    assert data.customer_id.is_unique
    assert set(data.split) == {"train", "validation", "test"}
    assert (
        data.loc[data.churn.eq(1), "next_purchase_days_capped"]
        .eq(cfg["features"]["horizon_days"])
        .all()
    )
    assert (
        data.loc[data.churn.eq(0), "next_purchase_days_capped"]
        .lt(cfg["features"]["horizon_days"])
        .all()
    )
    assert data.filter(like="affinity_").sum(axis=1).sub(1).abs().max() < 1e-8
    actual = pd.read_parquet(Path(cfg["paths"]["interim"]) / "transactions.parquet")
    ids = data.customer_id.head(20)
    targets = build_targets(actual, ids, pd.Timestamp(cfg["features"]["cutoff"]), cfg)
    assert np.array_equal(
        targets.churn, data.set_index("customer_id").loc[ids, "churn"]
    )


def test_sequence_scaling_and_unknown_sku(
    transactions: pd.DataFrame, cfg: dict
) -> None:
    """Keep padded events zero and learn high-cardinality encoding on fit rows."""
    features, orders = build_features(
        transactions, pd.Timestamp(cfg["features"]["cutoff"]), cfg
    )
    seq = make_sequences(orders, features.customer_id, cfg)
    arrays, params = scale_sequences(seq[:10], seq[10:])
    assert np.isfinite(arrays[0]).all()
    assert np.all(arrays[0][seq[:10, :, 0] == 0] == 0)
    assert len(params["mean"]) == 2
    encoder = FrequencyEncoder().fit(features[["top_sku"]].iloc[:1])
    assert (
        encoder.transform(pd.DataFrame({"top_sku": ["UNSEEN_TEST_INPUT"]}))[0, 0] == 0
    )
    assert len(encoder.get_feature_names_out()) == 1


def test_serialized_inference_matches_frozen_outputs(cfg: dict) -> None:
    """Verify persisted model probabilities against the untouched test predictions."""
    data = pd.read_parquet(Path(cfg["paths"]["processed"]) / "customers.parquet")
    frozen = pd.read_csv(
        Path(cfg["paths"]["artifacts"]) / "test_predictions.csv",
        dtype={"customer_id": str},
    )
    selected = read_json(Path(cfg["paths"]["artifacts"]) / "selection.json")
    model = joblib.load(Path(cfg["paths"]["artifacts"]) / f"{selected['model']}.joblib")
    rows = data.set_index("customer_id").loc[frozen.customer_id]
    probability = model.predict_proba(rows[selected["feature_columns"]])[:, 1]
    np.testing.assert_allclose(
        probability, frozen[f"{selected['model']}_probability"], atol=1e-10
    )
    measured = classification(rows.churn.to_numpy(), probability, selected["threshold"])
    saved = read_json(Path(cfg["paths"]["artifacts"]) / "evaluation.json")["churn"][
        selected["model"]
    ]
    assert measured["roc_auc"] == pytest.approx(saved["roc_auc"])


def test_prediction_without_sequence_does_not_invent_timing(cfg: dict) -> None:
    """Feature-only requests receive genuine tabular scores and absent timing."""
    data = pd.read_parquet(Path(cfg["paths"]["processed"]) / "customers.parquet").head(
        3
    )
    prediction = Predictor(cfg).score(data)
    assert prediction.next_purchase_days_capped.isna().all()
    assert prediction.next_category_proxy.isna().all()
    assert prediction.churn_probability.between(0, 1).all()
    assert np.allclose(
        prediction.churn_probability + prediction.purchase_probability_90d, 1
    )


def test_complete_training_workflow_on_training_only_fixture(
    tmp_path: Path, cfg: dict
) -> None:
    """Exercise all training paths on a subset of actual training customers only."""
    sample_cfg = copy.deepcopy(cfg)
    original = pd.read_parquet(Path(cfg["paths"]["processed"]) / "customers.parquet")
    subset = original[original.split == "train"].sample(400, random_state=cfg["seed"])
    seq = np.load(Path(cfg["paths"]["processed"]) / "sequences.npy")[subset.index]
    subset = subset.reset_index(drop=True)
    fit, test = train_test_split(
        subset.index, test_size=0.15, stratify=subset.churn, random_state=42
    )
    fit, val = train_test_split(
        fit, test_size=0.18, stratify=subset.loc[fit, "churn"], random_state=42
    )
    subset["split"] = "train"
    subset.loc[val, "split"] = "validation"
    subset.loc[test, "split"] = "test"
    processed = tmp_path / "processed"
    processed.mkdir()
    subset.to_parquet(processed / "customers.parquet", index=False)
    np.save(processed / "sequences.npy", seq)
    sample_cfg["paths"]["processed"] = str(processed)
    sample_cfg["paths"]["artifacts"] = str(tmp_path / "artifacts")
    sample_cfg["models"]["dl"].update(epochs=2, patience=1, batch_size=32)
    for name in ["random_forest", "xgboost", "lightgbm", "clv"]:
        sample_cfg["models"][name]["n_estimators"] = 5
    sample_cfg["models"]["searches"] = {
        "random_forest": [{"n_estimators": 5, "max_depth": 3, "min_samples_leaf": 3}],
        "xgboost": [{"n_estimators": 5}],
    }
    run_training(sample_cfg)
    report = read_json(tmp_path / "artifacts/evaluation.json")
    assert len(report["churn"]) == 7
    assert set(report["regression"]) == {"clv", "lstm"}
    checksum = digest(tmp_path / "artifacts/evaluation.json")
    run_training(sample_cfg)
    assert digest(tmp_path / "artifacts/evaluation.json") == checksum
    assert "churn" not in feature_columns(subset)
    with pytest.raises(ValueError):
        NeuralModel("invalid", sample_cfg["models"]["dl"], 42).fit(
            np.ones((3, 2)), np.ones(3), np.ones((2, 2)), np.ones(2)
        )


def test_report_regeneration_uses_frozen_evidence(tmp_path: Path, cfg: dict) -> None:
    """Regenerate figures and the final PDF from genuine saved evidence."""
    import shutil
    from src.utils.reporting import charts, diagrams, final_pdf

    report_cfg = copy.deepcopy(cfg)
    report_cfg["paths"]["docs"] = str(tmp_path)
    for source in Path(cfg["paths"]["docs"]).glob("*.png"):
        shutil.copyfile(source, tmp_path / source.name)
    charts(report_cfg)
    diagrams(tmp_path)
    final_pdf(report_cfg)
    assert (tmp_path / "final_report.pdf").read_bytes().startswith(b"%PDF")
    comparison = pd.read_csv(tmp_path / "model_comparison.csv", index_col=0)
    frozen = read_json(Path(cfg["paths"]["artifacts"]) / "evaluation.json")
    assert comparison.loc["lightgbm", "roc_auc"] == pytest.approx(
        frozen["churn"]["lightgbm"]["roc_auc"]
    )
    assert len(list(tmp_path.glob("*.png"))) >= 10
