"""Integration tests against real PostgreSQL and the actual trained artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from api.main import app
from src.data.store import predictions, save_predictions, transactions
from src.utils.common import config


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Open the actual application and restore any scored existing customer."""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Set DATABASE_URL to run PostgreSQL integration tests")
    with TestClient(app) as connection:
        yield connection
        cfg = config()
        features = pd.read_parquet(
            Path(cfg["paths"]["processed"]) / "customers.parquet"
        )
        scores = pd.read_csv(
            Path(cfg["paths"]["processed"]) / "predictions.csv",
            dtype={"customer_id": str},
        )
        first = features.head(1)
        save_predictions(
            app.state.db, first, scores[scores.customer_id.isin(first.customer_id)]
        )


@pytest.fixture(scope="module")
def customer() -> pd.Series:
    """Read an existing customer, never invent a prediction fixture."""
    return pd.read_parquet(
        Path(config()["paths"]["processed"]) / "customers.parquet"
    ).iloc[0]


def test_health_metadata_and_persistence(
    client: TestClient, customer: pd.Series
) -> None:
    """Check readiness and scores are read from actual persisted records."""
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/metadata").json()["evaluation"]["churn"]) == 7
    response = client.post("/predict", json={"customer_id": customer.customer_id})
    assert response.status_code == 200
    assert 0 <= response.json()["churn_probability"] <= 1
    with app.state.db.connect() as db:
        assert db.scalar(select(func.count()).select_from(transactions)) > 0
        stored = db.scalar(
            select(predictions.c.payload).where(
                predictions.c.customer_id == customer.customer_id
            )
        )
    assert stored == response.json()


def test_filters_and_supporting_views(client: TestClient) -> None:
    """All dashboard reads are real API data and filters match their criteria."""
    rows = client.get("/customers").json()
    assert rows
    first = rows[0]
    filtered = client.get(
        "/customers",
        params={
            "country": first["country"],
            "segment": first["segment_id"],
            "value_tier": first["value_tier"],
        },
    ).json()
    assert all(
        r["country"] == first["country"]
        and r["segment_id"] == first["segment_id"]
        and r["value_tier"] == first["value_tier"]
        for r in filtered
    )
    assert {r["kind"] for r in client.get("/trends").json()} == {
        "observed",
        "linear forecast",
    }
    assert client.get("/importance").json()


def test_shap_reconstructs_actual_probability(
    client: TestClient, customer: pd.Series
) -> None:
    """SHAP is numerically tied to the real model prediction; LIME exists."""
    response = client.get(f"/explain/{customer.customer_id}")
    assert response.status_code == 200
    result = response.json()
    assert result["shap_sum"] == pytest.approx(result["probability"], abs=1e-5)
    assert result["lime"] and result["contributions"]
    assert client.get("/explain/unknown-test-id").status_code == 404


def test_validation_rejects_malformed_records(
    client: TestClient, customer: pd.Series
) -> None:
    """Reject bad IDs, missing fields, unknown fields and non-finite values."""
    assert client.post("/predict", json={"customer_id": "../bad"}).status_code == 422
    assert (
        client.post("/predict", json={"customer_id": "unknown-test-id"}).status_code
        == 404
    )
    assert (
        client.post(
            "/predict",
            json={"customer_id": customer.customer_id, "features": {"recency": -1}},
        ).status_code
        == 422
    )
    columns = app.state.predictor.selection["feature_columns"]
    features = customer[columns].to_dict()
    for key, value in [
        ("country", ""),
        ("monetary", -1),
        ("frequency", 0),
        ("affinity_home", 2),
        ("recency", "nan"),
    ]:
        bad = {**features, key: value}
        assert (
            client.post(
                "/predict", json={"customer_id": customer.customer_id, "features": bad}
            ).status_code
            == 422
        )


def test_csv_batch_validation_and_scoring(
    client: TestClient, customer: pd.Series
) -> None:
    """Exercise actual uploads, size limits, all-row checks and complete features."""

    def upload(text: str) -> object:
        return client.post(
            "/predict/batch", files={"file": ("customers.csv", text, "text/csv")}
        )

    result = upload(f"customer_id\n{customer.customer_id}\n")
    assert result.status_code == 200 and result.json()["count"] == 1
    assert upload("wrong_column\nvalue\n").status_code == 422
    assert upload("customer_id\n").status_code == 422
    assert (
        upload(
            f"customer_id\n{customer.customer_id}\n{customer.customer_id}\n"
        ).status_code
        == 422
    )
    assert upload("customer_id,recency\n123,\n").status_code == 422
    assert upload("customer_id\n../invalid\n").status_code == 422
    assert upload("customer_id\n" + "x\n" * 501).status_code == 413
    assert (
        upload("x" * (app.state.cfg["api"]["max_upload_bytes"] + 1)).status_code == 413
    )
    columns = app.state.predictor.selection["feature_columns"]
    frame = customer[["customer_id", *columns]].to_frame().T
    response = upload(frame.to_csv(index=False))
    assert response.status_code == 200
    scored = response.json()["results"][0]
    assert scored["next_purchase_days_capped"] is None
    assert np.isfinite(scored["churn_probability"])
    assert (
        client.post(
            "/predict",
            json={
                "customer_id": customer.customer_id,
                "features": customer[columns].to_dict(),
            },
        ).status_code
        == 200
    )
