"""FastAPI service backed by PostgreSQL and the frozen trained artifacts."""

from __future__ import annotations

import io
import logging
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import ValidationError
from sqlalchemy import select

from api.schemas.prediction import PredictionRequest
from src.data.store import (
    customers,
    engine,
    health,
    initialize,
    predictions,
    records,
    save_predictions,
)
from src.explainability.explain import Explainer
from src.models.predict import Predictor
from src.utils.common import config, read_json, setup_logging

LOG = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models once and confirm database persistence before serving."""
    setup_logging()
    cfg = config()
    app.state.cfg = cfg
    app.state.db = engine()
    app.state.predictor = Predictor(cfg)
    initialize(app.state.db, cfg)
    app.state.explainer = Explainer(cfg)
    yield
    app.state.db.dispose()


app = FastAPI(
    title="Vantara Customer Behavior Prediction", version="1.0.0", lifespan=lifespan
)


@app.middleware("http")
async def log_request(request: Request, call_next: Any) -> Any:
    """Log request timing and status without uploading customer contents."""
    start = time.perf_counter()
    response = await call_next(request)
    LOG.log(
        logging.WARNING if response.status_code >= 400 else logging.INFO,
        "%s %s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - start) * 1000,
    )
    return response


@app.get("/health")
def get_health() -> dict[str, Any]:
    """Report readiness only when PostgreSQL responds."""
    try:
        ready = health(app.state.db)
    except Exception as exc:
        raise HTTPException(503, "Database unavailable") from exc
    return {
        "status": "ok" if ready else "unavailable",
        "model_version": app.state.predictor.version,
    }


@app.get("/metadata")
def get_metadata() -> dict[str, Any]:
    """Expose real model settings, frozen metrics and input field names."""
    cfg = app.state.cfg
    return {
        "model": app.state.predictor.selection,
        "evaluation": read_json(Path(cfg["paths"]["artifacts"]) / "evaluation.json"),
        "data": read_json(Path(cfg["paths"]["processed"]) / "feature_metadata.json"),
    }


def validate_features(item: PredictionRequest) -> pd.DataFrame:
    """Validate complete engineered records before model inference."""
    expected = app.state.predictor.selection["feature_columns"]
    features = item.features or {}
    missing = sorted(set(expected) - set(features))
    extra = sorted(set(features) - set(expected))
    if missing or extra:
        raise HTTPException(
            422, {"missing_features": missing, "unexpected_features": extra}
        )
    for key, value in features.items():
        if key in ["country", "top_sku"]:
            if not isinstance(value, str) or not value.strip():
                raise HTTPException(422, f"{key} must be a non-empty string")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise HTTPException(422, f"{key} must be a finite number")
        elif key != "frequency_trend" and value < 0:
            raise HTTPException(422, f"{key} must be non-negative")
        elif key.startswith("affinity_") and value > 1:
            raise HTTPException(422, f"{key} must be between 0 and 1")
    if features["frequency"] < 1 or features["monetary"] <= 0:
        raise HTTPException(
            422, "frequency must be at least one and monetary must be positive"
        )
    return pd.DataFrame([{"customer_id": item.customer_id, **features}])


def fetch_customer(customer_id: str) -> dict[str, Any]:
    """Read the persisted score; do not recompute models for dashboard reads."""
    with app.state.db.connect() as conn:
        row = conn.scalar(
            select(predictions.c.payload).where(
                predictions.c.customer_id == customer_id
            )
        )
    if row is None:
        raise HTTPException(
            404,
            "Unknown customer ID; provide a complete feature record to score a new customer",
        )
    return row


@app.post("/predict")
def predict(item: PredictionRequest) -> dict[str, Any]:
    """Return a stored prediction or score and persist a complete feature record."""
    if item.features is None:
        return fetch_customer(item.customer_id)
    frame = validate_features(item)
    scored = app.state.predictor.score(frame)
    save_predictions(app.state.db, frame, scored)
    return records(scored)[0]


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> dict[str, Any]:
    """Validate all CSV rows before scoring; reject empty or oversized uploads."""
    limit = app.state.cfg["api"]
    content = await file.read(limit["max_upload_bytes"] + 1)
    if len(content) > limit["max_upload_bytes"]:
        raise HTTPException(413, "CSV exceeds upload size limit")
    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            dtype={"customer_id": str, "country": str, "top_sku": str},
        )
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise HTTPException(422, "Upload a valid UTF-8 CSV") from exc
    if "customer_id" not in frame or frame.empty:
        raise HTTPException(
            422, "CSV requires a customer_id column and at least one row"
        )
    if len(frame) > limit["max_batch_rows"]:
        raise HTTPException(413, "Too many CSV rows")
    if frame.customer_id.duplicated().any():
        raise HTTPException(422, "Duplicate customer IDs in upload")
    if frame.isna().any().any():
        raise HTTPException(422, "Missing values in uploaded records")
    items = []
    for row in records(frame):
        customer_id = row.pop("customer_id")
        try:
            item = PredictionRequest(customer_id=customer_id, features=row or None)
        except ValidationError as exc:
            raise HTTPException(422, "Invalid customer ID or feature value") from exc
        items.append(item)
    if len(frame.columns) == 1:
        results = [fetch_customer(item.customer_id) for item in items]
    else:
        validated = pd.concat(
            [validate_features(item) for item in items], ignore_index=True
        )
        scored = app.state.predictor.score(validated)
        save_predictions(app.state.db, validated, scored)
        results = records(scored)
    return {"count": len(results), "results": results}


@app.get("/customers")
def list_customers(
    segment: int | None = None,
    country: str | None = None,
    value_tier: str | None = None,
) -> list[dict[str, Any]]:
    """Serve persisted scores with the required segment/country/value filters."""
    query = select(predictions.c.payload).join(customers)
    for name, value in [
        ("segment_id", segment),
        ("country", country),
        ("value_tier", value_tier),
    ]:
        if value is not None:
            query = query.where(customers.c[name] == value)
    with app.state.db.connect() as conn:
        return list(conn.scalars(query))


@app.get("/explain/{customer_id}")
def explain_customer(customer_id: str) -> dict[str, Any]:
    """Provide both SHAP and LIME for any persisted feature record."""
    with app.state.db.connect() as conn:
        features = conn.scalar(
            select(customers.c.features).where(customers.c.customer_id == customer_id)
        )
    if features is None:
        raise HTTPException(404, "Unknown customer ID")
    return app.state.explainer.explain(pd.DataFrame([features]))


@app.get("/trends")
def trends() -> list[dict[str, Any]]:
    """Return observed revenue and the clearly labelled model forecast."""
    return records(
        pd.read_csv(Path(app.state.cfg["paths"]["processed"]) / "revenue_trend.csv")
    )


@app.get("/importance")
def importance() -> list[dict[str, Any]]:
    """Return global SHAP importance computed on validation customers."""
    return records(
        pd.read_csv(
            Path(app.state.cfg["paths"]["artifacts"]) / "feature_importance.csv"
        )
    )
