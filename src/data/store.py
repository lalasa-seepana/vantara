"""PostgreSQL persistence for actual customers, transactions and scored results."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from src.utils.common import config

metadata = MetaData()
segments = Table(
    "segments",
    metadata,
    Column("segment_id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("profile", JSON, nullable=False),
)
customers = Table(
    "customers",
    metadata,
    Column("customer_id", String, primary_key=True),
    Column("country", String, nullable=False),
    Column("monetary", Float, nullable=False),
    Column("value_tier", String, nullable=False),
    Column("segment_id", ForeignKey("segments.segment_id"), nullable=False),
    Column("features", JSON, nullable=False),
)
transactions = Table(
    "transactions",
    metadata,
    Column("transaction_id", Integer, primary_key=True),
    Column(
        "customer_id", ForeignKey("customers.customer_id"), nullable=False, index=True
    ),
    Column("invoice", String, nullable=False),
    Column("stock_code", String, nullable=False),
    Column("invoice_date", DateTime, nullable=False),
    Column("quantity", Float, nullable=False),
    Column("price", Float, nullable=False),
    Column("revenue", Float, nullable=False),
    Column("is_return", Boolean, nullable=False),
)
predictions = Table(
    "predictions",
    metadata,
    Column("customer_id", ForeignKey("customers.customer_id"), primary_key=True),
    Column("model_version", String, nullable=False),
    Column("scored_at", DateTime(timezone=True), nullable=False),
    Column("payload", JSON, nullable=False),
)


def engine() -> Engine:
    """Require an environment-provided PostgreSQL URL; never embed a password."""
    url = os.getenv("DATABASE_URL")
    if not url or not url.startswith("postgresql"):
        raise ValueError("DATABASE_URL must point to PostgreSQL")
    return create_engine(url, pool_pre_ping=True)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas and NumPy values to JSON-safe native values."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def save_predictions(db: Engine, features: pd.DataFrame, scored: pd.DataFrame) -> None:
    """Atomically upsert features and predictions, including new feature records."""
    by_id = {r["customer_id"]: r for r in records(features)}
    now = datetime.now(timezone.utc)
    with db.begin() as conn:
        for row in records(scored):
            customer = {
                key: row[key]
                for key in [
                    "customer_id",
                    "country",
                    "monetary",
                    "value_tier",
                    "segment_id",
                ]
            }
            customer["features"] = by_id[row["customer_id"]]
            statement = insert(customers).values(**customer)
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=["customer_id"],
                    set_={k: v for k, v in customer.items() if k != "customer_id"},
                )
            )
            value = {
                "customer_id": row["customer_id"],
                "model_version": row["model_version"],
                "scored_at": now,
                "payload": row,
            }
            statement = insert(predictions).values(**value)
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=["customer_id"],
                    set_={k: v for k, v in value.items() if k != "customer_id"},
                )
            )


def initialize(db: Engine, cfg: dict[str, Any]) -> None:
    """Create schema and seed actual snapshot data only when the store is empty."""
    metadata.create_all(db)
    with db.connect() as conn:
        count = conn.scalar(select(func.count()).select_from(predictions))
    with db.connect() as conn:
        transaction_count = conn.scalar(select(func.count()).select_from(transactions))
    if count and transaction_count:
        return
    root = Path(cfg["paths"]["processed"])
    features = pd.read_parquet(root / "customers.parquet")
    scores = pd.read_csv(root / "predictions.csv", dtype={"customer_id": str})
    profiles = pd.read_csv(Path(cfg["paths"]["artifacts"]) / "segment_profiles.csv")
    profiles = profiles[profiles.algorithm == "kmeans"]
    with db.begin() as conn:
        for row in records(profiles):
            conn.execute(
                insert(segments)
                .values(segment_id=int(row["segment"]), name=row["name"], profile=row)
                .on_conflict_do_nothing()
            )
    if not count:
        save_predictions(db, features, scores)
    # Use COPY for the historical transaction table, avoiding one INSERT per line.
    tx = pd.read_parquet(Path(cfg["paths"]["interim"]) / "transactions.parquet")
    tx = tx[
        (tx.invoice_date < pd.Timestamp(cfg["features"]["cutoff"]))
        & tx.customer_id.isin(features.customer_id)
    ].copy()
    tx["transaction_id"] = tx.index + 1
    cols = [c.name for c in transactions.columns]
    with db.begin() as conn:
        existing = conn.scalar(select(func.count()).select_from(transactions))
    if not existing:
        raw = db.raw_connection()
        try:
            with raw.cursor().copy(
                "COPY transactions ("
                + ",".join(cols)
                + ") FROM STDIN WITH (FORMAT CSV)"
            ) as copy:
                copy.write(tx[cols].to_csv(index=False, header=False))
            raw.commit()
        finally:
            raw.close()


def health(db: Engine) -> bool:
    """Check database connectivity with a real query."""
    with db.connect() as conn:
        return conn.scalar(text("SELECT 1")) == 1


if __name__ == "__main__":
    initialize(engine(), config())
