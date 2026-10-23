"""Shared scoring logic for batch jobs and validated API feature records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.deep import transform_sequences
from src.segmentation.cluster import SEGMENT_FEATURES
from src.utils.common import read_json, write_json


class Predictor:
    """Load the required persisted artifacts once per process."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Load trained models; fail clearly if training has not completed."""
        self.cfg = cfg
        root = Path(cfg["paths"]["artifacts"])
        self.selection = read_json(root / "selection.json")
        self.churn = joblib.load(root / f"{self.selection['model']}.joblib")
        self.clv = joblib.load(root / "clv.joblib")
        self.segments = joblib.load(root / "segments.joblib")
        self.anomaly = joblib.load(root / "autoencoder.joblib")
        self.lstm = joblib.load(root / "lstm.joblib")
        self.catalog = pd.read_csv(
            Path(cfg["paths"]["processed"]) / "product_catalog.csv",
            dtype={"stock_code": str},
        )
        self.version = (
            f"{self.selection['model']}-{cfg['features']['cutoff']}-seed{cfg['seed']}"
        )

    def score(
        self, frame: pd.DataFrame, sequences: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Score real feature records; sequence outputs stay absent if unavailable."""
        columns = self.selection["feature_columns"]
        x = frame[columns]
        if self.selection["model"] == "ann":
            p = self.churn["model"].predict_proba(self.churn["transform"].transform(x))[
                :, 1
            ]
        else:
            p = self.churn.predict_proba(x)[:, 1]
        result = (
            frame[["customer_id", "country", "monetary", "recency", "frequency"]]
            .copy()
            .reset_index(drop=True)
        )
        result["churn_probability"] = p
        result["purchase_probability_90d"] = 1 - p
        result["high_risk"] = p >= self.selection["threshold"]
        result["predicted_clv_90d"] = self.clv.predict(x)
        result["retention_priority"] = p * np.maximum(result.predicted_clv_90d, 0)
        cluster_x = self.segments["scaler"].transform(np.log1p(frame[SEGMENT_FEATURES]))
        result["segment_id"] = self.segments["kmeans"].predict(cluster_x)
        result["segment"] = result.segment_id.map(self.segments["names"]["kmeans"])
        low, high = self.segments["value_tier_cutoffs"]
        result["value_tier"] = pd.cut(
            result.monetary,
            [-np.inf, low, high, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)
        errors = self.anomaly["model"].predict(self.anomaly["transform"].transform(x))
        result["anomaly_error"] = errors
        result["anomaly_flag"] = errors > self.anomaly["threshold"]
        names = self.selection["category_names"]
        if sequences is not None:
            seq = transform_sequences(sequences, self.lstm["transform"])
            result["next_purchase_days_capped"] = self.lstm["model"].predict(seq)
            result["next_category_proxy"] = [
                names[i] for i in self.lstm["model"].raw(seq)[:, 1:].argmax(axis=1)
            ]
        else:
            result["next_purchase_days_capped"] = np.nan
            result["next_category_proxy"] = None
        # History-based category fallback is explicitly labelled as such.
        affinity = frame[
            [f"affinity_{name}" for name in names if f"affinity_{name}" in frame]
        ]
        fallback = (
            affinity.idxmax(axis=1).str.replace("affinity_", "", regex=False).to_numpy()
        )
        result["recommendation_category"] = result.next_category_proxy.fillna(
            pd.Series(fallback)
        )
        result["recommendation_basis"] = (
            "LSTM category proxy"
            if sequences is not None
            else "Historical category affinity proxy"
        )
        recommendations = {}
        for name in names:
            selected = (
                self.catalog[self.catalog.category_proxy == name]
                .sort_values("orders", ascending=False)
                .drop_duplicates("stock_code")
                .head(self.cfg["features"]["recommendation_count"])
            )
            recommendations[name] = "; ".join(
                f"{r.stock_code}: {r.description}" for r in selected.itertuples()
            )
        result["recommendations"] = result.recommendation_category.map(
            recommendations
        ).fillna("")
        result["model_version"] = self.version
        result["as_of"] = self.cfg["features"]["cutoff"]
        return result


def run_scoring(cfg: dict[str, Any]) -> None:
    """Persist snapshot predictions and a simple data-derived revenue forecast."""
    processed = Path(cfg["paths"]["processed"])
    data = pd.read_parquet(processed / "customers.parquet")
    predictor = Predictor(cfg)
    predictor.score(data, np.load(processed / "sequences.npy")).to_csv(
        processed / "predictions.csv", index=False
    )
    transactions = pd.read_parquet(
        Path(cfg["paths"]["interim"]) / "transactions.parquet"
    )
    transactions = transactions[
        (transactions.invoice_date < pd.Timestamp(cfg["features"]["cutoff"]))
        & ~transactions.invalid_price
        & ~transactions.is_admin
    ]
    monthly = transactions.groupby(transactions.invoice_date.dt.to_period("M")).agg(
        revenue=("revenue", "sum"), invoices=("invoice", "nunique")
    )
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.reset_index().rename(columns={"invoice_date": "month"})
    monthly["kind"] = "observed"
    recent = monthly.tail(cfg["forecast"]["recent_months"])
    coefficients = np.polyfit(np.arange(len(recent)), recent.revenue.to_numpy(), 1)
    dates = pd.date_range(
        pd.Timestamp(cfg["features"]["cutoff"]),
        periods=cfg["forecast"]["months"],
        freq="MS",
    )
    forecasts = np.maximum(
        0, np.polyval(coefficients, np.arange(len(recent), len(recent) + len(dates)))
    )
    future = pd.DataFrame(
        {
            "month": dates,
            "revenue": forecasts,
            "invoices": np.nan,
            "kind": "linear forecast",
        }
    )
    pd.concat([monthly, future], ignore_index=True).to_csv(
        processed / "revenue_trend.csv", index=False
    )
    write_json(
        processed / "forecast_metadata.json",
        {
            "method": "Least-squares line over the last complete historical months; negative revenue forecasts clipped to zero",
            "fit_months": len(recent),
            "horizon_months": len(dates),
            "coefficients": coefficients.tolist(),
            "limitation": "Illustrative extrapolation, not validated demand or causal revenue uplift",
        },
    )
