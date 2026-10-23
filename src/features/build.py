"""Point-in-time customer features and fully observed 90-day labels."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.common import write_json

LOG = logging.getLogger(__name__)


def categories(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Create explicitly named description proxies, not official categories."""
    result = pd.Series(cfg["features"]["fallback_category"], index=df.index)
    remaining = pd.Series(True, index=df.index)
    for name, pattern in cfg["features"]["category_patterns"].items():
        mask = (
            df.description.fillna("").str.contains(pattern, case=False, regex=True)
            & remaining
        )
        result.loc[mask] = name
        remaining &= ~mask
    return result


def product_purchases(df: pd.DataFrame) -> pd.DataFrame:
    """Return genuine paid product purchases with valid customer IDs."""
    return df[
        (df.quantity > 0)
        & ~df.is_return
        & ~df.invalid_price
        & ~df.is_admin
        & df.customer_id.notna()
    ].copy()


def build_features(
    df: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use strictly pre-cutoff rows for every feature, catalog and sequence."""
    past = df[df.invoice_date < cutoff].copy()
    buys = product_purchases(past)
    if buys.empty:
        raise ValueError("No identified product purchases before cutoff")
    # Use the most common normalized description per SKU from history only.
    lookup = (
        buys.dropna(subset=["description"])
        .groupby(["stock_code", "description"])
        .size()
        .reset_index(name="count")
        .sort_values(
            ["stock_code", "count", "description"], ascending=[True, False, True]
        )
        .drop_duplicates("stock_code")
        .set_index("stock_code")
        .description
    )
    buys["description"] = buys.stock_code.map(lookup).fillna(buys.description)
    buys["category_proxy"] = categories(buys, cfg)
    grouped = buys.groupby("customer_id", sort=True)
    orders = buys.groupby(["customer_id", "invoice"], as_index=False).agg(
        invoice_date=("invoice_date", "min"),
        amount=("revenue", "sum"),
        units=("quantity", "sum"),
    )
    dominant = (
        buys.groupby(["customer_id", "invoice", "category_proxy"])
        .revenue.sum()
        .reset_index()
        .sort_values(["revenue", "category_proxy"], ascending=[False, True])
        .drop_duplicates(["customer_id", "invoice"])
    )
    orders = orders.merge(
        dominant[["customer_id", "invoice", "category_proxy"]],
        on=["customer_id", "invoice"],
        validate="one_to_one",
    )
    orders = orders.sort_values(
        ["customer_id", "invoice_date", "invoice"], kind="stable"
    )
    orders["gap_days"] = (
        orders.groupby("customer_id")
        .invoice_date.diff()
        .dt.total_seconds()
        .div(86400)
        .fillna(0)
    )
    feature = grouped.agg(
        monetary=("revenue", "sum"),
        units=("quantity", "sum"),
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
    )
    feature["recency"] = (cutoff - feature.last_purchase).dt.total_seconds() / 86400
    feature["tenure"] = (cutoff - feature.first_purchase).dt.total_seconds() / 86400
    feature["frequency"] = orders.groupby("customer_id").size()
    feature["average_spend"] = feature.monetary / feature.frequency
    feature["basket_size"] = feature.units / feature.frequency
    feature["gap_variance"] = (
        orders[orders.gap_days > 0].groupby("customer_id").gap_days.var(ddof=0)
    )
    feature["gap_variance"] = feature.gap_variance.fillna(0)
    months = cfg["features"]["trend_months"]
    periods = pd.period_range(
        end=(cutoff - pd.Timedelta(days=1)).to_period("M"), periods=months, freq="M"
    )
    counts = (
        orders.groupby(["customer_id", orders.invoice_date.dt.to_period("M")])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=periods, fill_value=0)
    )
    x = np.arange(months) - (months - 1) / 2
    feature["frequency_trend"] = counts.to_numpy() @ x / (x @ x)
    month_spend = (
        buys.groupby(["customer_id", buys.invoice_date.dt.month])
        .revenue.sum()
        .unstack(fill_value=0)
    )
    feature["seasonal_concentration"] = month_spend.max(axis=1) / month_spend.sum(
        axis=1
    )
    valid = past[past.customer_id.notna() & ~past.invalid_price & ~past.is_admin]
    returned = (
        valid[valid.is_return]
        .groupby("customer_id")
        .quantity.apply(lambda v: v.abs().sum())
    )
    feature["return_rate"] = (
        returned.reindex(feature.index, fill_value=0) / feature.units
    )
    feature["historical_clv"] = (
        valid.groupby("customer_id").revenue.sum().reindex(feature.index, fill_value=0)
    )
    # Only a customer's earlier observed price is used. Bulk prices are a confounder.
    prior_max = buys.groupby(["customer_id", "stock_code"]).price.transform(
        lambda s: s.cummax().shift()
    )
    buys["discount_proxy"] = (buys.price < prior_max).astype(float)
    feature["discount_sensitivity_proxy"] = buys.groupby(
        "customer_id"
    ).discount_proxy.mean()
    feature["discount_observations"] = prior_max.notna().groupby(buys.customer_id).sum()
    feature["country"] = (
        buys.sort_values("invoice_date").groupby("customer_id").country.last()
    )
    sku_counts = (
        buys.groupby(["customer_id", "stock_code"]).size().reset_index(name="count")
    )
    feature["top_sku"] = (
        sku_counts.sort_values(["count", "stock_code"], ascending=[False, True])
        .drop_duplicates("customer_id")
        .set_index("customer_id")
        .stock_code
    )
    affinity = (
        buys.groupby(["customer_id", "category_proxy"])
        .revenue.sum()
        .unstack(fill_value=0)
    )
    for category in [
        *cfg["features"]["category_patterns"],
        cfg["features"]["fallback_category"],
    ]:
        feature[f"affinity_{category}"] = (
            affinity.get(category, pd.Series(0, index=feature.index)) / feature.monetary
        )
    scales = cfg["features"]["engagement_scales"]
    feature["engagement_score"] = (
        100
        * (
            np.exp(-feature.recency / scales["recency"])
            + feature.frequency / (feature.frequency + scales["frequency"])
            + feature.monetary / (feature.monetary + scales["monetary"])
        )
        / 3
    )
    feature = feature.drop(
        columns=["first_purchase", "last_purchase", "units"]
    ).reset_index()
    if not np.isfinite(feature.select_dtypes("number")).all().all():
        raise ValueError("Non-finite engineered features")
    return feature, orders


def build_targets(
    df: pd.DataFrame, ids: pd.Series, cutoff: pd.Timestamp, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Label only a complete future horizon; do not invent censored outcomes."""
    horizon = cfg["features"]["horizon_days"]
    end = cutoff + pd.Timedelta(days=horizon)
    if df.invoice_date.max() < end:
        raise ValueError("Future observation window is incomplete")
    future = df[(df.invoice_date >= cutoff) & (df.invoice_date < end)]
    purchases = product_purchases(future).sort_values("invoice_date", kind="stable")
    targets = pd.DataFrame(index=pd.Index(ids, name="customer_id"))
    first = purchases.groupby("customer_id").invoice_date.min()
    targets["churn"] = (~targets.index.isin(first.index)).astype(int)
    targets["next_purchase_days_capped"] = (
        ((first - cutoff).dt.total_seconds() / 86400)
        .reindex(targets.index)
        .fillna(horizon)
    )
    net = (
        future[future.customer_id.notna() & ~future.invalid_price & ~future.is_admin]
        .groupby("customer_id")
        .revenue.sum()
    )
    targets["clv_90d"] = net.reindex(targets.index, fill_value=0)
    purchases["category_proxy"] = categories(purchases, cfg)
    first_invoice = purchases.drop_duplicates("customer_id")[["customer_id", "invoice"]]
    first_basket = purchases.merge(first_invoice, on=["customer_id", "invoice"])
    dominant = (
        first_basket.groupby(["customer_id", "category_proxy"])
        .revenue.sum()
        .reset_index()
        .sort_values(["revenue", "category_proxy"], ascending=[False, True])
        .drop_duplicates("customer_id")
        .set_index("customer_id")
        .category_proxy
    )
    targets["next_category_proxy"] = dominant.reindex(targets.index).fillna(
        "no_purchase"
    )
    return targets.reset_index()


def make_sequences(
    orders: pd.DataFrame, ids: pd.Series, cfg: dict[str, Any]
) -> np.ndarray:
    """Pad each historical invoice sequence; channel zero is a validity mask."""
    names = [
        *cfg["features"]["category_patterns"],
        cfg["features"]["fallback_category"],
    ]
    length = cfg["features"]["sequence_length"]
    result = np.zeros((len(ids), length, 3 + len(names)), dtype=np.float32)
    groups = {str(key): value for key, value in orders.groupby("customer_id")}
    for i, customer_id in enumerate(ids):
        rows = groups[str(customer_id)].tail(length)
        n = len(rows)
        result[i, :n, 0] = 1
        result[i, :n, 1] = np.log1p(rows.amount)
        result[i, :n, 2] = np.log1p(rows.gap_days)
        for j, category in enumerate(rows.category_proxy):
            result[i, j, 3 + names.index(category)] = 1
    return result


def run_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Persist customer-disjoint splits, features, labels and sequences."""
    cutoff = pd.Timestamp(cfg["features"]["cutoff"])
    features, orders = build_features(df, cutoff, cfg)
    targets = build_targets(df, features.customer_id, cutoff, cfg)
    merged = features.merge(targets, on="customer_id", validate="one_to_one")
    train, test = train_test_split(
        merged.index,
        test_size=cfg["split"]["test_size"],
        stratify=merged.churn,
        random_state=cfg["seed"],
    )
    train, val = train_test_split(
        train,
        test_size=cfg["split"]["validation_size"] / (1 - cfg["split"]["test_size"]),
        stratify=merged.loc[train, "churn"],
        random_state=cfg["seed"],
    )
    merged["split"] = "train"
    merged.loc[val, "split"] = "validation"
    merged.loc[test, "split"] = "test"
    output = Path(cfg["paths"]["processed"])
    output.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output / "customers.parquet", index=False)
    merged.to_csv(output / "customers.csv", index=False)
    orders.to_parquet(output / "purchase_sequences.parquet", index=False)
    np.save(output / "sequences.npy", make_sequences(orders, features.customer_id, cfg))
    catalog = product_purchases(df[df.invoice_date < cutoff])
    catalog["category_proxy"] = categories(catalog, cfg)
    catalog = (
        catalog.groupby(["stock_code", "category_proxy"])
        .agg(description=("description", "first"), orders=("invoice", "nunique"))
        .reset_index()
    )
    catalog.to_csv(output / "product_catalog.csv", index=False)
    write_json(
        output / "feature_metadata.json",
        {
            "cutoff_exclusive": str(cutoff),
            "horizon_days": cfg["features"]["horizon_days"],
            "customers": len(merged),
            "split_counts": merged.split.value_counts().to_dict(),
            "churn_rate": float(merged.churn.mean()),
            "category_definition": "Ordered description keyword proxies; no official categories exist",
            "discount_definition": "Share of lines below the same customer's earlier maximum SKU price; not a causal promotion response",
        },
    )
    LOG.info(
        "Built %s customers with complete targets and disjoint 70/15/15 splits",
        len(merged),
    )
    return merged
