"""Download, validate and clean the exact two-year UCI source."""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.utils.common import config, digest, setup_logging, write_json

LOG = logging.getLogger(__name__)
COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def download(cfg: dict[str, Any]) -> Path:
    """Fetch the original workbook only when it is absent; never overwrite it."""
    path = Path(cfg["paths"]["raw"])
    if not path.exists():
        LOG.info("Downloading Online Retail II from UCI")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(cfg["source"]["url"], timeout=120) as response:
            archive = ZipFile(io.BytesIO(response.read()))
        with path.open("xb") as output:
            output.write(archive.read("online_retail_II.xlsx"))
    if digest(path) != cfg["source"]["sha256"]:
        raise ValueError("Source checksum differs from the verified UCI workbook")
    return path


def load_source(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load both sheets, normalize types and sort before cleaning."""
    path = download(cfg)
    parts = pd.read_excel(path, sheet_name=cfg["source"]["sheets"], engine="openpyxl")
    df = pd.concat(parts.values(), ignore_index=True)
    if set(df.columns) != set(COLUMNS):
        raise ValueError(f"Unexpected workbook schema: {list(df.columns)}")
    df = df.rename(columns=COLUMNS)
    for column in ["invoice", "stock_code", "description", "country"]:
        df[column] = df[column].astype("string").str.strip()
    for column in ["quantity", "price", "customer_id"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["customer_id"] = df["customer_id"].astype("Int64").astype("string")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="raise")
    validate(df, cfg, exact_size=True)
    return df.sort_values("invoice_date", kind="stable").reset_index(drop=True)


def validate(df: pd.DataFrame, cfg: dict[str, Any], exact_size: bool = False) -> None:
    """Fail on schema, date, numeric or source null-rate violations."""
    if not set(COLUMNS.values()).issubset(df.columns) or df.empty:
        raise ValueError("Transaction schema is incomplete or empty")
    source = cfg["source"]
    if exact_size and len(df) != source["expected_rows"]:
        raise ValueError("Source row count does not match Online Retail II")
    for column in ["quantity", "price"]:
        if not np.isfinite(df[column]).all():
            raise ValueError(f"Non-finite {column}")
    required = ["invoice", "stock_code", "invoice_date", "country"]
    if df[required].isna().any().any():
        raise ValueError("Missing required transaction fields")
    if not df.invoice_date.between(
        pd.Timestamp(source["first_date"]),
        pd.Timestamp(source["end_date_exclusive"]),
        inclusive="left",
    ).all():
        raise ValueError("Invoice date outside the documented source period")
    for field in ["customer", "description"]:
        column = "customer_id" if field == "customer" else field
        if df[column].isna().mean() > source[f"max_missing_{field}_fraction"]:
            raise ValueError(f"Excessive missing {column}")


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flag unusual lines and preserve returns and anonymous product records."""
    validate(df, cfg)
    audit: dict[str, Any] = {
        "raw_rows": len(df),
        "missing_customer_rows": int(df.customer_id.isna().sum()),
    }
    columns = list(COLUMNS.values())
    result = df.drop_duplicates(columns).copy()
    audit["exact_duplicates_removed"] = len(df) - len(result)
    result["description"] = result.description.str.upper().str.replace(
        r"\s+", " ", regex=True
    )
    result["is_return"] = (
        result.quantity < 0
    ) | result.invoice.str.upper().str.startswith("C")
    result["invalid_price"] = result.price <= 0
    result["is_admin"] = result.stock_code.str.upper().isin(
        cfg["features"]["admin_codes"]
    ) | ~result.stock_code.str.match(r"^\d{5}[A-Za-z]*$")
    # Bounds use history only. Flags do not delete wholesale purchases.
    historical = result[result.invoice_date < pd.Timestamp(cfg["features"]["cutoff"])]
    for col in ["quantity", "price"]:
        q1, q3 = historical[col].quantile([0.25, 0.75])
        margin = cfg["features"]["iqr_multiplier"] * (q3 - q1)
        result[f"{col}_outlier"] = (result[col] < q1 - margin) | (
            result[col] > q3 + margin
        )
        audit[f"{col}_iqr_bounds"] = [float(q1 - margin), float(q3 + margin)]
    result["bulk_purchase"] = (
        (result.quantity >= cfg["features"]["bulk_quantity"])
        & ~result.is_return
        & ~result.invalid_price
    )
    result["revenue"] = result.quantity * result.price
    # A positive cancellation line must not count as a purchase or positive revenue.
    result.loc[result.is_return, "revenue"] = -result.loc[
        result.is_return, "revenue"
    ].abs()
    audit.update(
        {
            "clean_rows": len(result),
            "identified_customers": int(result.customer_id.nunique()),
            "return_rows": int(result.is_return.sum()),
            "invalid_price_rows": int(result.invalid_price.sum()),
            "admin_rows": int(result.is_admin.sum()),
            "bulk_rows_retained": int(result.bulk_purchase.sum()),
        }
    )
    return result.reset_index(drop=True), audit


def run_data(cfg: dict[str, Any]) -> pd.DataFrame:
    """Produce cleaned data and a checksum-backed audit."""
    raw = Path(cfg["paths"]["raw"])
    df = load_source(cfg)
    before = digest(raw)
    cleaned, audit = clean(df, cfg)
    interim = Path(cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(interim / "transactions.parquet", index=False)
    audit.update(
        {
            "source_sha256": before,
            "source_url": cfg["source"]["url"],
            "sheet_names": cfg["source"]["sheets"],
            "min_date": str(df.invoice_date.min()),
            "max_date": str(df.invoice_date.max()),
        }
    )
    write_json(interim / "validation_audit.json", audit)
    if digest(raw) != before:
        raise RuntimeError("Original workbook changed during processing")
    LOG.info("Validated %s source rows; retained %s clean rows", len(df), len(cleaned))
    return cleaned


def main() -> None:
    """Run data/features, or the complete pipeline, from one command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Train, score and produce all reports"
    )
    args = parser.parse_args()
    setup_logging()
    cfg = config()
    try:
        df = run_data(cfg)
        from src.features.build import run_features

        run_features(df, cfg)
        if args.all:
            from src.models.train import run_training
            from src.segmentation.cluster import run_segments
            from src.explainability.explain import run_explanations
            from src.models.predict import run_scoring
            from src.utils.reporting import run_reports

            run_training(cfg)
            run_segments(cfg)
            run_explanations(cfg)
            run_scoring(cfg)
            run_reports(cfg)
    except Exception:
        LOG.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()
