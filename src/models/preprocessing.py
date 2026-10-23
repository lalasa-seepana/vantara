"""Train-fitted transformations shared by all model and API paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Map high-cardinality SKU values to training-set customer frequencies."""

    def fit(self, x: pd.DataFrame, y: np.ndarray | None = None) -> FrequencyEncoder:
        """Learn frequencies only from the supplied fit partition."""
        self.frequencies_ = (
            pd.Series(np.asarray(x).ravel()).value_counts(normalize=True).to_dict()
        )
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        """Give unseen SKU values zero frequency."""
        return (
            pd.Series(np.asarray(x).ravel())
            .map(self.frequencies_)
            .fillna(0)
            .to_numpy()
            .reshape(-1, 1)
        )

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        """Name the encoded output for reports and explanations."""
        return np.array(["top_sku_frequency"], dtype=object)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Exclude outcomes, identifiers and duplicate/composite reporting fields."""
    excluded = {
        "customer_id",
        "churn",
        "clv_90d",
        "next_purchase_days_capped",
        "next_category_proxy",
        "split",
        "historical_clv",
        "engagement_score",
        "affinity_other",
        "discount_observations",
    }
    return [c for c in df if c not in excluded]


def preprocessor(columns: list[str], scaled: bool) -> ColumnTransformer:
    """One-hot encode country, frequency encode SKU and optionally scale numbers."""
    numeric = [c for c in columns if c not in ["country", "top_sku"]]
    numeric_step = StandardScaler() if scaled else "passthrough"
    sku_step = (
        Pipeline([("frequency", FrequencyEncoder()), ("scale", StandardScaler())])
        if scaled
        else FrequencyEncoder()
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_step, numeric),
            (
                "country",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["country"],
            ),
            ("sku", sku_step, ["top_sku"]),
        ],
        verbose_feature_names_out=False,
    )
