"""Compare K-Means and Gaussian mixtures and name segments from real profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils.common import write_json

SEGMENT_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "return_rate",
    "seasonal_concentration",
]


def run_segments(cfg: dict[str, Any]) -> None:
    """Select k from inertia and silhouette, then profile both algorithms."""
    data = pd.read_parquet(Path(cfg["paths"]["processed"]) / "customers.parquet")
    train = data[data.split == "train"]
    scaler = StandardScaler().fit(np.log1p(train[SEGMENT_FEATURES]))
    x = scaler.transform(np.log1p(train[SEGMENT_FEATURES]))
    xa = scaler.transform(np.log1p(data[SEGMENT_FEATURES]))
    settings = cfg["segmentation"]
    candidates, models = [], {}
    for k in settings["k_values"]:
        model = KMeans(
            n_clusters=k, n_init=settings["n_init"], random_state=cfg["seed"]
        ).fit(x)
        labels = model.labels_
        candidates.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(
                    x,
                    labels,
                    sample_size=min(len(x), settings["sample_size"]),
                    random_state=cfg["seed"],
                ),
                "davies_bouldin": davies_bouldin_score(x, labels),
                "smallest_cluster_fraction": float(
                    np.bincount(labels).min() / len(labels)
                ),
            }
        )
        models[k] = model
    table = pd.DataFrame(candidates)
    # Geometric elbow: maximum distance below the normalized endpoint chord.
    inertia = table.inertia.to_numpy()
    norm = (inertia - inertia.min()) / max(np.ptp(inertia), 1e-12)
    gap = np.linspace(1, 0, len(table)) - norm
    elbow_index = int(np.argmax(gap))
    table["elbow_distance"] = gap
    nearby = table.iloc[max(0, elbow_index - 1) : elbow_index + 2]
    eligible = nearby[
        nearby.smallest_cluster_fraction >= settings["min_cluster_fraction"]
    ]
    best = (
        (eligible if len(eligible) else nearby)
        .sort_values("silhouette", ascending=False)
        .iloc[0]
    )
    k = int(best.k)
    kmeans = models[k]
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=settings["n_init"],
        random_state=cfg["seed"],
    ).fit(x)
    metrics = [
        {
            "algorithm": "K-Means",
            "k": k,
            "silhouette": float(best.silhouette),
            "davies_bouldin": float(best.davies_bouldin),
        }
    ]
    gmm_labels = gmm.predict(x)
    metrics.append(
        {
            "algorithm": "Gaussian mixture",
            "k": k,
            "silhouette": silhouette_score(
                x,
                gmm_labels,
                sample_size=min(len(x), settings["sample_size"]),
                random_state=cfg["seed"],
            ),
            "davies_bouldin": davies_bouldin_score(x, gmm_labels),
        }
    )
    output = Path(cfg["paths"]["artifacts"])
    table.to_csv(output / "cluster_search.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "cluster_metrics.csv", index=False)
    centers = train[SEGMENT_FEATURES].median()
    profile_tables = []
    names = {}
    assignments = data[["customer_id"]].copy()
    for algorithm, model in [("kmeans", kmeans), ("gmm", gmm)]:
        labels = model.predict(xa)
        assignments[f"{algorithm}_segment"] = labels
        profiling = data.assign(segment=labels)
        # Names are learned from training profiles; summaries cover all assigned customers.
        learned = (
            profiling[profiling.split == "train"]
            .groupby("segment")[SEGMENT_FEATURES]
            .median()
        )
        names[algorithm] = {}
        for cluster, values in learned.iterrows():
            activity = "Lapsed" if values.recency > centers.recency else "Recent"
            habit = "frequent" if values.frequency > centers.frequency else "occasional"
            value = (
                "high value" if values.monetary > centers.monetary else "lower value"
            )
            names[algorithm][
                int(cluster)
            ] = f"{activity}, {habit}, {value} ({int(cluster)})"
        profiles = (
            profiling.groupby("segment")
            .agg(
                customers=("customer_id", "size"),
                mean_recency=("recency", "mean"),
                mean_frequency=("frequency", "mean"),
                mean_monetary=("monetary", "mean"),
                mean_return_rate=("return_rate", "mean"),
                churn_rate=("churn", "mean"),
            )
            .reset_index()
        )
        profiles["name"] = profiles.segment.map(names[algorithm])
        profiles["algorithm"] = algorithm
        profile_tables.append(profiles)
    pd.concat(profile_tables).to_csv(output / "segment_profiles.csv", index=False)
    assignments.to_csv(output / "segment_assignments.csv", index=False)
    tiers = train.monetary.quantile(cfg["features"]["value_tier_quantiles"]).tolist()
    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "gmm": gmm,
            "names": names,
            "value_tier_cutoffs": tiers,
        },
        output / "segments.joblib",
    )
    write_json(
        output / "segment_selection.json",
        {
            "k": k,
            "elbow_k": int(table.iloc[elbow_index].k),
            "rule": "Best silhouette near the geometric inertia elbow, excluding tiny clusters when possible",
            "metrics_partition": "training customers only",
        },
    )
