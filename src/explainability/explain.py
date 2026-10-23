"""SHAP, LIME and plain-language explanations for the selected churn model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay

from src.utils.common import read_json, write_json


class Explainer:
    """Explain actual model inputs after the persisted preprocessing step."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Load the frozen model and a training-only explanation background."""
        output = Path(cfg["paths"]["artifacts"])
        self.cfg = cfg
        self.selection = read_json(output / "selection.json")
        self.name = self.selection["model"]
        stored = joblib.load(output / f"{self.name}.joblib")
        if self.name == "ann":
            self.prep, self.model = stored["transform"], stored["model"]
        else:
            self.prep, self.model = (
                stored.named_steps["preprocess"],
                stored.named_steps["model"],
            )
        data = pd.read_parquet(Path(cfg["paths"]["processed"]) / "customers.parquet")
        train = data[data.split == "train"]
        self.train_x = np.asarray(
            self.prep.transform(train[self.selection["feature_columns"]]), dtype=float
        )
        self.names = list(self.prep.get_feature_names_out())
        rng = np.random.default_rng(cfg["seed"])
        background = self.train_x[
            rng.choice(
                len(train),
                min(cfg["explain"]["background_rows"], len(train)),
                replace=False,
            )
        ]
        if self.name in ["random_forest", "decision_tree", "xgboost", "lightgbm"]:
            self.shap = shap.TreeExplainer(
                self.model,
                data=background,
                model_output="probability",
                feature_perturbation="interventional",
            )
            self.tree = True
        else:
            self.shap = shap.Explainer(
                lambda x: self.model.predict_proba(x)[:, 1],
                background,
                algorithm="permutation",
                seed=cfg["seed"],
            )
            self.tree = False
        self.lime = LimeTabularExplainer(
            self.train_x,
            feature_names=self.names,
            class_names=["purchase", "churn"],
            mode="classification",
            random_state=cfg["seed"],
            discretize_continuous=True,
        )

    def values(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return churn-probability SHAP contributions and baselines."""
        explained = self.shap(x, check_additivity=False) if self.tree else self.shap(x)
        values, base = explained.values, np.asarray(explained.base_values)
        if values.ndim == 3:
            values = values[:, :, 1]
            base = base[:, 1]
        return values, np.broadcast_to(base, (len(x),))

    def explain(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Explain one customer with both methods and a factual summary."""
        x = np.asarray(
            self.prep.transform(frame[self.selection["feature_columns"]]), dtype=float
        )
        values, base = self.values(x)
        probability = float(self.model.predict_proba(x)[0, 1])
        ranking = np.argsort(np.abs(values[0]))[::-1][
            : self.cfg["explain"]["max_features"]
        ]
        contributions = [
            {
                "feature": self.names[i],
                "value": float(x[0, i]),
                "contribution": float(values[0, i]),
            }
            for i in ranking
        ]
        # Reset LIME's generator so repeated explanations are reproducible.
        self.lime.random_state.seed(self.cfg["seed"])
        local = self.lime.explain_instance(
            x[0],
            self.model.predict_proba,
            num_features=self.cfg["explain"]["max_features"],
            num_samples=self.cfg["explain"]["lime_samples"],
        )
        positive = [
            c["feature"].replace("_", " ")
            for c in contributions
            if c["contribution"] > 0
        ][:3]
        state = (
            "flagged" if probability >= self.selection["threshold"] else "not flagged"
        )
        sentence = f"This customer is {state} for no purchase within 90 days (estimated probability {probability:.1%}). "
        sentence += (
            f"The main factors raising the estimate are {', '.join(positive)}."
            if positive
            else "The strongest displayed factors reduce the estimate from its baseline."
        )
        return {
            "customer_id": str(frame.iloc[0].customer_id),
            "model": self.name,
            "probability": probability,
            "base_probability": float(base[0]),
            "shap_sum": float(base[0] + values[0].sum()),
            "contributions": contributions,
            "lime": [
                {"condition": name, "weight": float(weight)}
                for name, weight in local.as_list()
            ],
            "lime_local_r2": float(local.score),
            "summary": sentence,
            "units": "SHAP contributions are changes in model probability; LIME is an approximate local fit. Neither establishes causality.",
        }


def run_explanations(cfg: dict[str, Any]) -> None:
    """Save real global, low/high/borderline local and dependence outputs."""
    docs, output = Path(cfg["paths"]["docs"]), Path(cfg["paths"]["artifacts"])
    docs.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(Path(cfg["paths"]["processed"]) / "customers.parquet")
    explainer = Explainer(cfg)
    validation = data[data.split == "validation"]
    sample = validation.sample(
        min(len(validation), cfg["explain"]["summary_rows"]), random_state=cfg["seed"]
    )
    x = np.asarray(
        explainer.prep.transform(sample[explainer.selection["feature_columns"]]),
        dtype=float,
    )
    values, _ = explainer.values(x)
    shap.summary_plot(
        values, x, feature_names=explainer.names, max_display=12, show=False
    )
    plt.title("Churn drivers on validation customers")
    plt.tight_layout()
    plt.savefig(docs / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    importance = pd.DataFrame(
        {"feature": explainer.names, "mean_absolute_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    xv = np.asarray(
        explainer.prep.transform(validation[explainer.selection["feature_columns"]]),
        dtype=float,
    )
    p = explainer.model.predict_proba(xv)[:, 1]
    representatives = {
        "low_risk": int(p.argmin()),
        "high_risk": int(p.argmax()),
        "borderline": int(np.abs(p - explainer.selection["threshold"]).argmin()),
    }
    examples = {}
    for label, idx in representatives.items():
        row = validation.iloc[[idx]]
        explanation = explainer.explain(row)
        examples[label] = explanation
        vx, baseline = explainer.values(xv[[idx]])
        shap.force_plot(
            baseline[0],
            vx[0],
            np.round(xv[idx], 2),
            feature_names=[
                name.replace("seasonal_concentration", "seasonal share")
                .replace("frequency_trend", "order trend")
                .replace("affinity_", "share ")
                .replace("_", " ")
                for name in explainer.names
            ],
            matplotlib=True,
            show=False,
            figsize=(18, 4),
            text_rotation=30,
            contribution_threshold=0.15,
        )
        plt.title(
            f"{label.replace('_', ' ').title()} | Customer {row.iloc[0].customer_id}"
        )
        plt.savefig(docs / f"shap_{label}.png", dpi=180, bbox_inches="tight")
        plt.close("all")
    write_json(output / "explanation_examples.json", examples)
    local = examples["borderline"]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    weights = local["lime"][::-1]
    ax.barh(
        [r["condition"] for r in weights],
        [r["weight"] for r in weights],
        color=["#b94d54" if r["weight"] > 0 else "#14877e" for r in weights],
    )
    ax.set(
        title=f"LIME local explanation | Customer {local['customer_id']}",
        xlabel="Local contribution to churn probability",
    )
    fig.savefig(docs / "lime_example.png", dpi=180)
    plt.close(fig)
    # Use the fitted underlying estimator on transformed training features.
    if explainer.name != "ann":
        indices = [explainer.names.index(f) for f in importance.feature.head(3)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        PartialDependenceDisplay.from_estimator(
            explainer.model,
            explainer.train_x,
            features=indices,
            feature_names=explainer.names,
            ax=axes,
            kind="average",
            method="brute",
            grid_resolution=30,
        )
        fig.suptitle("Average model response | Training customers")
        fig.savefig(docs / "partial_dependence.png", dpi=180)
        plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        for ax, name in zip(axes, importance.feature.head(3)):
            j = explainer.names.index(name)
            grid = np.linspace(*np.quantile(explainer.train_x[:, j], [0.05, 0.95]), 30)
            response = []
            for point in grid:
                copy = explainer.train_x.copy()
                copy[:, j] = point
                response.append(explainer.model.predict_proba(copy)[:, 1].mean())
            ax.plot(grid, response)
            ax.set(xlabel=name, ylabel="Mean churn probability")
        fig.savefig(docs / "partial_dependence.png", dpi=180)
        plt.close(fig)
