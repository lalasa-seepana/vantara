"""Regenerate evidence charts, diagrams, notebooks and the final PDF from artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import nbformat
import numpy as np
import pandas as pd
import seaborn as sns
from nbclient import NotebookClient
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.linear_model import LinearRegression

from src.models.preprocessing import feature_columns
from src.utils.common import ROOT, config, read_json

NAVY, TEAL, GOLD = "#17364d", "#16887c", "#d0883f"


def style() -> None:
    """Apply a consistent readable chart style."""
    sns.set_theme(style="whitegrid", palette=[TEAL, GOLD, NAVY, "#b85b61"])
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    """Save a labelled genuine result figure and close it."""
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diagrams(docs: Path) -> None:
    """Draw the implemented data flow, schema and ordered workflow."""
    layouts = {
        "architecture_diagram": (
            "Vantara system architecture",
            [
                "UCI Online Retail II\nUnchanged two-sheet workbook",
                "Pandas pipeline\nValidation, cleaning, as-of features",
                "Training and artifacts\nML, neural models, clustering, XAI",
                "Batch scoring\nReal customer predictions",
                "PostgreSQL\nCustomers, transactions, segments, predictions",
                "FastAPI\nValidation, stored scores, explanations",
                "Streamlit dashboard\nFilters, trends, SHAP/LIME, CSV/PDF",
            ],
        ),
        "workflow_diagram": (
            "Reproducible model workflow",
            [
                "01  Verify source and SHA-256",
                "02  Clean, flag and audit transactions",
                "03  Pre-cutoff features / future labels",
                "04  Customer-disjoint 70 / 15 / 15 split",
                "05  Five-fold training CV + inner early stopping",
                "06  Select model and threshold on validation",
                "07  Freeze selection / evaluate test once",
                "08  Explain, score, persist and report",
            ],
        ),
        "er_diagram": (
            "PostgreSQL entity relationship diagram",
            [
                "SEGMENTS\nPK segment_id\nname, profile JSON",
                "CUSTOMERS\nPK customer_id\nFK segment_id\ncountry, monetary, value_tier, features JSON",
                "TRANSACTIONS\nPK transaction_id\nFK customer_id\ninvoice, stock_code, invoice_date\nquantity, price, revenue, is_return",
                "PREDICTIONS\nPK / FK customer_id\nmodel_version, scored_at, payload JSON",
            ],
        ),
    }
    for filename, (title, labels) in layouts.items():
        if filename == "er_diagram":
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.set(xlim=(0, 1), ylim=(0, 1))
            positions = [
                (0.05, 0.62, 0.38, 0.26),
                (0.56, 0.57, 0.40, 0.34),
                (0.05, 0.08, 0.40, 0.38),
                (0.56, 0.10, 0.40, 0.28),
            ]
            edges = [
                ((0.43, 0.75), (0.56, 0.75), "1 : many"),
                ((0.65, 0.57), (0.36, 0.46), "1 : many"),
                ((0.78, 0.57), (0.78, 0.38), "1 : 1 current"),
            ]
        else:
            fig, ax = plt.subplots(figsize=(10, 11))
            ax.set(xlim=(0, 1), ylim=(0, 1))
            height = 0.085
            positions = [
                (0.14, 0.89 - i * 0.115, 0.72, height) for i in range(len(labels))
            ]
            edges = [
                ((0.5, positions[i][1]), (0.5, positions[i + 1][1] + height), "")
                for i in range(len(labels) - 1)
            ]
        for (x, y, w, h), label in zip(positions, labels):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.012",
                    facecolor="#eef5f5",
                    edgecolor=TEAL,
                    linewidth=1.5,
                )
            )
            ax.text(
                x + w / 2,
                y + h / 2,
                label,
                ha="center",
                va="center",
                fontsize=11,
                color=NAVY,
                linespacing=1.5,
            )
        for start, end, label in edges:
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": NAVY},
            )
            if label:
                ax.text(
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2 + 0.025,
                    label,
                    ha="center",
                    fontsize=9,
                    color=NAVY,
                )
        ax.set_title(title, pad=22, fontsize=18, color=NAVY)
        ax.axis("off")
        save(fig, docs / f"{filename}.png")


def charts(cfg: dict[str, Any]) -> None:
    """Create statistics and PNGs strictly from processed data and saved outputs."""
    style()
    docs, artifacts = Path(cfg["paths"]["docs"]), Path(cfg["paths"]["artifacts"])
    data = pd.read_parquet(Path(cfg["paths"]["processed"]) / "customers.parquet")
    train = data[data.split == "train"]
    numeric = train.select_dtypes("number")
    numeric.describe().T.to_csv(docs / "descriptive_statistics.csv")
    numeric.corr().to_csv(docs / "feature_correlations.csv")
    names = [c for c in feature_columns(data) if c not in ["country", "top_sku"]]
    values = train[names].to_numpy()
    vif = []
    for i, name in enumerate(names):
        variance = np.var(values[:, i])
        score = (
            LinearRegression()
            .fit(np.delete(values, i, axis=1), values[:, i])
            .score(np.delete(values, i, axis=1), values[:, i])
            if variance
            else 1
        )
        vif.append({"feature": name, "vif": 1 / max(1 - score, 1e-12)})
    pd.DataFrame(vif).sort_values("vif", ascending=False).to_csv(
        docs / "vif.csv", index=False
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for ax, name, label in zip(
        axes,
        ["recency", "frequency", "monetary"],
        ["Days since last purchase", "Distinct orders", "Product spend (GBP)"],
    ):
        vals = train[name] if name == "recency" else np.log1p(train[name])
        sns.histplot(vals, bins=35, ax=ax, color=TEAL)
        ax.set(
            xlabel=label if name == "recency" else f"log(1 + {label})",
            ylabel="Training customers",
            title=name.title(),
        )
    fig.suptitle("Customer purchase patterns | Training partition", fontsize=17)
    save(fig, docs / "eda_distributions.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    sns.scatterplot(
        data=train,
        x="recency",
        y="frequency",
        hue="churn",
        palette={0: TEAL, 1: GOLD},
        alpha=0.4,
        s=14,
        ax=axes[0],
    )
    axes[0].set(
        yscale="log",
        xlabel="Recency (days)",
        ylabel="Orders (log scale)",
        title="Recency and order frequency",
    )
    sns.scatterplot(
        x=np.log1p(train.monetary),
        y=np.sign(train.clv_90d) * np.log1p(train.clv_90d.abs()),
        alpha=0.4,
        s=14,
        ax=axes[1],
        color=TEAL,
    )
    axes[1].set(
        xlabel="log(1 + historical spend GBP)",
        ylabel="Signed log(1 + future net revenue GBP)",
        title="Historical spend and 90-day target",
    )
    save(fig, docs / "eda_relationships.png")
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    chosen = [
        "recency",
        "frequency",
        "monetary",
        "average_spend",
        "basket_size",
        "return_rate",
        "churn",
        "clv_90d",
    ]
    sns.heatmap(
        train[chosen].corr(),
        vmin=-1,
        vmax=1,
        center=0,
        cmap="BrBG",
        annot=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Feature and target correlation | Training customers")
    save(fig, docs / "eda_correlations.png")
    tx = pd.read_parquet(Path(cfg["paths"]["interim"]) / "transactions.parquet")
    tx = tx[
        (tx.invoice_date < pd.Timestamp(cfg["features"]["cutoff"]))
        & ~tx.invalid_price
        & ~tx.is_admin
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    country = tx.groupby("country").revenue.sum().sort_values().tail(10)
    axes[0].barh(country.index, country.values, color=TEAL)
    axes[0].set(
        title="Top countries by observed net revenue",
        xlabel="Net revenue (GBP)",
        xscale="symlog",
    )
    month = tx.groupby(tx.invoice_date.dt.month).revenue.sum()
    axes[1].bar(month.index, month.values, color=GOLD)
    axes[1].set(
        title="Observed seasonal revenue",
        xlabel="Calendar month",
        ylabel="Net revenue (GBP)",
        xticks=range(1, 13),
    )
    fig.suptitle(
        "Pre-cutoff product sales | Unequal month exposure across years", fontsize=14
    )
    save(fig, docs / "eda_country_season.png")
    evaluation = read_json(artifacts / "evaluation.json")
    table = pd.DataFrame(evaluation["churn"]).T.sort_values(
        ["roc_auc", "recall"], ascending=False
    )
    table.drop(columns="confusion_matrix").to_csv(docs / "model_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    axes[0].barh(table.index[::-1], table.roc_auc.to_numpy()[::-1], color=TEAL)
    axes[0].axvline(0.80, color=GOLD, linestyle="--", label="PRD target 0.80")
    axes[0].set(
        xlim=(0.5, 1), xlabel="Held-out ROC-AUC", title="Churn model comparison"
    )
    axes[0].legend()
    sns.heatmap(
        evaluation["churn"][evaluation["selected_model"]]["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Purchase", "Churn"],
        yticklabels=["Purchase", "Churn"],
        ax=axes[1],
    )
    axes[1].set(
        xlabel="Predicted class",
        ylabel="Observed class",
        title=f"Selected model: {evaluation['selected_model']}",
    )
    save(fig, docs / "model_evaluation.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    for ax, name in zip(axes, ["ann", "lstm", "autoencoder"]):
        loss = pd.read_csv(artifacts / f"{name}_loss.csv")
        ax.plot(loss.epoch, loss.training_loss, label="Training", color=TEAL)
        ax.plot(loss.epoch, loss.validation_loss, label="Validation", color=GOLD)
        ax.set(xlabel="Epoch", ylabel="Objective loss", title=name.upper())
        ax.legend()
    save(fig, docs / "neural_loss_curves.png")
    search = pd.read_csv(artifacts / "cluster_search.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    axes[0].plot(search.k, search.inertia, "o-", color=TEAL)
    axes[0].set(
        xlabel="Number of clusters",
        ylabel="Within-cluster squared distance",
        title="K-Means inertia",
    )
    axes[1].plot(search.k, search.silhouette, "o-", color=GOLD)
    axes[1].set(
        xlabel="Number of clusters",
        ylabel="Silhouette score",
        title="K-Means separation",
    )
    save(fig, docs / "cluster_selection.png")
    frozen = pd.read_csv(artifacts / "test_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].scatter(frozen.clv_90d, frozen.clv_prediction, color=TEAL, alpha=0.5, s=15)
    axes[0].set(
        xlabel="Observed 90-day net revenue (GBP)",
        ylabel="Predicted 90-day net revenue (GBP)",
        xscale="symlog",
        yscale="symlog",
        title="CLV regression | Held-out customers",
    )
    axes[1].hist(
        frozen.anomaly_error,
        bins=np.geomspace(
            max(frozen.anomaly_error.min(), 1e-6), frozen.anomaly_error.max(), 35
        ),
        color=TEAL,
    )
    axes[1].axvline(
        evaluation["anomaly"]["threshold"],
        color=GOLD,
        linestyle="--",
        label="Validation 99th percentile",
    )
    axes[1].set(
        xscale="log",
        xlabel="Reconstruction error (log scale)",
        ylabel="Held-out customers",
        title="Anomaly scores, not fraud labels",
    )
    axes[1].legend()
    save(fig, docs / "clv_anomaly.png")


def notebooks(cfg: dict[str, Any]) -> None:
    """Create and execute the three required notebooks against genuine artifacts."""
    directory = ROOT / "notebooks"
    prefix = "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
    definitions = {
        "01_eda": [
            (
                "markdown",
                "# Customer purchase patterns\nThis notebook reviews the training partition. The source workbook is unchanged. Future labels are shown only for training-customer analysis.",
            ),
            (
                "code",
                prefix
                + "data = pd.read_parquet(root / 'data/processed/customers.parquet')\ntrain = data[data['split'] == 'train']\ndisplay(train[['recency','frequency','monetary','churn','clv_90d']].describe())",
            ),
            (
                "code",
                "display(Image(filename=str(root / 'docs/eda_distributions.png')))\ndisplay(Image(filename=str(root / 'docs/eda_relationships.png')))\ndisplay(Image(filename=str(root / 'docs/eda_country_season.png')))\ndisplay(Image(filename=str(root / 'docs/eda_correlations.png')))\ndisplay(pd.read_csv(root / 'docs/vif.csv'))",
            ),
            (
                "markdown",
                "## Modeling hypotheses\n1. Longer recency is associated with higher churn risk.\n2. Order frequency and spend add useful information beyond recency.\n3. Gradient-boosted trees can capture effects that a linear baseline misses.\n4. A small customer count may limit the neural models.\n5. Large wholesale accounts may create high CLV errors.\nThese are analysis questions, not proven business effects. Correlated features are retained for tree models; L2 regularization helps the linear baseline. Historical CLV and engagement are excluded from predictors as duplicate/composite summaries, and one affinity column is removed to avoid the sum-to-one identity.",
            ),
        ],
        "02_feature_engineering": [
            (
                "markdown",
                "# Point-in-time feature engineering\nThe production path is `python -m src.data.pipeline`. This notebook reviews its output and verifies the cutoff, without replacing the scripted pipeline.",
            ),
            (
                "code",
                prefix
                + "import json\nmetadata = json.loads((root / 'data/processed/feature_metadata.json').read_text())\ndisplay(metadata)\ndata = pd.read_parquet(root / 'data/processed/customers.parquet')\ndisplay(data.head())\nassert data.customer_id.is_unique\ndisplay(data.groupby('split').agg(customers=('customer_id','size'), churn_rate=('churn','mean')))",
            ),
            (
                "code",
                "events = pd.read_parquet(root / 'data/processed/purchase_sequences.parquet')\nassert events.invoice_date.max() < pd.Timestamp(metadata['cutoff_exclusive'])\ndisplay(events.head())",
            ),
            (
                "markdown",
                "## Feature definitions\nRecency, distinct orders, total and average product spend, basket units, six-month order trend, gap variance, seasonal concentration, return rate, description-based affinity and price-based discount proxy use pre-cutoff records only. The engagement score combines fixed recency, order and spend scales. Historical CLV is observed net product revenue; predicted CLV means the next 90 days only. It is not a full lifetime forecast. Category and discount proxies do not represent official categories or known promotions.",
            ),
        ],
        "03_model_experiments": [
            (
                "markdown",
                "# Model experiments and frozen evaluation\nTraining uses customer-disjoint partitions and five-fold training-only validation. The selected model and threshold were frozen before final test evaluation. Re-reading saved results does not refit any model.",
            ),
            (
                "code",
                prefix
                + "import json\nselection = json.loads((root / 'models_artifacts/selection.json').read_text())\ndisplay({'selected_model': selection['model'], 'threshold': selection['threshold'], 'rule': selection['selection_rule']})\ndisplay(pd.read_csv(root / 'docs/model_comparison.csv'))\nlog = pd.read_csv(root / 'models_artifacts/experiment_log.csv')\ndisplay(log.groupby(['model','phase']).training_seconds.agg(['count','sum']))",
            ),
            (
                "code",
                "display(Image(filename=str(root / 'docs/model_evaluation.png')))\ndisplay(Image(filename=str(root / 'docs/neural_loss_curves.png')))\ndisplay(pd.read_csv(root / 'models_artifacts/cluster_metrics.csv'))\ndisplay(Image(filename=str(root / 'docs/shap_summary.png')))",
            ),
            (
                "markdown",
                "## Interpretation\nUse the final report for actual target status. LSTM timing is capped at 90 days; no-purchase customers have a censored target at that cap. Autoencoder flags are candidates for review, not confirmed fraud. Validation determines model selection; test results must not guide further tuning in this run.",
            ),
        ],
    }
    for name, cells in definitions.items():
        book = nbformat.v4.new_notebook(
            cells=[
                (
                    nbformat.v4.new_markdown_cell(value)
                    if kind == "markdown"
                    else nbformat.v4.new_code_cell(value)
                )
                for kind, value in cells
            ],
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
        )
        NotebookClient(
            book,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        nbformat.write(book, directory / f"{name}.ipynb")


def final_pdf(cfg: dict[str, Any]) -> None:
    """Build the final report including a three-page mathematical appendix."""
    docs, artifacts = Path(cfg["paths"]["docs"]), Path(cfg["paths"]["artifacts"])
    evaluation = read_json(artifacts / "evaluation.json")
    selection = read_json(artifacts / "selection.json")
    audit = read_json(Path(cfg["paths"]["interim"]) / "validation_audit.json")
    meta = read_json(Path(cfg["paths"]["processed"]) / "feature_metadata.json")
    checks = (
        read_json(docs / "runtime_checks.json")
        if (docs / "runtime_checks.json").exists()
        else {}
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Cover",
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=32,
            textColor=colors.HexColor(NAVY),
            spaceAfter=20,
        )
    )
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 15
    styles["BodyText"].spaceAfter = 8
    styles["Heading1"].textColor = colors.HexColor(NAVY)
    styles["Heading1"].fontSize = 18
    styles["Heading1"].leading = 23
    styles["Heading1"].spaceAfter = 14
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 17
    styles["Heading2"].spaceBefore = 12
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, leading=11)
    story: list[Any] = []

    def paragraph(value: str, kind: str = "BodyText") -> None:
        story.append(Paragraph(value, styles[kind]))

    def heading(value: str) -> None:
        paragraph(value, "Heading1")

    def table(
        headers: list[str], rows: list[list[Any]], widths: list[float] | None = None
    ) -> None:
        converted = [
            [Paragraph(escape(str(value)), small) for value in row]
            for row in [headers, *rows]
        ]
        item = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0ecef")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f4f7f8")],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cfdadb")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend([item, Spacer(1, 12)])

    def picture(
        name: str, width: float = 6.6 * inch, height: float | None = None
    ) -> None:
        from PIL import Image as PILImage

        with PILImage.open(docs / name) as image:
            w, h = image.size
        story.extend(
            [
                Image(str(docs / name), width=width, height=height or width * h / w),
                Spacer(1, 10),
            ]
        )

    def page() -> None:
        story.append(PageBreak())

    selected = evaluation["churn"][selection["model"]]
    clv = evaluation["regression"]["clv"]
    heading("Vantara Retail Solutions")
    paragraph("Customer behavior<br/>prediction platform", "Cover")
    paragraph(
        f"Final implementation report | Historical scoring date {cfg['features']['cutoff']}"
    )
    paragraph(
        f"The platform uses {meta['customers']:,} customers from the complete UCI Online Retail II workbook. It produces churn scores, 90-day customer value estimates, purchase timing and category proxies, customer segments, and anomaly flags. The FastAPI service stores scores in PostgreSQL and provides them to the Streamlit dashboard."
    )
    paragraph(
        f"The selected {selection['model']} model achieved test ROC-AUC {selected['roc_auc']:.4f} and recall {selected['recall']:.4f}. Both churn targets were met. CLV R-squared was {clv['r2']:.4f}, below the 0.60 target. This shortfall is reported without changing the held-out test set or tuning against it."
    )
    table(
        ["Measure", "Observed result", "PRD target"],
        [
            ["Churn ROC-AUC", f"{selected['roc_auc']:.4f}", ">= 0.80"],
            ["Churn recall", f"{selected['recall']:.4f}", ">= 0.70"],
            ["CLV R-squared", f"{clv['r2']:.4f}", ">= 0.60 (not met)"],
            ["API p95", checks.get("api_p95_display", "Not yet measured"), "< 400 ms"],
            [
                "Dashboard full segment view",
                checks.get("dashboard_display", "Not yet measured"),
                "< 3 seconds",
            ],
            [
                "Source package coverage",
                checks.get("coverage_display", "See coverage report"),
                ">= 70%",
            ],
        ],
        [170, 175, 130],
    )
    paragraph(
        "These results describe a historical prototype. No live retention campaign, marketing uplift or fraud-detection accuracy was measured."
    )
    paragraph(
        "Source: Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. DOI: 10.24432/C5CG6D. CC BY 4.0."
    )
    page()
    heading("Data provenance and cleaning")
    paragraph(
        f"The unchanged workbook contains {audit['raw_rows']:,} invoice lines across both required sheets. Its date range is {audit['min_date']} to {audit['max_date']}. Loading standardizes columns and types, then sorts chronologically before cleaning."
    )
    table(
        ["Audit item", "Actual count"],
        [
            ["Exact duplicate rows removed", audit["exact_duplicates_removed"]],
            ["Clean rows retained", audit["clean_rows"]],
            ["Rows without customer ID in source", audit["missing_customer_rows"]],
            ["Return rows retained", audit["return_rows"]],
            ["Invalid-price rows flagged", audit["invalid_price_rows"]],
            ["Administrative rows flagged", audit["admin_rows"]],
            ["Bulk purchase rows retained", audit["bulk_rows_retained"]],
        ],
        [340, 135],
    )
    paragraph(
        "Missing customer IDs are excluded from customer modeling but retained in product-level revenue analysis. Non-positive prices are flagged and excluded from spend features and revenue targets. Negative quantities and cancellation invoices are kept as returns. Administrative stock codes are excluded from product affinity and product revenue. Large purchases are retained when they have positive prices and are not cancellations; IQR bounds flag them for review rather than treating every wholesale order as an error."
    )
    paragraph(
        "Exact duplicate lines are removed across all eight source fields. Lines with different quantities, prices, descriptions or timestamps remain separate. The source cannot prove whether every identical repeated line is a legitimate repeat, so this is an explicit cleaning assumption. Description case and whitespace are standardized; each SKU uses its most common historical normalized description for feature building. No unknown spelling correction is invented."
    )
    paragraph("SHA-256 of the original workbook: " + audit["source_sha256"])
    paragraph(
        "The source has no official category or promotion fields. Categories are deterministic description keyword proxies. Discount sensitivity is the share of lines priced below the same customer's earlier maximum price for that SKU. It may reflect bulk pricing and does not estimate a causal response to a discount. The actual 2009-2011 period does not contain a leap day, despite the PRD's general leap-year note."
    )
    page()
    heading("Feature definitions and leakage controls")
    table(
        ["Feature", "Definition and reason"],
        [
            [
                "Recency and tenure",
                "Days since the last and first paid product purchase; describe customer activity.",
            ],
            [
                "Frequency",
                "Distinct paid product invoices; separates repeat and occasional buyers.",
            ],
            [
                "Total and average spend",
                "Gross positive product spend and spend per order; describe observed value.",
            ],
            [
                "Historical CLV",
                "Net historical product revenue including returns; reporting measure.",
            ],
            [
                "Basket size",
                "Paid units per order; distinguishes small and bulk baskets.",
            ],
            [
                "Purchase trend",
                "Slope of monthly order counts over six calendar months before the cutoff.",
            ],
            [
                "Gap variance",
                "Variance of positive time gaps between orders; describes irregular behavior.",
            ],
            [
                "Seasonal concentration",
                "Largest calendar-month spend share; signals concentrated shopping.",
            ],
            [
                "Category affinity",
                "Spend shares in keyword categories; supports recommendations.",
            ],
            [
                "Return rate",
                "Returned product units divided by purchased units; may exceed one.",
            ],
            [
                "Discount proxy",
                "Share of lines below an earlier personal SKU price; pricing sensitivity proxy.",
            ],
            [
                "Engagement",
                "Fixed-scale composite of recency, frequency and spend; dashboard summary.",
            ],
        ],
        [120, 355],
    )
    paragraph(
        "Every feature uses invoice_date strictly before 1 September 2011. Labels use the next 90 days. Churn means no paid product purchase in that window. CLV means net product revenue in the same window, not an unlimited lifetime forecast. LSTM timing is the waiting time capped at 90 days; customers without a purchase receive the cap and are not treated as observed eventual purchases."
    )
    paragraph(
        f"Customer-disjoint split counts: {escape(str(meta['split_counts']))}. The random seed is {cfg['seed']}. Stratification uses the churn label for classification and also balances purchase/no-purchase cases in regression folds. No customer appears in more than one outer partition. Encoders, scalers and model fitting are restricted to each training fold. XGBoost and neural CV runs use a further training-only early-stopping subset."
    )
    paragraph(
        "Country is one-hot encoded. The most frequent SKU is frequency encoded from fitting customers only; unknown SKUs receive zero. Trees use unscaled numeric features, while logistic regression, KNN and neural models use StandardScaler. Historical CLV, engagement and one redundant affinity column are excluded from model inputs. All required feature outputs remain available in the feature table."
    )
    page()
    heading("Exploratory analysis")
    picture("eda_distributions.png")
    picture("eda_relationships.png")
    vif = pd.read_csv(docs / "vif.csv")
    paragraph(
        f"The largest training-set VIF is {vif.iloc[0].vif:.2f} for {escape(vif.iloc[0].feature)}. Spend, frequency and basket features are related. The analysis records this dependence rather than claiming independent causal effects. Tree models retain useful related inputs; L2 regularization stabilizes the linear baseline."
    )
    paragraph(
        "The executed EDA notebook includes descriptive statistics, feature/target correlations, country and seasonal plots, VIF, and hypotheses about recency, repeat purchases, model flexibility and wholesale outliers. Seasonal totals have unequal month exposure and must not be read as a controlled seasonality estimate."
    )
    page()
    heading("Classical and neural model evaluation")
    comparison = pd.DataFrame(evaluation["churn"]).T.sort_values(
        ["roc_auc", "recall"], ascending=False
    )
    table(
        ["Model", "AUC", "Recall", "Precision", "F1", "Accuracy"],
        [
            [
                name,
                *[
                    f"{row[c]:.4f}"
                    for c in ["roc_auc", "recall", "precision", "f1", "accuracy"]
                ],
            ]
            for name, row in comparison.iterrows()
        ],
        [115, 72, 72, 72, 72, 72],
    )
    picture("model_evaluation.png")
    paragraph(
        f"Production choice: {selection['model']}, selected by highest validation ROC-AUC with recall as a tie-breaker. The threshold {selection['threshold']:.2f} maximized validation F1 among thresholds meeting the recall target. XGBoost has a slightly higher test AUC, but the production choice remains unchanged because using test results for selection would break the evaluation discipline."
    )
    paragraph(
        "The comparison contains all six required classical classifiers and the ANN. Random Forest and XGBoost were tuned through five-fold training-only searches. Every run records parameters, metrics and training time in experiment_log.csv. The test_predictions.csv file stores the actual final outputs, and evaluation.json freezes their metrics. The ANN includes batch normalization, dropout, L1/L2 regularization and early stopping."
    )
    page()
    heading("Value, sequence and anomaly models")
    timing = evaluation["regression"]["lstm"]
    table(
        ["Task", "MAE", "RMSE", "R-squared"],
        [
            [
                "90-day net value (GBP)",
                f"{clv['mae']:,.2f}",
                f"{clv['rmse']:,.2f}",
                f"{clv['r2']:.4f}",
            ],
            [
                "Capped waiting time (days)",
                f"{timing['mae']:.2f}",
                f"{timing['rmse']:.2f}",
                f"{timing['r2']:.4f}",
            ],
        ],
        [210, 88, 88, 89],
    )
    picture("clv_anomaly.png")
    paragraph(
        "The Random Forest CLV model misses the 0.60 R-squared target. Large wholesale values and returns produce a heavy-tailed target, while this model uses one fixed snapshot and a limited customer sample. These are plausible limits, not a measured causal explanation for every error. The result is reported as-is; further tuning requires a new evaluation design."
    )
    paragraph(
        f"The LSTM reads up to {cfg['features']['sequence_length']} time-ordered purchase events containing amount, gap and category proxy. It predicts capped waiting time and the next category conditional on a future purchase. Conditional category accuracy is {timing['category_accuracy_conditional_purchase']:.3f}. Its timing R-squared is modest, and improvement over a tuned static timing model has not been established. Purchase probability comes from one minus the churn estimate. Recommendations rank historical products within the LSTM category proxy; they are not a validated recommendation system."
    )
    paragraph(
        f"The autoencoder threshold is the {cfg['models']['dl']['anomaly_quantile']:.0%} validation-error percentile ({evaluation['anomaly']['threshold']:.4f}). It flagged {evaluation['anomaly']['test_flagged']} of {evaluation['test_customers']} held-out customers. No fraud labels exist, so anomaly precision or fraud recall cannot be reported. Flags require manual review."
    )
    page()
    heading("Neural training and customer segments")
    picture("neural_loss_curves.png")
    paragraph(
        "These curves are saved from the final actual training runs. Validation early stopping restores the lowest-loss epoch. Cross-validation scores for supervised ANN and LSTM runs are available in the experiment log. The autoencoder is unsupervised and its threshold is checked on validation customers."
    )
    picture("cluster_selection.png")
    clustering = pd.read_csv(artifacts / "cluster_metrics.csv")
    table(
        ["Method", "k", "Silhouette", "Davies-Bouldin"],
        [
            [r.algorithm, int(r.k), f"{r.silhouette:.3f}", f"{r.davies_bouldin:.3f}"]
            for r in clustering.itertuples()
        ],
        [170, 65, 120, 120],
    )
    paragraph(
        "K-Means uses scaled log-transformed recency, frequency, spend, return rate and seasonal concentration. Selection combines the inertia elbow with silhouette scores and a small-cluster check. A full-covariance Gaussian mixture checks the spherical-cluster assumption. Both are fitted on training customers, then assign all customers to business-readable profiles."
    )
    page()
    heading("Explainability and business use")
    picture("shap_summary.png", width=5.7 * inch)
    paragraph(
        "Global SHAP importance uses validation customers and a training-only background. Positive SHAP values raise the estimated probability of no purchase; negative values lower it. The customer panel provides both SHAP and LIME for any stored customer. Three saved SHAP force plots show low-risk, high-risk and borderline examples. Partial dependence plots show the top three drivers."
    )
    examples = read_json(artifacts / "explanation_examples.json")
    local = examples["borderline"]
    paragraph("Worked example: " + escape(local["summary"]))
    shap_features = [r["feature"] for r in local["contributions"][:3]]
    paragraph(
        f"For that same customer, SHAP's leading fields are {escape(', '.join(shap_features))}. The LIME plot records local threshold conditions and weights, with local fit R-squared {local['lime_local_r2']:.3f}. Differences are expected because SHAP allocates the model output around a background, while LIME fits an approximate local surrogate. Agreement does not prove causality."
    )
    paragraph(
        "The leaderboard puts flagged, high-value customers first and then sorts by estimated revenue at risk. It does not claim that contacting those customers will cause retention or recover that amount. The anomaly panel is a review queue, not a fraud verdict."
    )
    page()
    heading("System architecture and API")
    picture("architecture_diagram.png", width=4.7 * inch, height=5.2 * inch)
    table(
        ["Endpoint", "Purpose"],
        [
            ["GET /health", "Database and model readiness"],
            ["GET /metadata", "Actual model settings and frozen evaluation"],
            ["POST /predict", "Stored customer score or a complete new feature record"],
            [
                "POST /predict/batch",
                "Validated CSV scoring, with all rows checked first",
            ],
            [
                "GET /customers",
                "Stored scores with segment, country and value-tier filters",
            ],
            ["GET /explain/{customer_id}", "Individual SHAP and LIME output"],
            [
                "GET /trends and /importance",
                "Observed/forecast revenue and saved global SHAP values",
            ],
        ],
        [200, 275],
    )
    page()
    heading("Storage, deployment and verification")
    picture("er_diagram.png", width=6.5 * inch)
    paragraph(
        "PostgreSQL keeps four tables: customers, historical transactions, segment profiles and current predictions. Foreign keys preserve relationships. Batch writes update customer features and predictions in one transaction. Dashboard reads use persisted predictions rather than model inference. New feature-only records have no sequence outputs; those fields remain absent rather than fabricated."
    )
    paragraph(
        "Docker Compose defines database, API and dashboard services with health checks. Paths, database settings and log level use environment variables. A setup command creates an owner-readable local .env with a generated database password; credentials are not committed. The README gives setup, data regeneration, testing and API examples."
    )
    paragraph(
        "Verification status: "
        + escape(
            checks.get(
                "deployment_note",
                "Runtime checks are pending; do not treat configuration files as proof of successful deployment.",
            )
        )
    )
    paragraph(
        "Dashboard timing: "
        + escape(
            checks.get(
                "dashboard_note",
                "See runtime_checks.json for the measured browser timing.",
            )
        )
    )
    paragraph(
        "The test suite checks source preservation, returns, cutoff leakage, disjoint partitions, complete outcome windows, inference consistency, API validation, batch behavior and persistence. Coverage evidence is saved in docs/coverage.txt and docs/coverage.json when tests run. All notebooks are executed against actual saved artifacts."
    )
    page()
    heading("Limitations and handover")
    paragraph(
        "This is one historical cutoff with a customer-disjoint random split. It measures generalization to other customers at the same date, not performance at a later date. Churn is not a minority in this cohort; the actual rate is reported in feature_metadata.json. Class weighting is learned from fitting labels rather than assuming the PRD's expected class balance."
    )
    paragraph(
        "Source limitations include missing IDs, ambiguous identical invoice lines, adjustment rows, unofficial category proxies and no promotion or fraud labels. CLV is a fixed 90-day net revenue estimate. Anomaly thresholds are validated by held-out reconstruction errors, not by verified fraud outcomes. A linear revenue forecast is an illustrative extrapolation from the last six complete months, not a validated demand model."
    )
    paragraph(
        "Future work: evaluate later time cutoffs, use verified product categories and campaign records, test retention impact in a controlled A/B test, improve CLV under a new evaluation design, and measure recommendation ranking on held-out baskets. Production hardening may add authentication, role-based access, monitoring and CI/CD. Transformers and streaming ingestion remain deferred as stated in the PRD."
    )
    paragraph(
        "Handover artifacts include the unchanged source, cleaned and feature tables, all trained model artifacts, fitted encoders/scalers, experiment logs, executed notebooks, PNG findings, SHAP/LIME outputs, diagrams, this PDF, API/dashboard code, Docker configuration and tests. The genuine recorded dashboard walkthrough is included as docs/walkthrough.webm; its verification details are stored in runtime_checks.json."
    )
    heading("References")
    paragraph(
        "UCI source: https://archive.ics.uci.edu/dataset/502/online+retail+ii<br/>Dataset DOI: https://doi.org/10.24432/C5CG6D<br/>Scikit-learn 1.6: https://scikit-learn.org/1.6/<br/>XGBoost 3.0: https://xgboost.readthedocs.io/en/release_3.0.0/<br/>SHAP: https://shap.readthedocs.io/<br/>Project requirements: supplied Vantara PRD v1.0."
    )
    page()
    heading("Mathematical appendix 1 of 3")
    paragraph("Statistics and feature matrices", "Heading2")
    paragraph(
        "The customer feature matrix X has n rows and d input columns. Each row uses only events before the cutoff. Mean spend is total positive product revenue divided by the number of distinct paid orders. Gap variance is the mean squared deviation of positive inter-order gaps. With only one eligible gap it is zero; order frequency preserves the distinction from repeat behavior."
    )
    paragraph(
        "The six-month purchase trend is sum((t - mean(t)) * orders(t)) / sum((t - mean(t))^2). Seasonal concentration is max(monthly spend) / total spend. Return rate is abs(returned units) / purchased units. These are descriptive measures, not causal estimates."
    )
    paragraph(
        "Engagement = 100 / 3 * [exp(-recency / 90) + frequency / (frequency + 10) + monetary / (monetary + 2000)]. The scales are fixed in YAML; no future labels enter the score. StandardScaler transforms each numeric input as z = (x - training mean) / training standard deviation. Validation and test records use these saved training values."
    )
    paragraph("Churn and conditional probability", "Heading2")
    paragraph(
        "The target y is 1 when no eligible product purchase occurs in [cutoff, cutoff + 90 days). The classifier estimates p = P(y = 1 | X). The purchase probability is 1 - p. Logistic regression computes p = 1 / (1 + exp(-(w^T x + b))). Its binary cross-entropy is -mean(y log(p) + (1-y) log(1-p)), with class weights and L2 regularization in this implementation."
    )
    paragraph("Correlation and VIF", "Heading2")
    paragraph(
        "Pearson correlation standardizes covariance by the two standard deviations. VIF for feature j is 1 / (1 - R_j^2), where R_j^2 comes from regressing that feature on the remaining numeric features in training data. Large VIF warns that coefficients may be unstable; it does not invalidate every tree prediction. PCA is optional in the PRD and is not used in this implementation. For centered X, the covariance matrix is Sigma = X^T X / n. PCA uses the eigenvectors with the largest eigenvalues as projection directions; no PCA result is claimed here."
    )
    page()
    heading("Mathematical appendix 2 of 3")
    paragraph("Neural optimization and regularization", "Heading2")
    paragraph(
        "Backpropagation applies the chain rule to compute each weight's loss gradient. Gradient descent has the basic form w <- w - learning_rate * gradient. The implemented Adam optimizer uses moving first and second gradient moments to adapt this step. The objective also includes a configured L1 weight penalty and Adam weight decay for L2 regularization. Dropout randomly removes activations during training; it is disabled for inference. Batch normalization in the ANN uses stored training statistics at inference."
    )
    paragraph(
        "ANN output is a churn logit and the data loss is weighted binary cross-entropy. The LSTM reads log(1 + order amount), log(1 + gap days) and category-proxy indicators. Only real events enter the fitted sequence scaler. A validity mask and packed sequences prevent padding from being read as a real purchase."
    )
    paragraph(
        "The LSTM timing target is log(1 + min(waiting days, 90)) / log(91). Its data loss combines timing mean squared error and 0.2 times category cross-entropy. The category term uses only customers with a purchase in the horizon. The predicted timing value is transformed back with expm1 and clipped to [0, 90]. This is restricted waiting time, not an uncensored time-to-event model."
    )
    paragraph(
        "The autoencoder minimizes mean((reconstructed X - X)^2). Its customer anomaly score is the mean squared reconstruction error across transformed features. The threshold is the 99th percentile on validation customers. That controls the observed validation flag rate, not a real-world false-positive rate for fraud."
    )
    paragraph(
        "Early stopping keeps the epoch with the smallest validation data loss and stops after the configured patience. CV scoring folds are separate from early-stopping subsets. The actual final-run loss curves below document training behavior."
    )
    picture("neural_loss_curves.png")
    page()
    heading("Mathematical appendix 3 of 3")
    paragraph("Evaluation and segmentation", "Heading2")
    paragraph(
        "Precision = TP / (TP + FP), recall = TP / (TP + FN), and F1 = 2 * precision * recall / (precision + recall). Accuracy = (TP + TN) / all cases. ROC-AUC summarizes true-positive versus false-positive rates across thresholds. The report shows both threshold-free AUC and metrics at validation-selected thresholds."
    )
    paragraph(
        "MAE = mean(abs(y - prediction)); RMSE = sqrt(mean((y - prediction)^2)); R-squared = 1 - sum((y - prediction)^2) / sum((y - mean(y))^2). RMSE gives large wholesale errors more weight. Negative R-squared is possible and must not be replaced with zero."
    )
    paragraph(
        "K-Means minimizes the sum of squared distances to assigned centers. Silhouette for a point is (b - a) / max(a, b), where a is mean within-cluster distance and b is the smallest mean distance to another cluster. Davies-Bouldin compares within-cluster spread to between-center distance; lower is better. A Gaussian mixture estimates weighted full-covariance Gaussian components and assigns the most likely component."
    )
    paragraph("Explanation mathematics", "Heading2")
    paragraph(
        "SHAP allocates the difference between the model output and a background baseline to individual inputs. For the selected tree model, the baseline plus all feature contributions reconstructs the churn probability. Correlated features can share attribution, so the interpretation depends on the background and feature-dependence assumptions. The implementation checks the reconstructed output against the actual prediction."
    )
    paragraph(
        "LIME samples nearby transformed inputs and fits a weighted local surrogate. Its coefficients describe that local approximation, and its local R-squared measures fit to the model in that sampled neighborhood. Partial dependence replaces one feature with a grid value for each training customer and averages predictions; combinations may be unrealistic when features are correlated."
    )
    paragraph(
        "The simple revenue forecast fits monthly revenue = slope * month_index + intercept by least squares on the last six complete historical months. Future negative estimates are clipped to zero. It is shown separately from observed data and has no claimed forecast-accuracy target."
    )

    def footer(canvas: Any, doc: Any) -> None:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(NAVY))
        canvas.drawString(42, 24, "Vantara | Historical customer behavior prototype")
        canvas.drawRightString(570, 24, str(doc.page))

    SimpleDocTemplate(
        str(docs / "final_report.pdf"),
        pagesize=(612, 792),
        leftMargin=42,
        rightMargin=42,
        topMargin=40,
        bottomMargin=42,
        title="Vantara customer behavior prediction final report",
        author="Vantara project",
    ).build(story, onFirstPage=footer, onLaterPages=footer)


def run_reports(cfg: dict[str, Any]) -> None:
    """Regenerate the full report package without training or touching raw files."""
    charts(cfg)
    diagrams(Path(cfg["paths"]["docs"]))
    notebooks(cfg)
    final_pdf(cfg)


if __name__ == "__main__":
    run_reports(config())
