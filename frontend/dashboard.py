"""Business dashboard that consumes the FastAPI service only."""

from __future__ import annotations

import io
import os
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))


def get(path: str) -> Any:
    """Read service data and display actionable connection errors."""
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"The prediction service is unavailable: {exc}")
        st.stop()


@st.cache_data(ttl=60)
def customers() -> pd.DataFrame:
    """Cache persisted predictions briefly; never call a model in the UI."""
    return pd.DataFrame(get("/customers"))


def pdf_results(frame: pd.DataFrame) -> bytes:
    """Render genuine uploaded scoring results as a downloadable PDF."""
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Vantara customer scoring results", styles["Title"]),
        Paragraph(
            "Historical snapshot. Risk is the model probability of no purchase in the next 90 days. CLV is predicted 90-day net product revenue in GBP.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]
    rows = [["Customer", "Churn risk", "90-day CLV GBP", "Value tier"]]
    rows += [
        [
            str(r.customer_id),
            f"{r.churn_probability:.1%}",
            f"{r.predicted_clv_90d:,.2f}",
            r.value_tier,
        ]
        for r in frame.itertuples()
    ]
    table = Table(rows, repeatRows=1, colWidths=[110, 100, 140, 100])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17364d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f0f5f6")],
                ),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    SimpleDocTemplate(output).build(story)
    return output.getvalue()


def main() -> None:
    """Display all PRD views using stored scores and service explanations."""
    st.set_page_config(
        page_title="Vantara | Customer intelligence", page_icon="◈", layout="wide"
    )
    st.title("Vantara customer intelligence")
    st.caption("Retail retention, customer value and purchase behavior")
    frame = customers()
    if frame.empty:
        st.info(
            "No scored customers are available. Run the documented scoring pipeline."
        )
        st.stop()
    st.info(
        f"Historical dataset snapshot: {frame.as_of.iloc[0]}. Predictions cover the following 90 days. Amounts are GBP. Category and discount measures are proxies."
    )
    with st.sidebar:
        st.header("Explore customers")
        segment = st.multiselect("Segment", sorted(frame.segment.unique()))
        country = st.multiselect("Country", sorted(frame.country.unique()))
        tier = st.multiselect("Value tier", ["High", "Medium", "Low"])
        if st.button("Refresh stored results"):
            st.cache_data.clear()
            st.rerun()
    selected = frame.copy()
    for key, values in [
        ("segment", segment),
        ("country", country),
        ("value_tier", tier),
    ]:
        if values:
            selected = selected[selected[key].isin(values)]
    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{len(selected):,}")
    b.metric("Flagged for retention", f"{int(selected.high_risk.sum()):,}")
    c.metric("Predicted 90-day revenue", f"£{selected.predicted_clv_90d.sum():,.0f}")
    d.metric("Accounts for anomaly review", f"{int(selected.anomaly_flag.sum()):,}")
    tabs = st.tabs(
        [
            "Segments",
            "Retention priorities",
            "Sales and forecast",
            "Customer explanation",
            "Batch scoring",
        ]
    )
    with tabs[0]:
        st.subheader("Customer segments")
        if selected.empty:
            st.info("No customers match these filters.")
        else:
            summary = selected.groupby("segment", as_index=False).agg(
                customers=("customer_id", "size"),
                mean_spend=("monetary", "mean"),
                mean_risk=("churn_probability", "mean"),
            )
            st.plotly_chart(
                px.bar(
                    summary,
                    x="customers",
                    y="segment",
                    orientation="h",
                    color="mean_risk",
                    color_continuous_scale="Tealrose",
                    labels={
                        "customers": "Customers",
                        "segment": "Business profile",
                        "mean_risk": "Mean churn risk",
                    },
                ),
                use_container_width=True,
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.caption(
                "Segment names use training-customer medians. Profiles describe shopping patterns; they do not prove a customer will churn."
            )
    with tabs[1]:
        st.subheader("Retention priorities")
        st.caption(
            "Flagged customers come first, followed by high value and expected revenue at risk. This is a prioritization measure, not predicted campaign uplift."
        )
        leader = selected.assign(
            value_order=selected.value_tier.map({"High": 3, "Medium": 2, "Low": 1})
        ).sort_values(
            ["high_risk", "value_order", "retention_priority"], ascending=False
        )
        st.dataframe(
            leader[
                [
                    "customer_id",
                    "country",
                    "segment",
                    "value_tier",
                    "churn_probability",
                    "predicted_clv_90d",
                    "retention_priority",
                    "anomaly_flag",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    with tabs[2]:
        st.subheader("Observed sales and simple forecast")
        revenue = pd.DataFrame(get("/trends"))
        st.plotly_chart(
            px.line(
                revenue,
                x="month",
                y="revenue",
                color="kind",
                markers=True,
                labels={
                    "month": "Month",
                    "revenue": "Net product revenue (GBP)",
                    "kind": "Series",
                },
                color_discrete_map={
                    "observed": "#16887c",
                    "linear forecast": "#d0883f",
                },
            ),
            use_container_width=True,
        )
        st.caption(
            "Revenue includes anonymous product purchases. Forecast: a line fitted to the last six complete months. These charts cover the whole business; customer filters do not change them."
        )
        observed = revenue[revenue.kind == "observed"]
        st.plotly_chart(
            px.bar(
                observed,
                x="month",
                y="invoices",
                labels={"month": "Month", "invoices": "Distinct product invoices"},
            ),
            use_container_width=True,
        )
    with tabs[3]:
        st.subheader("Why this customer was scored")
        customer_id = st.text_input("Customer ID", value=str(frame.customer_id.iloc[0]))
        if st.button("Explain customer"):
            try:
                response = httpx.get(
                    f"{API_URL}/explain/{customer_id}", timeout=TIMEOUT
                )
                if response.status_code == 404:
                    st.warning("Customer ID was not found.")
                else:
                    response.raise_for_status()
                    explanation = response.json()
                    st.write(explanation["summary"])
                    contributions = pd.DataFrame(explanation["contributions"])
                    st.plotly_chart(
                        px.bar(
                            contributions,
                            x="contribution",
                            y="feature",
                            orientation="h",
                            title="SHAP contribution to churn probability",
                        ),
                        use_container_width=True,
                    )
                    local = pd.DataFrame(explanation["lime"])
                    st.plotly_chart(
                        px.bar(
                            local,
                            x="weight",
                            y="condition",
                            orientation="h",
                            title="LIME local comparison",
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        f"LIME local fit R²: {explanation['lime_local_r2']:.3f}. {explanation['units']}"
                    )
                    actual = frame[frame.customer_id == customer_id]
                    if not actual.empty:
                        st.dataframe(
                            actual[
                                [
                                    "purchase_probability_90d",
                                    "next_purchase_days_capped",
                                    "next_category_proxy",
                                    "recommendations",
                                    "recommendation_basis",
                                ]
                            ],
                            hide_index=True,
                        )
            except httpx.HTTPError as exc:
                st.error(f"Could not obtain explanation: {exc}")
        st.subheader("Global model drivers")
        st.plotly_chart(
            px.bar(
                pd.DataFrame(get("/importance")).head(12),
                x="mean_absolute_shap",
                y="feature",
                orientation="h",
                labels={
                    "mean_absolute_shap": "Mean absolute SHAP probability contribution"
                },
            ),
            use_container_width=True,
        )
    with tabs[4]:
        st.subheader("Score a CSV file")
        st.write(
            "Upload a customer_id column for existing customers, or customer_id plus every feature listed in API /metadata for new records. Maximum 500 rows and 2 MB."
        )
        st.download_button(
            "Download existing-customer example",
            frame[["customer_id"]].head(10).to_csv(index=False),
            "customer_ids.csv",
            "text/csv",
        )
        upload = st.file_uploader("Customer CSV", type=["csv"])
        if st.button("Score uploaded CSV", disabled=upload is None):
            try:
                response = httpx.post(
                    f"{API_URL}/predict/batch",
                    files={"file": (upload.name, upload.getvalue(), "text/csv")},
                    timeout=TIMEOUT,
                )
                if response.is_error:
                    st.error(
                        response.json().get("detail", "The upload could not be scored.")
                    )
                else:
                    st.session_state["batch_results"] = response.json()["results"]
            except httpx.HTTPError as exc:
                st.error(f"Batch scoring failed: {exc}")
        if "batch_results" in st.session_state:
            scored = pd.DataFrame(st.session_state["batch_results"])
            st.dataframe(scored, hide_index=True, use_container_width=True)
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False),
                "vantara_predictions.csv",
                "text/csv",
            )
            st.download_button(
                "Download scored PDF",
                pdf_results(scored),
                "vantara_predictions.pdf",
                "application/pdf",
            )
            st.caption(
                "New feature-only records have no sequence timing prediction. Their recommendations use historical affinity, as labelled in the output."
            )


if __name__ == "__main__":
    main()
