# Vantara Customer Intelligence Platform

This project implements the supplied Vantara PRD using the complete **Online Retail II** dataset. It includes a reproducible data pipeline, six classical churn classifiers, an ANN, an LSTM, an autoencoder, CLV regression, two clustering methods, SHAP/LIME explanations, a FastAPI service, and a Streamlit dashboard backed by PostgreSQL.

The application is a **historical prototype**, scored as of **1 September 2011**. It does not show current retail activity. All figures and model results come from actual source records and saved model outputs.

## Results

The production model is **LightGBM**, chosen on validation ROC-AUC before final test evaluation. On 784 held-out customers it achieved:

| Metric | Actual result | PRD target |
| --- | ---: | ---: |
| Churn ROC-AUC | 0.8002 | At least 0.80 |
| Churn recall | 0.9375 | At least 0.70 |
| Churn precision | 0.6796 | Reported with recall |
| Churn F1 | 0.7880 | Reported with recall |
| Churn accuracy | 0.7117 | Secondary metric |
| CLV R-squared | 0.5852 | At least 0.60: **not met** |
| CLV MAE | GBP 532.42 | Reported |
| CLV RMSE | GBP 3,311.30 | Reported |

Two fresh-browser segment views took 3.38 and 6.29 seconds, above the 3-second target; warm reloads took 0.63-2.23 seconds. Fresh API scoring p95 was 142.54 ms, within the 400 ms target. The CLV and initial dashboard-load shortfalls are explained in [the final report](docs/final_report.pdf). Test results were not used to change model selection or tune the models. The genuine [recorded walkthrough](docs/walkthrough.webm) shows the working dashboard. Runtime and coverage results are recorded in [runtime_checks.json](docs/runtime_checks.json). The complete classifier comparison is in [model_comparison.csv](docs/model_comparison.csv).

## Environment and setup

Use **Python 3.11** (this run used 3.11.14), Docker Engine, and Docker Compose. PostgreSQL is supplied by Compose. CPU-only PyTorch and XGBoost keep the environment smaller. Optional CatBoost, Transformers, demographic enrichment, streaming, authentication, and CI/CD are not included.

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.utils.setup
```

The setup command creates a private `.env` with a generated database password. It preserves an existing file. Do not commit or share this file. Direct and resolved dependency versions are pinned in `requirements.txt`.

The current local `.venv` uses an existing CPython 3.11.14 installation on this machine. A clean clone needs its own installed Python 3.11 or the supplied Docker image.

## Start the application

The delivered processed tables and model artifacts can be used directly:

```bash
docker compose up --build
```

If your installation exposes the older command name, use `docker-compose up --build`. After the first image build, `docker compose up` starts all three services. The API waits for PostgreSQL, imports the actual snapshot on first startup, and becomes healthy before the dashboard starts.

- Dashboard: http://localhost:8501
- API documentation and interactive requests: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json
- Readiness check: http://localhost:8000/health

Initial database import is separate from dashboard page loading. Docker stores the database in its named volume. `docker compose down` stops the services without deleting that volume. A changed model/data version requires a deliberate database refresh; the prototype does not silently replace existing scores.

For a local Python deployment with an existing PostgreSQL database:

```bash
export DATABASE_URL='postgresql+psycopg://USER:PASSWORD@localhost:5432/DATABASE'
python -m src.data.store
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another activated terminal:

```bash
export API_URL='http://127.0.0.1:8000'
streamlit run frontend/dashboard.py --server.address=127.0.0.1 --browser.gatherUsageStats=false --server.maxUploadSize=2
```

## Reproduce the data and results

The full scripted path is:

```bash
python -m src.data.pipeline --all
```

On a new checkout, this downloads the original workbook if absent, loads **both sheets**, validates and cleans records, builds features and labels, trains and logs models, evaluates once, creates segments and explanations, scores customers, and generates the report, PNGs and executed notebooks. No manual notebook cell is required. Jupyter kernels need permission to open local sockets.

The data/features-only command is:

```bash
python -m src.data.pipeline
```

The report-only command is:

```bash
python -m src.utils.reporting
```

A completed run keeps `models_artifacts/evaluation.json` as its evaluation freeze. Later pipeline runs reuse the trained model/evaluation artifacts, so ordinary report or service work does not repeat test selection. To run a new experiment, use a **new artifact directory** through `VANTARA_ARTIFACTS_PATH`, keep the original artifacts, and document the new evaluation design. Never tune repeatedly against this held-out test result.

The direct UCI workbook download is PRD Option A. It preserves both original sheets and avoids installing an extra download package. The PRD's exact source is a workbook, not an original CSV. No converted file is labelled as an original CSV.

## Source and data rules

Source: Chen, D. (2012). [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). UCI Machine Learning Repository. [DOI 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D). License: CC BY 4.0.

`data/raw/online_retail_II.xlsx` is unchanged. Its SHA-256 is:

```text
bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980
```

Both sheets contain 1,067,371 rows in total. Cleaning removes 34,335 exact duplicates and retains 1,033,036 rows. Missing customer IDs remain available for product-level analysis. Customer modeling uses identified product purchases. Returns are kept, and non-positive prices and administrative codes are flagged. IQR outliers are flagged; legitimate bulk purchases are preserved. Exact repeated lines cannot always be distinguished from legitimate identical purchases, so deduplication is a documented assumption.

The supplied PRD remains in its original location and was not edited. The project uses no data from unrelated local projects.

## Targets and feature meanings

- **Churn:** no paid product purchase in the next 90 days.
- **Predicted CLV:** net product revenue in the next 90 days, including returns. This is a finite-horizon estimate, not total future lifetime value.
- **Purchase probability:** one minus the churn probability.
- **Purchase timing:** LSTM waiting time capped at 90 days. Customers without a purchase are censored at that cap.
- **Next category:** a description-keyword proxy, conditional on a purchase. The source has no official category field.
- **Discount sensitivity:** a price-based proxy using only that customer's earlier SKU prices. No promotion labels or causal response claims are available.
- **Anomaly:** reconstruction error above the validation 99th percentile. It is a review flag, not confirmed fraud.

Features include recency, frequency, total/average spend, basket size, tenure, six-month order trend, gap variance, seasonal concentration, return rate, category affinity, discount proxy, historical net value and engagement. Definitions and reasons appear in the report and feature notebook. Product recommendations rank observed historical products inside the predicted category proxy; recommendation uplift has not been measured.

All features use timestamps strictly before the cutoff. Future labels use a complete 90-day window. The 5,224 customers are split 70/15/15 with no shared customers. Every supervised model uses five training-only stratified folds. Regression folds are stratified by purchase/no-purchase status. Encoders/scalers are fitted inside their training partitions. XGBoost and neural early stopping use separate training-only subsets during CV. The ANN and classical classifiers use the same feature columns.

`config/config.yaml` stores paths, random seed, thresholds, model settings and proxy definitions. Environment overrides include `CONFIG_PATH`, `VANTARA_RAW_PATH`, `VANTARA_INTERIM_PATH`, `VANTARA_PROCESSED_PATH`, `VANTARA_ARTIFACTS_PATH`, `VANTARA_DOCS_PATH`, `DATABASE_URL`, `API_URL`, `API_TIMEOUT` and `LOG_LEVEL`.

## API workflow

`POST /predict` accepts an existing customer ID:

```json
{"customer_id": "12346"}
```

This ID is an example request; use `/customers` to select any actual scored customer. If the ID is unknown, the service returns HTTP 404 instead of inventing a prediction.

A new feature record uses:

```text
{"customer_id": "YOUR_ID", "features": {all fields listed by GET /metadata}}
```

All numeric fields must be finite. Required fields cannot be missing, and unexpected fields are rejected. The service persists validated features and scores in one transaction. New feature-only inputs have no purchase sequence, so timing and LSTM category outputs remain null; recommendation fallback is clearly labelled.

`POST /predict/batch` accepts a multipart CSV field named `file`. Provide only `customer_id` for stored customers, or `customer_id` plus every required model feature for feature scoring. The limit is 500 rows and 2 MB. The service rejects malformed rows, missing values and duplicate IDs before writing results. The dashboard provides a downloadable example made from actual customer IDs.

`GET /customers` supports `segment`, `country` and `value_tier` filters. `GET /explain/{customer_id}` returns both SHAP and LIME. `/trends` and `/importance` provide genuine chart data. Full request/response details are in Swagger.

The dashboard includes segment filters, a retention leaderboard, revenue and invoice charts, a labelled forecast, customer search and explanations, batch upload, and PDF/CSV downloads. Revenue charts describe the whole business and are not changed by customer filters.

## Tests and code quality

With an actual PostgreSQL URL in `DATABASE_URL`:

```bash
python -m coverage run --source=src -m pytest -q tests
python -m coverage report --fail-under=70
python -m coverage json -o docs/coverage.json
python -m coverage report > docs/coverage.txt
python -m ruff check src api frontend tests
python -m black --check src api frontend tests
```

Without `DATABASE_URL`, PostgreSQL API tests are explicitly skipped. Do not describe that as a complete integration run. The tests use actual source records. A temporary training smoke test uses only a subset of the original **training** customers and does not replace the submitted models, metrics, or held-out test set.

The source checksum, future-data invariance, cutoff equality, return handling, incomplete label windows, split integrity, saved-model inference, SHAP reconstruction, API validation, CSV limits and persistence are checked. Keep a copy of the database when testing your own modified records: one existing snapshot customer is scored during integration tests and then restored from the delivered prediction table.

## Deliverables and folder structure

The PRD folder layout is retained. The root is named `vantara-project` because it is the existing workspace, instead of the illustrative `customer-behavior-prediction` name. Required outputs are stored under the specified folders; no separate application architecture was added. The temporary Docker verification cache is kept outside this project, under the local user cache.

| Folder | Contents |
| --- | --- |
| `data/raw/` | Original two-sheet workbook |
| `data/interim/` | Cleaned transactions and validation audit |
| `data/processed/` | Customer feature CSV/Parquet, sequences, catalog, predictions and revenue series |
| `notebooks/` | Three executed PRD notebooks |
| `src/data/` | Download, validation, cleaning and PostgreSQL storage |
| `src/features/` | Point-in-time features, targets and sequence preparation |
| `src/models/` | Classical/neural training, preprocessing and scoring |
| `src/segmentation/` | K-Means, GMM and business profiles |
| `src/explainability/` | SHAP, LIME and explanation templates |
| `src/utils/` | Configuration, structured logging, local setup and reporting |
| `api/` | FastAPI application and Pydantic requests |
| `frontend/` | Streamlit dashboard |
| `models_artifacts/` | Actual fitted models, encoders/scalers, logs and frozen metrics |
| `config/` | YAML settings |
| `tests/` | Source-backed unit and PostgreSQL integration tests |
| `docs/` | Final PDF, PNG findings, architecture/ER/workflow diagrams and verification evidence |

The lightweight HTTP routes remain in `api/main.py`; the PRD's `api/routers/` directory is reserved rather than adding unnecessary routing modules. Python package markers, `.gitignore` and `.dockerignore` support imports and protect private/generated files. The raw workbook is not committed; the documented source download restores it.

## Limitations and future work

The CLV target and the initial dashboard-load target were not met. LSTM timing accuracy is modest, and improvement over a tuned static timing model has not been established. Category and discount fields are proxies. Fraud labels, campaign results and live conversion outcomes are absent. The single-cutoff split measures other customers at the same historical time, not a later deployment period. The simple revenue overlay is an unvalidated linear extrapolation.

Future work may add later-time evaluation, verified categories and promotion data, recommendation ranking evaluation, controlled campaign tests, improved CLV under a new evaluation design, and production monitoring. Transformers, streaming ingestion, multi-tenant authentication, role-based access and production CI/CD remain deferred as the PRD specifies.
