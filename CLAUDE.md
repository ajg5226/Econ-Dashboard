# CLAUDE.md — Econ-Dashboard Codebase Guide

This file provides AI assistants with essential context for working on this codebase. Read it before making any changes.

---

## Project Overview

**Econ-Dashboard** is a production-grade U.S. recession prediction web application that combines traditional econometrics with machine learning to forecast recession probability **6 months forward**. It is designed for professional/institutional use (financial analysts, portfolio managers).

**Performance**: 0.994 AUC on out-of-sample data (2016–2025), 99.1% accuracy, <1% false positive rate, 100% recession recall.

---

## Repository Structure

```
/
├── app/                          # Streamlit web application
│   ├── main.py                   # Entry point: auth + routing
│   ├── auth.py                   # Authentication & RBAC
│   ├── config.yaml               # User credentials (bcrypt-hashed)
│   ├── pages/
│   │   ├── dashboard.py          # Recession probability time series
│   │   ├── indicators.py         # Economic indicators explorer
│   │   ├── model_performance.py  # Model metrics & calibration plots
│   │   └── settings.py           # Admin: data refresh, user management
│   └── utils/
│       ├── data_loader.py        # CSV persistence layer
│       ├── plotting.py           # Plotly/matplotlib helpers
│       └── cache_manager.py      # Streamlit cache utilities
│
├── recession_engine/             # Core prediction engine
│   ├── data_acquisition.py       # FRED API: 45+ indicators, 362 features
│   ├── ensemble_model.py         # Three-tier ensemble (Probit/RF/XGBoost)
│   ├── backtester.py             # Pseudo OOS + ALFRED vintage backtests
│   ├── model_monitor.py          # Drift detection & monitoring
│   └── run_recession_engine.py   # Standalone engine runner
│
├── scheduler/
│   ├── update_job.py             # Main update pipeline (data + retrain)
│   ├── scheduler_config.py       # Runtime config persistence
│   └── run_scheduler.sh          # Shell wrapper
│
├── pages/                        # Legacy Streamlit shim pages (thin wrappers)
│   ├── 1_Dashboard.py
│   ├── 2_Indicators.py
│   ├── 3_Model_Performance.py
│   └── 4_Settings.py
│
├── tests/                        # pytest suite
│   ├── test_data_acquisition.py
│   ├── test_ensemble_model.py
│   ├── test_backtester.py
│   ├── test_update_job.py
│   └── test_scheduler.py
│
├── data/                         # Persistent storage (git-tracked for deployed data)
│   ├── predictions.csv           # Model output (Date, Prob_Ensemble, CI_Lower, CI_Upper, …)
│   ├── indicators.csv            # Raw + engineered indicator data
│   └── models/                   # Serialized models + config JSON files
│       ├── *.pkl                 # probit, random_forest, xgboost, meta_model, scaler
│       ├── features.txt          # Active feature list
│       ├── runtime_config.json   # Scheduler settings
│       ├── threshold.json        # Decision threshold + confidence intervals
│       ├── ensemble_weights.json # Performance-weighted ensemble coefficients
│       ├── run_manifest.json     # Audit trail: git SHA, metrics, timestamp
│       └── monitor_report.json   # Latest monitoring report
│
├── .github/workflows/
│   ├── ci.yml                    # pytest on push/PR
│   └── scheduler.yml             # Weekly data refresh (Sunday 3 AM UTC)
│
├── .streamlit/config.toml        # Streamlit theme + server config
├── streamlit_app.py              # Streamlit Cloud entry point (imports app/main.py)
├── Dockerfile                    # Python 3.11-slim; exposes 8501
├── Procfile                      # Heroku: web: streamlit run app/main.py
├── requirements.txt              # Pinned Python dependencies
├── setup.sh                      # Automated dev environment setup
└── Documentation/                # Architecture, deployment, data docs
```

---

## Development Commands

### Local Setup
```bash
./setup.sh                          # Create venv, install deps, create data dirs
source venv/bin/activate
cp .env.example .env                # Add your FRED_API_KEY
```

### Run Application
```bash
streamlit run app/main.py           # Development (localhost:8501)
streamlit run streamlit_app.py      # Streamlit Cloud entry point
```

### Run Engine (standalone)
```bash
python scheduler/update_job.py      # Fetch FRED data + retrain + save predictions
python recession_engine/run_recession_engine.py  # Engine only
```

### Tests
```bash
python -m pytest tests/ -q          # All tests
python -m pytest tests/ -v          # Verbose
python -m pytest tests/test_ensemble_model.py -v  # Single module
```

### Docker
```bash
docker build -t recession-dashboard .
docker run -p 8501:8501 -e FRED_API_KEY='...' -v $(pwd)/data:/app/data recession-dashboard
```

---

## Key Architecture Decisions

### 1. Three-Tier Ensemble Model
The model stack in `recession_engine/ensemble_model.py`:
1. **Probit** (L1-regularized logistic regression) — interpretable, client-facing
2. **Random Forest** (200 trees, max_depth=10) — feature importance
3. **XGBoost** (300 rounds, lr=0.01) — maximum accuracy
4. **Meta-model** (logistic regression on base predictions) — final ensemble

All models are trained on 1970–2015 (expanding window) and evaluated on 2016–present. **Never change the train/test split without explicit intent** — this would invalidate the published AUC figures.

### 2. Feature Engineering (362 features from 45 indicators)
Each raw FRED series generates: levels, MoM/3M/6M/YoY changes, moving averages, 6M rolling volatility, percentile-based weakness flags, PCA composites, and at-risk transformations. See `data_acquisition.py` for the full pipeline.

### 3. Target Variable
The prediction target is `RECESSION_FORWARD_6M` — a binary label: will a recession start within the next 6 months? This is constructed from NBER recession dates and lagged 6 months. **Do not confuse with current recession status.**

### 4. Caching Strategy
- Predictions: `@st.cache_data(ttl=3600)` — refresh every hour
- Indicators: `@st.cache_data(ttl=86400)` — refresh every 24 hours
- Models (pkl files): `@st.cache_resource` — persist until manual admin refresh
- Call `st.cache_data.clear()` or `st.cache_resource.clear()` to force refresh

### 5. File-Based Persistence
There is no database. All state is files:
- `data/predictions.csv` — primary output consumed by the UI
- `data/indicators.csv` — feature store
- `data/models/*.pkl` — serialized model artifacts (joblib)
- `data/models/runtime_config.json` — scheduler config
- `data/models/run_manifest.json` — audit trail

---

## Authentication & RBAC

Authentication uses `streamlit-authenticator`. Credentials are stored in `app/config.yaml` with bcrypt-hashed passwords.

**Roles**:
- `admin` — can trigger data refresh, manage users, clear cache
- `viewer` (default) — read-only dashboard access

**Default credentials** (change before production):
- `admin` / `admin123`
- `analyst1`–`analyst4` / `analyst123`

**Important**: Authentication state is managed in `st.session_state`. The authenticator object is cached in session to avoid duplicate widget errors on Streamlit Cloud. See `auth.py` for implementation details.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FRED_API_KEY` | Yes | Federal Reserve FRED API key (free at fred.stlouisfed.org) |
| `SECRET_KEY` | No | App secret key |
| `SCHEDULER_INTERVAL` | No | `daily` / `weekly` / `monthly` (default: `weekly`) |
| `PREDICTION_HORIZON` | No | Months forward (default: `6`) |
| `TRAIN_END_DATE` | No | Training cutoff override (default: rolling) |

For **Streamlit Cloud**, set these in the app's Secrets settings (not `.env`).

---

## Styling & UI Conventions

- **Theme**: Blue (`#1f77b4`) primary, white background, light gray secondary. Defined in `.streamlit/config.toml`.
- **Plots**: Prefer Plotly for interactive charts. Matplotlib used only as fallback.
  - Recession periods shaded in background
  - Confidence interval bands as shaded regions
  - Hover tooltips with date + probability
- **Risk status indicators**: Use consistent emoji labels — 🟢 LOW (<20%), 🟡 MODERATE (20–40%), 🟠 ELEVATED (40–60%), 🔴 HIGH (>60%)
- **No custom CSS** unless absolutely necessary — prefer Streamlit's native components.

---

## Code Conventions

### General
- **Python 3.11** (fixed in Dockerfile and CI)
- Type hints on all public function signatures: `def fit(self, X: pd.DataFrame) -> None:`
- Docstrings on all public classes and methods
- Logging via `logging.getLogger(__name__)` — not `print()`
- Use `logger.info()` / `logger.warning()` / `logger.error()` consistently

### Error Handling
- Wrap FRED API calls in try/except — always log the error and continue gracefully
- Return empty DataFrames (not None) on data fetch failure
- Let Streamlit handle UI errors via `st.error()` / `st.warning()` — avoid silent failures
- Check for required columns before processing DataFrames

### Data Handling
- Forward-fill (`ffill`) is the standard imputation for missing time series values
- Always sort DataFrames by date before use
- Preserve the `Date` column as a proper datetime dtype
- Never hardcode date strings — use config values or datetime arithmetic

### Model Artifacts
- Save with timestamps: `probit_YYYYMMDD_HHMMSS.pkl`
- Always update `run_manifest.json` after a successful retrain (includes git SHA, metrics, timestamp)
- Features list saved as plain text `features.txt` for easy inspection
- The `scaler.pkl` must be retrained alongside models — never reuse an old scaler with new models

### Academic References
Key literature underlying model choices (cite in code comments where relevant):
- Estrella & Mishkin (1998) — yield curve recession prediction
- Wright (2006) — federal funds rate augmentation
- Gilchrist & Zakrajsek (2012) — credit spreads (EBP)
- Sahm (2019) — unemployment-based recession trigger

---

## Testing Guidelines

- Tests live in `tests/` and use `pytest`
- Mock FRED API calls in tests — never make real network requests in tests
- Model tests use synthetic/minimal data — don't require full 1970–present dataset
- CI runs on every push and PR via `.github/workflows/ci.yml`
- Tests must pass before merging to `main`

---

## Deployment Contexts

| Environment | Entry Point | Config |
|-------------|-------------|--------|
| Local dev | `app/main.py` | `.env` |
| Streamlit Cloud | `streamlit_app.py` | Secrets settings |
| Docker | `app/main.py` | Environment variables |
| Heroku | `app/main.py` (via Procfile) | Config vars |

**GitHub Actions scheduler** runs every Sunday at 3 AM UTC, executes `scheduler/update_job.py`, and commits updated `data/` files back to the repo.

---

## Common Tasks

### Add a new economic indicator
1. Add the FRED series ID and metadata to `INDICATORS` dict in `data_acquisition.py`
2. Add feature engineering logic to `_engineer_features()` if needed
3. Re-run `python scheduler/update_job.py` to regenerate `data/indicators.csv`
4. Retrain models (same command triggers retraining)

### Update model hyperparameters
Edit the model initialization in `ensemble_model.py` within `_train_base_models()`. Always re-run backtesting after changes to verify AUC doesn't regress.

### Change prediction horizon
Set `PREDICTION_HORIZON` env var or update `runtime_config.json`. The target variable `RECESSION_FORWARD_6M` must be regenerated.

### Change the decision threshold
Update `data/models/threshold.json`. The threshold is Youden's J optimized — don't use 0.5 unless specifically intended.

### Add a new dashboard page
1. Create `app/pages/new_page.py` with a `render()` function
2. Create a `pages/N_PageName.py` shim that calls `render()`
3. Streamlit auto-discovers pages in the `pages/` directory

### Rotate credentials
Edit `app/config.yaml` with bcrypt-hashed passwords. Use `streamlit-authenticator`'s `Hasher` utility to generate hashes.

---

## Known Constraints & Gotchas

- **PyTorch/LSTM disabled**: `torch` is commented out in `requirements.txt`. The LSTM model was removed due to insufficient training sample size for the small monthly dataset (~650 observations). Do not re-enable without thorough validation.
- **Markov-switching weight capped at 5%**: Contributes to false positives if weighted higher. See commit `601de4f`.
- **FRED API rate limit**: 120 requests/minute. `data_acquisition.py` has throttling built in — don't bypass it.
- **Class imbalance**: Recessions occur ~12% of months. All models use class-weighted loss. Accuracy alone is a misleading metric — always evaluate AUC and recall.
- **Streamlit widget state**: Avoid creating widgets inside `if` blocks where the condition can toggle — this causes duplicate widget key errors. Cache authenticator in session state (see `auth.py`).
- **Model promotion guard**: `update_job.py` compares new model performance to the incumbent on a holdout set before promoting. New models only replace old ones if they improve (see commit `0fb0cf6`). Do not bypass this check.
- **data/ is git-tracked**: The `data/` directory is committed so deployed Streamlit Cloud instances have pre-built models and predictions without needing to run the full pipeline on cold start.
