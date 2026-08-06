# ForecastPH — PSE Stock Forecast Pipeline (Backend)

ForecastPH forecasts next-session closing prices for selected PSE-listed companies. **This repository
is the data/ML pipeline only** — it scrapes, cleans, engineers features, trains, evaluates, and
produces forecasts, then exports the results as static JSON to `public/forecasts/`.

The UI lives in a separate repo, [pse-stock-price-forecast-dashboard-vercel-frontend](https://github.com/AlvinTubtub/pse-stock-price-forecast-dashboard-vercel-frontend),
deployed on Vercel. It contains no Python and calls no backend — it fetches the JSON files this
repository's GitHub Actions workflows commit to `public/forecasts/` straight from GitHub's raw
content API at request/build time. See that repo's README for the frontend, and "Wiring to the
frontend" below for how the two repos stay in sync.

**This repo is a pure, read-only-to-the-frontend data pipeline.** There is no server, no API, and no
upload/retraining capability triggered from the UI — every number the frontend shows comes from files
already committed here by the automated pipeline described below. `scripts/export_forecast_artifacts.py`
is the only new piece: it reshapes this pipeline's existing outputs (`data/raw/`, `prediction_cache/`,
`best_models.json`, `latest_processed.json`, `statistical_tests.json`) into the flat JSON contract the
frontend expects, and runs as the last step of both GitHub Actions workflows below.

> The previous Streamlit UI (`app.py`, `ui/`, `pages_app/`) has been removed from this repository —
> it's fully superseded by the frontend repo above.

## Wiring to the frontend (two-repo setup)

Because the frontend is a separate repo, committing `public/forecasts/` here doesn't by itself
trigger a Vercel rebuild — the frontend reads this repo's raw GitHub content directly, on a 1-hour
revalidation window, but that's a safety net, not the primary refresh path. For near-immediate
updates after every pipeline run:

1. In the **frontend** Vercel project: Project Settings → Git → **Deploy Hooks** → create one
   (`main` branch) → copy the generated URL.
2. In **this** repo: Settings → Secrets and variables → Actions → New repository secret →
   name it `VERCEL_DEPLOY_HOOK_URL`, paste the URL.
3. Both workflows already call it (`Trigger Vercel redeploy` step, added alongside
   `export_forecast_artifacts.py`) — no further changes needed. If the secret isn't set, that step
   just logs a warning and skips instead of failing the run.

## Features

- Home dashboard with project overview
- Company list and sector browsing
- Company details with historical charts, next-day forecast, and actual-vs-predicted backtests
- Model comparison across forecasting methods (RMSE/MAE/MASE/R²)
- Educational section explaining OHLCV and forecasting models
- About page for project context and capstone background
- A "data last refreshed" indicator sourced directly from the automated pipeline's own run metadata

## Architecture

```text
Cron-job.org (Mon–Fri, 4:00 PM PHT)         GitHub Actions cron (Sun, 8:00 AM PHT)
        │  POST repository_dispatch                  │  schedule trigger
        ▼                                             ▼
.github/workflows/update_pipeline.yml       .github/workflows/train_models.yml
        │  ("Fast Pipeline")                          │  ("Heavy Training")
        ▼                                             ▼
run_pipeline.py --no-train                  services.model_selector.train_and_select_all()
  1. Download latest PSE EDGE disclosures      1. Feature engineering
  2. Extract and validate PDF tables           2. Retrain Lag-Informed Regression, ARIMA, LSTM
  3. Update OHLCV datasets                     3. Evaluate (RMSE, MAE, MASE, R²)
     (data/raw/<SYMBOL>.csv)                   4. Statistical significance suite (DM/HLN,
  4. Update latest_processed.json                 Friedman, Wilcoxon-Holm, consistency check)
        │                                       5. Select the best model per company
        ▼                                       6. Update prediction_cache/ + best_models.json
Commit changed artifacts only                    + statistical_tests.json
(idempotent — no-op if nothing changed)               │
        │                                             ▼
        │                                    Commit changed artifacts only
        │                                    (idempotent — no-op if nothing changed)
        ▼                                             │
        └──────────────────┬──────────────────────────┘
                            ▼
        Streamlit Community Cloud auto-redeploys from the new commit
                            │
                            ▼
        Dashboard reflects the latest data/models — no user interaction required
```

Data refreshes daily (Mon-Fri); models retrain weekly (Sun) — see
"Automated Pipeline" below for why, and the runtime numbers behind that
split.

Streamlit (`app.py`, `pages_app/`, `ui/`) only ever:
- loads `data/raw/*.csv`, `models/`, `prediction_cache/`, `best_models.json`, `latest_processed.json`
- displays the Company List, Company Details, Forecast Results, Model Performance, charts, and dashboard metrics built from what it loaded

Streamlit never downloads PDFs, processes data, retrains models, executes any forecasting pipeline, or writes anything back to the repository. There is no "Update Data" page, no upload widget, and no button anywhere in the app that triggers processing — the only way data changes is a commit from the automated pipeline landing in the repo.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- Statsmodels (ADF test, CV-scored ARIMA order search, Ljung-Box diagnostics)
- Torch
- Joblib (model persistence)
- pdfplumber, requests (PDF ingestion pipeline)

## Model Training Pipeline

Training happens exclusively inside the automated pipeline (never in Streamlit):

```text
PSE EDGE PDF -> PDF Extraction -> CSV Generation -> Data Validation
    -> Feature Engineering -> Model Training (x3) -> Model Evaluation
    -> Statistical Significance Tests -> Best Model Selection
    -> Saved Models -> Streamlit Dashboard (read-only)
```

All three models predict next-day **ΔClose** = Close(t+1) − Close(t)
rather than the raw Close level, and reconstruct
`Predicted Close(t+1) = Close(t) + Predicted ΔClose(t+1)` before any
metric is computed — this matches the capstone paper's methodology and
keeps RMSE/MAE/MASE/R² reported in peso terms, not on an internal
differenced/scaled target.

- `services/feature_engineering.py` — lag features + technical indicators
  (EMA 10/20, RSI 14, MACD/Signal, Bollinger Bands, daily return, rolling
  volatility, High-Low and Open-Close spreads) plus the expanded return
  feature set (lagged returns 1-20, rolling return mean/volatility at
  5/10/20, high-low range %, log volume, rolling volume means), shared by
  every model.
- `services/time_series_cv.py` — shared expanding-window rolling-origin
  CV splitter (5 folds, shrinking only for short series), used by both
  the regression's lambda selection and the ARIMA order search.
- `services/forecasting/lag_regression.py` — training-only StandardScaler
  -> PACF-assisted lag selection -> LASSO as the *final* estimator (no
  secondary OLS refit), lambda chosen by 5-fold expanding-window CV.
- `services/forecasting/arima_model.py` — ADF stationarity test +
  (p, d, q) search restricted to p<=3, d<=2, q<=3, scored by
  expanding-window rolling-origin CV with walk-forward one-step
  forecasting (not AIC), plus Ljung-Box residual diagnostics.
- `services/forecasting/lstm_model.py` — single LSTM layer + linear head,
  predicting scaled ΔClose (Min-Max scaler fit on the training split
  only), hyperparameters chosen by grid search over lookback
  (5/10/20/30), hidden units (25/50/100), learning rate (0.01/0.001), and
  batch size (16/32) — 48 configurations, each trained with mini-batches,
  up to 200 epochs, early stopping (patience 10), seed 42.
- `services/evaluation.py` — shared RMSE/MAE/MASE/R² metrics (computed on
  reconstructed peso prices; MASE is scaled against the in-sample naive
  one-step forecast), plus the cross-model statistical-significance
  suite: Diebold-Mariano with Newey-West HAC variance and the
  Harvey-Leybourne-Newbold small-sample correction (within each company),
  Holm-Bonferroni correction, a Friedman rank test and Holm-adjusted
  Wilcoxon signed-rank post-hoc tests (across companies), and a
  best-model consistency check (lowest RMSE on >=8 of 15 companies).
- `services/model_selector.py` — orchestrates training all three models
  per ticker, saves them under `models/`, caches predictions under
  `prediction_cache/`, writes `best_models.json`, and — once every ticker
  is trained — runs and saves the statistical-significance suite to
  `statistical_tests.json`.

Training is now decoupled from data ingestion: `train_and_select_all()`
is called directly by `.github/workflows/train_models.yml` (Heavy
Training, weekly), *not* by every run of
`services/pdf_pipeline/pipeline.py`/`run_pipeline.py` (Fast Pipeline,
Monday-Friday — see "Automated Pipeline" below for the full split and
why). `run_pipeline.py` still supports training inline via its
`train_models=True` default, for local/manual use:

```bash
python -m services.model_selector          # train + save models for every data/raw/*.csv, no ingestion
python run_pipeline.py                     # ingest + train in one go (local/dev; CI never does both together)
python run_pipeline.py --no-train           # ingest new data only, skip retraining (what the Fast Pipeline runs)
```

**Runtime**: the CV-based ARIMA order search and the 48-configuration
LSTM grid are considerably more expensive than the models this pipeline
originally shipped with. Measured on real PSE data (~1,600 trading days):
ARIMA's order search is the dominant cost at roughly 7 minutes/ticker,
the LSTM grid roughly 3 minutes/ticker, and the regression well under a
second — around 10 minutes/ticker, so ~2.5 hours for all 15 tickers.
`train_and_select_all()` retrains *every* ticker in `data/raw/` on every
call (not just ones with new data), which is exactly why it now runs
weekly (Heavy Training) rather than on every data update (Fast
Pipeline) — see "Automated Pipeline" below. If weekly still isn't often
enough, or ~2.5h/week stops being acceptable, the straightforward options
are: (a) only retrain tickers that changed since the last training run,
(b) fan the 15 tickers out across a GitHub Actions matrix job, or (c)
narrow the CV/grid-search space — none of which this refactor does by
default, to keep the implementation an exact match for the paper's
specified search
spaces.

## Project Structure

```text
pse-streamlit-2/
├── app.py
├── run_pipeline.py            # headless CLI entrypoint for the Fast Pipeline (and local dev)
├── requirements.txt
├── requirements-fast.txt      # deps for the Fast Pipeline (no ML stack)
├── requirements-pipeline.txt  # deps for the Heavy Training pipeline / local full-pipeline dev
├── README.md
├── .github/
│   └── workflows/
│       ├── update_pipeline.yml   # Fast Pipeline — Mon-Fri, triggered by Cron-job.org
│       └── train_models.yml      # Heavy Training — weekly, GitHub Actions native cron
├── data/
│   ├── raw/                # <TICKER>.csv — the data the dashboard reads
│   ├── pdf_reports/        # staged PSE EDGE EOD PDFs (gitignored, except bundled samples)
│   └── pdf_pipeline/       # intermediate ETL artifacts + pipeline.log (gitignored)
├── models/
│   ├── lag_regression/     # <TICKER>.pkl
│   ├── arima/              # <TICKER>.pkl
│   └── lstm/                # <TICKER>.pth
├── prediction_cache/        # <TICKER>.json — cached metrics/predictions the dashboard loads
├── best_models.json         # {"<TICKER>": "<best model label>"} per ticker, lowest RMSE
├── statistical_tests.json   # cross-model significance tests (DM/HLN, Friedman, Wilcoxon-Holm, consistency check)
├── latest_processed.json    # metadata about the most recent automated pipeline run
├── pages_app/
│   ├── about.py
│   ├── companies.py
│   ├── compare.py
│   ├── details.py
│   ├── home.py
│   └── learn.py
├── services/
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── feature_engineering.py   # lag + technical-indicator + return features, shared by all models
│   ├── time_series_cv.py        # shared expanding-window rolling-origin CV splitter
│   ├── evaluation.py            # shared RMSE/MAE/MASE/R² metrics + statistical significance suite
│   ├── model_selector.py        # trains all 3 models per ticker, saves them, picks the best
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── lag_regression.py
│   │   ├── arima_model.py
│   │   └── lstm_model.py
│   └── pdf_pipeline/       # PDF ingestion pipeline (download, parser, cleaner, validator, merge)
│       ├── config.py
│       ├── downloader.py
│       ├── parser.py
│       ├── cleaner.py
│       ├── validator.py
│       ├── merge.py
│       └── pipeline.py     # single orchestration layer: ingestion -> training -> metadata
└── ui/
```

## Requirements

- Python 3.10 or newer
- pip

## Installation (running the dashboard locally)

Clone or download the repository, then open a terminal in the project folder.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If the `streamlit` command isn't available: `python -m streamlit run app.py`.

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

Since Streamlit never generates data itself, a fresh clone shows whatever
`data/raw/`, `models/`, and `prediction_cache/` were last committed —
normally the output of the most recent automated pipeline run.

## Data Format

Each company CSV in `data/raw/` contains: `Date, Open, High, Low, Close, Volume`, named by ticker symbol (e.g. `ALI.csv`, `BPI.csv`, ...). The full list of 15 tracked tickers: ALI, APX, BPI, GLO, ICT, JFC, MBT, MEG, MER, NIKL, PGOLD, SCC, SECB, SHLPH, SMPH.

## Notes

- `SECB.csv` replaces `BDO.csv` throughout the project.
- `services/data_loader.py` maps ticker symbols to the company metadata used in the dashboard.
- If a company CSV or trained model is missing, the app shows a placeholder or an in-app "not processed yet" message instead of failing — see [Architecture](#architecture).

## Automated Pipeline

The pipeline is split across two workflows so the expensive part doesn't
run daily:

| | `.github/workflows/update_pipeline.yml` ("Fast Pipeline") | `.github/workflows/train_models.yml` ("Heavy Training") |
|---|---|---|
| Does | PDF ingestion -> `data/raw/` CSVs only (`python run_pipeline.py --no-train`) | Retrains all 3 models on current `data/raw/` (`python -m services.model_selector`) |
| Schedule | Monday-Friday, 4:00 PM Philippine Time | Sunday, 8:00 AM Philippine Time |
| Trigger | External: [Cron-job.org](https://cron-job.org) `repository_dispatch` (no GitHub-native cron) | GitHub Actions' own `schedule: cron` |
| Dependencies | `requirements-fast.txt` (pandas/numpy/pdfplumber/requests) | `requirements-pipeline.txt` (adds scikit-learn/statsmodels/torch) |
| Typical runtime | A couple of minutes | ~2.5 hours (see "Model Training Pipeline" above) |
| Commits | `data/raw/`, `latest_processed.json` | `models/`, `prediction_cache/`, `best_models.json`, `statistical_tests.json` |

Both share the `pse-pipeline` concurrency group, so they queue instead of
racing each other if a run overlaps. Since PSE doesn't trade weekends,
Heavy Training doesn't re-fetch PDFs itself — by Sunday, `data/raw/` is
already current through Friday's close via the week's Fast Pipeline runs.

### Setting up the Cron-job.org trigger (Fast Pipeline only)

Heavy Training needs no external setup — it's a native GitHub Actions
`schedule: cron` trigger, already configured in `train_models.yml`. Only
the Fast Pipeline needs Cron-job.org:

1. Create a GitHub Personal Access Token with `repo` + `workflow` scope (a fine-grained token scoped to just this repo's contents+actions permissions also works).
2. In Cron-job.org, create a new job with:
   - **Schedule**: Monday–Friday, 16:00 (4:00 PM) — set the job's timezone to `Asia/Manila`.
   - **Request type**: Custom HTTP request (`POST`)
   - **URL**: `https://api.github.com/repos/<OWNER>/<REPO>/dispatches`
   - **Headers**:
     - `Accept: application/vnd.github+json`
     - `Authorization: Bearer <YOUR_GITHUB_PAT>`
     - `X-GitHub-Api-Version: 2022-11-28`
   - **Body**: `{"event_type": "run-pipeline"}`
3. Save. Cron-job.org will now POST to GitHub on that schedule, which fires the `repository_dispatch` trigger and starts the Fast Pipeline workflow — no polling, no GitHub Actions schedule minute-drift.

**Never commit the PAT to this repository.** Store it only in Cron-job.org's own encrypted request-header field.

GitHub Actions' own cron (used by Heavy Training) can be delayed by a few
minutes during periods of high platform load — not a concern for a
weekly, non-latency-sensitive job, which is why it's only used there and
not for the Fast Pipeline's tighter Monday-Friday schedule.

### What each workflow does

**Fast Pipeline** (Monday-Friday):

1. Checks out the repo and installs `requirements-fast.txt`.
2. Runs `python run_pipeline.py --no-train`, which downloads new EOD reports, extracts, cleans, validates, and merges them into `data/raw/`, then writes `latest_processed.json`. No model retraining.
3. Verifies at least one non-empty CSV exists in `data/raw/` — if not, the job fails loudly instead of silently pushing nothing.
4. Stages `data/raw/` and `latest_processed.json`, then checks `git diff --cached`. If nothing changed (e.g. a market holiday, or the pipeline already ran for that data), the job **finishes successfully without committing** — idempotent, no empty commits ever.
5. If something changed, commits and pushes.
6. Uploads `data/pdf_pipeline/pipeline.log` as a build artifact either way, for troubleshooting.
7. Streamlit Community Cloud picks up the new commit and redeploys automatically — no separate step needed on this repo's side.

**Heavy Training** (Sunday):

1. Checks out the repo (already current through Friday, via the week's Fast Pipeline commits) and installs `requirements-pipeline.txt`.
2. Runs `python -m services.model_selector`, which retrains all three models for every ticker in `data/raw/`, evaluates them, selects the best model per company, runs the cross-model statistical-significance suite, and writes `prediction_cache/`, `best_models.json`, and `statistical_tests.json`.
3. Verifies `best_models.json` and `prediction_cache/` were actually populated.
4. Stages `models/`, `prediction_cache/`, `best_models.json`, and `statistical_tests.json`, then checks `git diff --cached` — same idempotency guarantee as the Fast Pipeline.
5. If something changed, commits and pushes; Streamlit redeploys automatically.

Both workflows are granted only `contents: write` — nothing else.

### Manual / ops trigger (workflow_dispatch)

For a maintainer testing or backfilling outside the scheduled runs:
**Actions → PSE Fast Data Pipeline → Run workflow** (optional `start_date`/`end_date` inputs, YYYY-MM-DD) or **Actions → PSE Weekly Model Training → Run workflow** (no inputs — trains on whatever `data/raw/` currently has). Both are operator actions taken directly in GitHub, entirely outside the deployed Streamlit app.

### Running it locally

```bash
pip install -r requirements-fast.txt       # PDF ingestion only
pip install -r requirements-pipeline.txt   # adds the ML stack, for training
pip install -r requirements.txt            # or this, if you also want to run the app

python run_pipeline.py                     # fetch new reports, process, train, evaluate, select
python run_pipeline.py --no-download       # only process what's already in data/pdf_reports/
python run_pipeline.py --no-train          # skip retraining (only refresh data/raw/ CSVs)
python run_pipeline.py --start-date 2026-07-01 --end-date 2026-07-27
python -m services.model_selector          # train + save models for every data/raw/*.csv, no ingestion
```

Exit code `0` means success (including "nothing new to do"); exit code `1` means a real failure — check `data/pdf_pipeline/pipeline.log`.

### Disabling automation

- In GitHub: **Actions → \<workflow name\> → ⋯ → Disable workflow** (do this for each of the two workflows independently).
- Or delete/rename the corresponding file under `.github/workflows/`.
- Independently, pause or delete the Cron-job.org job — that alone stops new Fast Pipeline runs from being triggered, without touching anything in this repo (Heavy Training is unaffected, since it doesn't depend on Cron-job.org).

### Idempotency / duplicate-run protection

Re-running either workflow on data/models it already has is safe and a
no-op at the commit layer: `merge_into_raw()` upserts by date (identical
rows produce an identical file), retraining on unchanged data reproduces
bit-for-bit-equivalent models, and each workflow's `git diff --cached`
check means an unchanged working tree never produces a commit —
including two accidental triggers on the same day.

## About the Forecasting Models

The dashboard compares three forecasting approaches, each predicting
next-day ΔClose and reconstructing a peso price from it (see "Model
Training Pipeline" above for each model's methodology):

- Lag-Informed Regression
- ARIMA
- LSTM

against a naive (yesterday's close) baseline, using:

- RMSE
- MAE
- MASE
- R²

Cross-model significance is assessed with Diebold-Mariano (Newey-West HAC
variance, HLN small-sample correction) within each company, and a
Friedman rank test with Holm-adjusted Wilcoxon signed-rank post-hoc tests
across all companies — see `statistical_tests.json`, written by
`services/model_selector.py` on every pipeline run.

## Disclaimer

This dashboard is intended for academic, educational, and analytical decision-support purposes only. It is not financial advice and should not be used as the sole basis for investment decisions.

## License

For academic and internal project use.
