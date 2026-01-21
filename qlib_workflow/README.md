# Qlib rotation workflow

This workflow converts your existing ETF data into Qlib format, then uses Qlib
to run an AutoML-style grid search and MLOps recording so you can compare the
best Qlib-discovered rotation strategy against your baseline rotation strategy.

## Prerequisites

1. Install dependencies:
   `python3 -m pip install pyqlib`

2. Prepare Qlib data directory:
   - Set `provider_uri` in `config.toml` to where Qlib `.bin` data will live.
   - Qlib reads data from `provider_uri` after conversion. See Qlib docs on
     `provider_uri` and the `.bin` format. [Qlib data docs](https://qlib.readthedocs.io/en/latest/component/data.html?utm_source=openai)

## Configuration

Edit `qlib_workflow/config.toml`:

- `provider_uri`: local qlib data path
- `data_source.sqlite_path`: existing DB (e.g. `data/etf_momentum.sqlite3`)
- `data_source.symbol_map`: map DB codes -> Qlib instruments (e.g. `159915 -> SZ159915`)
- `data_source.csv_dir`: temp CSV folder for Qlib conversion
- `data_source.qlib_dir`: Qlib `.bin` output directory (same as `provider_uri`)
- `symbols`: universe (use qlib instrument codes, e.g. `SZ159915`)
- `backtest`: date range, rebalance weekday, lookback, topk
- `optimize`: grid for lookback/topk/weekday

## Step 1: Convert DB data to Qlib format

Export CSVs from your existing DB:
`python3 qlib_workflow/export_qlib_csv.py`

Convert CSVs to Qlib `.bin` format:
`python3 qlib_workflow/dump_qlib_bin.py`

Qlib conversion uses `dump_bin.py` (official script). See Qlib docs for details.
[dump_bin docs](https://qlib.readthedocs.io/en/latest/component/data.html?utm_source=openai)

## Step 2: Standard ML pipeline + visual report

Run the standard Qlib ML pipeline (model -> signal -> backtest -> report):
`python3 qlib_workflow/ml_pipeline.py`

Qlib will generate the report under the recorder artifacts (look for `analysis/` and `report/`).
The exact path is controlled by Qlib's experiment manager (default in Qlib workspace).

## Step 3: Complete AutoML pipeline

Run the full AutoML pipeline (multiple handlers/models/strategies + reports):
`python3 qlib_workflow/automl_pipeline.py`

Run summary is saved at:
`qlib_workflow/outputs/automl_runs/runs.json`

Ranking table (CSV/HTML):
`qlib_workflow/outputs/automl_runs/runs.csv`
`qlib_workflow/outputs/automl_runs/runs.html`
`qlib_workflow/outputs/automl_runs/ranking.png`

Baseline comparison (CSV/HTML):
`qlib_workflow/outputs/automl_runs/baseline.csv`
`qlib_workflow/outputs/automl_runs/compare.csv`
`qlib_workflow/outputs/automl_runs/compare.html`
`qlib_workflow/outputs/automl_runs/compare.png`

Best config snapshot:
`qlib_workflow/outputs/automl_runs/best_config.json`

Summary:
`qlib_workflow/outputs/automl_runs/summary.md`

## Optional: AutoML + baseline comparison

If you still want a simple grid-search AutoML + baseline comparison (CSV outputs):
`python3 qlib_workflow/automl_compare.py`

Outputs are saved under `qlib_workflow/outputs/YYYYMMDD_HHMMSS/`.

## Notes

- `rebalance_weekday` is the *execution day* (0=Mon..4=Fri).
- Decision day is the previous trading day close, so the signal does not use
  the execution day's price data.
