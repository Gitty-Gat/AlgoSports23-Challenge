# Repository Layout

## Core
- `src/` Python source code
- `run_pipeline.py` main pipeline entrypoint
- `phase1_execute.py` phase execution orchestrator

## Active Runtime Artifacts (root)
- `Submission.zip`
- `Predictions.csv` / `predictions.csv`
- `Rankings.xlsx` / `rankings.xlsx`
- `run_report.md`
- `final_report.pdf`
- `run_metadata.json`
- `selected_config.json`
- `runner_up_configs.json`
- `oof_predictions.csv` / `oof_predictions.parquet`

## Organized Artifacts
- `artifacts/logs/` run logs
- `artifacts/timing/` timing summaries
- `artifacts/archives/` historical zip bundles
- `artifacts/submissions/history/` historical submission zips
- `artifacts/submissions/legacy_dirs/` historical submission directories
- `artifacts/scratch/` temporary/diagnostic artifacts

## Reports
- `legacy_reports/` archived prior reports
- root reports are current/latest

## Docs
- `docs/MODEL_IMPROVEMENT_AGENTS.md`
- `docs/REPO_LAYOUT.md`
- `docs/BRANCHES.md`
- `docs/r_legacy/` legacy R/EDA files
