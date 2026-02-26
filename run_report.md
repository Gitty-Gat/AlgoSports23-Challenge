# run_report

- git commit hash: `9ea582d`
- git branch: `kdecay_elo_simple`
- baseline reference commit: `9ea582d`

## budgets used
- MAX_TOTAL_SECONDS: 360.0
- MAX_TUNING_SECONDS: 120.0
- MAX_MODEL_FITS: 200
- tuning_elapsed_seconds: 5.8309
- total_elapsed_seconds_at_write: 18.235
- model_fit_count: 185
- tuning_stop_reason: None

## chosen params
- decay_type: linear
- A: 1.0
- G: 100
- tau: 50
- HOME_ADV: 50.0
- ridge_alpha: 100.0

## cv metrics
- chosen_rmse: 36.449115
- chosen_mae: 28.805594
- chosen_oof_pred_std: 22.933702
- chosen_oof_actual_std: 43.617887
- best_baseline_rmse (A=0): 36.467407
- best_baseline_mae (A=0): 28.809210
- best_kdecay_rmse (A>0): 36.44911500195051
- best_kdecay_mae (A>0): 28.80559371343184

## output validations
- predictions.csv rows: 75
- predictions.csv Team1_WinMargin missing: 0
- predictions.csv Team1_WinMargin numeric: True
- rankings.xlsx rows: 165
- rankings.xlsx Rank missing: 0
- rankings.xlsx rank_set_valid: True
- final_report.pdf exists: True
- final_report.pdf size_bytes: 87760
- derby clipping applied: False
- derby clip bounds (train 0.5/99.5 pct): (-110.61, 107.8299999999997)

## tuning meta
```json
{
  "tuning_elapsed_seconds": 5.8309,
  "model_fit_count": 185,
  "tuning_stop_reason": null,
  "n_eval_rows": 37,
  "n_unique_elo_configs": 25,
  "n_stage1": 25,
  "n_stage2": 12
}
```

## top cv rows (first 10)
```
             config_key  decay_type   A   G  tau  alpha      rmse       mae  pred_std  actual_std  n_folds  stage
      linear_A1.00_G100      linear 1.0 100   50  100.0 36.449115 28.805594 22.933702   43.617887        5 stage2
      linear_A1.00_G150      linear 1.0 150   50  100.0 36.455564 28.813138 22.856599   43.617887        5 stage2
      linear_A0.00_G100      linear 0.0 100   50  100.0 36.467407 28.809210 22.710171   43.617887        5 stage2
      linear_A1.00_G150      linear 1.0 150   50   50.0 36.629358 28.911319 22.959385   43.617887        5 stage2
      linear_A1.00_G100      linear 1.0 100   50   50.0 36.632533 28.911367 22.984216   43.617887        5 stage2
      linear_A0.00_G100      linear 0.0 100   50   50.0 36.783657 29.016944 22.314686   43.617887        5 stage2
      linear_A1.00_G150      linear 1.0 150   50   10.0 37.442356 29.425917 22.320814   43.617887        5 stage1
      linear_A1.00_G100      linear 1.0 100   50   10.0 37.466011 29.449123 22.287772   43.617887        5 stage1
exponential_A1.00_tau75 exponential 1.0 100   75   10.0 37.534640 29.494670 22.163491   43.617887        5 stage1
exponential_A1.00_tau50 exponential 1.0 100   50   10.0 37.555561 29.513659 22.052506   43.617887        5 stage1
```