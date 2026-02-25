# AlgoSports23 Run Report

- Timestamp (UTC): 2026-02-25T02:38:35.319904+00:00
- Working directory: `C:\algosports23\AlgoSports23-Challenge`
- Thread limits: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- Runtime config: fast_mode=0, max_total=600s, max_scan=120s, max_fits=400, bag_n=2, optimizations=1

## Files Used / Found
- train: `C:\algosports23\AlgoSports23-Challenge\Train.csv` (exists=True)
- pred: `C:\algosports23\AlgoSports23-Challenge\Predictions.csv` (exists=True)
- rankings: `C:\algosports23\AlgoSports23-Challenge\Rankings.xlsx` (exists=True)

## Fast Profile Findings
- Fast profile timing summary loaded from `timing_summary_fast.json`.
- Baseline fast profile timing summary loaded from `timing_summary_fast_baseline.json`.

### Fast Profile Before vs After (selected phases / fit counts)

```text
     kind                                name  before_sec  after_sec  delta_sec
    phase    _scan_and_select_core_candidates    5.230000   5.523647   0.293646
    phase                    _core_scan_score    5.226144   5.516053   0.289908
    phase _generate_inner_oof_and_outer_preds    5.539930   3.973330  -1.566600
    phase                     _predict_family   10.500256   9.416226  -1.084030
    phase                 _build_split_tables    0.446848   0.333938  -0.112910
    phase         _prepare_variant_outer_data    0.549957   0.440423  -0.109535
fit_count                              histgb   48.000000  51.000000   3.000000
fit_count                          histgb_bag    8.000000   6.000000  -2.000000
fit_count                          histgb_q50    8.000000   6.000000  -2.000000
fit_count                               huber    8.000000   6.000000  -2.000000
fit_count                               ridge   48.000000  51.000000   3.000000
```

### Optimized Fast Timing (focus phases)

```text
Runtime timing summary
Hotspot totals (sec):
- _scan_and_select_core_candidates: 5.524s (calls=2)
- _core_scan_score: 5.516s (calls=54)
- _generate_inner_oof_and_outer_preds: 3.973s (calls=6)
- _predict_family: 9.416s (calls=173)
- _build_split_tables: 0.334s (calls=16)
- _prepare_variant_outer_data: 0.440s (calls=8)
Predict family totals (sec):
- histgb: 6.102s (calls=67)
- huber: 1.509s (calls=13)
- histgb_q50: 0.963s (calls=13)
- histgb_bag: 0.720s (calls=13)
- ridge: 0.122s (calls=67)
Model fit counts by family:
- histgb: 51
- histgb_bag: 6
- histgb_q50: 6
- huber: 6
- ridge: 51
Top 10 slowest call sites / phases:
1. run_nextgen_pipeline_total [total]: 16.902s
2. _scan_and_select_core_candidates: 2.904s
3. _scan_and_select_core_candidates: 2.620s
4. _generate_inner_oof_and_outer_preds: 1.473s
5. _generate_inner_oof_and_outer_preds: 1.428s
6. _generate_inner_oof_and_outer_preds: 1.061s
7. _predict_family_bundle [ridge|huber|histgb|histgb_bag|histgb_q50]: 0.882s
8. _predict_family_bundle [ridge|huber|histgb|histgb_bag|histgb_q50]: 0.877s
9. _predict_family_bundle [ridge|huber|histgb|histgb_bag|histgb_q50]: 0.608s
10. _predict_family_bundle [ridge|huber|histgb|histgb_bag|histgb_q50]: 0.594s
```

## Final Run Timing Summary

```text
Runtime timing summary
Hotspot totals (sec):
- _scan_and_select_core_candidates: 29.713s (calls=5)
- _core_scan_score: 29.679s (calls=1205)
- _generate_inner_oof_and_outer_preds: 0.081s (calls=15)
- _predict_family: 28.813s (calls=3217)
- _build_split_tables: 1.597s (calls=80)
- _prepare_variant_outer_data: 1.981s (calls=20)
Predict family totals (sec):
- histgb: 28.543s (calls=1516)
- ridge: 0.266s (calls=1516)
- huber: 0.002s (calls=61)
- histgb_q50: 0.001s (calls=61)
- histgb_bag: 0.001s (calls=61)
- histgb_q20: 0.000s (calls=1)
- histgb_q80: 0.000s (calls=1)
Model fit counts by family:
- histgb: 200
- ridge: 200
Top 10 slowest call sites / phases:
1. run_nextgen_pipeline_total [total]: 48.166s
2. _scan_and_select_core_candidates: 28.955s
3. _core_scan_score: 0.371s
4. _predict_family_bundle [ridge|histgb]: 0.371s
5. _predict_family [histgb]: 0.369s
6. _core_scan_score: 0.365s
7. _predict_family_bundle [ridge|histgb]: 0.364s
8. _predict_family [histgb]: 0.363s
9. _core_scan_score: 0.350s
10. _predict_family_bundle [ridge|histgb]: 0.349s
```

## Budgets / Safety Controls
- Elapsed wall time (pipeline): 48.17s
- Fit count used: 400
- Budgets triggered: histgb_bag_auto_reduce, max_fits
- Budget events:
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400
  - max_fits: fit cap reached before ridge; requested=1, used=400, budget=400
  - max_fits: fit cap reached before histgb; requested=1, used=400, budget=400

## Chosen Model / CV Metrics
- Selected model family: Regime-aware simplex stack (Ridge/Huber/HistGB/HistGB-bag) + q50 blend
- Selected Elo variant: elo_base_static
- Feature profile / half-life: no_extra_recency / None
- Calibration / scale / regime stack: pooled / pooled / False
- Nested outer RMSE / MAE: 43.55695 / 34.52281

## Sanity Checks
- predictions.csv rows=75; Team1_WinMargin numeric+nonmissing=True
- rankings.xlsx rows=165; Rank exactly 1..165=True

## Output File Paths
- `C:\algosports23\AlgoSports23-Challenge\predictions.csv` (exists=True, bytes=4188)
- `C:\algosports23\AlgoSports23-Challenge\rankings.xlsx` (exists=True, bytes=8741)
- `C:\algosports23\AlgoSports23-Challenge\final_report.pdf` (exists=True, bytes=201538)
- `C:\algosports23\AlgoSports23-Challenge\run_report.md` (exists=True, bytes=7000)
