# AlgoSports23 Run Report

- Timestamp (UTC): 2026-03-02T06:11:33.800210+00:00
- Working directory: `C:\algosports23\AlgoSports23-Challenge`
- Thread limits: `OMP_NUM_THREADS=`, `MKL_NUM_THREADS=`
- Runtime config: fast_mode=0, max_total=1800s, max_scan=600s, max_fits=1500, bag_n=4, optimizations=1

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
- _scan_and_select_core_candidates: 121.786s (calls=6)
- _core_scan_score: 121.571s (calls=10320)
- _generate_inner_oof_and_outer_preds: 0.506s (calls=96)
- _predict_family: 102.666s (calls=54152)
- _build_split_tables: 4.645s (calls=96)
- _prepare_variant_outer_data: 5.121s (calls=24)
Predict family totals (sec):
- gbr: 37.479s (calls=13345)
- huber: 36.728s (calls=13345)
- histgb: 27.620s (calls=13345)
- ridge: 0.824s (calls=13345)
- histgb_q50: 0.009s (calls=385)
- histgb_bag: 0.008s (calls=385)
- histgb_q20: 0.000s (calls=1)
- histgb_q80: 0.000s (calls=1)
Model fit counts by family:
- gbr: 375
- histgb: 375
- huber: 375
- ridge: 375
Top 10 slowest call sites / phases:
1. run_nextgen_pipeline_total [total]: 354.178s
2. _scan_and_select_core_candidates: 104.795s
3. _scan_and_select_core_candidates: 3.418s
4. _scan_and_select_core_candidates: 3.410s
5. _scan_and_select_core_candidates: 3.403s
6. _scan_and_select_core_candidates: 3.383s
7. _scan_and_select_core_candidates: 3.378s
8. _core_scan_score: 0.500s
9. _core_scan_score: 0.500s
10. _predict_family_bundle [ridge|huber|histgb|gbr]: 0.498s
```

## Budgets / Safety Controls
- Elapsed wall time (pipeline): 354.31s
- Fit count used: 1500
- Budgets triggered: histgb_bag_auto_reduce, max_fits
- Budget events:
  - max_fits: fit cap reached before ridge; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before huber; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before histgb; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before gbr; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before ridge; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before huber; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before histgb; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before gbr; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before ridge; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before huber; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before histgb; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before gbr; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before ridge; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before huber; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before histgb; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before gbr; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before ridge; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before huber; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before histgb; requested=1, used=1500, budget=1500
  - max_fits: fit cap reached before gbr; requested=1, used=1500, budget=1500

## Chosen Model / CV Metrics
- Selected model family: Regime-aware simplex stack (Ridge/Huber/HistGB/GBR/HistGB-bag) + q50 blend
- Selected Elo variant: elo_base_static
- Feature profile / half-life: no_extra_recency / None
- Selected postprocess module: expand_regime_affine_v2_heterosk
- Selected ETA: 0.55
- Calibration / scale / regime stack: regime / regime / True
- Nested outer RMSE / MAE: 43.17846 / 34.35478
- Phase metrics:
  - mean_rmse=42.727143
  - corr=0.156726
  - mean_tail_dispersion=0.073853
  - mean_tail_bias=-5.414612
  - fold2_rmse=44.365581
  - cap_hit_rate=0.600000
- Regime gate change applied: tail_dispersion < 0.60 and tail_improve < 1.0 => invalid.
- ETA grid evaluated: 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
- Selected feature count: 122
- Selected feature list: conf_strength_away, conf_strength_diff, conf_strength_diff__abs, conf_strength_diff__cu, conf_strength_diff__sq, conf_strength_home, cross_conf_flag, d_key, elo_away_pre, elo_diff_pre, elo_diff_pre__abs, elo_diff_pre__cu, elo_diff_pre__hinge_gt_-150, elo_diff_pre__hinge_gt_-45, elo_diff_pre__hinge_gt_-90, elo_diff_pre__hinge_gt_0, elo_diff_pre__hinge_gt_150, elo_diff_pre__hinge_gt_45, elo_diff_pre__hinge_gt_90, elo_diff_pre__hinge_lt_-150, elo_diff_pre__hinge_lt_-45, elo_diff_pre__hinge_lt_-90, elo_diff_pre__hinge_lt_0, elo_diff_pre__hinge_lt_150, elo_diff_pre__hinge_lt_45, elo_diff_pre__hinge_lt_90, elo_diff_pre__sq, elo_home_pre, elo_k_eff_pre, elo_neutral_diff_pre, elo_neutral_diff_pre__abs, elo_neutral_diff_pre__cu, elo_neutral_diff_pre__hinge_gt_-150, elo_neutral_diff_pre__hinge_gt_-45, elo_neutral_diff_pre__hinge_gt_-90, elo_neutral_diff_pre__hinge_gt_0, elo_neutral_diff_pre__hinge_gt_150, elo_neutral_diff_pre__hinge_gt_45, elo_neutral_diff_pre__hinge_gt_90, elo_neutral_diff_pre__hinge_lt_-150, elo_neutral_diff_pre__hinge_lt_-45, elo_neutral_diff_pre__hinge_lt_-90, elo_neutral_diff_pre__hinge_lt_0, elo_neutral_diff_pre__hinge_lt_150, elo_neutral_diff_pre__hinge_lt_45, elo_neutral_diff_pre__hinge_lt_90, elo_neutral_diff_pre__sq, elo_prob_home_pre, elok_diff, ema_margin_diff, ema_margin_diff__abs, ema_margin_diff__cu, ema_margin_diff__sq, games_played_away, games_played_diff, games_played_home, games_played_min, massey_away_rating, massey_diff, massey_diff__abs, massey_diff__cu, massey_diff__hinge_gt_-20, massey_diff__hinge_gt_-40, massey_diff__hinge_gt_-80, massey_diff__hinge_gt_0, massey_diff__hinge_gt_20, massey_diff__hinge_gt_40, massey_diff__hinge_gt_80, massey_diff__hinge_lt_-20, massey_diff__hinge_lt_-40, massey_diff__hinge_lt_-80, massey_diff__hinge_lt_0, massey_diff__hinge_lt_20, massey_diff__hinge_lt_40, massey_diff__hinge_lt_80, massey_diff__sq, massey_home_rating, mean_margin_away, mean_margin_diff, mean_margin_diff__abs, mean_margin_diff__cu, mean_margin_diff__sq, mean_margin_home, offdef_defense_away, offdef_defense_home, offdef_margin_neutral, offdef_margin_neutral__abs, offdef_margin_neutral__cu, offdef_margin_neutral__sq, offdef_net_away, offdef_net_diff, offdef_net_diff__abs, offdef_net_diff__cu, offdef_net_diff__hinge_gt_-10, offdef_net_diff__hinge_gt_-30, offdef_net_diff__hinge_gt_-60, offdef_net_diff__hinge_gt_0, offdef_net_diff__hinge_gt_10, offdef_net_diff__hinge_gt_30, offdef_net_diff__hinge_gt_60, offdef_net_diff__hinge_lt_-10, offdef_net_diff__hinge_lt_-30, offdef_net_diff__hinge_lt_-60, offdef_net_diff__hinge_lt_0, offdef_net_diff__hinge_lt_10, offdef_net_diff__hinge_lt_30, offdef_net_diff__hinge_lt_60, offdef_net_diff__sq, offdef_net_home, offdef_offense_away, offdef_offense_home, oppadj_margin_away, oppadj_margin_diff, oppadj_margin_diff__abs, oppadj_margin_diff__cu, oppadj_margin_diff__sq, oppadj_margin_home, same_conf_flag, volatility_away, volatility_diff, volatility_home, volatility_sum

## Sanity Checks
- predictions.csv rows=75; Team1_WinMargin numeric+nonmissing=True
- rankings.xlsx rows=165; Rank exactly 1..165=True

## Output File Paths
- `C:\algosports23\AlgoSports23-Challenge\predictions.csv` (exists=True, bytes=4263)
- `C:\algosports23\AlgoSports23-Challenge\rankings.xlsx` (exists=True, bytes=8740)
- `C:\algosports23\AlgoSports23-Challenge\final_report.pdf` (exists=True, bytes=222914)
- `C:\algosports23\AlgoSports23-Challenge\run_report.md` (exists=True, bytes=10425)
