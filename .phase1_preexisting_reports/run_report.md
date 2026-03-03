# AlgoSports23 Run Report

- Timestamp (UTC): 2026-03-02T07:23:29.998215+00:00
- Working directory: `.`
- Thread limits: `OMP_NUM_THREADS=`, `MKL_NUM_THREADS=`
- Runtime config: fast_mode=0, max_total=1200s, max_scan=400s, max_fits=2500, bag_n=2, optimizations=1

## Files Used / Found
- train: `Train.csv` (exists=True)
- pred: `Predictions.csv` (exists=True)
- rankings: `Rankings.xlsx` (exists=True)

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
- _scan_and_select_core_candidates: 137.848s (calls=6)
- _core_scan_score: 137.763s (calls=3480)
- _generate_inner_oof_and_outer_preds: 78.755s (calls=48)
- _predict_family: 211.260s (calls=10520)
- _build_split_tables: 4.753s (calls=96)
- _prepare_variant_outer_data: 5.270s (calls=24)
Predict family totals (sec):
- histgb: 139.886s (calls=4873)
- histgb_q50: 23.931s (calls=193)
- gbr: 20.922s (calls=193)
- histgb_bag: 17.403s (calls=193)
- huber: 7.398s (calls=193)
- ridge: 1.720s (calls=4873)
- histgb_q80: 0.000s (calls=1)
- histgb_q20: 0.000s (calls=1)
Model fit counts by family:
- gbr: 32
- histgb: 1170
- histgb_bag: 64
- histgb_q50: 32
- huber: 32
- ridge: 1170
Top 10 slowest call sites / phases:
1. run_nextgen_pipeline_total [total]: 323.794s
2. _scan_and_select_core_candidates: 93.804s
3. _scan_and_select_core_candidates: 40.928s
4. _generate_inner_oof_and_outer_preds: 10.233s
5. _generate_inner_oof_and_outer_preds: 9.977s
6. _generate_inner_oof_and_outer_preds: 9.931s
7. _generate_inner_oof_and_outer_preds: 9.764s
8. _generate_inner_oof_and_outer_preds: 9.665s
9. _generate_inner_oof_and_outer_preds: 9.661s
10. _generate_inner_oof_and_outer_preds: 9.648s
```

## Budgets / Safety Controls
- Elapsed wall time (pipeline): 328.44s
- Fit count used: 2500
- Budgets triggered: histgb_bag_auto_reduce, max_fits
- Budget events:
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before ridge; requested=1, used=2500, budget=2500
  - max_fits: fit cap reached before histgb; requested=1, used=2500, budget=2500

## Chosen Model / CV Metrics
- Selected model family: Regime-aware simplex stack (Ridge/Huber/HistGB/GBR/HistGB-bag) + q50 blend
- Selected Elo variant: elo_base_static
- Feature profile / half-life: compact_recency / None
- Selected postprocess module: expand_affine_global
- Selected ETA: 0.55
- Calibration / scale / regime stack: pooled / none / False
- Nested outer RMSE / MAE: 44.18581 / 34.81801
- Phase metrics:
  - mean_rmse=43.562967
  - corr=0.192085
  - mean_tail_dispersion=0.109066
  - mean_tail_bias=-10.568591
  - fold2_rmse=44.365581
  - cap_hit_rate=0.800000
- Regime gate change applied: tail_dispersion < 0.60 and tail_improve < 1.0 => invalid.
- ETA grid evaluated: 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
- Selected feature count: 169
- Selected feature list: conf_strength_away, conf_strength_diff, conf_strength_diff__abs, conf_strength_diff__cu, conf_strength_diff__sq, conf_strength_home, consistency_ratio_away, consistency_ratio_diff, consistency_ratio_diff__abs, consistency_ratio_diff__cu, consistency_ratio_diff__sq, consistency_ratio_home, cross_conf_flag, d_key, elo_away_pre, elo_diff_pre, elo_diff_pre__abs, elo_diff_pre__cu, elo_diff_pre__hinge_gt_-150, elo_diff_pre__hinge_gt_-45, elo_diff_pre__hinge_gt_-90, elo_diff_pre__hinge_gt_0, elo_diff_pre__hinge_gt_150, elo_diff_pre__hinge_gt_45, elo_diff_pre__hinge_gt_90, elo_diff_pre__hinge_lt_-150, elo_diff_pre__hinge_lt_-45, elo_diff_pre__hinge_lt_-90, elo_diff_pre__hinge_lt_0, elo_diff_pre__hinge_lt_150, elo_diff_pre__hinge_lt_45, elo_diff_pre__hinge_lt_90, elo_diff_pre__sq, elo_home_pre, elo_k_eff_pre, elo_neutral_diff_pre, elo_neutral_diff_pre__abs, elo_neutral_diff_pre__cu, elo_neutral_diff_pre__hinge_gt_-150, elo_neutral_diff_pre__hinge_gt_-45, elo_neutral_diff_pre__hinge_gt_-90, elo_neutral_diff_pre__hinge_gt_0, elo_neutral_diff_pre__hinge_gt_150, elo_neutral_diff_pre__hinge_gt_45, elo_neutral_diff_pre__hinge_gt_90, elo_neutral_diff_pre__hinge_lt_-150, elo_neutral_diff_pre__hinge_lt_-45, elo_neutral_diff_pre__hinge_lt_-90, elo_neutral_diff_pre__hinge_lt_0, elo_neutral_diff_pre__hinge_lt_150, elo_neutral_diff_pre__hinge_lt_45, elo_neutral_diff_pre__hinge_lt_90, elo_neutral_diff_pre__sq, elo_prob_home_pre, elok_diff, ema_margin_a20_away, ema_margin_a20_diff, ema_margin_a20_home, ema_margin_diff, ema_margin_diff__abs, ema_margin_diff__cu, ema_margin_diff__sq, ema_oppadj_a20_away, ema_oppadj_a20_diff, ema_oppadj_a20_home, ema_winrate_a20_away, ema_winrate_a20_diff, ema_winrate_a20_home, games_played_away, games_played_diff, games_played_home, games_played_min, last5_margin_away, last5_margin_diff, last5_margin_home, last5_oppadj_away, last5_oppadj_diff, last5_oppadj_home, massey_away_rating, massey_diff, massey_diff__abs, massey_diff__cu, massey_diff__hinge_gt_-20, massey_diff__hinge_gt_-40, massey_diff__hinge_gt_-80, massey_diff__hinge_gt_0, massey_diff__hinge_gt_20, massey_diff__hinge_gt_40, massey_diff__hinge_gt_80, massey_diff__hinge_lt_-20, massey_diff__hinge_lt_-40, massey_diff__hinge_lt_-80, massey_diff__hinge_lt_0, massey_diff__hinge_lt_20, massey_diff__hinge_lt_40, massey_diff__hinge_lt_80, massey_diff__sq, massey_home_rating, mean_margin_away, mean_margin_diff, mean_margin_diff__abs, mean_margin_diff__cu, mean_margin_diff__sq, mean_margin_home, offdef_defense_away, offdef_defense_home, offdef_margin_neutral, offdef_margin_neutral__abs, offdef_margin_neutral__cu, offdef_margin_neutral__sq, offdef_net_away, offdef_net_diff, offdef_net_diff__abs, offdef_net_diff__cu, offdef_net_diff__hinge_gt_-10, offdef_net_diff__hinge_gt_-30, offdef_net_diff__hinge_gt_-60, offdef_net_diff__hinge_gt_0, offdef_net_diff__hinge_gt_10, offdef_net_diff__hinge_gt_30, offdef_net_diff__hinge_gt_60, offdef_net_diff__hinge_lt_-10, offdef_net_diff__hinge_lt_-30, offdef_net_diff__hinge_lt_-60, offdef_net_diff__hinge_lt_0, offdef_net_diff__hinge_lt_10, offdef_net_diff__hinge_lt_30, offdef_net_diff__hinge_lt_60, offdef_net_diff__sq, offdef_net_home, offdef_offense_away, offdef_offense_home, oppadj_margin_away, oppadj_margin_diff, oppadj_margin_diff__abs, oppadj_margin_diff__cu, oppadj_margin_diff__sq, oppadj_margin_home, same_conf_flag, trend_margin_last5_vs_season_away, trend_margin_last5_vs_season_diff, trend_margin_last5_vs_season_diff__abs, trend_margin_last5_vs_season_diff__cu, trend_margin_last5_vs_season_diff__hinge_gt_-10, trend_margin_last5_vs_season_diff__hinge_gt_-20, trend_margin_last5_vs_season_diff__hinge_gt_-40, trend_margin_last5_vs_season_diff__hinge_gt_0, trend_margin_last5_vs_season_diff__hinge_gt_10, trend_margin_last5_vs_season_diff__hinge_gt_20, trend_margin_last5_vs_season_diff__hinge_gt_40, trend_margin_last5_vs_season_diff__hinge_lt_-10, trend_margin_last5_vs_season_diff__hinge_lt_-20, trend_margin_last5_vs_season_diff__hinge_lt_-40, trend_margin_last5_vs_season_diff__hinge_lt_0, trend_margin_last5_vs_season_diff__hinge_lt_10, trend_margin_last5_vs_season_diff__hinge_lt_20, trend_margin_last5_vs_season_diff__hinge_lt_40, trend_margin_last5_vs_season_diff__sq, trend_margin_last5_vs_season_home, trend_oppadj_last5_vs_season_away, trend_oppadj_last5_vs_season_diff, trend_oppadj_last5_vs_season_diff__abs, trend_oppadj_last5_vs_season_diff__cu, trend_oppadj_last5_vs_season_diff__sq, trend_oppadj_last5_vs_season_home, volatility_away, volatility_diff, volatility_home, volatility_sum

## Sanity Checks
- predictions.csv rows=75; Team1_WinMargin numeric+nonmissing=True
- rankings.xlsx rows=165; Rank exactly 1..165=True

## Output File Paths
- `predictions.csv` (exists=True, bytes=4188)
- `rankings.xlsx` (exists=True, bytes=8740)
- `final_report.pdf` (exists=True, bytes=218328)
- `run_report.md` (exists=True, bytes=10201)
