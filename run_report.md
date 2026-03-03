# AlgoSports23 Run Report

- Timestamp (UTC): 2026-03-03T06:29:17.664299+00:00
- Working directory: `C:\algosports23\AlgoSports23-Challenge`
- Thread limits: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- Runtime config: fast_mode=0, max_total=3600s, max_scan=1200s, max_fits=8000, bag_n=4, optimizations=1

## Files Used / Found
- train: `C:\algosports23\AlgoSports23-Challenge\Train.csv` (exists=True)
- pred: `C:\algosports23\AlgoSports23-Challenge\Predictions.csv` (exists=True)
- rankings: `C:\algosports23\AlgoSports23-Challenge\Rankings.xlsx` (exists=True)

## Fast Profile Findings
- `timing_summary_fast.json` not found in this run session.

## Final Run Timing Summary

```text
Runtime timing summary
Hotspot totals (sec):
- _scan_and_select_core_candidates: 880.585s (calls=6)
- _core_scan_score: 880.524s (calls=1810)
- _generate_inner_oof_and_outer_preds: 740.003s (calls=42)
- _predict_family: 1624.179s (calls=7174)
- _build_split_tables: 1.159s (calls=24)
- _prepare_variant_outer_data: 1.277s (calls=6)
Predict family totals (sec):
- histgb: 921.910s (calls=3079)
- gbr: 267.273s (calls=169)
- histgb_bag: 193.709s (calls=169)
- histgb_q50: 104.436s (calls=169)
- xgb: 91.988s (calls=169)
- huber: 39.921s (calls=169)
- ridge: 3.864s (calls=3079)
- histgb_q20: 0.461s (calls=1)
- histgb_q80: 0.455s (calls=1)
- enet_f2: 0.162s (calls=169)
Model fit counts by family:
- enet_f2: 169
- gbr: 169
- histgb: 3079
- histgb_bag: 676
- histgb_q20: 1
- histgb_q50: 169
- histgb_q80: 1
- huber: 169
- ridge: 3079
- xgb: 169
Top 10 slowest call sites / phases:
1. run_nextgen_pipeline_total [total]: 1649.914s
2. _scan_and_select_core_candidates: 266.806s
3. _scan_and_select_core_candidates: 191.869s
4. _scan_and_select_core_candidates: 163.262s
5. _scan_and_select_core_candidates: 117.614s
6. _scan_and_select_core_candidates: 106.838s
7. _generate_inner_oof_and_outer_preds: 45.915s
8. _generate_inner_oof_and_outer_preds: 35.939s
9. _generate_inner_oof_and_outer_preds: 35.911s
10. _generate_inner_oof_and_outer_preds: 34.960s
```

## Budgets / Safety Controls
- Elapsed wall time (pipeline): 1649.96s
- Fit count used: 7681
- Budgets triggered: none

## Chosen Model / CV Metrics
- Selected model family: Regime-aware simplex stack (Ridge/Huber/HistGB/GBR/XGB/HistGB-bag) + q50 blend
- Selected Elo variant: elo_dynamic_k_teamha_decay
- Feature profile / half-life: signal_core / None
- Selected postprocess module: expand_affine_global
- Selected ETA: 0.55
- Calibration / scale / regime stack: none / none / False
- Nested outer RMSE / MAE: 39.04914 / 30.66425
- Phase metrics:
  - mean_rmse=37.971373
  - corr=0.498899
  - mean_tail_dispersion=0.501990
  - mean_tail_bias=-8.713067
  - fold2_rmse=39.692873
  - cap_hit_rate=0.200000
- Regime gate change applied: tail_dispersion < 0.60 and tail_improve < 1.0 => invalid.
- ETA grid evaluated: 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
- Selected feature count: 86
- Selected feature list: conf_strength_away, conf_strength_diff, conf_strength_home, consistency_ratio_away, consistency_ratio_diff, consistency_ratio_home, cross_conf_flag, elo_away_pre, elo_diff_pre, elo_home_pre, elo_k_eff_pre, elo_neutral_diff_pre, elo_prob_home_pre, elok_diff, ema_margin_a10_away, ema_margin_a10_diff, ema_margin_a10_home, ema_margin_a20_away, ema_margin_a20_diff, ema_margin_a20_home, ema_margin_a35_away, ema_margin_a35_diff, ema_margin_a35_home, ema_margin_diff, ema_oppadj_a10_away, ema_oppadj_a10_diff, ema_oppadj_a10_home, ema_oppadj_a20_away, ema_oppadj_a20_diff, ema_oppadj_a20_home, ema_oppadj_a35_away, ema_oppadj_a35_diff, ema_oppadj_a35_home, ema_winrate_a10_away, ema_winrate_a10_diff, ema_winrate_a10_home, ema_winrate_a20_away, ema_winrate_a20_diff, ema_winrate_a20_home, ema_winrate_a35_away, ema_winrate_a35_diff, ema_winrate_a35_home, games_played_away, games_played_diff, games_played_home, games_played_min, last3_margin_away, last3_margin_diff, last3_margin_home, last5_margin_away, last5_margin_diff, last5_margin_home, last5_oppadj_away, last5_oppadj_diff, last5_oppadj_home, last8_margin_away, last8_margin_diff, last8_margin_home, massey_away_rating, massey_diff, massey_home_rating, mean_margin_away, mean_margin_diff, mean_margin_home, offdef_defense_away, offdef_defense_home, offdef_margin_neutral, offdef_net_away, offdef_net_diff, offdef_net_home, offdef_offense_away, offdef_offense_home, oppadj_margin_away, oppadj_margin_diff, oppadj_margin_home, same_conf_flag, trend_margin_last5_vs_season_away, trend_margin_last5_vs_season_diff, trend_margin_last5_vs_season_home, trend_oppadj_last5_vs_season_away, trend_oppadj_last5_vs_season_diff, trend_oppadj_last5_vs_season_home, volatility_away, volatility_diff, volatility_home, volatility_sum

## Sanity Checks
- predictions.csv rows=75; Team1_WinMargin numeric+nonmissing=True
- rankings.xlsx rows=165; Rank exactly 1..165=True

## Output File Paths
- `C:\algosports23\AlgoSports23-Challenge\predictions.csv` (exists=True, bytes=4419)
- `C:\algosports23\AlgoSports23-Challenge\rankings.xlsx` (exists=True, bytes=8747)
- `C:\algosports23\AlgoSports23-Challenge\final_report.pdf` (exists=True, bytes=227838)
- `C:\algosports23\AlgoSports23-Challenge\run_report.md` (exists=True, bytes=4310)
