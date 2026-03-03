# Phase 1 Governance Evidence

- Model number: `13`
- Submission rotation: `Submission.zip` -> `Submission.zip13`
- New `Submission.zip` contents: `Predictions.csv`, `Rankings.xlsx`
- Prior reports archived in `legacy_reports/` with suffix `model13`

## Full Metric Table
```json
{
  "cap_hit_rate": 0.0,
  "corr": 0.5211267916294419,
  "fold2_rmse": 38.75282138866413,
  "mean_rmse": 36.674008467820364,
  "mean_tail_bias": -8.987592668184956,
  "mean_tail_dispersion": 0.4400479943178377,
  "mean_tail_rmse": 57.40130241115113
}
```

## Fold RMSE / Tail / Corr
```text
 outer_fold      rmse  tail_rmse  tail_dispersion  tail_bias     corr
          1 40.079992  61.330096         0.483466 -13.443260 0.501040
          2 38.752821  62.441905         0.432510 -18.042264 0.490473
          3 35.416082  52.946853         0.384778  -2.148300 0.452889
          4 31.142204  46.803046         0.537296  -1.862891 0.649686
          5 37.978943  63.484611         0.362190  -9.441248 0.596664
```

## Model + Hyperparameters
- Model family: `Regime-aware simplex stack (Ridge/Huber/HistGB/GBR/XGB/HistGB-bag) + q50 blend`
```json
{
  "elo_variant": "elo_base_static",
  "feature_profile": "full_recency",
  "half_life_days": null,
  "histgb_idx": 0,
  "histgb_params": {
    "l2_regularization": 0.3,
    "learning_rate": 0.04,
    "max_depth": 3,
    "max_iter": 280,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 10
  },
  "ridge_alpha": 0.5
}
```
```json
{
  "affine_fit": {
    "a": 1.8611491716600397,
    "eta": 0.55,
    "gamma": 0.0,
    "naive_r": 1.7338981024150724,
    "r_base": 0.5767351602768022,
    "s": 1.0,
    "s_cap": 1.0,
    "s_cap_hit": 0.0,
    "s_uncapped": 1.0
  },
  "calibration_mode": "none",
  "eta": 0.55,
  "module": "expand_affine_global",
  "q50_blend": 0.2,
  "regime_stack": false,
  "scale_mode": "none",
  "winsor_q": 0.0
}
```

## Feature Profile + List
- Feature profile used: `F` (`full_recency`)
- Feature count: `201`
```text
conf_strength_away, conf_strength_diff, conf_strength_diff__abs, conf_strength_diff__cu, conf_strength_diff__sq, conf_strength_home, consistency_ratio_away, consistency_ratio_diff, consistency_ratio_diff__abs, consistency_ratio_diff__cu, consistency_ratio_diff__sq, consistency_ratio_home, cross_conf_flag, d_key, elo_away_pre, elo_diff_pre, elo_diff_pre__abs, elo_diff_pre__cu, elo_diff_pre__hinge_gt_-150, elo_diff_pre__hinge_gt_-45, elo_diff_pre__hinge_gt_-90, elo_diff_pre__hinge_gt_0, elo_diff_pre__hinge_gt_150, elo_diff_pre__hinge_gt_45, elo_diff_pre__hinge_gt_90, elo_diff_pre__hinge_lt_-150, elo_diff_pre__hinge_lt_-45, elo_diff_pre__hinge_lt_-90, elo_diff_pre__hinge_lt_0, elo_diff_pre__hinge_lt_150, elo_diff_pre__hinge_lt_45, elo_diff_pre__hinge_lt_90, elo_diff_pre__sq, elo_home_pre, elo_k_eff_pre, elo_neutral_diff_pre, elo_neutral_diff_pre__abs, elo_neutral_diff_pre__cu, elo_neutral_diff_pre__hinge_gt_-150, elo_neutral_diff_pre__hinge_gt_-45, elo_neutral_diff_pre__hinge_gt_-90, elo_neutral_diff_pre__hinge_gt_0, elo_neutral_diff_pre__hinge_gt_150, elo_neutral_diff_pre__hinge_gt_45, elo_neutral_diff_pre__hinge_gt_90, elo_neutral_diff_pre__hinge_lt_-150, elo_neutral_diff_pre__hinge_lt_-45, elo_neutral_diff_pre__hinge_lt_-90, elo_neutral_diff_pre__hinge_lt_0, elo_neutral_diff_pre__hinge_lt_150, elo_neutral_diff_pre__hinge_lt_45, elo_neutral_diff_pre__hinge_lt_90, elo_neutral_diff_pre__sq, elo_prob_home_pre, elok_diff, ema_margin_a10_away, ema_margin_a10_diff, ema_margin_a10_home, ema_margin_a20_away, ema_margin_a20_diff, ema_margin_a20_home, ema_margin_a35_away, ema_margin_a35_diff, ema_margin_a35_home, ema_margin_diff, ema_margin_diff__abs, ema_margin_diff__cu, ema_margin_diff__sq, ema_margin_diff__x__massey_diff, ema_oppadj_a10_away, ema_oppadj_a10_diff, ema_oppadj_a10_home, ema_oppadj_a20_away, ema_oppadj_a20_diff, ema_oppadj_a20_home, ema_oppadj_a35_away, ema_oppadj_a35_diff, ema_oppadj_a35_home, ema_winrate_a10_away, ema_winrate_a10_diff, ema_winrate_a10_home, ema_winrate_a20_away, ema_winrate_a20_diff, ema_winrate_a20_home, ema_winrate_a35_away, ema_winrate_a35_diff, ema_winrate_a35_home, games_played_away, games_played_diff, games_played_home, games_played_min, last3_margin_away, last3_margin_diff, last3_margin_home, last5_margin_away, last5_margin_diff, last5_margin_home, last5_oppadj_away, last5_oppadj_diff, last5_oppadj_home, last8_margin_away, last8_margin_diff, last8_margin_home, massey_away_rating, massey_diff, massey_diff__abs, massey_diff__cu, massey_diff__hinge_gt_-20, massey_diff__hinge_gt_-40, massey_diff__hinge_gt_-80, massey_diff__hinge_gt_0, massey_diff__hinge_gt_20, massey_diff__hinge_gt_40, massey_diff__hinge_gt_80, massey_diff__hinge_lt_-20, massey_diff__hinge_lt_-40, massey_diff__hinge_lt_-80, massey_diff__hinge_lt_0, massey_diff__hinge_lt_20, massey_diff__hinge_lt_40, massey_diff__hinge_lt_80, massey_diff__sq, massey_home_rating, mean_margin_away, mean_margin_diff, mean_margin_diff__abs, mean_margin_diff__cu, mean_margin_diff__sq, mean_margin_home, offdef_defense_away, offdef_defense_home, offdef_margin_neutral, offdef_margin_neutral__abs, offdef_margin_neutral__cu, offdef_margin_neutral__sq, offdef_net_away, offdef_net_diff, offdef_net_diff__abs, offdef_net_diff__cu, offdef_net_diff__hinge_gt_-10, offdef_net_diff__hinge_gt_-30, offdef_net_diff__hinge_gt_-60, offdef_net_diff__hinge_gt_0, offdef_net_diff__hinge_gt_10, offdef_net_diff__hinge_gt_30, offdef_net_diff__hinge_gt_60, offdef_net_diff__hinge_lt_-10, offdef_net_diff__hinge_lt_-30, offdef_net_diff__hinge_lt_-60, offdef_net_diff__hinge_lt_0, offdef_net_diff__hinge_lt_10, offdef_net_diff__hinge_lt_30, offdef_net_diff__hinge_lt_60, offdef_net_diff__sq, offdef_net_diff__x__ema_margin_diff, offdef_net_diff__x__oppadj_margin_diff, offdef_net_diff__x__trend_margin_last5_vs_season_diff, offdef_net_home, offdef_offense_away, offdef_offense_home, oppadj_margin_away, oppadj_margin_diff, oppadj_margin_diff__abs, oppadj_margin_diff__cu, oppadj_margin_diff__sq, oppadj_margin_diff__x__ema_margin_diff, oppadj_margin_diff__x__massey_diff, oppadj_margin_home, same_conf_flag, trend_margin_last5_vs_season_away, trend_margin_last5_vs_season_diff, trend_margin_last5_vs_season_diff__abs, trend_margin_last5_vs_season_diff__cu, trend_margin_last5_vs_season_diff__hinge_gt_-10, trend_margin_last5_vs_season_diff__hinge_gt_-20, trend_margin_last5_vs_season_diff__hinge_gt_-40, trend_margin_last5_vs_season_diff__hinge_gt_0, trend_margin_last5_vs_season_diff__hinge_gt_10, trend_margin_last5_vs_season_diff__hinge_gt_20, trend_margin_last5_vs_season_diff__hinge_gt_40, trend_margin_last5_vs_season_diff__hinge_lt_-10, trend_margin_last5_vs_season_diff__hinge_lt_-20, trend_margin_last5_vs_season_diff__hinge_lt_-40, trend_margin_last5_vs_season_diff__hinge_lt_0, trend_margin_last5_vs_season_diff__hinge_lt_10, trend_margin_last5_vs_season_diff__hinge_lt_20, trend_margin_last5_vs_season_diff__hinge_lt_40, trend_margin_last5_vs_season_diff__sq, trend_margin_last5_vs_season_diff__x__consistency_ratio_diff, trend_margin_last5_vs_season_home, trend_oppadj_last5_vs_season_away, trend_oppadj_last5_vs_season_diff, trend_oppadj_last5_vs_season_diff__abs, trend_oppadj_last5_vs_season_diff__cu, trend_oppadj_last5_vs_season_diff__sq, trend_oppadj_last5_vs_season_diff__x__consistency_ratio_diff, trend_oppadj_last5_vs_season_home, volatility_away, volatility_diff, volatility_home, volatility_sum
```

## Run Cleanliness
- budget_triggered: `[]`
- fallback_triggered: `False`
- clean_run: `True`
- fit_count: `3011`