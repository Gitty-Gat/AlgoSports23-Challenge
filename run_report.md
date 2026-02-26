# run_report

- date: 2026-02-26
- git commit hash: `019eccd`
- git branch: `kdecay_elo_simple`

## budgets used + stop reason
- ALGOSPORTS_MAX_TOTAL_SECONDS: 420.0
- ALGOSPORTS_MAX_TUNING_SECONDS: 180.0
- ALGOSPORTS_MAX_MODEL_FITS: 250
- total_elapsed_seconds: 12.045
- tuning_elapsed_seconds: 12.045
- model_fit_count: 250
- stop_reason: MAX_MODEL_FITS
- stop_events: ["MAX_MODEL_FITS @ total=2.29s tuning=2.29s fits=250"]

## chosen system/weights/hyperparams
```json
{
  "strategy_type": "single",
  "force_scale_policy": true,
  "final_combo_choice": {
    "label": "single::p_elok::scale1",
    "strategy_type": "single",
    "scale_on": true,
    "family_col": "p_elok",
    "alpha": null,
    "rmse": 36.26705607310889,
    "mae": 28.833515960302954,
    "bias": -0.12243001702471933,
    "pred_std": 15.403647301977209,
    "actual_std": 42.599845454527504,
    "dispersion_ratio": 0.3615892766188454,
    "n": 417,
    "n_meta_splits": 2
  },
  "selected_family_labels": {
    "elo": "elo_base",
    "elok": "elok_2_linear_A0.5_G100_tau50"
  },
  "selected_family_params": {
    "elo": {
      "label": "elo_base",
      "params": {
        "elo_cfg": {
          "home_adv": 50.0,
          "k_factor": 24.0,
          "initial_rating": 1500.0,
          "use_mov": true,
          "decay_type": "linear",
          "A": 0.0,
          "G": 100,
          "tau": 50,
          "season_gap_days": 60
        }
      }
    },
    "elok": {
      "label": "elok_2_linear_A0.5_G100_tau50",
      "params": {
        "elo_cfg": {
          "home_adv": 50.0,
          "k_factor": 24.0,
          "initial_rating": 1500.0,
          "use_mov": true,
          "decay_type": "linear",
          "A": 0.5,
          "G": 100,
          "tau": 50.0,
          "season_gap_days": 60
        }
      }
    }
  },
  "HOME_ADV": 50.0,
  "ELO_K": 24.0
}
```

## outer CV RMSE/MAE and dispersion metrics
- selected_oof_rmse: 36.71211866491186
- selected_oof_mae: 28.88956227703776
- selected_oof_bias: 1.1903291676333025
- selected_oof_pred_std: 26.154379933141648
- selected_oof_actual_std: 43.61788687802306
- selected_oof_dispersion_ratio: 0.5996251034875231

## output validations
- predictions.csv rows: 75
- predictions.csv missing Team1_WinMargin: 0
- predictions.csv numeric Team1_WinMargin: True
- rankings.xlsx rows: 165
- rankings.xlsx missing Rank: 0
- rankings.xlsx rank_set_valid: True
- final_report.pdf exists: True
- final_report.pdf size_bytes: 111011

## derby prediction distribution summary
```json
{
  "mean": -3.383975180871826,
  "std": 18.054486675821867,
  "min": -43.39010623826734,
  "q05": -36.74341323794989,
  "q25": -13.195042201193198,
  "median": -2.627404137788435,
  "q75": 6.92492647738138,
  "q95": 28.007210428891362,
  "max": 31.865579125533603
}
```

## single-system outer summary
```
    name  folds  rmse_mean  rmse_std  mae_mean  mae_std  bias_mean  pred_std_mean  actual_std_mean  dispersion_ratio_mean
p_massey      5  35.996530  3.592353 28.687755 2.552562   1.021449      30.037855         43.11186               0.700806
p_offdef      5  36.281863  3.831768 28.905221 2.637193  -0.228265      29.125983         43.11186               0.679711
  p_elok      5  37.254336  4.209471 29.789632 3.245876  -2.016206      26.645679         43.11186               0.621498
   p_elo      5  37.659025  4.521819 30.002095 3.544922  -2.056920      28.790207         43.11186               0.670618
p_colley      5  38.945072  4.372118 30.924564 3.535453   1.063153      30.007502         43.11186               0.700370
```

## combination outer summary
```
          name  folds  rmse_mean  rmse_std  mae_mean  mae_std  bias_mean  pred_std_mean  actual_std_mean  dispersion_ratio_mean
        single      5  35.833862  3.916211 28.349102 2.827912   0.970604      25.280271         43.11186               0.587662
static_simplex      5  36.117331  4.238842 28.611895 3.076029   0.081510      24.740766         43.11186               0.575701
dynamic_regime      5  36.205700  4.410976 28.699646 3.282131   0.180903      24.829127         43.11186               0.577569
 ridge_stacker      5  36.790019  4.515260 29.240292 3.358388  -4.424056      22.093857         43.11186               0.518120
```

## per-outer-fold selected metrics
```
 fold      rmse       mae      bias  pred_std  actual_std  dispersion_ratio   n chosen_strategy_type             chosen_label
    1 38.510049 29.894341 -0.114590 26.087456   45.098415          0.578456 193               single single::p_massey::scale0
    2 39.556542 31.068069  3.434843 32.051200   44.240814          0.724471 205               single single::p_massey::scale0
    3 32.954796 27.009399  1.118122 22.281260   38.591190          0.577367 151               single single::p_massey::scale1
    4 30.479314 24.046373 -0.826661 23.710557   40.496026          0.585503 102               single single::p_massey::scale1
    5 37.668610 29.727329  1.241306 22.270881   47.132857          0.472513 164               single single::p_massey::scale1
```

## final tuning combo candidates (top 15)
```
                 label strategy_type  scale_on family_col alpha      rmse       mae      bias  pred_std  actual_std  dispersion_ratio   n  n_meta_splits
single::p_elok::scale1        single      True     p_elok  None 36.267056 28.833516 -0.122430 15.403647   42.599845          0.361589 417              2
 single::p_elo::scale1        single      True      p_elo  None 36.424029 28.990399 -0.029814 14.826750   42.599845          0.348047 417              2
```

## final tuning family candidates (top 20)
```
family                              label      rmse       mae      bias  pred_std  actual_std  dispersion_ratio   n  inner_folds_used
  elok      elok_2_linear_A0.5_G100_tau50 39.668146 31.638670 -1.723394 30.092977   43.082448          0.698497 746                 3
  elok      elok_1_linear_A0.25_G50_tau50 40.093826 31.939060 -1.690632 31.184806   43.082448          0.723840 746                 3
   elo                           elo_base 40.405413 32.190579 -1.714509 31.742396   43.082448          0.736783 746                 3
  elok elok_3_exponential_A0.5_G100_tau50 44.430115 35.737238 -2.607746 32.100452   43.683765          0.734837 329                 1
```