# Run Report

- Timestamp (UTC): 2026-02-27T02:47:29.836702+00:00
- Git hash: `1aaa14e93a225970a75736c5fd1f73a703e08fc3`
- selected_config.json: `C:\algosports23\AlgoSports23-Challenge\selected_config.json`
- selected_config digest (sha256): `c9cd6d96cd2b0d5f61ce260a8899c833df412254a0e567abeb850e9350468520`
- oof_predictions.csv: `C:\algosports23\AlgoSports23-Challenge\oof_predictions.csv`
- oof_predictions.parquet: `C:\algosports23\AlgoSports23-Challenge\oof_predictions.parquet`
- runner_up_configs.json: `C:\algosports23\AlgoSports23-Challenge\runner_up_configs.json`

## Selection Lock Proof
- Rating-stage selected config id: `massey_alpha=1.0|use_elo=1|elo_home_adv=60.0|elo_k=24.0|elo_decay_a=0.25|elo_decay_g=50.0`
- Rating-stage tail lambda: `0.1`
- Rating-stage xtail lambda: `0.05`
- Rating-stage stability lambda: `0.1`
- Rating-stage dispersion lambda: `0.05`
- Rating-stage cap-hit lambda: `0.0`
- Rating-stage tail-bias lambda: `0.0`
- Rating-stage tail-dispersion lambda: `0.05`
- Model-stage selected config id: `massey_alpha=1.0|use_elo=1|elo_home_adv=60.0|elo_k=24.0|elo_decay_a=0.25|elo_decay_g=50.0||elasticnet|alpha=1.0|l1_ratio=0.12||module=expand_affine_global`
- Model-stage tail lambda: `0.1`
- Model-stage xtail lambda: `0.05`
- Model-stage stability lambda: `0.1`
- Model-stage dispersion lambda: `0.05`
- Model-stage cap-hit lambda: `0.0`
- Model-stage tail-bias lambda: `0.0`
- Model-stage tail-dispersion lambda: `0.05`
- Model-stage best-of-three applied: `True`
- Model-stage best-of-three feasible count: `3`
- Selection gate rejections count: `55`
- Massey-only gate applied: `False`
- Full-train stage performed refit-only on locked selected configuration.

## Outer Fold Table (Selected Config)

```text
 fold      rmse       mae  tail_rmse  dispersion_ratio  max_abs_pred
    1 39.227631 30.836249  61.450020          0.564425     96.034739
    2 39.477395 30.708436  59.858585          0.743125     89.784898
    3 33.511316 27.344348  47.705685          0.732711     76.043479
    4 30.674944 23.800730  46.441971          0.757707     82.409385
    5 36.832247 29.900925  57.985479          0.647885     86.326471
```

## Summary Stats
- mean_rmse: 35.944707
- std_rmse: 3.398146
- max_fold_rmse: 39.477395
- mean_disp_ratio: 0.689171
- mean_tailrmse: 54.688348

## Diagnostics
- Mean MAE: 28.518138
- Mean Extreme Tail RMSE: 62.118894
- Mean tail dispersion ratio: 0.480557
- Mean bias: -0.938902
- Mean tail bias: -9.415731
- Weighted mean RMSE: 35.265046
- Weighted mean Tail RMSE: 53.671063
- Weighted mean dispersion ratio: 0.698246
- Weighted mean tail bias: -9.004297
- Cap-hit rate: 0.250000
- Selected expansion affine r_base: 0.49057701866105907
- Selected expansion affine naive_r: 2.0384159101649693
- Selected expansion affine s_star: 1.0533373677158486
- Selected expansion affine chosen_s: 1.4268911371154784
- Selected expansion affine eta: 0.7
- Selected piecewise t: None
- Selected piecewise k: None
- Selected piecewise q: None
- Gate rejected configs: 55 / 75
- Gate reason `invalid_postprocess`: 38
- Gate reason `invalid_postprocess;cap_hit_gate`: 9
- Gate reason `cap_hit_gate`: 6
- Gate reason `invalid_postprocess;underdisp_no_blowout_expansion_gate`: 2
- Invalid postprocess reason `module_none_for_underdispersed_base`: 75
- Invalid postprocess reason `blowout_taildisp_gate`: 67
- Invalid postprocess reason `underdispersed_but_no_expansion`: 8
- Invalid postprocess reason `heterosk_nonzero_no_gain`: 1

## Validation Proofs
- predictions.csv exists: True
- predictions rows: 75
- predictions Team1_WinMargin int: True
- predictions Team1_WinMargin missing: 0
- rankings.xlsx exists: True
- rankings rows: 165
- rank permutation 1..165: True
- final_report.pdf exists: True
- final_report.pdf size_bytes: 123367
- oof_predictions.csv exists: True
- oof_predictions.parquet exists: True
- runner_up_configs.json exists: True
- oof_predictions required cols: True