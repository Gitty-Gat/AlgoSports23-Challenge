from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


SEED = 23


def _rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def make_estimator(model_name: str, seed: int = SEED):
    model_name = model_name.lower()
    if model_name == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=6.0, random_state=seed)),
            ]
        )
    if model_name == "elasticnet":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.08, l1_ratio=0.25, max_iter=20000, random_state=seed)),
            ]
        )
    if model_name == "huber":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(alpha=0.0005, epsilon=1.35, max_iter=300)),
            ]
        )
    if model_name == "histgb":
        return HistGradientBoostingRegressor(
            random_state=seed,
            learning_rate=0.05,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            max_iter=300,
            l2_regularization=0.5,
            loss="squared_error",
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(err))
    sign_acc = float(np.mean((y_pred > 0) == (y_true > 0)))
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "sign_acc": sign_acc,
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "pred_abs_p99": float(np.quantile(np.abs(y_pred), 0.99)),
    }


def extract_model_complexity(estimator, n_features: int) -> Tuple[int, Optional[int]]:
    nonzero = None
    model_obj = estimator
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        model_obj = estimator.named_steps["model"]
    if hasattr(model_obj, "coef_"):
        coef = np.asarray(model_obj.coef_).ravel()
        nonzero = int(np.sum(np.abs(coef) > 1e-8))
    return int(n_features), nonzero


def build_feature_set_definitions(all_columns: Sequence[str]) -> Dict[str, List[str]]:
    cols = list(all_columns)
    conf_dummy_cols = [c for c in cols if c.startswith("conf_home_") or c.startswith("conf_away_")]
    ratings_cols = [c for c in ["elo_diff_pre", "elo_prob_home_pre", "massey_diff", "conf_strength_diff"] if c in cols]
    offdef_cols = [c for c in ["offdef_margin_neutral", "offdef_margin_with_side", "offdef_net_diff"] if c in cols]
    schedule_cols = [
        c
        for c in [
            "sos_elo_diff",
            "oppadj_margin_diff",
            "games_played_diff",
            "winrate_diff",
            "mean_margin_diff",
            "ema_margin_diff",
        ]
        if c in cols
    ]
    rest_cols = [c for c in ["rest_days_diff", "volatility_diff"] if c in cols]
    efficiency_only = [c for c in schedule_cols + rest_cols if c in cols] + [c for c in ["conf_strength_diff"] if c in cols] + conf_dummy_cols
    full_engineered = [
        c
        for c in cols
        if c
        not in {
            "GameID",
            "Date",
            "HomeConf",
            "HomeID",
            "HomeTeam",
            "HomePts",
            "AwayConf",
            "AwayID",
            "AwayTeam",
            "AwayPts",
            "HomeWinMargin",
            "Team1_Conf",
            "Team1_ID",
            "Team1",
            "Team2_Conf",
            "Team2_ID",
            "Team2",
            "Team1_WinMargin",
        }
        and (pd.api.types.is_numeric_dtype(pd.Series(dtype=np.float64)) or True)
    ]
    # `full_engineered` from columns list; restrict to engineered numeric columns known by prefixes and names.
    full_engineered = [
        c
        for c in cols
        if c in set(ratings_cols + offdef_cols + schedule_cols + rest_cols)
        or c.startswith("elo_")
        or c.startswith("mean_margin_")
        or c.startswith("ema_margin_")
        or c.startswith("oppadj_margin_")
        or c.startswith("sos_elo_")
        or c.startswith("winrate_")
        or c.startswith("volatility_")
        or c.startswith("games_played_")
        or c.startswith("rest_days_")
        or c.startswith("massey_")
        or c.startswith("offdef_")
        or c.startswith("conf_strength_")
        or c.startswith("conf_home_")
        or c.startswith("conf_away_")
    ]

    defs = {
        "ratings_only": sorted(dict.fromkeys(ratings_cols + [c for c in ["elo_home_pre", "elo_away_pre"] if c in cols])),
        "ratings_plus_schedule": sorted(dict.fromkeys(ratings_cols + schedule_cols)),
        "ratings_plus_offdef": sorted(dict.fromkeys(ratings_cols + offdef_cols)),
        "ratings_offdef_sched_rest": sorted(dict.fromkeys(ratings_cols + offdef_cols + schedule_cols + rest_cols)),
        "efficiency_only": sorted(dict.fromkeys(efficiency_only)),
        "full": sorted(dict.fromkeys(full_engineered)),
    }
    return {k: v for k, v in defs.items() if len(v) > 0}


def _safe_xy(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str = "HomeWinMargin") -> Tuple[pd.DataFrame, np.ndarray]:
    X = df.reindex(columns=list(feature_cols), fill_value=0.0).astype(float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


@dataclass
class SweepOutputs:
    config_summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    config_registry: Dict[str, dict]
    selection: dict


def evaluate_model_sweep(
    fold_datasets: Sequence[Mapping[str, object]],
    feature_sets: Mapping[str, Sequence[str]],
    model_names: Sequence[str],
    seed: int = SEED,
    include_ensemble: bool = True,
) -> SweepOutputs:
    fold_metric_records: List[dict] = []
    oof_records: List[dict] = []
    config_registry: Dict[str, dict] = {}
    config_fold_preds: Dict[Tuple[str, int], pd.DataFrame] = {}

    for feature_set_name, feature_cols in feature_sets.items():
        for model_name in model_names:
            config_name = f"{model_name}__{feature_set_name}"
            config_registry[config_name] = {
                "config_name": config_name,
                "model_name": model_name,
                "feature_set": feature_set_name,
                "feature_cols": list(feature_cols),
                "is_ensemble": False,
            }

            for fold_data in fold_datasets:
                fold_id = int(fold_data["fold"])
                train_df = fold_data["train"]
                val_df = fold_data["val"]
                train_plus_df = fold_data.get("train_plus")
                val_plus_df = fold_data.get("val_plus")
                train_minus_df = fold_data.get("train_minus")
                val_minus_df = fold_data.get("val_minus")

                X_train, y_train = _safe_xy(train_df, feature_cols)
                X_val, y_val = _safe_xy(val_df, feature_cols)

                t0 = time.perf_counter()
                est = make_estimator(model_name, seed=seed)
                est.fit(X_train, y_train)
                train_time = time.perf_counter() - t0

                t1 = time.perf_counter()
                y_hat = np.asarray(est.predict(X_val), dtype=float)
                pred_time = time.perf_counter() - t1

                m = compute_regression_metrics(y_val, y_hat)
                n_features, nonzero = extract_model_complexity(est, n_features=X_train.shape[1])

                sens_mean_abs = np.nan
                sens_max_abs = np.nan
                if all(x is not None for x in [train_plus_df, val_plus_df, train_minus_df, val_minus_df]):
                    X_train_p, y_train_p = _safe_xy(train_plus_df, feature_cols)
                    X_val_p, _ = _safe_xy(val_plus_df, feature_cols)
                    X_train_m, y_train_m = _safe_xy(train_minus_df, feature_cols)
                    X_val_m, _ = _safe_xy(val_minus_df, feature_cols)
                    est_p = make_estimator(model_name, seed=seed)
                    est_m = make_estimator(model_name, seed=seed)
                    est_p.fit(X_train_p, y_train_p)
                    est_m.fit(X_train_m, y_train_m)
                    y_hat_p = np.asarray(est_p.predict(X_val_p), dtype=float)
                    y_hat_m = np.asarray(est_m.predict(X_val_m), dtype=float)
                    sens = y_hat_p - y_hat_m
                    sens_mean_abs = float(np.mean(np.abs(sens)))
                    sens_max_abs = float(np.max(np.abs(sens)))

                feature_bytes = int(X_train.memory_usage(deep=True).sum() + X_val.memory_usage(deep=True).sum())
                rss = _rss_bytes()

                fold_metric_records.append(
                    {
                        "config_name": config_name,
                        "feature_set": feature_set_name,
                        "model_name": model_name,
                        "fold": fold_id,
                        "rmse": m["rmse"],
                        "mae": m["mae"],
                        "bias": m["bias"],
                        "sign_acc": m["sign_acc"],
                        "pred_abs_p99": m["pred_abs_p99"],
                        "feature_build_time_sec": float(fold_data.get("feature_build_time_sec", 0.0)),
                        "train_time_sec": float(train_time),
                        "predict_time_sec": float(pred_time),
                        "feature_memory_bytes": feature_bytes,
                        "rss_bytes": rss,
                        "n_features": n_features,
                        "nonzero_coef": nonzero,
                        "sens_home_adv_pm10_mean_abs": sens_mean_abs,
                        "sens_home_adv_pm10_max_abs": sens_max_abs,
                    }
                )

                fold_pred_df = pd.DataFrame(
                    {
                        "config_name": config_name,
                        "fold": fold_id,
                        "row_index": val_df.index.to_numpy(),
                        "y_true": y_val,
                        "y_pred": y_hat,
                    }
                )
                config_fold_preds[(config_name, fold_id)] = fold_pred_df.copy()
                oof_records.extend(fold_pred_df.to_dict(orient="records"))

    fold_metrics = pd.DataFrame(fold_metric_records)
    if fold_metrics.empty:
        raise RuntimeError("No fold metrics generated.")

    config_summary = (
        fold_metrics.groupby(["config_name", "feature_set", "model_name"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            bias_mean=("bias", "mean"),
            sign_acc_mean=("sign_acc", "mean"),
            pred_abs_p99_mean=("pred_abs_p99", "mean"),
            feature_build_time_sec=("feature_build_time_sec", "sum"),
            train_time_sec=("train_time_sec", "sum"),
            predict_time_sec=("predict_time_sec", "sum"),
            feature_memory_bytes_peak=("feature_memory_bytes", "max"),
            rss_bytes_peak=("rss_bytes", "max"),
            n_features=("n_features", "max"),
            nonzero_coef_mean=("nonzero_coef", "mean"),
            sens_home_adv_pm10_mean_abs=("sens_home_adv_pm10_mean_abs", "mean"),
            sens_home_adv_pm10_max_abs=("sens_home_adv_pm10_max_abs", "mean"),
            n_folds=("fold", "count"),
        )
        .sort_values(["rmse_mean", "mae_mean", "rmse_std"], kind="mergesort")
        .reset_index(drop=True)
    )

    oof_predictions = pd.DataFrame(oof_records)

    if include_ensemble and len(config_summary) >= 2:
        top2 = config_summary.nsmallest(2, "rmse_mean")["config_name"].tolist()
        ens_name = f"ensemble_mean__{top2[0]}__{top2[1]}"
        config_registry[ens_name] = {
            "config_name": ens_name,
            "model_name": "ensemble_mean",
            "feature_set": "derived",
            "feature_cols": None,
            "is_ensemble": True,
            "members": top2,
        }
        ens_fold_rows = []
        ens_oof_rows = []
        all_fold_ids = sorted(fold_metrics["fold"].unique().tolist())
        for fold_id in all_fold_ids:
            p1 = config_fold_preds[(top2[0], fold_id)].sort_values("row_index").reset_index(drop=True)
            p2 = config_fold_preds[(top2[1], fold_id)].sort_values("row_index").reset_index(drop=True)
            if not np.array_equal(p1["row_index"].to_numpy(), p2["row_index"].to_numpy()):
                raise RuntimeError("Ensemble member row alignment mismatch.")
            y_true = p1["y_true"].to_numpy(dtype=float)
            y_pred = 0.5 * (p1["y_pred"].to_numpy(dtype=float) + p2["y_pred"].to_numpy(dtype=float))
            m = compute_regression_metrics(y_true, y_pred)
            fm1 = fold_metrics[(fold_metrics["config_name"] == top2[0]) & (fold_metrics["fold"] == fold_id)].iloc[0]
            fm2 = fold_metrics[(fold_metrics["config_name"] == top2[1]) & (fold_metrics["fold"] == fold_id)].iloc[0]
            ens_fold_rows.append(
                {
                    "config_name": ens_name,
                    "feature_set": "ensemble",
                    "model_name": "ensemble_mean",
                    "fold": int(fold_id),
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "bias": m["bias"],
                    "sign_acc": m["sign_acc"],
                    "pred_abs_p99": m["pred_abs_p99"],
                    "feature_build_time_sec": float(max(fm1["feature_build_time_sec"], fm2["feature_build_time_sec"])),
                    "train_time_sec": float(fm1["train_time_sec"] + fm2["train_time_sec"]),
                    "predict_time_sec": float(fm1["predict_time_sec"] + fm2["predict_time_sec"]),
                    "feature_memory_bytes": float(max(fm1["feature_memory_bytes"], fm2["feature_memory_bytes"])),
                    "rss_bytes": float(max(fm1["rss_bytes"], fm2["rss_bytes"])) if pd.notna(fm1["rss_bytes"]) and pd.notna(fm2["rss_bytes"]) else np.nan,
                    "n_features": float(fm1["n_features"] + fm2["n_features"]),
                    "nonzero_coef": np.nan,
                    "sens_home_adv_pm10_mean_abs": float(np.nanmean([fm1["sens_home_adv_pm10_mean_abs"], fm2["sens_home_adv_pm10_mean_abs"]])),
                    "sens_home_adv_pm10_max_abs": float(np.nanmean([fm1["sens_home_adv_pm10_max_abs"], fm2["sens_home_adv_pm10_max_abs"]])),
                }
            )
            ens_oof_df = pd.DataFrame(
                {
                    "config_name": ens_name,
                    "fold": int(fold_id),
                    "row_index": p1["row_index"].to_numpy(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
            ens_oof_rows.extend(ens_oof_df.to_dict(orient="records"))

        fold_metrics = pd.concat([fold_metrics, pd.DataFrame(ens_fold_rows)], ignore_index=True)
        oof_predictions = pd.concat([oof_predictions, pd.DataFrame(ens_oof_rows)], ignore_index=True)
        ens_summary = (
            pd.DataFrame(ens_fold_rows)
            .groupby(["config_name", "feature_set", "model_name"], as_index=False)
            .agg(
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                bias_mean=("bias", "mean"),
                sign_acc_mean=("sign_acc", "mean"),
                pred_abs_p99_mean=("pred_abs_p99", "mean"),
                feature_build_time_sec=("feature_build_time_sec", "sum"),
                train_time_sec=("train_time_sec", "sum"),
                predict_time_sec=("predict_time_sec", "sum"),
                feature_memory_bytes_peak=("feature_memory_bytes", "max"),
                rss_bytes_peak=("rss_bytes", "max"),
                n_features=("n_features", "max"),
                nonzero_coef_mean=("nonzero_coef", "mean"),
                sens_home_adv_pm10_mean_abs=("sens_home_adv_pm10_mean_abs", "mean"),
                sens_home_adv_pm10_max_abs=("sens_home_adv_pm10_max_abs", "mean"),
                n_folds=("fold", "count"),
            )
        )
        config_summary = (
            pd.concat([config_summary, ens_summary], ignore_index=True)
            .sort_values(["rmse_mean", "mae_mean", "rmse_std"], kind="mergesort")
            .reset_index(drop=True)
        )

    config_summary["selection_score"] = (
        config_summary["rmse_mean"]
        + 0.20 * config_summary["rmse_std"].fillna(0.0)
        + 0.10 * config_summary["mae_std"].fillna(0.0)
        + 0.08 * config_summary["pred_abs_p99_mean"].fillna(0.0)
        + 0.10 * config_summary["bias_mean"].abs().fillna(0.0)
    )
    config_summary = config_summary.sort_values(["selection_score", "rmse_mean", "mae_mean"], kind="mergesort").reset_index(drop=True)

    chosen_row = config_summary.iloc[0].to_dict()
    selection = {
        "chosen_config_name": chosen_row["config_name"],
        "selection_score": float(chosen_row["selection_score"]),
        "rmse_mean": float(chosen_row["rmse_mean"]),
        "rmse_std": float(chosen_row["rmse_std"]) if pd.notna(chosen_row["rmse_std"]) else np.nan,
        "mae_mean": float(chosen_row["mae_mean"]),
        "mae_std": float(chosen_row["mae_std"]) if pd.notna(chosen_row["mae_std"]) else np.nan,
        "rationale": "Minimize RMSE primarily, with penalties for fold instability, extreme predictions, and bias.",
    }

    return SweepOutputs(
        config_summary=config_summary,
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
        config_registry=config_registry,
        selection=selection,
    )


def fit_single_config_and_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    config: Mapping[str, object],
    seed: int = SEED,
) -> Tuple[np.ndarray, dict]:
    if config.get("is_ensemble", False):
        raise ValueError("Use fit_ensemble_and_predict for ensemble configs.")
    feature_cols = list(config["feature_cols"])
    X_train, y_train = _safe_xy(train_df, feature_cols)
    X_pred = pred_df.reindex(columns=feature_cols, fill_value=0.0).astype(float)
    est = make_estimator(str(config["model_name"]), seed=seed)
    est.fit(X_train, y_train)
    pred = np.asarray(est.predict(X_pred), dtype=float)
    n_features, nonzero = extract_model_complexity(est, n_features=X_train.shape[1])
    meta = {"estimator": est, "n_features": n_features, "nonzero_coef": nonzero, "feature_cols": feature_cols}
    return pred, meta


def fit_config_and_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    chosen_config_name: str,
    config_registry: Mapping[str, Mapping[str, object]],
    seed: int = SEED,
) -> Tuple[np.ndarray, dict]:
    config = config_registry[chosen_config_name]
    if not config.get("is_ensemble", False):
        return fit_single_config_and_predict(train_df, pred_df, config, seed=seed)

    members = list(config["members"])
    preds = []
    member_meta = {}
    for m in members:
        p, meta = fit_single_config_and_predict(train_df, pred_df, config_registry[m], seed=seed)
        preds.append(p)
        member_meta[m] = meta
    pred = np.mean(np.column_stack(preds), axis=1)
    return pred, {"ensemble_members": members, "member_meta": member_meta}

