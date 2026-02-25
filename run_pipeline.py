from __future__ import annotations

import math
import os
import random
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.features import (
    apply_static_models_to_derby_rows,
    apply_static_models_to_train_like_rows,
    assemble_model_table,
    build_derby_sequential_features,
    build_team_universe,
    build_train_sequential_features,
    fit_static_models_for_fold,
    make_expanding_time_folds,
    parse_and_sort_train,
    parse_predictions,
)
from src.ratings import tune_home_advantage_elo
from src.nextgen_pipeline import run_nextgen_pipeline


SEED = 23
ROOT = Path(__file__).resolve().parent
BASE_MODELS = ["ridge", "elasticnet", "huber", "histgb"]
MAPPING_VARIANTS = ["linear", "nonlinear"]
META_EXCLUDE = {
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


@dataclass
class SweepResult:
    fold_metrics: pd.DataFrame
    config_summary: pd.DataFrame
    oof_frame: pd.DataFrame


@dataclass
class EnsembleResult:
    variant: str
    base_pred_cols: List[str]
    oof_raw_progressive: np.ndarray
    oof_raw_global_weighted: np.ndarray
    progressive_weights: pd.DataFrame
    global_weights: np.ndarray


def set_deterministic(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_ls_la(root: Path) -> None:
    print("ls -la")
    for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        st = p.stat()
        mode = "d" if p.is_dir() else "-"
        mtime = pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
        print(f"{mode} {st.st_size:12d} {mtime} {p.name}")


def assert_required_files(root: Path) -> Dict[str, Path]:
    required = {
        "train": root / "Train.csv",
        "pred": root / "Predictions.csv",
        "rankings": root / "Rankings.xlsx",
    }
    for k, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {k} -> {p}")
    return required


def load_inputs(paths: Mapping[str, Path]) -> Dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(paths["train"]),
        "pred": pd.read_csv(paths["pred"]),
        "rankings": pd.read_excel(paths["rankings"]),
    }


def validate_team_coverage(team_universe: pd.DataFrame, train_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict[str, List[int]]:
    known = set(team_universe["TeamID"].astype(int))
    train_ids = set(train_df["HomeID"].astype(int)) | set(train_df["AwayID"].astype(int))
    pred_ids = set(pred_df["Team1_ID"].astype(int)) | set(pred_df["Team2_ID"].astype(int))
    return {
        "missing_train": sorted(train_ids - known),
        "missing_pred": sorted(pred_ids - known),
    }


def _share_time(total_sec: float, n_parts: int) -> float:
    return float(total_sec) / max(int(n_parts), 1)


def build_fold_datasets(
    train_df: pd.DataFrame,
    folds: List[dict],
    seq_base: pd.DataFrame,
    seq_plus: pd.DataFrame,
    seq_minus: pd.DataFrame,
    team_ids: List[int],
    conf_values: List[str],
    seq_build_times: Mapping[str, float],
) -> tuple[list[dict], pd.DataFrame]:
    fold_datasets: List[dict] = []
    component_val_rows: List[pd.DataFrame] = []
    seq_time_share = _share_time(sum(seq_build_times.values()), len(folds))

    for fold in folds:
        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        fit_games = train_df.loc[train_idx].copy()
        train_rows = train_df.loc[train_idx].copy()
        val_rows = train_df.loc[val_idx].copy()

        static_models = fit_static_models_for_fold(fit_games, team_ids=team_ids, conf_values=conf_values)
        train_static = apply_static_models_to_train_like_rows(train_rows, static_models, neutral_site=False)
        val_static = apply_static_models_to_train_like_rows(val_rows, static_models, neutral_site=False)

        train_tbl = assemble_model_table(train_rows, seq_base.loc[train_idx], train_static)
        val_tbl = assemble_model_table(val_rows, seq_base.loc[val_idx], val_static)
        train_tbl.index = train_rows.index
        val_tbl.index = val_rows.index

        train_tbl_plus = assemble_model_table(train_rows, seq_plus.loc[train_idx], train_static)
        val_tbl_plus = assemble_model_table(val_rows, seq_plus.loc[val_idx], val_static)
        train_tbl_minus = assemble_model_table(train_rows, seq_minus.loc[train_idx], train_static)
        val_tbl_minus = assemble_model_table(val_rows, seq_minus.loc[val_idx], val_static)
        train_tbl_plus.index = train_rows.index
        val_tbl_plus.index = val_rows.index
        train_tbl_minus.index = train_rows.index
        val_tbl_minus.index = val_rows.index

        for tbl in [train_tbl, val_tbl, train_tbl_plus, val_tbl_plus, train_tbl_minus, val_tbl_minus]:
            if "Date" in tbl.columns:
                tbl["Date"] = pd.to_datetime(tbl["Date"])

        fold_datasets.append(
            {
                "fold": fold["fold"],
                "train": train_tbl,
                "val": val_tbl,
                "train_plus": train_tbl_plus,
                "val_plus": val_tbl_plus,
                "train_minus": train_tbl_minus,
                "val_minus": val_tbl_minus,
                "feature_build_time_sec": float(seq_time_share),
                "fold_meta": fold,
            }
        )

        comp = val_tbl[["elo_diff_pre", "massey_diff", "offdef_net_diff", "HomeWinMargin"]].copy()
        comp["fold"] = fold["fold"]
        comp["row_index"] = comp.index
        component_val_rows.append(comp.reset_index(drop=True))

    comp_df = pd.concat(component_val_rows, ignore_index=True) if component_val_rows else pd.DataFrame()
    return fold_datasets, comp_df


def build_full_train_and_derby_tables(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    seq_full_train: pd.DataFrame,
    seq_derby: pd.DataFrame,
    team_ids: List[int],
    conf_values: List[str],
):
    static_models_full = fit_static_models_for_fold(train_df, team_ids=team_ids, conf_values=conf_values)
    train_static_full = apply_static_models_to_train_like_rows(train_df, static_models_full, neutral_site=False)
    pred_static = apply_static_models_to_derby_rows(pred_df, static_models_full)
    train_tbl_full = assemble_model_table(train_df, seq_full_train, train_static_full)
    derby_tbl = assemble_model_table(pred_df, seq_derby, pred_static)
    train_tbl_full.index = train_df.index
    derby_tbl.index = pred_df.index
    if "Date" in train_tbl_full.columns:
        train_tbl_full["Date"] = pd.to_datetime(train_tbl_full["Date"])
    if "Date" in derby_tbl.columns:
        derby_tbl["Date"] = pd.to_datetime(derby_tbl["Date"])
    return train_tbl_full, derby_tbl, static_models_full, train_static_full, pred_static


def compute_ranking_weights_from_oof(component_oof: pd.DataFrame) -> Dict[str, float]:
    cols = ["elo_diff_pre", "massey_diff", "offdef_net_diff"]
    X = component_oof[cols].astype(float)
    y = component_oof["HomeWinMargin"].astype(float).to_numpy()
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=2.0, random_state=SEED))])
    model.fit(X, y)
    coef = np.abs(np.asarray(model.named_steps["ridge"].coef_, dtype=float))
    if coef.sum() <= 0:
        coef = np.ones_like(coef)
    coef = coef / coef.sum()
    return {"elo": float(coef[0]), "massey": float(coef[1]), "net": float(coef[2])}


def zscore_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = float(s.std(ddof=0))
    if not math.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - float(s.mean())) / sd


def build_rankings_output(
    rankings_df: pd.DataFrame,
    team_universe: pd.DataFrame,
    final_elo: Mapping[int, float],
    final_massey: Mapping[int, float],
    final_net: Mapping[int, float],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    out = rankings_df.copy()
    out["TeamID"] = out["TeamID"].astype(int)
    out = out.merge(team_universe.rename(columns={"Team": "Team_universe"}), on="TeamID", how="left")
    elo_s = out["TeamID"].map(final_elo).fillna(1500.0).astype(float)
    massey_s = out["TeamID"].map(final_massey).fillna(0.0).astype(float)
    net_s = out["TeamID"].map(final_net).fillna(0.0).astype(float)
    out["_elo_z"] = zscore_series(elo_s)
    out["_massey_z"] = zscore_series(massey_s)
    out["_net_z"] = zscore_series(net_s)
    out["_score"] = (
        float(weights["elo"]) * out["_elo_z"]
        + float(weights["massey"]) * out["_massey_z"]
        + float(weights["net"]) * out["_net_z"]
    )
    out = out.sort_values(["_score", "TeamID"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1, dtype=int)
    out = out.sort_values("TeamID", kind="mergesort").reset_index(drop=True)
    out["Rank"] = out["Rank"].astype(int)
    out = out.drop(columns=[c for c in out.columns if c.startswith("_") or c == "Team_universe"])
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    resid_sd = float(np.std(resid, ddof=0))
    outlier_freq = 0.0 if resid_sd <= 1e-12 else float(np.mean(np.abs(resid) > 2.5 * resid_sd))
    try:
        skew = float(stats.skew(resid, bias=False))
    except Exception:
        skew = float("nan")
    try:
        kurt = float(stats.kurtosis(resid, fisher=True, bias=False))
    except Exception:
        kurt = float("nan")
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "bias": _bias(y_true, y_pred),
        "resid_skew": skew,
        "resid_kurtosis": kurt,
        "outlier_freq_2p5sd": outlier_freq,
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
    }


def make_estimator(model_name: str, seed: int = SEED):
    m = model_name.lower()
    if m == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=8.0, random_state=seed))])
    if m == "elasticnet":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=30000, random_state=seed)),
            ]
        )
    if m == "huber":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=1000)),
            ]
        )
    if m == "histgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=8,
            l2_regularization=0.8,
            max_iter=400,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unknown model {model_name}")


def add_nonlinear_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    poly_cols = [
        "elo_diff_pre",
        "massey_diff",
        "offdef_net_diff",
        "offdef_margin_neutral",
        "ema_margin_diff",
        "mean_margin_diff",
        "oppadj_margin_diff",
        "conf_strength_diff",
    ]
    for col in poly_cols:
        if col not in out.columns:
            continue
        x = out[col].astype(float)
        z = x / 50.0
        out[f"{col}__sq"] = z * z
        out[f"{col}__cu"] = z * z * z
        out[f"{col}__abs"] = np.abs(z)

    for col, knots in [
        ("elo_diff_pre", [-120, -80, -40, 0, 40, 80, 120]),
        ("massey_diff", [-80, -40, -20, 0, 20, 40, 80]),
        ("offdef_net_diff", [-60, -30, -10, 0, 10, 30, 60]),
    ]:
        if col not in out.columns:
            continue
        x = out[col].astype(float).to_numpy()
        abs_x = np.abs(x)
        abs_bins = [0, 10, 20, 40, 60, 100, np.inf]
        for lo, hi in zip(abs_bins[:-1], abs_bins[1:]):
            name = f"{col}__absbin_{int(lo)}_{'inf' if not np.isfinite(hi) else int(hi)}"
            out[name] = ((abs_x >= lo) & (abs_x < hi)).astype(float)
        for k in knots:
            out[f"{col}__hinge_gt_{k}"] = np.maximum(x - float(k), 0.0) / 50.0
            out[f"{col}__hinge_lt_{k}"] = np.maximum(float(k) - x, 0.0) / 50.0
    return out


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in META_EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def _xy(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str = "HomeWinMargin") -> tuple[pd.DataFrame, np.ndarray]:
    X = df.reindex(columns=list(feature_cols), fill_value=0.0).astype(float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def evaluate_base_models(
    train_df: pd.DataFrame,
    fold_datasets: Sequence[Mapping[str, object]],
    feature_cols_by_variant: Mapping[str, Sequence[str]],
) -> SweepResult:
    y_all = train_df["HomeWinMargin"].to_numpy(dtype=float)
    fold_id_by_row = np.full(len(train_df), -1, dtype=int)
    for fold in fold_datasets:
        fold_id_by_row[np.asarray(fold["fold_meta"]["val_idx"], dtype=int)] = int(fold["fold"])
    valid_mask_all = fold_id_by_row > 0
    if valid_mask_all.sum() == 0:
        raise RuntimeError("No validation-covered rows found in time-aware folds.")

    oof_frame = pd.DataFrame(
        {
            "row_index": np.arange(len(train_df))[valid_mask_all],
            "fold": fold_id_by_row[valid_mask_all],
            "y_true": y_all[valid_mask_all],
        }
    )
    fold_rows: List[dict] = []

    for variant in MAPPING_VARIANTS:
        feature_cols = list(feature_cols_by_variant[variant])
        for model_name in BASE_MODELS:
            config_name = f"{model_name}__{variant}"
            pred_col = f"pred__{config_name}"
            oof_pred = np.full(len(train_df), np.nan, dtype=float)

            for fd in fold_datasets:
                fold_id = int(fd["fold"])
                train_tbl = fd["train_aug"][variant]
                val_tbl = fd["val_aug"][variant]
                X_train, y_train = _xy(train_tbl, feature_cols)
                X_val, y_val = _xy(val_tbl, feature_cols)
                est = make_estimator(model_name, seed=SEED)
                est.fit(X_train, y_train)
                y_hat = np.asarray(est.predict(X_val), dtype=float)
                oof_pred[val_tbl.index.to_numpy(dtype=int)] = y_hat
                met = regression_metrics(y_val, y_hat)
                fold_rows.append(
                    {
                        "config_name": config_name,
                        "model_name": model_name,
                        "mapping": variant,
                        "fold": fold_id,
                        "rmse": met["rmse"],
                        "mae": met["mae"],
                        "bias": met["bias"],
                        "resid_skew": met["resid_skew"],
                        "resid_kurtosis": met["resid_kurtosis"],
                        "outlier_freq_2p5sd": met["outlier_freq_2p5sd"],
                        "n_features": len(feature_cols),
                    }
                )

            if np.isnan(oof_pred[valid_mask_all]).any():
                raise RuntimeError(f"Missing OOF predictions for {config_name}")
            oof_frame[pred_col] = oof_pred[valid_mask_all]

    oof_frame["rating_diff"] = train_df.get("elo_diff_pre", pd.Series(np.zeros(len(train_df)))).astype(float).to_numpy()[valid_mask_all]
    oof_frame["rating_diff_abs"] = np.abs(oof_frame["rating_diff"].astype(float))

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (
        fold_metrics.groupby(["config_name", "model_name", "mapping"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            bias_mean=("bias", "mean"),
            bias_abs_mean=("bias", lambda s: float(np.mean(np.abs(s)))),
            resid_skew_mean=("resid_skew", "mean"),
            resid_kurtosis_mean=("resid_kurtosis", "mean"),
            outlier_freq_2p5sd_mean=("outlier_freq_2p5sd", "mean"),
            n_features=("n_features", "max"),
            n_folds=("fold", "count"),
        )
        .sort_values(["rmse_mean", "mae_mean", "rmse_std"], kind="mergesort")
        .reset_index(drop=True)
    )
    glob_rows = []
    for _, r in summary.iterrows():
        pred_col = f"pred__{r['config_name']}"
        met = regression_metrics(oof_frame["y_true"].to_numpy(dtype=float), oof_frame[pred_col].to_numpy(dtype=float))
        glob_rows.append(
            {
                "config_name": r["config_name"],
                "rmse_oof": met["rmse"],
                "mae_oof": met["mae"],
                "bias_oof": met["bias"],
                "resid_skew_oof": met["resid_skew"],
                "resid_kurtosis_oof": met["resid_kurtosis"],
                "outlier_freq_2p5sd_oof": met["outlier_freq_2p5sd"],
            }
        )
    summary = summary.merge(pd.DataFrame(glob_rows), on="config_name", how="left")
    summary = summary.sort_values(["rmse_mean", "mae_mean", "rmse_std"], kind="mergesort").reset_index(drop=True)
    return SweepResult(fold_metrics=fold_metrics, config_summary=summary, oof_frame=oof_frame)


def _normalize_simplex(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    w[~np.isfinite(w)] = 0.0
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    return np.full_like(w, 1.0 / len(w)) if s <= 0 else w / s


def optimize_simplex_weights(Z: np.ndarray, y: np.ndarray, init: Optional[np.ndarray] = None) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    n = Z.shape[1]
    init = _normalize_simplex(np.full(n, 1.0 / n) if init is None else np.asarray(init, dtype=float))

    def obj(w: np.ndarray) -> float:
        resid = y - Z.dot(np.asarray(w, dtype=float))
        return float(np.mean(resid**2))

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(obj, x0=init, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 300, "ftol": 1e-12})
    if res.success and np.isfinite(res.fun):
        return _normalize_simplex(res.x)

    # Deterministic fallback: test one-hot and equal weights, choose lowest MSE.
    candidates = [init, np.full(n, 1.0 / n)]
    for i in range(n):
        e = np.zeros(n, dtype=float)
        e[i] = 1.0
        candidates.append(e)
    best_w = init
    best_obj = float("inf")
    for w in candidates:
        val = obj(_normalize_simplex(w))
        if val < best_obj:
            best_obj = val
            best_w = _normalize_simplex(w)
    return best_w


def build_ensemble_from_oof(oof_frame: pd.DataFrame, variant: str) -> EnsembleResult:
    base_pred_cols = [f"pred__{m}__{variant}" for m in BASE_MODELS]
    df = oof_frame.sort_values(["fold", "row_index"], kind="mergesort").reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    y_all = df["y_true"].to_numpy(dtype=float)
    Z_all = df[base_pred_cols].to_numpy(dtype=float)
    global_weights = optimize_simplex_weights(Z_all, y_all)
    oof_raw_global_weighted = Z_all.dot(global_weights)

    progressive_raw = np.zeros(len(df), dtype=float)
    rows: List[dict] = []
    fold_array = df["fold"].to_numpy(dtype=int)
    for fold_id in sorted(np.unique(fold_array).tolist()):
        val_mask = fold_array == fold_id
        prior_mask = fold_array < fold_id
        if prior_mask.sum() >= len(BASE_MODELS) * 5:
            w = optimize_simplex_weights(Z_all[prior_mask], y_all[prior_mask], init=global_weights)
            source = "prior_folds"
        else:
            w = np.full(len(BASE_MODELS), 1.0 / len(BASE_MODELS))
            source = "uniform_warm_start"
        progressive_raw[val_mask] = Z_all[val_mask].dot(w)
        row = {"fold": int(fold_id), "source": source}
        for i, m in enumerate(BASE_MODELS):
            row[f"w_{m}"] = float(w[i])
        rows.append(row)

    prog_orig = np.zeros(len(df), dtype=float)
    glob_orig = np.zeros(len(df), dtype=float)
    idx_orig = df["_orig_idx"].to_numpy(dtype=int)
    prog_orig[idx_orig] = progressive_raw
    glob_orig[idx_orig] = oof_raw_global_weighted
    return EnsembleResult(
        variant=variant,
        base_pred_cols=base_pred_cols,
        oof_raw_progressive=prog_orig,
        oof_raw_global_weighted=glob_orig,
        progressive_weights=pd.DataFrame(rows),
        global_weights=global_weights,
    )


def fit_linear_calibration(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(pred) < 2 or np.std(pred) < 1e-9:
        return 0.0, 1.0
    X = np.column_stack([np.ones(len(pred)), pred])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if not np.isfinite(a):
        a = 0.0
    if not np.isfinite(b):
        b = 1.0
    return a, b


def progressive_calibration(oof_frame: pd.DataFrame, raw_pred: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, tuple[float, float]]:
    df = oof_frame[["row_index", "fold", "y_true"]].copy()
    df["raw_pred"] = np.asarray(raw_pred, dtype=float)
    df = df.sort_values(["fold", "row_index"], kind="mergesort").reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    y_all = df["y_true"].to_numpy(dtype=float)
    p_all = df["raw_pred"].to_numpy(dtype=float)

    cal = np.zeros(len(df), dtype=float)
    rows: List[dict] = []
    fold_arr = df["fold"].to_numpy(dtype=int)
    for fold_id in sorted(np.unique(fold_arr).tolist()):
        val_mask = fold_arr == fold_id
        prior_mask = fold_arr < fold_id
        if prior_mask.sum() >= 30:
            a, b = fit_linear_calibration(p_all[prior_mask], y_all[prior_mask])
            source = "prior_folds"
        else:
            a, b = 0.0, 1.0
            source = "identity_warm_start"
        cal[val_mask] = a + b * p_all[val_mask]
        rows.append({"fold": int(fold_id), "intercept_a": float(a), "slope_b": float(b), "source": source})

    cal_orig = np.zeros(len(df), dtype=float)
    cal_orig[df["_orig_idx"].to_numpy(dtype=int)] = cal
    global_params = fit_linear_calibration(np.asarray(raw_pred, dtype=float), oof_frame["y_true"].to_numpy(dtype=float))
    return cal_orig, pd.DataFrame(rows), global_params


def fit_variance_model_linear(x_abs: np.ndarray, sq_resid: np.ndarray) -> tuple[float, float]:
    x_abs = np.asarray(x_abs, dtype=float)
    sq_resid = np.asarray(sq_resid, dtype=float)
    if len(x_abs) < 2:
        base = float(np.mean(sq_resid)) if len(sq_resid) else 100.0
        return max(base, 1.0), 0.0
    X = np.column_stack([np.ones(len(x_abs)), x_abs])
    coef, *_ = np.linalg.lstsq(X, sq_resid, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if not np.isfinite(a):
        a = float(np.mean(sq_resid))
    if not np.isfinite(b):
        b = 0.0
    return max(a, 1.0), max(b, 0.0)


def predict_variance_linear(x_abs: np.ndarray, params: tuple[float, float], floor_var: float = 9.0) -> np.ndarray:
    a, b = params
    var = a + b * np.asarray(x_abs, dtype=float)
    return np.clip(var, floor_var, None)


def progressive_variance_estimation(
    oof_frame: pd.DataFrame,
    pred_for_resid: np.ndarray,
    warm_var: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, tuple[float, float]]:
    df = oof_frame[["row_index", "fold", "y_true", "rating_diff_abs"]].copy()
    df["pred"] = np.asarray(pred_for_resid, dtype=float)
    df = df.sort_values(["fold", "row_index"], kind="mergesort").reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    resid_all = df["y_true"].to_numpy(dtype=float) - df["pred"].to_numpy(dtype=float)
    x_all = df["rating_diff_abs"].to_numpy(dtype=float)
    default_var = float(np.var(df["y_true"].to_numpy(dtype=float)))
    if warm_var is not None and np.isfinite(warm_var) and warm_var > 0:
        default_var = float(warm_var)
    if not np.isfinite(default_var) or default_var <= 0:
        default_var = 100.0

    var_hat = np.zeros(len(df), dtype=float)
    rows: List[dict] = []
    fold_arr = df["fold"].to_numpy(dtype=int)
    for fold_id in sorted(np.unique(fold_arr).tolist()):
        val_mask = fold_arr == fold_id
        prior_mask = fold_arr < fold_id
        if prior_mask.sum() >= 40:
            params = fit_variance_model_linear(x_all[prior_mask], resid_all[prior_mask] ** 2)
            source = "prior_folds"
        else:
            params = (default_var, 0.0)
            source = "constant_warm_start"
        var_hat[val_mask] = predict_variance_linear(x_all[val_mask], params, floor_var=max(default_var * 0.02, 9.0))
        rows.append({"fold": int(fold_id), "var_intercept": float(params[0]), "var_slope": float(params[1]), "source": source})

    var_orig = np.zeros(len(df), dtype=float)
    var_orig[df["_orig_idx"].to_numpy(dtype=int)] = var_hat
    global_params = fit_variance_model_linear(oof_frame["rating_diff_abs"].to_numpy(dtype=float), (oof_frame["y_true"].to_numpy(dtype=float) - np.asarray(pred_for_resid, dtype=float)) ** 2)
    sd_orig = np.sqrt(np.clip(var_orig, 1e-9, None))
    return var_orig, sd_orig, pd.DataFrame(rows), global_params


def compute_shrink_factor(var_hat: np.ndarray, lambda_shrink: float, var_ref: float) -> np.ndarray:
    var_ref = float(max(var_ref, 1e-6))
    sf = 1.0 / (1.0 + float(lambda_shrink) * (np.asarray(var_hat, dtype=float) / var_ref))
    return np.clip(sf, 0.45, 1.0)


def simulate_trimmed_mean(means: np.ndarray, sds: np.ndarray, seed: int, n_draws: int = 2000, trim_frac: float = 0.05) -> np.ndarray:
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((len(means), n_draws))
    sims = means[:, None] + sds[:, None] * z
    sims.sort(axis=1)
    lo = int(round(n_draws * trim_frac))
    hi = int(round(n_draws * (1.0 - trim_frac)))
    lo = max(lo, 0)
    hi = min(max(hi, lo + 1), n_draws)
    return sims[:, lo:hi].mean(axis=1)


def choose_shrink_lambda(y_true: np.ndarray, calibrated_pred: np.ndarray, var_hat: np.ndarray, sd_hat: np.ndarray) -> tuple[float, pd.DataFrame, Dict[str, np.ndarray]]:
    y_true = np.asarray(y_true, dtype=float)
    calibrated_pred = np.asarray(calibrated_pred, dtype=float)
    var_hat = np.asarray(var_hat, dtype=float)
    sd_hat = np.asarray(sd_hat, dtype=float)
    var_ref = float(np.median(var_hat[np.isfinite(var_hat)])) if np.isfinite(var_hat).any() else 1.0
    var_ref = max(var_ref, 1.0)
    rows: List[dict] = []
    for lam in np.linspace(0.0, 1.5, 31):
        sf = compute_shrink_factor(var_hat, float(lam), var_ref)
        shrunk = calibrated_pred * sf
        sim = simulate_trimmed_mean(shrunk, sd_hat, seed=SEED + 900 + int(round(lam * 1000)))
        rows.append(
            {
                "lambda_shrink": float(lam),
                "rmse_shrunk": _rmse(y_true, shrunk),
                "mae_shrunk": _mae(y_true, shrunk),
                "rmse_sim_trimmed": _rmse(y_true, sim),
                "mae_sim_trimmed": _mae(y_true, sim),
                "shrink_mean": float(np.mean(sf)),
                "shrink_min": float(np.min(sf)),
            }
        )
    grid = pd.DataFrame(rows).sort_values(["rmse_sim_trimmed", "mae_sim_trimmed", "lambda_shrink"], kind="mergesort").reset_index(drop=True)
    return float(grid.iloc[0]["lambda_shrink"]), grid, {"var_ref": np.array([var_ref])}


def apply_postprocessing_stack(
    raw_pred: np.ndarray,
    rating_diff_abs: np.ndarray,
    calib_params: tuple[float, float],
    var_params: tuple[float, float],
    lambda_shrink: float,
    var_ref: float,
    sim_seed: int,
) -> Dict[str, np.ndarray]:
    raw_pred = np.asarray(raw_pred, dtype=float)
    a, b = calib_params
    calibrated = a + b * raw_pred
    var_hat = predict_variance_linear(np.asarray(rating_diff_abs, dtype=float), var_params)
    sd_hat = np.sqrt(np.clip(var_hat, 1e-9, None))
    sf = compute_shrink_factor(var_hat, lambda_shrink=lambda_shrink, var_ref=var_ref)
    shrunk = calibrated * sf
    sim_trim = simulate_trimmed_mean(shrunk, sd_hat, seed=sim_seed)
    return {
        "raw": raw_pred,
        "calibrated": calibrated,
        "var_hat": var_hat,
        "sd_hat": sd_hat,
        "shrink_factor": sf,
        "shrunk": shrunk,
        "sim_trimmed": sim_trim,
    }


def augment_fold_and_full_tables(fold_datasets: Sequence[dict], full_train_tbl: pd.DataFrame, derby_tbl: pd.DataFrame) -> tuple[list[dict], dict]:
    out_folds: List[dict] = []
    for fd in fold_datasets:
        fd2 = dict(fd)
        fd2["train_aug"] = {"linear": fd["train"], "nonlinear": add_nonlinear_features(fd["train"])}
        fd2["val_aug"] = {"linear": fd["val"], "nonlinear": add_nonlinear_features(fd["val"])}
        fd2["train_plus_aug"] = {"linear": fd["train_plus"], "nonlinear": add_nonlinear_features(fd["train_plus"])}
        fd2["val_plus_aug"] = {"linear": fd["val_plus"], "nonlinear": add_nonlinear_features(fd["val_plus"])}
        fd2["train_minus_aug"] = {"linear": fd["train_minus"], "nonlinear": add_nonlinear_features(fd["train_minus"])}
        fd2["val_minus_aug"] = {"linear": fd["val_minus"], "nonlinear": add_nonlinear_features(fd["val_minus"])}
        out_folds.append(fd2)
    full_aug = {
        "train": {"linear": full_train_tbl, "nonlinear": add_nonlinear_features(full_train_tbl)},
        "derby": {"linear": derby_tbl, "nonlinear": add_nonlinear_features(derby_tbl)},
    }
    return out_folds, full_aug


def fit_full_base_models_and_predict(train_tbl: pd.DataFrame, pred_tbl: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, np.ndarray]:
    X_train, y_train = _xy(train_tbl, feature_cols)
    X_pred = pred_tbl.reindex(columns=list(feature_cols), fill_value=0.0).astype(float)
    out: Dict[str, np.ndarray] = {}
    for model_name in BASE_MODELS:
        est = make_estimator(model_name, seed=SEED)
        est.fit(X_train, y_train)
        out[model_name] = np.asarray(est.predict(X_pred), dtype=float)
    return out


def summarize_mapping_comparison(config_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mapping in MAPPING_VARIANTS:
        sub = config_summary[config_summary["mapping"] == mapping].sort_values(["rmse_mean", "mae_mean"], kind="mergesort")
        if sub.empty:
            continue
        best = sub.iloc[0]
        rows.append(
            {
                "mapping": mapping,
                "best_config": str(best["config_name"]),
                "rmse_mean": float(best["rmse_mean"]),
                "mae_mean": float(best["mae_mean"]),
                "bias_mean": float(best["bias_mean"]),
                "n_features": int(best["n_features"]),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 2 and (out["mapping"] == "linear").any():
        lin_rmse = float(out.loc[out["mapping"] == "linear", "rmse_mean"].iloc[0])
        out["rmse_delta_vs_linear"] = out["rmse_mean"] - lin_rmse
        if (out["mapping"] == "nonlinear").any():
            nl_rmse = float(out.loc[out["mapping"] == "nonlinear", "rmse_mean"].iloc[0])
            out["nonlinear_improves"] = bool(nl_rmse < lin_rmse)
    return out


def build_stage_metrics_table(y_true: np.ndarray, stages: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, pred in stages.items():
        m = regression_metrics(y_true, pred)
        rows.append(
            {
                "stage": name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "bias": m["bias"],
                "resid_skew": m["resid_skew"],
                "resid_kurtosis": m["resid_kurtosis"],
                "outlier_freq_2p5sd": m["outlier_freq_2p5sd"],
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "mae"], kind="mergesort").reset_index(drop=True)


def format_df_for_text(df: pd.DataFrame, max_rows: int = 20, width: int = 140) -> str:
    if df is None or df.empty:
        return "(empty)"
    use = df.head(max_rows).copy()
    txt = use.to_string(index=False)
    lines: List[str] = []
    for line in txt.splitlines():
        if len(line) <= width:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=width, break_long_words=False, replace_whitespace=False))
    if len(df) > max_rows:
        lines.append(f"... ({len(df) - max_rows} more rows)")
    return "\n".join(lines)


def _new_text_page(pdf: PdfPages, title: str, lines: List[str], footer: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.05, 0.97, title, fontsize=16, fontweight="bold", va="top")
    y = 0.94
    for line in lines:
        if y < 0.05:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            fig.text(0.05, 0.97, f"{title} (cont.)", fontsize=16, fontweight="bold", va="top")
            y = 0.94
        fig.text(0.05, y, line, fontsize=9, va="top", family="monospace")
        y -= 0.018
    if footer:
        fig.text(0.05, 0.02, footer, fontsize=8, va="bottom")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_with_table(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows_per_page: int = 28, float_fmt: str = "{:.3f}") -> None:
    if df is None or df.empty:
        _new_text_page(pdf, title, ["(empty)"])
        return
    n_pages = int(math.ceil(len(df) / max_rows_per_page))
    for i in range(n_pages):
        chunk = df.iloc[i * max_rows_per_page : (i + 1) * max_rows_per_page].copy()
        disp = chunk.copy()
        for c in disp.columns:
            if pd.api.types.is_float_dtype(disp[c]):
                disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else float_fmt.format(float(x)))
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(f"{title} ({i + 1}/{n_pages})", fontsize=12, fontweight="bold", pad=12)
        tbl = ax.table(cellText=disp.values, colLabels=[str(c) for c in disp.columns], loc="center", cellLoc="center", colLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_final_report_pdf(
    out_path: Path,
    *,
    train_df: pd.DataFrame,
    pred_input: pd.DataFrame,
    folds: Sequence[dict],
    best_home_adv: int,
    home_adv_detail: pd.DataFrame,
    feature_cols_by_variant: Mapping[str, Sequence[str]],
    sweep: SweepResult,
    mapping_comparison: pd.DataFrame,
    ensemble_linear: EnsembleResult,
    ensemble_nonlinear: EnsembleResult,
    ensemble_comparison: pd.DataFrame,
    calibration_fold_df: pd.DataFrame,
    calibration_global: tuple[float, float],
    variance_fold_df: pd.DataFrame,
    variance_global: tuple[float, float],
    shrink_grid: pd.DataFrame,
    lambda_shrink: float,
    var_ref: float,
    stage_metrics: pd.DataFrame,
    heavy_tail_compare: pd.DataFrame,
    oof_frame: pd.DataFrame,
    oof_stages: Mapping[str, np.ndarray],
    final_derby_stage: Mapping[str, np.ndarray],
    final_pred_int: np.ndarray,
    home_adv_sensitivity: pd.DataFrame,
    weight_perturb_summary: pd.DataFrame,
    rankings_out: pd.DataFrame,
    ranking_weights: Mapping[str, float],
    component_oof_weights: Mapping[str, float],
    note_reportlab_missing: bool,
) -> None:
    y_true = oof_frame["y_true"].to_numpy(dtype=float)
    pred_input_dates = pd.to_datetime(pred_input["Date"])
    final_oof_pred = np.asarray(oof_stages["sim_trimmed"], dtype=float)
    final_resid = y_true - final_oof_pred
    raw_ens = np.asarray(oof_stages["ensemble_raw_progressive"], dtype=float)
    cal_ens = np.asarray(oof_stages["calibrated_progressive"], dtype=float)
    rating_abs = oof_frame["rating_diff_abs"].to_numpy(dtype=float)

    with PdfPages(out_path) as pdf:
        if note_reportlab_missing:
            _new_text_page(
                pdf,
                "Environment Note",
                [
                    "reportlab package is not installed in this environment.",
                    "PDF generated with matplotlib PdfPages fallback (same required content included).",
                ],
            )

        prior_rmse = 279.39
        best_stage = stage_metrics.sort_values(["rmse", "mae"], kind="mergesort").iloc[0]
        _new_text_page(
            pdf,
            "Executive Summary",
            [
                "1. Executive summary",
                "- Objective metric: RMSE on margin prediction (primary). MAE is secondary.",
                f"- Prior stated RMSE baseline: {prior_rmse:.2f}",
                f"- Best time-aware CV stage RMSE: {float(best_stage['rmse']):.3f}",
                f"- Improvement vs prior baseline: {prior_rmse - float(best_stage['rmse']):.3f}",
                "- Final derby output uses nonlinear mapping + constrained ensemble + calibration + uncertainty-aware shrinkage.",
                "",
                "2. Data constraints (home vs neutral, no derby labels)",
                f"- Train games: {len(train_df)} ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})",
                f"- Derby games: {len(pred_input)} on {pred_input_dates.min().date()}",
                "- No derby labels available. Validation uses chronological expanding folds of Train.csv only.",
                "- Derby predictions are neutral-site (home advantage set to 0 in derby feature generation).",
                "",
                "3. Feature engineering",
                "- Pregame-only sequential Elo, Massey, off/def net, strength of schedule, EMA margin, rest/volatility retained.",
                f"- Elo HOME_ADV tuned via time-aware CV: {best_home_adv} Elo points.",
                f"- Feature counts: linear={len(feature_cols_by_variant['linear'])}, nonlinear={len(feature_cols_by_variant['nonlinear'])}.",
                "- Nonlinear mapping implemented with polynomial + piecewise hinge/bin transforms of rating differentials.",
            ],
        )

        fold_lines = ["Chronological folds:"]
        for f in folds:
            fold_lines.append(
                f" fold={f['fold']} train_n={len(f['train_idx'])} val_n={len(f['val_idx'])} "
                f"train_end={pd.Timestamp(f['train_end_date']).date()} "
                f"val={pd.Timestamp(f['val_start_date']).date()}..{pd.Timestamp(f['val_end_date']).date()}"
            )
        ha_summary = (
            home_adv_detail[home_adv_detail["fold"] >= 0]
            .groupby("home_adv", as_index=False)
            .agg(brier_mean=("brier", "mean"), brier_std=("brier", "std"), logloss_mean=("logloss", "mean"))
            .sort_values(["brier_mean", "brier_std", "home_adv"], kind="mergesort")
        )
        fold_lines.extend(["", "HOME_ADV tuning summary:", format_df_for_text(ha_summary, max_rows=20, width=130)])
        _new_text_page(pdf, "Validation Design", fold_lines)

        model_cols = [
            "config_name",
            "model_name",
            "mapping",
            "rmse_mean",
            "mae_mean",
            "bias_mean",
            "rmse_std",
            "resid_skew_oof",
            "resid_kurtosis_oof",
            "outlier_freq_2p5sd_oof",
        ]
        _page_with_table(pdf, "4. Model comparison table (all configs, CV metrics)", sweep.config_summary[model_cols].copy(), max_rows_per_page=24)

        _new_text_page(
            pdf,
            "Modeling Highlights",
            [
                "4. Model comparison summary",
                "Selection rule: lowest CV RMSE first, MAE second.",
                "",
                "Nonlinear mapping comparison:",
                format_df_for_text(mapping_comparison, max_rows=10, width=130),
                "",
                "7. Ensemble weighting results (linear vs nonlinear)",
                format_df_for_text(ensemble_comparison, max_rows=10, width=130),
                "",
                "Final nonlinear global ensemble weights (simplex constrained):",
                f" Ridge={ensemble_nonlinear.global_weights[0]:.4f}",
                f" ElasticNet={ensemble_nonlinear.global_weights[1]:.4f}",
                f" Huber={ensemble_nonlinear.global_weights[2]:.4f}",
                f" HistGB={ensemble_nonlinear.global_weights[3]:.4f}",
            ],
        )
        _page_with_table(pdf, "7. Ensemble weighting results - nonlinear progressive fold weights", ensemble_nonlinear.progressive_weights, max_rows_per_page=20)

        cal_avg_slope = float(calibration_fold_df["slope_b"].mean())
        cal_avg_intercept = float(calibration_fold_df["intercept_a"].mean())
        cal_interp = "overconfident" if cal_avg_slope < 1.0 else "underconfident" if cal_avg_slope > 1.0 else "well-calibrated"
        _new_text_page(
            pdf,
            "Calibration and Variance",
            [
                "5. Calibration analysis",
                "- Fold calibration regression: actual = a + b * predicted (progressive prior-fold fit).",
                f"- Global calibration for final derby: a={calibration_global[0]:.4f}, b={calibration_global[1]:.4f}",
                f"- Average fold slope b={cal_avg_slope:.4f}, average intercept a={cal_avg_intercept:.4f}",
                f"- Interpretation: average slope implies raw ensemble is {cal_interp}.",
                f"- Calibration RMSE effect: raw={_rmse(y_true, raw_ens):.3f} -> calibrated={_rmse(y_true, cal_ens):.3f}",
                "",
                "6. Variance modeling",
                "- squared_residual ~ |rating_diff| (using |elo_diff_pre|; nonnegative slope enforced).",
                f"- Global variance model: var = {variance_global[0]:.4f} + {variance_global[1]:.6f} * |rating_diff|",
                f"- Shrink lambda selected by OOF RMSE of simulated trimmed mean: {lambda_shrink:.3f}",
                f"- Variance reference scale for shrinking: {var_ref:.3f}",
                "",
                "8. Uncertainty-aware prediction",
                "- Per-fold residual SD estimated from variance model.",
                "- 2000 Normal draws per game, 5% trimmed mean candidate prediction.",
                "- Compared raw, calibrated, variance-shrunk, simulated-trimmed stages via OOF RMSE.",
            ],
        )
        _page_with_table(pdf, "5. Calibration fold slopes/intercepts", calibration_fold_df, max_rows_per_page=20)
        _page_with_table(pdf, "6. Variance model fold parameters", variance_fold_df, max_rows_per_page=20)
        _page_with_table(pdf, "8. Shrink lambda sweep and simulation RMSE", shrink_grid, max_rows_per_page=24)
        _page_with_table(pdf, "Stage comparison (OOF)", stage_metrics, max_rows_per_page=20)

        _new_text_page(
            pdf,
            "Robust Regression",
            [
                "6. Robust regression / heavy tail control",
                "- Explicitly compared Ridge (L2) and Huber (robust loss) on nonlinear feature mapping.",
                "- Residual skewness, kurtosis, and outlier frequency reported below.",
                "",
                format_df_for_text(heavy_tail_compare, max_rows=10, width=130),
                "",
                "11. Key decisions & rationale",
                "- RMSE is the governing selection criterion for configs and downstream stack comparisons.",
                "- Simplex-constrained weights enforce stable convex blending across model families.",
                "- Calibration and variance-aware shrinkage correct scale/bias and reduce extreme-margin risk.",
            ],
        )

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("8. Diagnostic plots - residuals and variance", fontsize=14, fontweight="bold")
        axes[0, 0].hist(final_resid, bins=30, color="#4C78A8", alpha=0.85, edgecolor="white")
        axes[0, 0].set_title("Residual histogram (final OOF stage)")
        axes[0, 0].set_xlabel("Residual")
        axes[0, 0].set_ylabel("Count")
        ridge_pred = oof_frame["pred__ridge__nonlinear"].to_numpy(dtype=float)
        huber_pred = oof_frame["pred__huber__nonlinear"].to_numpy(dtype=float)
        axes[0, 1].hist(y_true - ridge_pred, bins=30, density=True, alpha=0.45, label="Ridge")
        axes[0, 1].hist(y_true - huber_pred, bins=30, density=True, alpha=0.45, label="Huber")
        axes[0, 1].set_title("Ridge vs Huber residual distributions")
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].scatter(final_oof_pred, final_resid, s=10, alpha=0.5)
        axes[1, 0].axhline(0, color="black", lw=1)
        axes[1, 0].set_title("Residual vs fitted")
        axes[1, 0].set_xlabel("Fitted")
        axes[1, 0].set_ylabel("Residual")
        axes[1, 1].scatter(rating_abs, (y_true - cal_ens) ** 2, s=10, alpha=0.3)
        x_line = np.linspace(float(np.min(rating_abs)), float(np.max(rating_abs)), 200)
        axes[1, 1].plot(x_line, variance_global[0] + variance_global[1] * x_line, color="#E45756", lw=2)
        axes[1, 1].set_title("Squared residual vs |rating_diff|")
        axes[1, 1].set_xlabel("|elo_diff_pre|")
        axes[1, 1].set_ylabel("Squared residual")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("8. Diagnostic plots - QQ, calibration, derby distribution", fontsize=14, fontweight="bold")
        (osm, osr), (slope, intercept, rqq) = stats.probplot(final_resid, dist="norm")
        axes[0, 0].scatter(osm, osr, s=10, alpha=0.6)
        axes[0, 0].plot(osm, slope * np.asarray(osm) + intercept, color="#E45756", lw=2)
        axes[0, 0].set_title(f"QQ-style residual plot (r={rqq:.3f})")
        axes[0, 1].scatter(raw_ens, y_true, s=10, alpha=0.45)
        xx = np.linspace(float(np.min(raw_ens)), float(np.max(raw_ens)), 200)
        axes[0, 1].plot(xx, xx, color="black", lw=1, linestyle="--")
        axes[0, 1].plot(xx, calibration_global[0] + calibration_global[1] * xx, color="#E45756", lw=2)
        axes[0, 1].set_title("Calibration line: actual vs raw ensemble")
        derby_final = np.asarray(final_derby_stage["sim_trimmed"], dtype=float)
        axes[1, 0].hist(derby_final, bins=20, color="#72B7B2", alpha=0.85, edgecolor="white")
        axes[1, 0].set_title("10. Final derby prediction distribution")
        axes[1, 0].set_xlabel("Predicted margin")
        axes[1, 1].scatter(derby_final, np.asarray(final_derby_stage["shrink_factor"], dtype=float), s=14, alpha=0.7)
        axes[1, 1].set_ylim(0.4, 1.02)
        axes[1, 1].set_title("Shrink factor vs derby prediction")
        axes[1, 1].set_xlabel("Predicted margin")
        axes[1, 1].set_ylabel("Shrink factor")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("9. Sensitivity analyses", fontsize=14, fontweight="bold")
        base_final = np.asarray(final_derby_stage["sim_trimmed"], dtype=float)
        axes[0, 0].hist(home_adv_sensitivity["delta_plus"], bins=15, alpha=0.6, label="+10 Elo HA")
        axes[0, 0].hist(home_adv_sensitivity["delta_minus"], bins=15, alpha=0.6, label="-10 Elo HA")
        axes[0, 0].axvline(0, color="black", lw=1)
        axes[0, 0].set_title("HOME_ADV +/-10 sensitivity")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].scatter(base_final, home_adv_sensitivity["delta_plus"], s=12, alpha=0.6, label="+10")
        axes[0, 1].scatter(base_final, home_adv_sensitivity["delta_minus"], s=12, alpha=0.6, label="-10")
        axes[0, 1].axhline(0, color="black", lw=1)
        axes[0, 1].set_title("Sensitivity vs base prediction")
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].hist(weight_perturb_summary["mean_abs_shift"], bins=20, color="#F58518", alpha=0.8, edgecolor="white")
        axes[1, 0].set_title("Ensemble weights perturbation: mean abs shift")
        axes[1, 1].scatter(weight_perturb_summary["weight_l2_shift"], weight_perturb_summary["mean_abs_shift"], s=12, alpha=0.6)
        axes[1, 1].set_title("Prediction sensitivity vs weight L2 shift")
        axes[1, 1].set_xlabel("Weight L2 shift")
        axes[1, 1].set_ylabel("Mean abs prediction shift")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        derby_desc = pd.Series(final_pred_int).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
        _new_text_page(
            pdf,
            "Final Decisions And Limitations",
            [
                "10. Final derby prediction distribution",
                f"- Count={len(final_pred_int)}, mean={np.mean(final_pred_int):.3f}, std={np.std(final_pred_int):.3f}, min={np.min(final_pred_int)}, max={np.max(final_pred_int)}",
                f"- Quantiles 1/5/50/95/99: {derby_desc.get('1%', np.nan):.3f}, {derby_desc.get('5%', np.nan):.3f}, {derby_desc.get('50%', np.nan):.3f}, {derby_desc.get('95%', np.nan):.3f}, {derby_desc.get('99%', np.nan):.3f}",
                "",
                "11. Key decisions & rationale",
                "- Final derby stack keeps all required layers: nonlinear transform, ensemble weighting, calibration, uncertainty shrinkage/simulation.",
                "- Winsorization uses Train margin 1st/99th percentiles before rounding.",
                "",
                "12. Limitations and future improvements",
                "- Derby labels unavailable; all tuning based on Train-only time-aware CV proxies.",
                "- Global deployment calibration/weights are fit on pooled OOF predictions; nested tuning could further reduce optimism.",
                "- No injuries/lineups/travel features included.",
                "- QuantileRegressor comparison omitted to keep deterministic runtime bounded.",
                "",
                "Rankings construction",
                f"- Ranking weights (Elo/Massey/Net): {ranking_weights}",
                f"- Component OOF ridge weighting proxy: {component_oof_weights}",
            ],
        )

        appendix_lines = [
            "Appendix",
            "First 10 rows of predictions.csv:",
            format_df_for_text(pred_input.assign(Team1_WinMargin=final_pred_int).head(10), max_rows=12, width=140),
            "",
            "Top 20 teams in rankings (sorted by Rank):",
            format_df_for_text(rankings_out.sort_values('Rank', kind='mergesort').head(20), max_rows=22, width=120),
        ]
        _new_text_page(pdf, "Appendix", appendix_lines)


def validate_outputs_and_print_proof(root: Path) -> None:
    pred_path = root / "predictions.csv"
    rank_path = root / "rankings.xlsx"
    report_path = root / "final_report.pdf"

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not rank_path.exists():
        raise FileNotFoundError(rank_path)
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    if report_path.stat().st_size <= 0:
        raise AssertionError("final_report.pdf must have size > 0")

    pred = pd.read_csv(pred_path)
    rank = pd.read_excel(rank_path)

    if len(pred) != 75:
        raise AssertionError(f"predictions.csv must have 75 rows, found {len(pred)}")
    if pred["Team1_WinMargin"].isna().any():
        raise AssertionError("predictions.csv has missing Team1_WinMargin")
    if not pd.to_numeric(pred["Team1_WinMargin"], errors="coerce").notna().all():
        raise AssertionError("predictions.csv Team1_WinMargin contains non-numeric values")

    if len(rank) != 165:
        raise AssertionError(f"rankings.xlsx must have 165 rows, found {len(rank)}")
    rank_vals = pd.to_numeric(rank["Rank"], errors="coerce")
    if rank_vals.isna().any():
        raise AssertionError("rankings.xlsx Rank has missing/non-numeric values")
    if not np.allclose(rank_vals, np.round(rank_vals)):
        raise AssertionError("rankings.xlsx Rank must be integer-valued")
    rank_int = rank_vals.astype(int)
    if set(rank_int.tolist()) != set(range(1, 166)):
        raise AssertionError("rankings.xlsx Rank must be exactly unique 1..165")

    print("\nPROOF OF COMPLETION")
    pred_numeric_ok = bool(pd.to_numeric(pred["Team1_WinMargin"], errors="coerce").notna().all())
    print(
        f"- predictions.csv exists, rows={len(pred)}, Team1_WinMargin no missing={not pred['Team1_WinMargin'].isna().any()}, numeric={pred_numeric_ok}"
    )
    print(
        f"- rankings.xlsx exists, rows={len(rank)}, Rank exactly integers 1..165={set(rank_int.tolist()) == set(range(1,166))}"
    )
    print(f"- final_report.pdf exists and file size > 0: {report_path.stat().st_size > 0} (bytes={report_path.stat().st_size})")
    print("\nhead(10) predictions:")
    print(pred.head(10).to_string(index=False))
    print("\nhead(10) rankings sorted by Rank:")
    print(rank.assign(Rank=rank_int).sort_values("Rank", kind="mergesort").head(10).to_string(index=False))


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    set_deterministic(SEED)

    # Next-generation nested time-aware pipeline (kept in a separate module to preserve this file's legacy helpers/report code).
    result = run_nextgen_pipeline(ROOT, seed=SEED)
    validate_outputs_and_print_proof(ROOT)
    print("\nFINAL PIPELINE SUMMARY")
    print(f"- chosen model family: {result.get('selected_model_family')}")
    print(f"- chosen Elo variant: {result.get('selected_elo_variant')}")
    print(f"- nested outer RMSE/MAE: {result.get('nested_outer_rmse'):.5f} / {result.get('nested_outer_mae'):.5f}")
    print(f"- calibration type: {result.get('calibration_type')}")
    print(f"- scale correction type: {result.get('scale_correction_type')}")
    print(f"- regime-specific stack used: {result.get('regime_specific_stack_used')}")
    return

    print("pwd")
    print(str(ROOT))
    print_ls_la(ROOT)

    paths = assert_required_files(ROOT)
    inputs = load_inputs(paths)
    train_raw = inputs["train"]
    pred_raw = inputs["pred"]
    rankings_raw = inputs["rankings"]
    print("\nLoaded input shapes:")
    print(f"Train.csv: {train_raw.shape}")
    print(f"Predictions.csv: {pred_raw.shape}")
    print(f"Rankings.xlsx: {rankings_raw.shape}")

    train_df = parse_and_sort_train(train_raw)
    pred_df = parse_predictions(pred_raw)
    team_universe = build_team_universe(rankings_raw)
    coverage = validate_team_coverage(team_universe, train_df, pred_df)
    if coverage["missing_train"] or coverage["missing_pred"]:
        print(f"WARNING missing_train={coverage['missing_train']}")
        print(f"WARNING missing_pred={coverage['missing_pred']}")

    team_ids = team_universe["TeamID"].astype(int).tolist()
    conf_values = sorted(set(train_df["HomeConf"].astype(str)) | set(train_df["AwayConf"].astype(str)))
    folds = make_expanding_time_folds(train_df, n_folds=5)
    print("\nExpanding-window folds:")
    for f in folds:
        print(
            f" fold={f['fold']} train_n={len(f['train_idx'])} val_n={len(f['val_idx'])} "
            f"train_end={pd.Timestamp(f['train_end_date']).date()} "
            f"val={pd.Timestamp(f['val_start_date']).date()}..{pd.Timestamp(f['val_end_date']).date()}"
        )

    best_home_adv, home_adv_detail = tune_home_advantage_elo(train_df, folds, candidate_values=range(0, 121, 10), k_factor=24.0)
    print(f"\nSelected HOME_ADV={best_home_adv} Elo points via time-aware Elo CV.")

    seq_times = {"base": 0.0, "plus": 0.0, "minus": 0.0}
    seq_build_base = build_train_sequential_features(train_df, home_adv=float(best_home_adv), team_ids=team_ids)
    seq_build_plus = build_train_sequential_features(train_df, home_adv=float(best_home_adv + 10), team_ids=team_ids)
    seq_build_minus = build_train_sequential_features(train_df, home_adv=float(best_home_adv - 10), team_ids=team_ids)

    fold_datasets, component_oof_rows = build_fold_datasets(
        train_df=train_df,
        folds=folds,
        seq_base=seq_build_base.features,
        seq_plus=seq_build_plus.features,
        seq_minus=seq_build_minus.features,
        team_ids=team_ids,
        conf_values=conf_values,
        seq_build_times=seq_times,
    )

    seq_derby_base = build_derby_sequential_features(pred_df, seq_build_base.final_states, seq_build_base.final_elo)
    full_train_tbl, derby_tbl, static_models_full, train_static_full, pred_static_full = build_full_train_and_derby_tables(
        train_df=train_df,
        pred_df=pred_df,
        seq_full_train=seq_build_base.features,
        seq_derby=seq_derby_base,
        team_ids=team_ids,
        conf_values=conf_values,
    )

    fold_datasets_aug, full_aug = augment_fold_and_full_tables(fold_datasets, full_train_tbl, derby_tbl)
    feature_cols_by_variant = {
        "linear": select_feature_columns(full_aug["train"]["linear"]),
        "nonlinear": select_feature_columns(full_aug["train"]["nonlinear"]),
    }
    print(f"\nFeature counts: linear={len(feature_cols_by_variant['linear'])}, nonlinear={len(feature_cols_by_variant['nonlinear'])}")

    sweep = evaluate_base_models(train_df=full_train_tbl, fold_datasets=fold_datasets_aug, feature_cols_by_variant=feature_cols_by_variant)
    print("\nTop configs by CV RMSE:")
    print(
        sweep.config_summary[
            ["config_name", "rmse_mean", "mae_mean", "bias_mean", "rmse_std", "resid_kurtosis_oof", "outlier_freq_2p5sd_oof"]
        ].head(12).to_string(index=False)
    )
    best_base = sweep.config_summary.sort_values(["rmse_mean", "mae_mean"], kind="mergesort").iloc[0]
    print(f"\nBase model chosen by RMSE: {best_base['config_name']} (RMSE={best_base['rmse_mean']:.3f})")

    mapping_comparison = summarize_mapping_comparison(sweep.config_summary)

    ensemble_linear = build_ensemble_from_oof(sweep.oof_frame, variant="linear")
    ensemble_nonlinear = build_ensemble_from_oof(sweep.oof_frame, variant="nonlinear")
    y_oof = sweep.oof_frame["y_true"].to_numpy(dtype=float)
    ensemble_comparison = pd.DataFrame(
        [
            {
                "variant": "linear",
                "rmse_progressive": _rmse(y_oof, ensemble_linear.oof_raw_progressive),
                "mae_progressive": _mae(y_oof, ensemble_linear.oof_raw_progressive),
                "rmse_global_weighted_oof": _rmse(y_oof, ensemble_linear.oof_raw_global_weighted),
                "mae_global_weighted_oof": _mae(y_oof, ensemble_linear.oof_raw_global_weighted),
                **{f"w_{m}": float(ensemble_linear.global_weights[i]) for i, m in enumerate(BASE_MODELS)},
            },
            {
                "variant": "nonlinear",
                "rmse_progressive": _rmse(y_oof, ensemble_nonlinear.oof_raw_progressive),
                "mae_progressive": _mae(y_oof, ensemble_nonlinear.oof_raw_progressive),
                "rmse_global_weighted_oof": _rmse(y_oof, ensemble_nonlinear.oof_raw_global_weighted),
                "mae_global_weighted_oof": _mae(y_oof, ensemble_nonlinear.oof_raw_global_weighted),
                **{f"w_{m}": float(ensemble_nonlinear.global_weights[i]) for i, m in enumerate(BASE_MODELS)},
            },
        ]
    ).sort_values(["rmse_progressive", "rmse_global_weighted_oof"], kind="mergesort")
    print("\nConstrained ensemble comparison:")
    print(ensemble_comparison.to_string(index=False))

    raw_ensemble_prog = ensemble_nonlinear.oof_raw_progressive
    raw_ensemble_global = ensemble_nonlinear.oof_raw_global_weighted
    calibrated_prog, calibration_fold_df, calibration_global = progressive_calibration(sweep.oof_frame, raw_ensemble_global)
    var_hat_prog, sd_hat_prog, variance_fold_df, variance_global = progressive_variance_estimation(
        sweep.oof_frame, pred_for_resid=calibrated_prog, warm_var=float(np.var(y_oof - calibrated_prog))
    )
    lambda_shrink, shrink_grid, shrink_aux = choose_shrink_lambda(y_oof, calibrated_prog, var_hat_prog, sd_hat_prog)
    var_ref = float(shrink_aux["var_ref"][0])
    shrink_factor_prog = compute_shrink_factor(var_hat_prog, lambda_shrink=lambda_shrink, var_ref=var_ref)
    shrunk_prog = calibrated_prog * shrink_factor_prog
    sim_trim_prog = simulate_trimmed_mean(shrunk_prog, sd_hat_prog, seed=SEED + 1001)

    oof_stages = {
        "ensemble_raw_progressive": raw_ensemble_prog,
        "ensemble_raw_global_weighted": raw_ensemble_global,
        "calibrated_progressive": calibrated_prog,
        "variance_shrunk": shrunk_prog,
        "sim_trimmed": sim_trim_prog,
    }
    stage_metrics = build_stage_metrics_table(y_oof, oof_stages)
    print("\nPost-ensemble stage comparison (OOF):")
    print(stage_metrics.to_string(index=False))

    heavy_tail_compare = sweep.config_summary[
        sweep.config_summary["config_name"].isin(["ridge__nonlinear", "huber__nonlinear"])
    ][
        ["config_name", "rmse_mean", "mae_mean", "bias_mean", "resid_skew_oof", "resid_kurtosis_oof", "outlier_freq_2p5sd_oof"]
    ].sort_values(["rmse_mean", "mae_mean"], kind="mergesort")
    print("\nHeavy-tail control comparison (Ridge vs Huber):")
    print(heavy_tail_compare.to_string(index=False))

    full_train_nl = full_aug["train"]["nonlinear"]
    derby_nl = full_aug["derby"]["nonlinear"]
    full_pred_by_model = fit_full_base_models_and_predict(full_train_nl, derby_nl, feature_cols_by_variant["nonlinear"])
    P_derby = np.column_stack([full_pred_by_model[m] for m in BASE_MODELS])
    raw_derby = P_derby.dot(ensemble_nonlinear.global_weights)
    global_post = apply_postprocessing_stack(
        raw_pred=raw_derby,
        rating_diff_abs=np.abs(derby_tbl["elo_diff_pre"].astype(float).to_numpy()),
        calib_params=calibration_global,
        var_params=variance_global,
        lambda_shrink=lambda_shrink,
        var_ref=var_ref,
        sim_seed=SEED + 2001,
    )

    seq_derby_plus = build_derby_sequential_features(pred_df, seq_build_plus.final_states, seq_build_plus.final_elo)
    seq_derby_minus = build_derby_sequential_features(pred_df, seq_build_minus.final_states, seq_build_minus.final_elo)
    full_train_tbl_plus = assemble_model_table(train_df, seq_build_plus.features, train_static_full)
    full_train_tbl_minus = assemble_model_table(train_df, seq_build_minus.features, train_static_full)
    derby_tbl_plus = assemble_model_table(pred_df, seq_derby_plus, pred_static_full)
    derby_tbl_minus = assemble_model_table(pred_df, seq_derby_minus, pred_static_full)
    for tbl in [full_train_tbl_plus, full_train_tbl_minus]:
        tbl.index = train_df.index
    for tbl in [derby_tbl_plus, derby_tbl_minus]:
        tbl.index = pred_df.index
    full_train_plus_nl = add_nonlinear_features(full_train_tbl_plus)
    full_train_minus_nl = add_nonlinear_features(full_train_tbl_minus)
    derby_plus_nl = add_nonlinear_features(derby_tbl_plus)
    derby_minus_nl = add_nonlinear_features(derby_tbl_minus)

    plus_by_model = fit_full_base_models_and_predict(full_train_plus_nl, derby_plus_nl, feature_cols_by_variant["nonlinear"])
    minus_by_model = fit_full_base_models_and_predict(full_train_minus_nl, derby_minus_nl, feature_cols_by_variant["nonlinear"])
    raw_plus = np.column_stack([plus_by_model[m] for m in BASE_MODELS]).dot(ensemble_nonlinear.global_weights)
    raw_minus = np.column_stack([minus_by_model[m] for m in BASE_MODELS]).dot(ensemble_nonlinear.global_weights)
    post_plus = apply_postprocessing_stack(
        raw_pred=raw_plus,
        rating_diff_abs=np.abs(derby_tbl_plus["elo_diff_pre"].astype(float).to_numpy()),
        calib_params=calibration_global,
        var_params=variance_global,
        lambda_shrink=lambda_shrink,
        var_ref=var_ref,
        sim_seed=SEED + 2002,
    )
    post_minus = apply_postprocessing_stack(
        raw_pred=raw_minus,
        rating_diff_abs=np.abs(derby_tbl_minus["elo_diff_pre"].astype(float).to_numpy()),
        calib_params=calibration_global,
        var_params=variance_global,
        lambda_shrink=lambda_shrink,
        var_ref=var_ref,
        sim_seed=SEED + 2003,
    )
    home_adv_sensitivity = pd.DataFrame(
        {
            "GameID": pred_df["GameID"].astype(int).to_numpy(),
            "base_pred": global_post["sim_trimmed"],
            "pred_plus": post_plus["sim_trimmed"],
            "pred_minus": post_minus["sim_trimmed"],
        }
    )
    home_adv_sensitivity["delta_plus"] = home_adv_sensitivity["pred_plus"] - home_adv_sensitivity["base_pred"]
    home_adv_sensitivity["delta_minus"] = home_adv_sensitivity["pred_minus"] - home_adv_sensitivity["base_pred"]

    rng = np.random.default_rng(SEED + 3001)
    w_base = ensemble_nonlinear.global_weights.copy()
    base_final_derby = np.asarray(global_post["sim_trimmed"], dtype=float)
    perturb_rows = []
    for i in range(200):
        w = _normalize_simplex(w_base + rng.normal(0.0, 0.04, size=len(w_base)))
        raw_p = P_derby.dot(w)
        post_p = apply_postprocessing_stack(
            raw_pred=raw_p,
            rating_diff_abs=np.abs(derby_tbl["elo_diff_pre"].astype(float).to_numpy()),
            calib_params=calibration_global,
            var_params=variance_global,
            lambda_shrink=lambda_shrink,
            var_ref=var_ref,
            sim_seed=SEED + 4000 + i,
        )
        pred_p = np.asarray(post_p["sim_trimmed"], dtype=float)
        delta = pred_p - base_final_derby
        row = {
            "sample_id": i,
            "weight_l2_shift": float(np.sqrt(np.sum((w - w_base) ** 2))),
            "mean_abs_shift": float(np.mean(np.abs(delta))),
            "max_abs_shift": float(np.max(np.abs(delta))),
        }
        for j, m in enumerate(BASE_MODELS):
            row[f"w_{m}"] = float(w[j])
        perturb_rows.append(row)
    weight_perturb_summary = pd.DataFrame(perturb_rows)

    clip_lo = float(train_df["HomeWinMargin"].quantile(0.01))
    clip_hi = float(train_df["HomeWinMargin"].quantile(0.99))
    final_derby_float = np.clip(global_post["sim_trimmed"], clip_lo, clip_hi)
    final_derby_int = np.rint(final_derby_float).astype(int)
    pred_out = pred_raw.copy()
    pred_out["Team1_WinMargin"] = final_derby_int
    pred_out.to_csv(ROOT / "predictions.csv", index=False)
    print(f"\nWrote predictions.csv with winsorized range [{clip_lo:.2f}, {clip_hi:.2f}] and integer rounding.")

    component_oof_weights = compute_ranking_weights_from_oof(component_oof_rows.copy())
    ranking_weights = component_oof_weights.copy()
    final_net_map = static_models_full.offdef.net_rating_map()
    rankings_out = build_rankings_output(
        rankings_df=rankings_raw,
        team_universe=team_universe,
        final_elo=seq_build_base.final_elo,
        final_massey=static_models_full.massey.team_rating,
        final_net=final_net_map,
        weights=ranking_weights,
    )
    rankings_out.to_excel(ROOT / "rankings.xlsx", index=False)
    print("Wrote rankings.xlsx with Rank 1..165.")

    note_reportlab_missing = False
    try:
        import reportlab  # noqa: F401
    except Exception:
        note_reportlab_missing = True
    build_final_report_pdf(
        ROOT / "final_report.pdf",
        train_df=train_df,
        pred_input=pred_raw.copy(),
        folds=folds,
        best_home_adv=int(best_home_adv),
        home_adv_detail=home_adv_detail.copy(),
        feature_cols_by_variant=feature_cols_by_variant,
        sweep=sweep,
        mapping_comparison=mapping_comparison,
        ensemble_linear=ensemble_linear,
        ensemble_nonlinear=ensemble_nonlinear,
        ensemble_comparison=ensemble_comparison,
        calibration_fold_df=calibration_fold_df.copy(),
        calibration_global=calibration_global,
        variance_fold_df=variance_fold_df.copy(),
        variance_global=variance_global,
        shrink_grid=shrink_grid.copy(),
        lambda_shrink=float(lambda_shrink),
        var_ref=float(var_ref),
        stage_metrics=stage_metrics.copy(),
        heavy_tail_compare=heavy_tail_compare.copy(),
        oof_frame=sweep.oof_frame.copy(),
        oof_stages=oof_stages,
        final_derby_stage=global_post,
        final_pred_int=final_derby_int,
        home_adv_sensitivity=home_adv_sensitivity.copy(),
        weight_perturb_summary=weight_perturb_summary.copy(),
        rankings_out=rankings_out.copy(),
        ranking_weights=ranking_weights,
        component_oof_weights=component_oof_weights,
        note_reportlab_missing=note_reportlab_missing,
    )
    print("Wrote final_report.pdf")

    validate_outputs_and_print_proof(ROOT)


if __name__ == "__main__":
    main()
