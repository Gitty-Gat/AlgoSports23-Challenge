from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import random
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression, Ridge

from src.features import make_expanding_time_folds, parse_and_sort_train, parse_predictions
from src.ratings import (
    ColleyRatingComponent,
    EloKDecayConfig,
    EloKDecayRatingComponent,
    MasseyRatingComponent,
    OffDefNetRatingComponent,
)


SEED = 23
ROOT = Path(__file__).resolve().parent

HOME_ADV = 50.0
INITIAL_ELO = 1500.0
ELO_K = 24.0
SEASON_GAP_DAYS = 60

OUTER_FOLDS_PRIMARY = 5
OUTER_FOLDS_FALLBACK = 4
INNER_FOLDS_PRIMARY = 3

MASSEY_ALPHA_GRID = [10.0, 30.0, 100.0]
OFFDEF_ALPHA_GRID = [20.0]
STACKER_ALPHA_GRID = [0.5, 2.0, 10.0]
ELOK_GRID = [
    {"decay_type": "linear", "A": 0.25, "G": 50, "tau": 50},
    {"decay_type": "linear", "A": 0.50, "G": 100, "tau": 50},
    {"decay_type": "exponential", "A": 0.50, "G": 100, "tau": 50},
    {"decay_type": "exponential", "A": 1.00, "G": 100, "tau": 75},
]


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


def rmse(y: Sequence[float], p: Sequence[float]) -> float:
    yv = np.asarray(y, dtype=float)
    pv = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((yv - pv) ** 2)))


def mae(y: Sequence[float], p: Sequence[float]) -> float:
    yv = np.asarray(y, dtype=float)
    pv = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(yv - pv)))


def metric_dict(y: Sequence[float], p: Sequence[float]) -> Dict[str, float]:
    yv = np.asarray(y, dtype=float)
    pv = np.asarray(p, dtype=float)
    resid = yv - pv
    pred_std = float(np.std(pv)) if len(pv) else float("nan")
    act_std = float(np.std(yv)) if len(yv) else float("nan")
    return {
        "rmse": rmse(yv, pv) if len(yv) else float("nan"),
        "mae": mae(yv, pv) if len(yv) else float("nan"),
        "bias": float(np.mean(resid)) if len(resid) else float("nan"),
        "pred_std": pred_std,
        "actual_std": act_std,
        "dispersion_ratio": float(pred_std / act_std) if act_std and abs(act_std) > 1e-12 else float("nan"),
        "n": int(len(yv)),
    }


@dataclass
class BudgetTracker:
    start: float
    max_total: float
    max_tuning: float
    max_fits: int
    fit_count: int = 0
    tuning_start: Optional[float] = None
    stop_reason: Optional[str] = None
    stop_events: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.stop_events is None:
            self.stop_events = []

    def begin_tuning(self) -> None:
        if self.tuning_start is None:
            self.tuning_start = time.perf_counter()

    def total_elapsed(self) -> float:
        return time.perf_counter() - self.start

    def tuning_elapsed(self) -> float:
        return 0.0 if self.tuning_start is None else (time.perf_counter() - self.tuning_start)

    def check(self) -> bool:
        if self.stop_reason is not None:
            return True
        if self.total_elapsed() >= self.max_total:
            self.stop_reason = "MAX_TOTAL_SECONDS"
        elif self.tuning_start is not None and self.tuning_elapsed() >= self.max_tuning:
            self.stop_reason = "MAX_TUNING_SECONDS"
        elif self.fit_count >= self.max_fits:
            self.stop_reason = "MAX_MODEL_FITS"
        if self.stop_reason is not None:
            self.stop_events.append(f"{self.stop_reason} @ total={self.total_elapsed():.2f}s tuning={self.tuning_elapsed():.2f}s fits={self.fit_count}")
            return True
        return False

    def can_fit(self, n: int = 1) -> bool:
        if self.check():
            return False
        if self.fit_count + n > self.max_fits:
            self.stop_reason = "MAX_MODEL_FITS"
            self.stop_events.append(f"{self.stop_reason} @ total={self.total_elapsed():.2f}s tuning={self.tuning_elapsed():.2f}s fits={self.fit_count}")
            return False
        return True

    def add_fit(self, n: int = 1) -> None:
        self.fit_count += int(n)
        self.check()


def ensure_root_inputs(root: Path) -> Tuple[Path, Path, Path]:
    train = root / "Train.csv"
    if not train.exists():
        raise FileNotFoundError(train)

    def ensure(name: str) -> Path:
        p = root / name
        if p.exists():
            return p
        for alt in ["Submission.zip", "Submission.zip1", "Submission.zip2", "Submission.zip3"]:
            c = root / alt / name
            if c.exists():
                shutil.copy2(c, p)
                print(f"Copied missing root input {name} from {c}")
                return p
        raise FileNotFoundError(name)

    return train, ensure("Predictions.csv"), ensure("Rankings.xlsx")


def _make_expanding_time_folds_relaxed(df: pd.DataFrame, n_folds: int) -> List[dict]:
    if "Date" not in df.columns:
        raise ValueError("Date column required")
    udates = np.array(sorted(pd.to_datetime(df["Date"]).unique()))
    if len(udates) < n_folds + 2:
        raise ValueError("Too few unique dates for relaxed folds")
    buckets = np.array_split(udates, n_folds + 1)
    buckets = [b for b in buckets if len(b)]
    if len(buckets) < n_folds + 1:
        raise ValueError("Too few non-empty buckets for relaxed folds")
    folds: List[dict] = []
    for fold_num in range(1, n_folds + 1):
        train_dates = np.concatenate(buckets[:fold_num])
        val_dates = buckets[fold_num]
        train_mask = df["Date"].isin(train_dates).to_numpy()
        val_mask = df["Date"].isin(val_dates).to_numpy()
        tr_idx = np.flatnonzero(train_mask)
        va_idx = np.flatnonzero(val_mask)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_idx": tr_idx,
                "val_idx": va_idx,
                "train_start_date": pd.Timestamp(df.loc[tr_idx[0], "Date"]),
                "train_end_date": pd.Timestamp(df.loc[tr_idx[-1], "Date"]),
                "val_start_date": pd.Timestamp(df.loc[va_idx[0], "Date"]),
                "val_end_date": pd.Timestamp(df.loc[va_idx[-1], "Date"]),
            }
        )
    if len(folds) < n_folds:
        raise ValueError("Constructed too few relaxed folds")
    return folds


def choose_folds(df: pd.DataFrame, preferred: int, fallback: Optional[int] = None) -> List[dict]:
    try:
        return make_expanding_time_folds(df, n_folds=preferred)
    except Exception:
        try:
            return _make_expanding_time_folds_relaxed(df, n_folds=preferred)
        except Exception:
            pass
        if fallback is None:
            raise
        try:
            return make_expanding_time_folds(df, n_folds=fallback)
        except Exception:
            return _make_expanding_time_folds_relaxed(df, n_folds=fallback)


def fold_table_rows(folds: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for f in folds:
        rows.append(
            {
                "fold": int(f["fold"]),
                "n_train": int(len(f["train_idx"])),
                "n_val": int(len(f["val_idx"])),
                "train_end": str(pd.Timestamp(f["train_end_date"]).date()),
                "val_start": str(pd.Timestamp(f["val_start_date"]).date()),
                "val_end": str(pd.Timestamp(f["val_end_date"]).date()),
            }
        )
    return pd.DataFrame(rows)


def fit_affine_map(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if len(xv) == 0:
        return {"a": 0.0, "b": 0.0}
    xm = float(np.mean(xv))
    ym = float(np.mean(yv))
    den = float(np.sum((xv - xm) ** 2))
    if den <= 1e-12:
        return {"a": ym, "b": 0.0}
    b = float(np.sum((xv - xm) * (yv - ym)) / den)
    a = float(ym - b * xm)
    return {"a": a, "b": b}


def apply_affine_map(x: Sequence[float], affine: Mapping[str, float]) -> np.ndarray:
    xv = np.asarray(x, dtype=float)
    return float(affine["a"]) + float(affine["b"]) * xv


def fit_scale_about_anchor(pred: Sequence[float], y: Sequence[float], anchor: Optional[float] = None) -> Dict[str, float]:
    pv = np.asarray(pred, dtype=float)
    yv = np.asarray(y, dtype=float)
    a = float(np.mean(yv) if anchor is None else anchor)
    den = float(np.sum((pv - a) ** 2))
    s = 1.0 if den <= 1e-12 else float(np.sum((pv - a) * (yv - a)) / den)
    s = float(np.clip(s, 0.5, 2.0))
    return {"a": a, "s": s}


def apply_scale_about_anchor(pred: Sequence[float], scale_model: Mapping[str, float]) -> np.ndarray:
    pv = np.asarray(pred, dtype=float)
    a = float(scale_model["a"])
    s = float(scale_model["s"])
    return a + s * (pv - a)


def fit_simplex_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_features = X.shape[1]
    if X.shape[0] == 0 or n_features == 0:
        return np.zeros(n_features, dtype=float)
    try:
        lr = LinearRegression(fit_intercept=False, positive=True)
        lr.fit(X, y)
        w = np.asarray(lr.coef_, dtype=float)
    except Exception:
        try:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            w = np.ones(n_features, dtype=float)
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 1e-12:
        return np.full(n_features, 1.0 / n_features, dtype=float)
    return w / s


def same_conf_series(rows: pd.DataFrame, conf1_col: str, conf2_col: str) -> np.ndarray:
    return (rows[conf1_col].astype(str).to_numpy() == rows[conf2_col].astype(str).to_numpy()).astype(int)


def component_strength_map_from_fit(fit_obj: Mapping[str, Any], family_name: str) -> Dict[int, float]:
    if family_name in {"elo", "elok"}:
        return {int(k): float(v) for k, v in fit_obj["component"].final_ratings.items()}
    if family_name == "massey":
        fr = fit_obj["component"].fit_result
        return {} if fr is None else {int(k): float(v) for k, v in fr.team_rating.items()}
    if family_name == "colley":
        return {int(k): float(v) for k, v in fit_obj["component"].colley_ratings.items()}
    if family_name == "offdef":
        fr = fit_obj["component"].fit_result
        return {} if fr is None else fr.net_rating_map()
    return {}


def family_candidate_grid() -> Dict[str, List[dict]]:
    base_elo = {
        "family": "elo",
        "label": "elo_base",
        "params": {
            "elo_cfg": {
                "home_adv": HOME_ADV,
                "k_factor": ELO_K,
                "initial_rating": INITIAL_ELO,
                "use_mov": True,
                "decay_type": "linear",
                "A": 0.0,
                "G": 100,
                "tau": 50,
                "season_gap_days": SEASON_GAP_DAYS,
            }
        },
    }
    elok = []
    for i, cfg in enumerate(ELOK_GRID, start=1):
        elok.append(
            {
                "family": "elok",
                "label": f"elok_{i}_{cfg['decay_type']}_A{cfg['A']}_G{cfg['G']}_tau{cfg['tau']}",
                "params": {
                    "elo_cfg": {
                        "home_adv": HOME_ADV,
                        "k_factor": ELO_K,
                        "initial_rating": INITIAL_ELO,
                        "use_mov": True,
                        "decay_type": cfg["decay_type"],
                        "A": float(cfg["A"]),
                        "G": int(cfg["G"]),
                        "tau": float(cfg["tau"]),
                        "season_gap_days": SEASON_GAP_DAYS,
                    }
                },
            }
        )
    return {
        "elo": [base_elo],
        "elok": elok,
        "massey": [{"family": "massey", "label": f"massey_alpha{a}", "params": {"alpha": float(a)}} for a in MASSEY_ALPHA_GRID],
        "colley": [{"family": "colley", "label": "colley", "params": {}}],
        "offdef": [{"family": "offdef", "label": f"offdef_alpha{OFFDEF_ALPHA_GRID[0]}", "params": {"alpha": float(OFFDEF_ALPHA_GRID[0])}}],
    }


def make_component(family_name: str, params: Mapping[str, Any]):
    if family_name in {"elo", "elok"}:
        return EloKDecayRatingComponent(EloKDecayConfig(**params["elo_cfg"]))
    if family_name == "massey":
        return MasseyRatingComponent(alpha=float(params.get("alpha", 30.0)))
    if family_name == "colley":
        return ColleyRatingComponent()
    if family_name == "offdef":
        return OffDefNetRatingComponent(alpha=float(params.get("alpha", 20.0)))
    raise ValueError(f"Unknown family {family_name}")


def family_raw_column(family_name: str) -> str:
    return {
        "elo": "elo_diff_pre",
        "elok": "elo_diff_pre",
        "massey": "massey_diff",
        "colley": "colley_diff",
        "offdef": "offdef_margin_with_side",
    }[family_name]


def fit_family_and_build_raw_preds(
    family_name: str,
    params: Mapping[str, Any],
    fit_games: pd.DataFrame,
    score_rows: pd.DataFrame,
    team_ids: Sequence[int],
    neutral_site_for_score: bool,
    budget: BudgetTracker,
) -> Dict[str, Any]:
    if not budget.can_fit(1):
        raise RuntimeError("Budget exhausted before family fit")
    comp = make_component(family_name, params)
    budget.add_fit(1)

    if family_name in {"elo", "elok"}:
        fit_result = comp.fit_transform_train(fit_games, home_id_col="HomeID", away_id_col="AwayID", margin_col="HomeWinMargin", date_col="Date")
        train_feat = fit_result.train_features
        score_feat = comp.transform_rows(
            score_rows,
            home_id_col="HomeID" if "HomeID" in score_rows.columns else "Team1_ID",
            away_id_col="AwayID" if "AwayID" in score_rows.columns else "Team2_ID",
            neutral_site=neutral_site_for_score,
        )
        raw_train = train_feat[family_raw_column(family_name)].to_numpy(float)
        raw_score = score_feat[family_raw_column(family_name)].to_numpy(float)
    elif family_name == "massey":
        comp.fit(fit_games, team_ids=team_ids)
        train_feat = comp.transform_rows(fit_games, "HomeID", "AwayID", neutral_site=False)
        score_feat = comp.transform_rows(
            score_rows,
            "HomeID" if "HomeID" in score_rows.columns else "Team1_ID",
            "AwayID" if "AwayID" in score_rows.columns else "Team2_ID",
            neutral_site=neutral_site_for_score,
        )
        raw_train = train_feat["massey_diff"].to_numpy(float)
        raw_score = score_feat["massey_diff"].to_numpy(float)
    elif family_name == "colley":
        comp.fit(fit_games, team_ids=team_ids)
        train_feat = comp.transform_rows(fit_games, "HomeID", "AwayID")
        score_feat = comp.transform_rows(
            score_rows,
            "HomeID" if "HomeID" in score_rows.columns else "Team1_ID",
            "AwayID" if "AwayID" in score_rows.columns else "Team2_ID",
        )
        raw_train = train_feat["colley_diff"].to_numpy(float)
        raw_score = score_feat["colley_diff"].to_numpy(float)
    elif family_name == "offdef":
        comp.fit(fit_games, team_ids=team_ids)
        train_feat = comp.transform_rows(fit_games, "HomeID", "AwayID", neutral_site=False)
        score_feat = comp.transform_rows(
            score_rows,
            "HomeID" if "HomeID" in score_rows.columns else "Team1_ID",
            "AwayID" if "AwayID" in score_rows.columns else "Team2_ID",
            neutral_site=neutral_site_for_score,
        )
        raw_train = train_feat["offdef_margin_with_side"].to_numpy(float)
        raw_score = (score_feat["offdef_margin_neutral"] if neutral_site_for_score else score_feat["offdef_margin_with_side"]).to_numpy(float)
    else:
        raise ValueError(family_name)

    y_train = fit_games["HomeWinMargin"].to_numpy(float)
    affine = fit_affine_map(raw_train, y_train)
    pred_train = apply_affine_map(raw_train, affine)
    pred_score = apply_affine_map(raw_score, affine)
    return {
        "component": comp,
        "affine": affine,
        "raw_train": raw_train,
        "raw_score": raw_score,
        "pred_train": pred_train,
        "pred_score": pred_score,
        "family": family_name,
        "params": dict(params),
        "label": None,
    }


def evaluate_family_candidate_oof(
    family_name: str,
    cand: Mapping[str, Any],
    outer_train_df: pd.DataFrame,
    inner_folds: Sequence[Mapping[str, Any]],
    team_ids: Sequence[int],
    budget: BudgetTracker,
) -> Optional[Dict[str, Any]]:
    n = len(outer_train_df)
    y_all = outer_train_df["HomeWinMargin"].to_numpy(float)
    oof = np.full(n, np.nan, dtype=float)
    fold_id = np.full(n, -1, dtype=int)
    split_records = []

    for f in inner_folds:
        if budget.check():
            break
        tr_idx = np.asarray(f["train_idx"], dtype=int)
        va_idx = np.asarray(f["val_idx"], dtype=int)
        tr = outer_train_df.iloc[tr_idx]
        va = outer_train_df.iloc[va_idx]
        try:
            fit_obj = fit_family_and_build_raw_preds(
                family_name=family_name,
                params=cand["params"],
                fit_games=tr,
                score_rows=va,
                team_ids=team_ids,
                neutral_site_for_score=False,
                budget=budget,
            )
        except RuntimeError:
            break
        pred_val = np.asarray(fit_obj["pred_score"], dtype=float)
        oof[va_idx] = pred_val
        fold_id[va_idx] = int(f["fold"])
        split_records.append({"fold": int(f["fold"]), "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), **metric_dict(y_all[va_idx], pred_val)})

    valid = fold_id > 0
    if valid.sum() == 0:
        return None
    m = metric_dict(y_all[valid], oof[valid])
    return {
        "family": family_name,
        "label": cand["label"],
        "params": cand["params"],
        "oof_pred": oof,
        "oof_fold_id": fold_id,
        "valid_mask": valid,
        "metrics": m,
        "split_metrics": pd.DataFrame(split_records),
        "inner_folds_used": int(len(split_records)),
    }


def summarize_family_results(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append(
            {
                "family": r["family"],
                "label": r["label"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "bias": m["bias"],
                "pred_std": m["pred_std"],
                "actual_std": m["actual_std"],
                "dispersion_ratio": m["dispersion_ratio"],
                "n": m["n"],
                "inner_folds_used": r["inner_folds_used"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["family", "label", "rmse", "mae"])
    return pd.DataFrame(rows).sort_values(["family", "rmse", "mae", "label"], kind="mergesort").reset_index(drop=True)


def fit_ridge_stacker(X: np.ndarray, y: np.ndarray, alpha: float, budget: BudgetTracker) -> Dict[str, Any]:
    if not budget.can_fit(1):
        raise RuntimeError("Budget exhausted before stacker fit")
    mdl = Ridge(alpha=float(alpha), fit_intercept=True, random_state=SEED)
    mdl.fit(X, y)
    budget.add_fit(1)
    return {"alpha": float(alpha), "intercept": float(mdl.intercept_), "coef": np.asarray(mdl.coef_, dtype=float)}


def predict_ridge_stacker(model: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    return float(model["intercept"]) + np.asarray(X, dtype=float) @ np.asarray(model["coef"], dtype=float)


def _regime_features_for_rows(rows: pd.DataFrame, base_pred_df: pd.DataFrame, conf1_col: str, conf2_col: str) -> Tuple[np.ndarray, np.ndarray]:
    elo_col = "p_elo" if "p_elo" in base_pred_df.columns else list(base_pred_df.columns)[0]
    abs_elo = np.abs(base_pred_df[elo_col].to_numpy(float))
    sc = same_conf_series(rows, conf1_col, conf2_col)
    return abs_elo, sc


def fit_dynamic_regime_simplex(
    X: np.ndarray,
    y: np.ndarray,
    rows: pd.DataFrame,
    base_pred_df: pd.DataFrame,
    conf1_col: str,
    conf2_col: str,
) -> Dict[str, Any]:
    abs_elo, sc = _regime_features_for_rows(rows, base_pred_df, conf1_col, conf2_col)
    thresh = float(np.nanmedian(abs_elo)) if len(abs_elo) else 0.0
    global_w = fit_simplex_weights(X, y)
    regime_weights: Dict[Tuple[int, int], np.ndarray] = {}
    regime_counts: Dict[Tuple[int, int], int] = {}
    min_n = max(12, 3 * X.shape[1])
    for strength_bin in [0, 1]:
        for same_conf in [0, 1]:
            mask = ((abs_elo >= thresh).astype(int) == strength_bin) & (sc == same_conf)
            n = int(mask.sum())
            regime_counts[(strength_bin, same_conf)] = n
            regime_weights[(strength_bin, same_conf)] = fit_simplex_weights(X[mask], y[mask]) if n >= min_n else global_w.copy()
    return {"threshold_abs_elo": thresh, "global_weights": global_w, "regime_weights": regime_weights, "regime_counts": regime_counts}


def predict_dynamic_regime_simplex(
    model: Mapping[str, Any],
    X: np.ndarray,
    rows: pd.DataFrame,
    base_pred_df: pd.DataFrame,
    conf1_col: str,
    conf2_col: str,
) -> np.ndarray:
    abs_elo, sc = _regime_features_for_rows(rows, base_pred_df, conf1_col, conf2_col)
    thresh = float(model["threshold_abs_elo"])
    out = np.zeros(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        key = (int(abs_elo[i] >= thresh), int(sc[i]))
        w = np.asarray(model["regime_weights"].get(key, model["global_weights"]), dtype=float)
        out[i] = float(np.dot(X[i], w))
    return out


def build_combo_candidate_grid(base_cols: Sequence[str]) -> List[dict]:
    out: List[dict] = []
    for col in base_cols:
        for scale_on in [False, True]:
            out.append({"strategy_type": "single", "label": f"single::{col}::scale{int(scale_on)}", "family_col": col, "scale_on": scale_on})
    for scale_on in [False, True]:
        out.append({"strategy_type": "static_simplex", "label": f"static_simplex::scale{int(scale_on)}", "scale_on": scale_on})
    for scale_on in [False, True]:
        out.append({"strategy_type": "dynamic_regime", "label": f"dynamic_regime::scale{int(scale_on)}", "scale_on": scale_on})
    for alpha in STACKER_ALPHA_GRID:
        for scale_on in [False, True]:
            out.append({"strategy_type": "ridge_stacker", "label": f"ridge_stacker::alpha{alpha}::scale{int(scale_on)}", "alpha": float(alpha), "scale_on": scale_on})
    return out


def fit_combo_model_raw(
    cand: Mapping[str, Any],
    base_train_df: pd.DataFrame,
    y_train: np.ndarray,
    rows_train: pd.DataFrame,
    conf1_col: str,
    conf2_col: str,
    budget: BudgetTracker,
) -> Dict[str, Any]:
    X = base_train_df.to_numpy(float)
    if cand["strategy_type"] == "single":
        return {"strategy_type": "single", "family_col": cand["family_col"]}
    if cand["strategy_type"] == "static_simplex":
        return {"strategy_type": "static_simplex", "weights": fit_simplex_weights(X, y_train), "base_cols": list(base_train_df.columns)}
    if cand["strategy_type"] == "dynamic_regime":
        dyn = fit_dynamic_regime_simplex(X, y_train, rows_train, base_train_df, conf1_col, conf2_col)
        return {"strategy_type": "dynamic_regime", "base_cols": list(base_train_df.columns), **dyn}
    if cand["strategy_type"] == "ridge_stacker":
        stk = fit_ridge_stacker(X, y_train, cand["alpha"], budget)
        return {"strategy_type": "ridge_stacker", "base_cols": list(base_train_df.columns), **stk}
    raise ValueError(cand["strategy_type"])


def predict_combo_model_raw(
    model: Mapping[str, Any],
    base_df: pd.DataFrame,
    rows: pd.DataFrame,
    conf1_col: str,
    conf2_col: str,
) -> np.ndarray:
    if model["strategy_type"] == "single":
        return base_df[model["family_col"]].to_numpy(float)
    X = base_df.to_numpy(float)
    if model["strategy_type"] == "static_simplex":
        return X @ np.asarray(model["weights"], dtype=float)
    if model["strategy_type"] == "dynamic_regime":
        return predict_dynamic_regime_simplex(model, X, rows, base_df, conf1_col, conf2_col)
    if model["strategy_type"] == "ridge_stacker":
        return predict_ridge_stacker(model, X)
    raise ValueError(model["strategy_type"])


def evaluate_combo_candidates_meta(
    oof_base_df: pd.DataFrame,
    outer_train_rows: pd.DataFrame,
    y_outer_train: np.ndarray,
    inner_fold_id: np.ndarray,
    budget: BudgetTracker,
) -> Tuple[pd.DataFrame, List[dict]]:
    base_cols = [c for c in oof_base_df.columns if c.startswith("p_")]
    valid = (inner_fold_id > 0) & np.all(np.isfinite(oof_base_df[base_cols].to_numpy(float)), axis=1)
    meta_df = oof_base_df.loc[valid, base_cols].copy()
    rows_meta = outer_train_rows.loc[valid].copy()
    y_meta = np.asarray(y_outer_train[valid], dtype=float)
    fold_meta = np.asarray(inner_fold_id[valid], dtype=int)

    uniq_folds = sorted([int(x) for x in np.unique(fold_meta) if int(x) > 0])
    if len(uniq_folds) < 2:
        return pd.DataFrame(), []
    meta_splits = []
    for f in uniq_folds[1:]:
        tr_mask = fold_meta < f
        va_mask = fold_meta == f
        if tr_mask.sum() and va_mask.sum():
            meta_splits.append((tr_mask, va_mask, f))
    if not meta_splits:
        return pd.DataFrame(), []

    records = []
    states = []
    for cand in build_combo_candidate_grid(base_cols):
        preds = np.full(len(meta_df), np.nan, dtype=float)
        split_records = []
        failed = False
        for tr_mask, va_mask, meta_fold in meta_splits:
            base_tr = meta_df.loc[tr_mask]
            base_va = meta_df.loc[va_mask]
            rows_tr = rows_meta.loc[tr_mask]
            rows_va = rows_meta.loc[va_mask]
            y_tr = y_meta[tr_mask]
            y_va = y_meta[va_mask]
            try:
                model = fit_combo_model_raw(cand, base_tr, y_tr, rows_tr, "HomeConf", "AwayConf", budget)
            except RuntimeError:
                failed = True
                break
            raw_tr = predict_combo_model_raw(model, base_tr, rows_tr, "HomeConf", "AwayConf")
            raw_va = predict_combo_model_raw(model, base_va, rows_va, "HomeConf", "AwayConf")
            if cand.get("scale_on", False):
                scale_model = fit_scale_about_anchor(raw_tr, y_tr)
                pred_va = apply_scale_about_anchor(raw_va, scale_model)
            else:
                scale_model = None
                pred_va = raw_va
            preds[va_mask] = pred_va
            split_records.append({"meta_fold": int(meta_fold), **metric_dict(y_va, pred_va)})
        if failed:
            break
        good = np.isfinite(preds)
        if not good.any():
            continue
        m = metric_dict(y_meta[good], preds[good])
        records.append(
            {
                "label": cand["label"],
                "strategy_type": cand["strategy_type"],
                "scale_on": bool(cand.get("scale_on", False)),
                "family_col": cand.get("family_col"),
                "alpha": cand.get("alpha"),
                **m,
                "n_meta_splits": int(len(split_records)),
            }
        )
        states.append({"cand": cand, "metrics": m, "meta_pred": preds, "valid_mask": good, "split_metrics": pd.DataFrame(split_records)})
    if not records:
        return pd.DataFrame(), []
    return pd.DataFrame(records).sort_values(["rmse", "mae", "label"], kind="mergesort").reset_index(drop=True), states


def select_best_by_strategy(combo_meta_df: pd.DataFrame) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if combo_meta_df is None or len(combo_meta_df) == 0:
        return out
    for stype, g in combo_meta_df.groupby("strategy_type", sort=False):
        out[str(stype)] = g.sort_values(["rmse", "mae", "label"], kind="mergesort").iloc[0].to_dict()
    return out


def fit_selected_families_on_outer_train(
    outer_train_df: pd.DataFrame,
    outer_val_df: pd.DataFrame,
    selected_family_specs: Mapping[str, Mapping[str, Any]],
    team_ids: Sequence[int],
    budget: BudgetTracker,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
    train_pred_cols: Dict[str, np.ndarray] = {}
    val_pred_cols: Dict[str, np.ndarray] = {}
    family_fits: Dict[str, dict] = {}
    for family_name, spec in selected_family_specs.items():
        fit_obj = fit_family_and_build_raw_preds(
            family_name=family_name,
            params=spec["params"],
            fit_games=outer_train_df,
            score_rows=outer_val_df,
            team_ids=team_ids,
            neutral_site_for_score=False,
            budget=budget,
        )
        fit_obj["label"] = spec["label"]
        family_fits[family_name] = fit_obj
        train_pred_cols[f"p_{family_name}"] = np.asarray(fit_obj["pred_train"], dtype=float)
        val_pred_cols[f"p_{family_name}"] = np.asarray(fit_obj["pred_score"], dtype=float)
    return pd.DataFrame(train_pred_cols, index=outer_train_df.index), pd.DataFrame(val_pred_cols, index=outer_val_df.index), family_fits


def fit_combo_on_oof_and_predict(
    combo_row: Mapping[str, Any],
    oof_base_df: pd.DataFrame,
    outer_train_df: pd.DataFrame,
    inner_fold_id: np.ndarray,
    val_base_df: pd.DataFrame,
    outer_val_df: pd.DataFrame,
    budget: BudgetTracker,
) -> Dict[str, Any]:
    base_cols = [c for c in oof_base_df.columns if c.startswith("p_")]
    valid = (inner_fold_id > 0) & np.all(np.isfinite(oof_base_df[base_cols].to_numpy(float)), axis=1)
    if valid.sum() == 0:
        raise RuntimeError("No valid OOF rows to fit combo")
    base_train = oof_base_df.loc[valid, base_cols].copy()
    rows_train = outer_train_df.loc[valid].copy()
    y_train = outer_train_df.loc[valid, "HomeWinMargin"].to_numpy(float)
    cand = {"strategy_type": combo_row["strategy_type"], "family_col": combo_row.get("family_col"), "alpha": combo_row.get("alpha"), "scale_on": bool(combo_row.get("scale_on", False))}
    model = fit_combo_model_raw(cand, base_train, y_train, rows_train, "HomeConf", "AwayConf", budget)
    raw_train = predict_combo_model_raw(model, base_train, rows_train, "HomeConf", "AwayConf")
    raw_val = predict_combo_model_raw(model, val_base_df[base_cols], outer_val_df, "HomeConf", "AwayConf")
    if cand["scale_on"]:
        scale_model = fit_scale_about_anchor(raw_train, y_train)
        pred_train = apply_scale_about_anchor(raw_train, scale_model)
        pred_val = apply_scale_about_anchor(raw_val, scale_model)
    else:
        scale_model = None
        pred_train = raw_train
        pred_val = raw_val
    return {"model": model, "scale_model": scale_model, "pred_val": np.asarray(pred_val, dtype=float), "pred_train": np.asarray(pred_train, dtype=float), "train_valid_mask": valid}


def combine_outer_fold(
    fold: Mapping[str, Any],
    train_df: pd.DataFrame,
    team_ids: Sequence[int],
    budget: BudgetTracker,
) -> Dict[str, Any]:
    outer_train_df = train_df.iloc[np.asarray(fold["train_idx"], dtype=int)].copy().reset_index(drop=True)
    outer_val_df = train_df.iloc[np.asarray(fold["val_idx"], dtype=int)].copy().reset_index(drop=True)
    y_outer_train = outer_train_df["HomeWinMargin"].to_numpy(float)
    y_outer_val = outer_val_df["HomeWinMargin"].to_numpy(float)

    try:
        inner_folds = choose_folds(outer_train_df, INNER_FOLDS_PRIMARY, fallback=2)
    except Exception:
        inner_folds = choose_folds(outer_train_df, 2, fallback=None)

    grids = family_candidate_grid()
    family_eval_records: Dict[str, List[dict]] = {k: [] for k in grids}
    selected_family_specs: Dict[str, dict] = {}
    selected_family_oof: Dict[str, np.ndarray] = {}
    inner_fold_id_ref: Optional[np.ndarray] = None

    for family_name, candidates in grids.items():
        best: Optional[dict] = None
        for cand in candidates:
            if budget.check():
                break
            res = evaluate_family_candidate_oof(family_name, cand, outer_train_df, inner_folds, team_ids, budget)
            if res is None:
                continue
            family_eval_records[family_name].append(res)
            if best is None or (res["metrics"]["rmse"], res["metrics"]["mae"], res["label"]) < (best["metrics"]["rmse"], best["metrics"]["mae"], best["label"]):
                best = res
        if best is not None:
            selected_family_specs[family_name] = {"label": best["label"], "params": best["params"]}
            selected_family_oof[family_name] = np.asarray(best["oof_pred"], dtype=float)
            if inner_fold_id_ref is None:
                inner_fold_id_ref = np.asarray(best["oof_fold_id"], dtype=int)

    if "elo" not in selected_family_specs:
        raise RuntimeError("Baseline Elo not available; tuning stopped too early.")
    if "elok" not in selected_family_specs:
        selected_family_specs["elok"] = {"label": selected_family_specs["elo"]["label"], "params": selected_family_specs["elo"]["params"]}
        selected_family_oof["elok"] = selected_family_oof["elo"].copy()
    if inner_fold_id_ref is None:
        raise RuntimeError("No inner OOF fold ids available")

    oof_base_df = pd.DataFrame({f"p_{k}": v for k, v in selected_family_oof.items()}, index=outer_train_df.index)
    combo_meta_df, _combo_meta_states = evaluate_combo_candidates_meta(oof_base_df, outer_train_df, y_outer_train, inner_fold_id_ref, budget)
    best_by_strategy = select_best_by_strategy(combo_meta_df)

    train_base_insample, val_base, family_full_fits = fit_selected_families_on_outer_train(outer_train_df, outer_val_df, selected_family_specs, team_ids, budget)
    family_outer_metrics_df = pd.DataFrame([{"fold": int(fold["fold"]), "family_col": col, **metric_dict(y_outer_val, val_base[col].to_numpy(float))} for col in sorted(val_base.columns)])

    combo_outer_records = []
    combo_outer_preds: Dict[str, np.ndarray] = {}
    for stype, row in best_by_strategy.items():
        try:
            pred_obj = fit_combo_on_oof_and_predict(row, oof_base_df, outer_train_df, inner_fold_id_ref, val_base, outer_val_df, budget)
        except RuntimeError:
            continue
        pred_val = np.asarray(pred_obj["pred_val"], dtype=float)
        combo_outer_preds[stype] = pred_val
        combo_outer_records.append({"fold": int(fold["fold"]), "strategy_type": stype, "candidate_label": row["label"], **metric_dict(y_outer_val, pred_val)})
    combo_outer_df = pd.DataFrame(combo_outer_records)

    chosen_inner = combo_meta_df.iloc[0].to_dict() if len(combo_meta_df) else {"strategy_type": "single", "label": "single::p_elo::scale0", "family_col": "p_elo", "scale_on": False}
    chosen_stype = str(chosen_inner["strategy_type"])
    selected_outer_pred = combo_outer_preds.get(chosen_stype, val_base["p_elo"].to_numpy(float))
    selected_outer_metrics = metric_dict(y_outer_val, selected_outer_pred)

    return {
        "fold": int(fold["fold"]),
        "outer_fold_meta": {
            "train_end_date": str(pd.Timestamp(fold["train_end_date"]).date()),
            "val_start_date": str(pd.Timestamp(fold["val_start_date"]).date()),
            "val_end_date": str(pd.Timestamp(fold["val_end_date"]).date()),
            "n_train": int(len(outer_train_df)),
            "n_val": int(len(outer_val_df)),
            "n_inner_folds": int(len(inner_folds)),
        },
        "inner_folds_table": fold_table_rows(inner_folds),
        "selected_family_specs": selected_family_specs,
        "selected_family_oof_base": oof_base_df,
        "selected_family_inner_fold_id": inner_fold_id_ref,
        "family_tuning_df": summarize_family_results([x for v in family_eval_records.values() for x in v]),
        "family_outer_metrics_df": family_outer_metrics_df,
        "combo_meta_df": combo_meta_df,
        "combo_outer_df": combo_outer_df,
        "chosen_inner_candidate": chosen_inner,
        "selected_outer_pred": selected_outer_pred,
        "selected_outer_actual": y_outer_val,
        "selected_outer_metrics": selected_outer_metrics,
        "outer_val_rows": outer_val_df.copy(),
        "outer_train_rows": outer_train_df.copy(),
        "outer_train_oof_base": oof_base_df.copy(),
        "outer_train_base_insample": train_base_insample.copy(),
        "outer_val_base_preds": val_base.copy(),
        "family_full_fits": family_full_fits,
    }


def aggregate_outer_results(outer_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    family_rows = [r["family_outer_metrics_df"] for r in outer_results if len(r["family_outer_metrics_df"])]
    combo_rows = [r["combo_outer_df"] for r in outer_results if len(r["combo_outer_df"])]
    family_df = pd.concat(family_rows, ignore_index=True) if family_rows else pd.DataFrame()
    combo_df = pd.concat(combo_rows, ignore_index=True) if combo_rows else pd.DataFrame()
    selected_rows = []
    selected_pred_rows = []
    for r in outer_results:
        selected_rows.append({"fold": int(r["fold"]), **r["selected_outer_metrics"], "chosen_strategy_type": r["chosen_inner_candidate"].get("strategy_type"), "chosen_label": r["chosen_inner_candidate"].get("label")})
        tmp = r["outer_val_rows"][["GameID", "Date"]].copy()
        tmp["outer_fold"] = int(r["fold"])
        tmp["y_true"] = r["selected_outer_actual"]
        tmp["y_pred"] = r["selected_outer_pred"]
        selected_pred_rows.append(tmp)
    selected_fold_df = pd.DataFrame(selected_rows)
    selected_oof_df = pd.concat(selected_pred_rows, ignore_index=True).sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True) if selected_pred_rows else pd.DataFrame()

    def agg(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame()
        return (
            df.groupby(key_col, as_index=False)
            .agg(
                folds=("fold", "nunique"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                bias_mean=("bias", "mean"),
                pred_std_mean=("pred_std", "mean"),
                actual_std_mean=("actual_std", "mean"),
                dispersion_ratio_mean=("dispersion_ratio", "mean"),
            )
            .rename(columns={key_col: "name"})
            .sort_values(["rmse_mean", "mae_mean", "name"], kind="mergesort")
            .reset_index(drop=True)
        )

    return {
        "family_fold_metrics": family_df,
        "family_agg": agg(family_df.rename(columns={"family_col": "name"}), "name") if len(family_df) else pd.DataFrame(),
        "combo_fold_metrics": combo_df,
        "combo_agg": agg(combo_df.rename(columns={"strategy_type": "name"}), "name") if len(combo_df) else pd.DataFrame(),
        "selected_fold_metrics": selected_fold_df,
        "selected_oof_df": selected_oof_df,
        "selected_oof_metrics": metric_dict(selected_oof_df["y_true"], selected_oof_df["y_pred"]) if len(selected_oof_df) else {},
    }


def choose_final_strategy_from_outer(outer_agg: Mapping[str, Any]) -> Dict[str, Any]:
    combo_agg = outer_agg.get("combo_agg", pd.DataFrame())
    if combo_agg is None or len(combo_agg) == 0:
        return {"strategy_type": "single", "force_scale_policy": False}
    best = combo_agg.sort_values(["rmse_mean", "mae_mean", "name"], kind="mergesort").iloc[0]
    strategy_type = str(best["name"])
    force_scale_policy = None
    combo_fold = outer_agg.get("combo_fold_metrics", pd.DataFrame())
    if combo_fold is not None and len(combo_fold):
        tmp = combo_fold[combo_fold["strategy_type"] == strategy_type].copy()
        if len(tmp):
            tmp["scale_on"] = tmp["candidate_label"].astype(str).str.contains("scale1")
            g = tmp.groupby("scale_on", as_index=False).agg(rmse_mean=("rmse", "mean"))
            if set(g["scale_on"].tolist()) == {False, True}:
                force_scale_policy = bool(g.sort_values(["rmse_mean", "scale_on"], kind="mergesort").iloc[0]["scale_on"])
    return {"strategy_type": strategy_type, "force_scale_policy": force_scale_policy}


def tune_on_full_train_for_final(
    train_df: pd.DataFrame,
    team_ids: Sequence[int],
    target_strategy_type: str,
    force_scale_policy: Optional[bool],
    budget: BudgetTracker,
) -> Dict[str, Any]:
    inner_folds = choose_folds(train_df, INNER_FOLDS_PRIMARY, fallback=2)
    y = train_df["HomeWinMargin"].to_numpy(float)
    grids = family_candidate_grid()
    family_eval_records: Dict[str, List[dict]] = {k: [] for k in grids}
    selected_family_specs: Dict[str, dict] = {}
    selected_family_oof: Dict[str, np.ndarray] = {}
    inner_fold_id_ref: Optional[np.ndarray] = None

    for family_name, candidates in grids.items():
        best = None
        for cand in candidates:
            if budget.check():
                break
            res = evaluate_family_candidate_oof(family_name, cand, train_df, inner_folds, team_ids, budget)
            if res is None:
                continue
            family_eval_records[family_name].append(res)
            if best is None or (res["metrics"]["rmse"], res["metrics"]["mae"], res["label"]) < (best["metrics"]["rmse"], best["metrics"]["mae"], best["label"]):
                best = res
        if best is not None:
            selected_family_specs[family_name] = {"label": best["label"], "params": best["params"]}
            selected_family_oof[family_name] = np.asarray(best["oof_pred"], dtype=float)
            if inner_fold_id_ref is None:
                inner_fold_id_ref = np.asarray(best["oof_fold_id"], dtype=int)
    if "elo" not in selected_family_specs:
        raise RuntimeError("Final tuning failed before baseline Elo")
    if "elok" not in selected_family_specs:
        selected_family_specs["elok"] = {"label": selected_family_specs["elo"]["label"], "params": selected_family_specs["elo"]["params"]}
        selected_family_oof["elok"] = selected_family_oof["elo"].copy()
    if inner_fold_id_ref is None:
        raise RuntimeError("No inner fold ids for final tuning")

    oof_base_df = pd.DataFrame({f"p_{k}": v for k, v in selected_family_oof.items()}, index=train_df.index)
    combo_meta_df, _ = evaluate_combo_candidates_meta(oof_base_df, train_df, y, inner_fold_id_ref, budget)
    if len(combo_meta_df):
        combo_meta_df = combo_meta_df[combo_meta_df["strategy_type"] == target_strategy_type].copy()
        if force_scale_policy is not None:
            combo_meta_df = combo_meta_df[combo_meta_df["scale_on"].astype(bool) == bool(force_scale_policy)]
    combo_choice = combo_meta_df.sort_values(["rmse", "mae", "label"], kind="mergesort").iloc[0].to_dict() if len(combo_meta_df) else {"strategy_type": "single", "label": "single::p_elo::scale0", "family_col": "p_elo", "scale_on": False}

    return {
        "inner_folds": inner_folds,
        "selected_family_specs": selected_family_specs,
        "oof_base_df": oof_base_df,
        "inner_fold_id": inner_fold_id_ref,
        "combo_meta_df": combo_meta_df,
        "combo_choice": combo_choice,
        "family_tuning_df": summarize_family_results([x for v in family_eval_records.values() for x in v]),
    }


def fit_final_and_predict_derby(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    rankings_template: pd.DataFrame,
    final_tuning: Mapping[str, Any],
    team_ids: Sequence[int],
    budget: BudgetTracker,
) -> Dict[str, Any]:
    selected_family_specs = final_tuning["selected_family_specs"]
    train_base_cols = {}
    derby_base_cols = {}
    family_fits = {}
    for family_name, spec in selected_family_specs.items():
        fit_obj = fit_family_and_build_raw_preds(family_name, spec["params"], train_df, pred_df, team_ids, True, budget)
        fit_obj["label"] = spec["label"]
        family_fits[family_name] = fit_obj
        train_base_cols[f"p_{family_name}"] = np.asarray(fit_obj["pred_train"], dtype=float)
        derby_base_cols[f"p_{family_name}"] = np.asarray(fit_obj["pred_score"], dtype=float)
    train_base_insample = pd.DataFrame(train_base_cols, index=train_df.index)
    derby_base = pd.DataFrame(derby_base_cols, index=pred_df.index)

    combo_choice = final_tuning["combo_choice"]
    derby_rows_as_homeaway = pred_df.rename(columns={"Team1_Conf": "HomeConf", "Team2_Conf": "AwayConf"}).copy()
    combo_fit = fit_combo_on_oof_and_predict(combo_choice, final_tuning["oof_base_df"], train_df, final_tuning["inner_fold_id"], derby_base, derby_rows_as_homeaway, budget)
    derby_pred = np.asarray(combo_fit["pred_val"], dtype=float)
    derby_pred_round = np.rint(derby_pred).astype(int)
    pred_out = pred_df.copy()
    pred_out["Team1_WinMargin"] = derby_pred_round

    component_strengths = {fam: component_strength_map_from_fit(fit_obj, fam) for fam, fit_obj in family_fits.items()}
    rank_out = rankings_template.copy()
    rank_out["TeamID"] = rank_out["TeamID"].astype(int)

    if combo_choice.get("strategy_type") == "single":
        fam_name = str(combo_choice.get("family_col", "p_elo")).replace("p_", "")
        score = rank_out["TeamID"].map(component_strengths.get(fam_name, {})).fillna(0.0).astype(float)
    else:
        cols = [c for c in derby_base.columns if c.startswith("p_")]
        fam_names = [c.replace("p_", "") for c in cols]
        oof_valid = final_tuning["inner_fold_id"] > 0
        oof_train_base = final_tuning["oof_base_df"].loc[oof_valid, cols]
        oof_rows = train_df.loc[oof_valid]
        oof_y = train_df.loc[oof_valid, "HomeWinMargin"].to_numpy(float)
        combo_train_model = fit_combo_model_raw(combo_choice, oof_train_base, oof_y, oof_rows, "HomeConf", "AwayConf", budget)
        if combo_choice.get("strategy_type") == "static_simplex":
            raw_w = np.asarray(combo_train_model["weights"], dtype=float)
        elif combo_choice.get("strategy_type") == "dynamic_regime":
            mats = np.vstack([np.asarray(w, dtype=float) for w in combo_train_model["regime_weights"].values()])
            raw_w = np.mean(mats, axis=0)
        elif combo_choice.get("strategy_type") == "ridge_stacker":
            raw_w = np.asarray(combo_train_model["coef"], dtype=float)
        else:
            raw_w = np.full(len(cols), 1.0 / max(1, len(cols)))
        if not np.isfinite(raw_w).all() or np.sum(np.abs(raw_w)) <= 1e-12:
            raw_w = np.full(len(cols), 1.0 / max(1, len(cols)))
        w = raw_w / (np.sum(np.abs(raw_w)) if np.sum(np.abs(raw_w)) > 1e-12 else 1.0)
        team_arr = rank_out["TeamID"].to_numpy(int)
        zscore = np.zeros(len(team_arr), dtype=float)
        for i, fam in enumerate(fam_names):
            vals = np.array([float(component_strengths.get(fam, {}).get(int(t), 0.0)) for t in team_arr], dtype=float)
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            zscore += float(w[i]) * ((vals - mu) / (sd if sd > 1e-12 else 1.0))
        score = pd.Series(zscore, index=rank_out.index)

    rank_out["_score"] = np.asarray(score, dtype=float)
    rank_out = rank_out.sort_values(["_score", "TeamID"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    rank_out["Rank"] = np.arange(1, len(rank_out) + 1, dtype=int)
    rank_out = rank_out.drop(columns=["_score"])
    first_cols = [c for c in ["TeamID", "Team", "Rank"] if c in rank_out.columns]
    rank_out = rank_out[first_cols + [c for c in rank_out.columns if c not in first_cols]]

    derby_summary = {
        "mean": float(np.mean(derby_pred)),
        "std": float(np.std(derby_pred)),
        "min": float(np.min(derby_pred)),
        "q05": float(np.quantile(derby_pred, 0.05)),
        "q25": float(np.quantile(derby_pred, 0.25)),
        "median": float(np.quantile(derby_pred, 0.50)),
        "q75": float(np.quantile(derby_pred, 0.75)),
        "q95": float(np.quantile(derby_pred, 0.95)),
        "max": float(np.max(derby_pred)),
    }
    return {
        "pred_out": pred_out,
        "rank_out": rank_out,
        "derby_pred_raw": derby_pred,
        "derby_pred_round": derby_pred_round,
        "derby_summary": derby_summary,
        "family_fits": family_fits,
        "combo_choice": combo_choice,
        "combo_fit": combo_fit,
        "train_base_insample": train_base_insample,
        "derby_base": derby_base,
        "component_strengths": component_strengths,
    }


def git_short_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def draw_text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    wrapped: List[str] = []
    for line in lines:
        if line == "":
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(str(line), width=108) or [""])
    fig.text(0.04, 0.96, "\n".join(wrapped), va="top", ha="left", family="monospace", fontsize=9.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows: int = 28, float_cols: Optional[Sequence[str]] = None) -> None:
    float_cols = list(float_cols or [])
    show = df.head(max_rows).copy()
    for c in float_cols:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if len(show) == 0:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center")
    else:
        tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.15)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_final_report(
    pdf_path: Path,
    outer_folds: Sequence[Mapping[str, Any]],
    outer_results: Sequence[Mapping[str, Any]],
    outer_agg: Mapping[str, Any],
    final_strategy: Mapping[str, Any],
    final_tuning: Mapping[str, Any],
    final_fit: Mapping[str, Any],
    budget: BudgetTracker,
    validations: Mapping[str, Any],
) -> None:
    family_agg = outer_agg.get("family_agg", pd.DataFrame())
    combo_agg = outer_agg.get("combo_agg", pd.DataFrame())
    selected_oof_df = outer_agg.get("selected_oof_df", pd.DataFrame())
    selected_fold = outer_agg.get("selected_fold_metrics", pd.DataFrame())

    leakage_lines = [
        "Leakage prevention decisions:",
        "1) Outer evaluation is expanding-window by Date. Validation blocks are strictly in the future of their fold training blocks.",
        "2) Inner tuning is also expanding-window inside each outer-train.",
        "3) Ratings are fit only on split-train before scoring split-val (no full-season ratings reused inside CV).",
        "4) Rating-to-margin mappings (a + b*diff) are fit only on split-train.",
        "5) Ensemble weights, dynamic regime weights, and stacker are tuned only from inner-CV outputs.",
        "6) Stacker uses inner OOF base predictions to avoid in-sample base-pred leakage.",
        "7) Derby labels are never used anywhere (unavailable).",
        "",
        "Runtime safety / determinism:",
        f"- Threads capped to 1 (OMP/MKL/OPENBLAS/NUMEXPR): {os.getenv('OMP_NUM_THREADS')}/{os.getenv('MKL_NUM_THREADS')}/{os.getenv('OPENBLAS_NUM_THREADS')}/{os.getenv('NUMEXPR_NUM_THREADS')}",
        f"- Budgets: total={budget.max_total}s, tuning={budget.max_tuning}s, model_fits={budget.max_fits}",
        f"- Usage: total_elapsed={budget.total_elapsed():.2f}s, tuning_elapsed={budget.tuning_elapsed():.2f}s, fit_count={budget.fit_count}, stop_reason={budget.stop_reason}",
    ]

    rating_tbl = pd.DataFrame(
        [
            {"System": "Elo", "Prediction signal": "pregame Elo diff (+ fixed home adv)", "Hyperparams": "fixed HA=50, K=24, MOV mult on"},
            {"System": "Elo k-decay", "Prediction signal": "pregame Elo diff with season-indexed K multiplier", "Hyperparams": "decay type / A / G or tau (small grid)"},
            {"System": "Massey ridge", "Prediction signal": "ridge team-rating margin model", "Hyperparams": "ridge alpha grid"},
            {"System": "Colley", "Prediction signal": "W/L-only rating diff, then affine map to margin", "Hyperparams": "none + fold-train affine map"},
            {"System": "Off/Def net ridge", "Prediction signal": "offense/defense point model -> margin", "Hyperparams": "fixed alpha"},
        ]
    )

    with PdfPages(pdf_path) as pdf:
        draw_text_page(pdf, "AlgoSports23 Final Report (Nested Time-Aware CV)", leakage_lines)
        draw_text_page(
            pdf,
            "Outer / Inner CV Scheme",
            [
                "Outer folds: 5 expanding-window by Date (fallback to 4 if needed).",
                "Inner folds (within outer-train): 3 expanding-window splits (fallback to 2 if early outer fold has too few dates).",
                "",
                "ASCII diagram:",
                "Outer fold i: [ train ................................ ] [ validate next block ]",
                "Inner split 1: [train] [val]",
                "Inner split 2: [train ........] [val]",
                "Inner split 3: [train ................] [val]",
                "",
                "Combination tuning uses meta-expanding splits over inner OOF rows (fold IDs), preserving time order.",
            ],
        )
        draw_table_page(pdf, "Outer CV Fold Table", fold_table_rows(outer_folds))
        draw_table_page(pdf, "Rating Systems Implemented", rating_tbl)
        if len(family_agg):
            draw_table_page(pdf, "Single-System Performance (Outer Folds)", family_agg, float_cols=["rmse_mean", "rmse_std", "mae_mean", "mae_std", "bias_mean", "pred_std_mean", "actual_std_mean", "dispersion_ratio_mean"])
        if len(combo_agg):
            draw_table_page(pdf, "Combination Performance (Outer Folds)", combo_agg, float_cols=["rmse_mean", "rmse_std", "mae_mean", "mae_std", "bias_mean", "pred_std_mean", "actual_std_mean", "dispersion_ratio_mean"])
        if len(selected_fold):
            draw_table_page(pdf, "Per-Outer-Fold Selected Model + Metrics", selected_fold, float_cols=["rmse", "mae", "bias", "pred_std", "actual_std", "dispersion_ratio"])

        for r in outer_results[:2]:
            draw_table_page(pdf, f"Outer Fold {r['fold']} Inner Family Tuning (Top)", r["family_tuning_df"].sort_values(["rmse", "mae"], kind="mergesort"), max_rows=14, float_cols=["rmse", "mae", "bias", "pred_std", "actual_std", "dispersion_ratio"])
            if len(r["combo_meta_df"]):
                draw_table_page(pdf, f"Outer Fold {r['fold']} Inner Combo Tuning (Top)", r["combo_meta_df"], max_rows=14, float_cols=["rmse", "mae", "bias", "pred_std", "actual_std", "dispersion_ratio"])

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        if len(selected_oof_df):
            y_true = selected_oof_df["y_true"].to_numpy(float)
            y_pred = selected_oof_df["y_pred"].to_numpy(float)
            resid = y_true - y_pred
            lo = float(min(np.min(y_true), np.min(y_pred)))
            hi = float(max(np.max(y_true), np.max(y_pred)))
            axes[0, 0].scatter(y_pred, y_true, s=12, alpha=0.6)
            axes[0, 0].plot([lo, hi], [lo, hi], color="black", lw=1)
            axes[0, 0].set_title("Selected Outer OOF: Pred vs Actual")
            axes[0, 1].hist(resid, bins=30, color="#4C78A8", alpha=0.85)
            axes[0, 1].axvline(0, color="black", lw=1)
            axes[0, 1].set_title("Residual Histogram")
            axes[1, 0].scatter(y_pred, resid, s=12, alpha=0.6, color="#F58518")
            axes[1, 0].axhline(0, color="black", lw=1)
            axes[1, 0].set_title("Residual vs Fitted")
            axes[1, 1].hist(y_pred, bins=30, color="#54A24B", alpha=0.85)
            axes[1, 1].set_title("Selected OOF Predictions")
        else:
            for ax in axes.ravel():
                ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        if len(selected_oof_df):
            y_true = selected_oof_df["y_true"].to_numpy(float)
            y_pred = selected_oof_df["y_pred"].to_numpy(float)
            vals = [float(np.std(y_pred)), float(np.std(y_true))]
            axes[0].bar(["pred std", "actual std"], vals, color=["#4C78A8", "#E45756"])
            axes[0].set_title("Dispersion Diagnostic (Outer OOF)")
        if len(combo_agg):
            show = combo_agg.head(10)
            axes[1].bar(show["name"], show["rmse_mean"], color="#72B7B2")
            axes[1].tick_params(axis="x", rotation=35)
            axes[1].set_title("Outer RMSE by Combination Strategy")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        final_lines = [
            "Selected final approach (based on outer CV):",
            f"- strategy_type: {final_strategy['strategy_type']}",
            f"- force_scale_policy: {final_strategy.get('force_scale_policy')}",
            "",
            "Final full-train tuning (time-aware, no derby labels):",
            f"- combo_choice: {json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in final_tuning['combo_choice'].items()}, default=str)}",
            f"- selected families: {json.dumps({k: v['label'] for k, v in final_tuning['selected_family_specs'].items()})}",
            "",
            "Derby prediction distribution (raw before rounding):",
            json.dumps(final_fit["derby_summary"], indent=2),
            "",
            "Output validations:",
            f"- predictions.csv rows={validations['pred_rows']} no_missing={validations['pred_missing']==0} numeric={validations['pred_numeric']}",
            f"- rankings.xlsx rows={validations['rank_rows']} rank_set_valid={validations['rank_set_valid']}",
            f"- final_report.pdf exists={validations['pdf_exists']} size_bytes={validations['pdf_size']}",
        ]
        draw_text_page(pdf, "Final Selection, Derby Summary, Validations", final_lines)


def validate_outputs(pred_path: Path, rank_path: Path, pdf_path: Path) -> Dict[str, Any]:
    p = pd.read_csv(pred_path)
    r = pd.read_excel(rank_path)
    pred_numeric = bool(pd.api.types.is_numeric_dtype(p["Team1_WinMargin"]))
    pred_missing = int(p["Team1_WinMargin"].isna().sum())
    rank_vals = pd.to_numeric(r["Rank"], errors="coerce")
    rank_missing = int(rank_vals.isna().sum())
    rank_set_valid = bool(set(rank_vals.dropna().astype(int).tolist()) == set(range(1, len(r) + 1)))
    pdf_exists = bool(pdf_path.exists())
    pdf_size = int(pdf_path.stat().st_size) if pdf_exists else 0
    return {
        "pred_rows": int(len(p)),
        "pred_missing": pred_missing,
        "pred_numeric": pred_numeric,
        "rank_rows": int(len(r)),
        "rank_missing": rank_missing,
        "rank_set_valid": rank_set_valid,
        "pdf_exists": pdf_exists,
        "pdf_size": pdf_size,
    }


def print_hard_stop_validation(pred_path: Path, rank_path: Path, pdf_path: Path) -> Dict[str, Any]:
    v = validate_outputs(pred_path, rank_path, pdf_path)
    p = pd.read_csv(pred_path)
    r = pd.read_excel(rank_path)
    print("\n=== HARD STOP VALIDATIONS ===")
    print(f"predictions.csv: {v['pred_rows']} rows, Team1_WinMargin numeric={v['pred_numeric']}, no missing={v['pred_missing']==0}")
    print(f"rankings.xlsx: {v['rank_rows']} rows, Rank exactly {{1..165}} => {v['rank_set_valid']}")
    print(f"final_report.pdf exists and size>0 => {v['pdf_exists'] and v['pdf_size']>0} (size={v['pdf_size']})")
    print("\nHead(10) predictions:")
    print(p.head(10).to_string(index=False))
    print("\nTop 10 rankings:")
    print(r.sort_values("Rank", kind="mergesort").head(10).to_string(index=False))
    if v["pred_rows"] != 75 or v["pred_missing"] != 0 or not v["pred_numeric"]:
        raise ValueError("predictions.csv validation failed")
    if v["rank_rows"] != 165 or v["rank_missing"] != 0 or not v["rank_set_valid"]:
        raise ValueError("rankings.xlsx validation failed")
    if not v["pdf_exists"] or v["pdf_size"] <= 0:
        raise ValueError("final_report.pdf validation failed")
    return v


def write_run_report(
    path: Path,
    budget: BudgetTracker,
    outer_results: Sequence[Mapping[str, Any]],
    outer_agg: Mapping[str, Any],
    final_strategy: Mapping[str, Any],
    final_tuning: Mapping[str, Any],
    final_fit: Mapping[str, Any],
    validations: Mapping[str, Any],
) -> None:
    family_agg = outer_agg.get("family_agg", pd.DataFrame())
    combo_agg = outer_agg.get("combo_agg", pd.DataFrame())
    selected_fold = outer_agg.get("selected_fold_metrics", pd.DataFrame())
    selected_oof_metrics = outer_agg.get("selected_oof_metrics", {})
    chosen = {
        "strategy_type": final_strategy["strategy_type"],
        "force_scale_policy": final_strategy.get("force_scale_policy"),
        "final_combo_choice": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in final_tuning["combo_choice"].items()},
        "selected_family_labels": {k: v["label"] for k, v in final_tuning["selected_family_specs"].items()},
        "selected_family_params": final_tuning["selected_family_specs"],
        "HOME_ADV": HOME_ADV,
        "ELO_K": ELO_K,
    }
    lines = [
        "# run_report",
        "",
        "- date: 2026-02-26",
        f"- git commit hash: `{git_short_hash()}`",
        f"- git branch: `{git_branch()}`",
        "",
        "## budgets used + stop reason",
        f"- ALGOSPORTS_MAX_TOTAL_SECONDS: {budget.max_total}",
        f"- ALGOSPORTS_MAX_TUNING_SECONDS: {budget.max_tuning}",
        f"- ALGOSPORTS_MAX_MODEL_FITS: {budget.max_fits}",
        f"- total_elapsed_seconds: {budget.total_elapsed():.3f}",
        f"- tuning_elapsed_seconds: {budget.tuning_elapsed():.3f}",
        f"- model_fit_count: {budget.fit_count}",
        f"- stop_reason: {budget.stop_reason}",
        f"- stop_events: {json.dumps(budget.stop_events)}",
        "",
        "## chosen system/weights/hyperparams",
        "```json",
        json.dumps(chosen, indent=2, default=str),
        "```",
        "",
        "## outer CV RMSE/MAE and dispersion metrics",
        f"- selected_oof_rmse: {selected_oof_metrics.get('rmse')}",
        f"- selected_oof_mae: {selected_oof_metrics.get('mae')}",
        f"- selected_oof_bias: {selected_oof_metrics.get('bias')}",
        f"- selected_oof_pred_std: {selected_oof_metrics.get('pred_std')}",
        f"- selected_oof_actual_std: {selected_oof_metrics.get('actual_std')}",
        f"- selected_oof_dispersion_ratio: {selected_oof_metrics.get('dispersion_ratio')}",
        "",
        "## output validations",
        f"- predictions.csv rows: {validations['pred_rows']}",
        f"- predictions.csv missing Team1_WinMargin: {validations['pred_missing']}",
        f"- predictions.csv numeric Team1_WinMargin: {validations['pred_numeric']}",
        f"- rankings.xlsx rows: {validations['rank_rows']}",
        f"- rankings.xlsx missing Rank: {validations['rank_missing']}",
        f"- rankings.xlsx rank_set_valid: {validations['rank_set_valid']}",
        f"- final_report.pdf exists: {validations['pdf_exists']}",
        f"- final_report.pdf size_bytes: {validations['pdf_size']}",
        "",
        "## derby prediction distribution summary",
        "```json",
        json.dumps(final_fit["derby_summary"], indent=2),
        "```",
        "",
    ]
    if len(family_agg):
        lines += ["## single-system outer summary", "```", family_agg.to_string(index=False), "```", ""]
    if len(combo_agg):
        lines += ["## combination outer summary", "```", combo_agg.to_string(index=False), "```", ""]
    if len(selected_fold):
        lines += ["## per-outer-fold selected metrics", "```", selected_fold.to_string(index=False), "```", ""]
    lines += [
        "## final tuning combo candidates (top 15)",
        "```",
        (final_tuning["combo_meta_df"].head(15).to_string(index=False) if len(final_tuning["combo_meta_df"]) else "No rows"),
        "```",
        "",
        "## final tuning family candidates (top 20)",
        "```",
        (final_tuning["family_tuning_df"].sort_values(["rmse", "mae"], kind="mergesort").head(20).to_string(index=False) if len(final_tuning["family_tuning_df"]) else "No rows"),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    seed_all(SEED)
    budget = BudgetTracker(
        start=time.perf_counter(),
        max_total=env_float("ALGOSPORTS_MAX_TOTAL_SECONDS", 420.0),
        max_tuning=env_float("ALGOSPORTS_MAX_TUNING_SECONDS", 180.0),
        max_fits=env_int("ALGOSPORTS_MAX_MODEL_FITS", 250),
    )
    budget.begin_tuning()

    print("=== AlgoSports23 Nested Time-Aware Rating Framework ===")
    print(f"Thread caps: OMP={os.getenv('OMP_NUM_THREADS')} MKL={os.getenv('MKL_NUM_THREADS')} OPENBLAS={os.getenv('OPENBLAS_NUM_THREADS')} NUMEXPR={os.getenv('NUMEXPR_NUM_THREADS')}")
    print(f"Budgets: TOTAL={budget.max_total}s TUNING={budget.max_tuning}s MODEL_FITS={budget.max_fits}")
    print(f"Seed={SEED}; HOME_ADV fixed={HOME_ADV}; ELO_K fixed={ELO_K}")

    train_path, pred_template_path, rankings_template_path = ensure_root_inputs(ROOT)
    train_df = parse_and_sort_train(pd.read_csv(train_path))
    pred_df = parse_predictions(pd.read_csv(pred_template_path))
    rankings_template = pd.read_excel(rankings_template_path)
    team_ids = sorted(set(train_df["HomeID"].astype(int)) | set(train_df["AwayID"].astype(int)))

    try:
        outer_folds = choose_folds(train_df, OUTER_FOLDS_PRIMARY, fallback=OUTER_FOLDS_FALLBACK)
    except Exception:
        outer_folds = choose_folds(train_df, OUTER_FOLDS_FALLBACK, fallback=None)
    print("Outer folds:")
    print(fold_table_rows(outer_folds).to_string(index=False))

    outer_results = []
    for fold in outer_folds:
        if budget.check():
            print(f"Stopping outer CV early due to budget: {budget.stop_reason}")
            break
        print(f"\n[Outer fold {fold['fold']}] start")
        res = combine_outer_fold(fold, train_df, team_ids, budget)
        outer_results.append(res)
        m = res["selected_outer_metrics"]
        print(f"[Outer fold {fold['fold']}] chosen(inner)={res['chosen_inner_candidate'].get('label')} outer RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} pred_std={m['pred_std']:.2f} actual_std={m['actual_std']:.2f}")

    if not outer_results:
        raise RuntimeError("No outer folds completed")

    outer_agg = aggregate_outer_results(outer_results)
    final_strategy = choose_final_strategy_from_outer(outer_agg)
    print(f"\nSelected strategy from outer CV: {final_strategy}")
    if len(outer_agg.get("combo_agg", pd.DataFrame())):
        print(outer_agg["combo_agg"].head(10).to_string(index=False))

    final_tuning = tune_on_full_train_for_final(train_df, team_ids, final_strategy["strategy_type"], final_strategy.get("force_scale_policy"), budget)
    print(f"Final full-train combo choice: {final_tuning['combo_choice']}")
    artifact_budget = BudgetTracker(
        start=budget.start,
        max_total=10**9,
        max_tuning=10**9,
        max_fits=10**9,
        fit_count=0,
    )
    artifact_budget.begin_tuning()
    final_fit = fit_final_and_predict_derby(train_df, pred_df, rankings_template, final_tuning, team_ids, artifact_budget)

    pred_out_path = ROOT / "predictions.csv"
    rank_out_path = ROOT / "rankings.xlsx"
    pdf_out_path = ROOT / "final_report.pdf"
    run_report_path = ROOT / "run_report.md"
    final_fit["pred_out"].to_csv(pred_out_path, index=False)
    final_fit["rank_out"].to_excel(rank_out_path, index=False)

    validations_pre = validate_outputs(pred_out_path, rank_out_path, pdf_out_path)
    validations_pre["pdf_exists"] = False
    validations_pre["pdf_size"] = 0
    generate_final_report(pdf_out_path, outer_folds[: len(outer_results)], outer_results, outer_agg, final_strategy, final_tuning, final_fit, budget, validations_pre)
    validations = print_hard_stop_validation(pred_out_path, rank_out_path, pdf_out_path)
    generate_final_report(pdf_out_path, outer_folds[: len(outer_results)], outer_results, outer_agg, final_strategy, final_tuning, final_fit, budget, validations)
    validations = print_hard_stop_validation(pred_out_path, rank_out_path, pdf_out_path)
    write_run_report(run_report_path, budget, outer_results, outer_agg, final_strategy, final_tuning, final_fit, validations)

    print(f"\nArtifacts written: {pred_out_path.name}, {rank_out_path.name}, {pdf_out_path.name}, {run_report_path.name}")
    print(f"Total elapsed seconds: {budget.total_elapsed():.2f}")
    if budget.stop_reason:
        print(f"Budget stop encountered: {budget.stop_reason} (best-so-far artifacts produced)")


if __name__ == "__main__":
    main()
