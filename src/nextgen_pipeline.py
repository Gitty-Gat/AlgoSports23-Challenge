from __future__ import annotations

import json
import math
import os
import random
import re
import textwrap
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
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
from src.ratings import EloVariantConfig


SEED_DEFAULT = 23
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
MEAN_MODEL_COLS = ["pred_ridge", "pred_huber", "pred_histgb", "pred_histgb_bag"]


@dataclass(frozen=True)
class CoreCandidate:
    elo_variant: str
    feature_profile: str
    half_life_days: Optional[int]
    ridge_alpha: float
    histgb_idx: int

    def key(self) -> str:
        hl = "none" if self.half_life_days is None else str(int(self.half_life_days))
        return f"{self.elo_variant}|{self.feature_profile}|hl={hl}|ridge={self.ridge_alpha:g}|hgb={self.histgb_idx}"


@dataclass(frozen=True)
class PostprocessCandidate:
    regime_stack: bool
    calibration_mode: str
    scale_mode: str
    q50_blend: float
    winsor_q: float

    def key(self) -> str:
        return (
            f"stack={'reg' if self.regime_stack else 'pool'}|cal={self.calibration_mode}|"
            f"scale={self.scale_mode}|q50={self.q50_blend:.2f}|win={self.winsor_q:.3f}"
        )


@dataclass
class PostprocessModel:
    candidate: PostprocessCandidate
    thresholds: dict
    stack_fit: dict
    cal_fit: dict
    scale_fit: dict
    winsor_bounds: tuple[float, float]
    regime_summary: pd.DataFrame
    calibration_summary: pd.DataFrame
    scale_summary: pd.DataFrame


@dataclass
class RuntimeProfiler:
    totals: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    model_fit_counts: Dict[str, int] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)

    def add(self, phase: str, elapsed_sec: float, *, detail: Optional[str] = None) -> None:
        if not np.isfinite(elapsed_sec):
            return
        self.totals[phase] = float(self.totals.get(phase, 0.0) + float(elapsed_sec))
        self.counts[phase] = int(self.counts.get(phase, 0) + 1)
        self.events.append({"phase": str(phase), "detail": "" if detail is None else str(detail), "sec": float(elapsed_sec)})

    def add_model_fit(self, family: str, n: int = 1) -> None:
        self.model_fit_counts[str(family)] = int(self.model_fit_counts.get(str(family), 0) + int(n))


@dataclass
class RuntimeConfig:
    max_total_seconds: int = 600
    max_scan_seconds: int = 120
    max_fits: int = 400
    fast_mode: bool = True
    enable_optimizations: bool = True
    histgb_max_iter_cap: int = 60
    histgb_bag_n_models: int = 2
    scan_prune_keep_top_n: int = 25
    scan_prune_keep_frac: float = 0.35
    scan_top_candidates_return: int = 3
    fast_outer_folds: int = 1
    fast_inner_folds: int = 1
    fast_scan_topn: int = 20
    fast_disable_q20_q80: bool = True
    fast_post_grid_limit: int = 8


@dataclass
class BudgetController:
    cfg: RuntimeConfig
    start_perf: float = field(default_factory=time.perf_counter)
    fit_count: int = 0
    triggered: Dict[str, Any] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

    def elapsed(self) -> float:
        return float(time.perf_counter() - self.start_perf)

    def note(self, code: str, message: str) -> None:
        if code not in self.triggered:
            self.triggered[code] = True
        self.events.append(f"{code}: {message}")

    def total_exceeded(self) -> bool:
        if self.cfg.max_total_seconds <= 0:
            return False
        hit = self.elapsed() > float(self.cfg.max_total_seconds)
        if hit:
            self.note("max_total_seconds", f"elapsed={self.elapsed():.1f}s > budget={self.cfg.max_total_seconds}s")
        return hit

    def scan_exceeded(self, scan_start_elapsed: float) -> bool:
        if self.cfg.max_scan_seconds <= 0:
            return False
        scan_elapsed = self.elapsed() - float(scan_start_elapsed)
        hit = scan_elapsed > float(self.cfg.max_scan_seconds)
        if hit:
            self.note("max_scan_seconds", f"scan_elapsed={scan_elapsed:.1f}s > budget={self.cfg.max_scan_seconds}s")
        return hit

    def try_reserve_fits(self, n: int, family: str) -> bool:
        n = int(max(n, 0))
        if n == 0:
            return True
        if self.cfg.max_fits > 0 and (self.fit_count + n) > int(self.cfg.max_fits):
            self.note(
                "max_fits",
                f"fit cap reached before {family}; requested={n}, used={self.fit_count}, budget={self.cfg.max_fits}",
            )
            return False
        self.fit_count += n
        return True

    def remaining_fits(self) -> int:
        if self.cfg.max_fits <= 0:
            return 10**9
        return max(int(self.cfg.max_fits) - int(self.fit_count), 0)


@dataclass
class PreparedMatrixBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    sample_weight: np.ndarray


_ACTIVE_PROFILER: Optional[RuntimeProfiler] = None
_ACTIVE_RUNTIME_CFG: Optional[RuntimeConfig] = None
_ACTIVE_BUDGET: Optional[BudgetController] = None
_PIPELINE_START_PERF: Optional[float] = None
_PREP_CACHE: Dict[Any, PreparedMatrixBundle] = {}
_BUILD_SPLIT_TABLES_CACHE: Dict[Any, tuple[pd.DataFrame, pd.DataFrame, Any]] = {}
_PREPARE_VARIANT_CACHE: Dict[Any, dict] = {}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _load_runtime_config_from_env() -> RuntimeConfig:
    fast_mode = _env_bool("ALGOSPORTS_FAST_MODE", True)
    enable_optimizations = _env_bool("ALGOSPORTS_ENABLE_OPTIMIZATIONS", True)
    bag_default = 1 if fast_mode else 2
    cfg = RuntimeConfig(
        max_total_seconds=_env_int("ALGOSPORTS_MAX_TOTAL_SECONDS", 600),
        max_scan_seconds=_env_int("ALGOSPORTS_MAX_SCAN_SECONDS", 120),
        max_fits=_env_int("ALGOSPORTS_MAX_FITS", 400),
        fast_mode=bool(fast_mode),
        enable_optimizations=bool(enable_optimizations),
        histgb_max_iter_cap=_env_int("ALGOSPORTS_FAST_HISTGB_MAX_ITER_CAP", 60),
        histgb_bag_n_models=_env_int("ALGOSPORTS_HISTGB_BAG_N_MODELS", bag_default),
        scan_prune_keep_top_n=_env_int("ALGOSPORTS_SCAN_PRUNE_TOPN", 25),
        scan_prune_keep_frac=_env_float("ALGOSPORTS_SCAN_PRUNE_FRAC", 0.35),
        scan_top_candidates_return=_env_int("ALGOSPORTS_SCAN_RETURN_TOP", 3),
        fast_outer_folds=_env_int("ALGOSPORTS_FAST_OUTER_FOLDS", 1),
        fast_inner_folds=_env_int("ALGOSPORTS_FAST_INNER_FOLDS", 1),
        fast_scan_topn=_env_int("ALGOSPORTS_FAST_SCAN_TOPN", 20),
        fast_disable_q20_q80=_env_bool("ALGOSPORTS_FAST_SKIP_Q20_Q80", True),
        fast_post_grid_limit=_env_int("ALGOSPORTS_FAST_POST_GRID_LIMIT", 8),
    )
    cfg.histgb_bag_n_models = max(1, int(cfg.histgb_bag_n_models))
    cfg.scan_prune_keep_top_n = max(1, int(cfg.scan_prune_keep_top_n))
    cfg.scan_top_candidates_return = max(1, int(cfg.scan_top_candidates_return))
    cfg.fast_outer_folds = max(1, int(cfg.fast_outer_folds))
    cfg.fast_inner_folds = max(1, int(cfg.fast_inner_folds))
    cfg.fast_scan_topn = max(1, int(cfg.fast_scan_topn))
    cfg.fast_post_grid_limit = max(1, int(cfg.fast_post_grid_limit))
    cfg.scan_prune_keep_frac = float(np.clip(float(cfg.scan_prune_keep_frac), 0.05, 1.0))
    return cfg


def now_seconds() -> float:
    if _PIPELINE_START_PERF is None:
        return float(time.perf_counter())
    return float(time.perf_counter() - float(_PIPELINE_START_PERF))


def _budget() -> Optional[BudgetController]:
    return _ACTIVE_BUDGET


def _runtime_cfg() -> RuntimeConfig:
    return _ACTIVE_RUNTIME_CFG if _ACTIVE_RUNTIME_CFG is not None else RuntimeConfig()


def _budget_total_exceeded() -> bool:
    b = _budget()
    return bool(b.total_exceeded()) if b is not None else False


def _budget_scan_exceeded(scan_start_elapsed: float) -> bool:
    b = _budget()
    return bool(b.scan_exceeded(scan_start_elapsed)) if b is not None else False



@contextmanager
def _timed_phase(phase: str, *, detail: Optional[str] = None):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        profiler = _ACTIVE_PROFILER
        if profiler is not None:
            profiler.add(phase, time.perf_counter() - t0, detail=detail)


def _record_model_fit_count(family: str, n: int = 1) -> None:
    profiler = _ACTIVE_PROFILER
    if profiler is not None:
        profiler.add_model_fit(family, n=n)


def _timing_summary_dict(profiler: Optional[RuntimeProfiler], top_n: int = 10) -> dict:
    if profiler is None:
        return {
            "phase_totals_sec": {},
            "phase_counts": {},
            "predict_family_totals_sec": {},
            "predict_family_call_counts": {},
            "model_fit_counts_by_family": {},
            "top_slowest_events": [],
        }
    pf_events = [e for e in profiler.events if e["phase"] == "_predict_family"]
    pf_totals: Dict[str, float] = {}
    pf_counts: Dict[str, int] = {}
    for e in pf_events:
        fam = str(e.get("detail") or "unknown")
        pf_totals[fam] = float(pf_totals.get(fam, 0.0) + float(e["sec"]))
        pf_counts[fam] = int(pf_counts.get(fam, 0) + 1)
    top_events = sorted(profiler.events, key=lambda x: float(x.get("sec", 0.0)), reverse=True)[: max(int(top_n), 1)]
    return {
        "phase_totals_sec": {k: float(v) for k, v in sorted(profiler.totals.items(), key=lambda kv: kv[0])},
        "phase_counts": {k: int(v) for k, v in sorted(profiler.counts.items(), key=lambda kv: kv[0])},
        "predict_family_totals_sec": {k: float(v) for k, v in sorted(pf_totals.items(), key=lambda kv: kv[1], reverse=True)},
        "predict_family_call_counts": {k: int(v) for k, v in sorted(pf_counts.items(), key=lambda kv: kv[0])},
        "model_fit_counts_by_family": {k: int(v) for k, v in sorted(profiler.model_fit_counts.items(), key=lambda kv: kv[0])},
        "top_slowest_events": [
            {"phase": str(e.get("phase", "")), "detail": str(e.get("detail", "")), "sec": float(e.get("sec", 0.0))}
            for e in top_events
        ],
    }


def _timing_summary_lines(summary: Mapping[str, Any], focus_phases: Optional[Sequence[str]] = None, top_n: int = 10) -> List[str]:
    focus = list(focus_phases or [])
    phase_totals = dict(summary.get("phase_totals_sec", {}) or {})
    phase_counts = dict(summary.get("phase_counts", {}) or {})
    pf_totals = dict(summary.get("predict_family_totals_sec", {}) or {})
    pf_call_counts = dict(summary.get("predict_family_call_counts", {}) or {})
    fit_counts = dict(summary.get("model_fit_counts_by_family", {}) or {})
    lines = ["Runtime timing summary"]
    if focus:
        lines.append("Hotspot totals (sec):")
        for name in focus:
            lines.append(
                f"- {name}: {float(phase_totals.get(name, 0.0)):.3f}s "
                f"(calls={int(phase_counts.get(name, 0))})"
            )
    else:
        lines.append("Phase totals (top by total sec):")
        for name, sec in sorted(phase_totals.items(), key=lambda kv: float(kv[1]), reverse=True)[:10]:
            lines.append(f"- {name}: {float(sec):.3f}s (calls={int(phase_counts.get(name, 0))})")
    lines.append("Predict family totals (sec):")
    for fam, sec in sorted(pf_totals.items(), key=lambda kv: float(kv[1]), reverse=True):
        lines.append(f"- {fam}: {float(sec):.3f}s (calls={int(pf_call_counts.get(fam, 0))})")
    lines.append("Model fit counts by family:")
    for fam, cnt in sorted(fit_counts.items(), key=lambda kv: kv[0]):
        lines.append(f"- {fam}: {int(cnt)}")
    lines.append(f"Top {int(top_n)} slowest call sites / phases:")
    for i, e in enumerate((summary.get("top_slowest_events", []) or [])[: max(int(top_n), 1)], start=1):
        detail = str(e.get("detail", "") or "")
        suffix = f" [{detail}]" if detail else ""
        lines.append(f"{i}. {str(e.get('phase', ''))}{suffix}: {float(e.get('sec', 0.0)):.3f}s")
    return lines


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "bias": _bias(y_true, y_pred),
        "pred_std": float(np.std(y_pred)),
        "actual_std": float(np.std(y_true)),
        "resid_std": float(np.std(resid)),
    }


def _fit_linear_calibration(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    if pred.size < 2 or float(np.std(pred)) < 1e-9:
        return 0.0, 1.0
    X = np.column_stack([np.ones(pred.size), pred])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if not np.isfinite(a):
        a = 0.0
    if not np.isfinite(b):
        b = 1.0
    return float(a), float(np.clip(b, 0.3, 2.2))


def _normalize_simplex(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    w[~np.isfinite(w)] = 0.0
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return np.full_like(w, 1.0 / max(len(w), 1))
    return w / s


def _optimize_simplex_weights(Z: np.ndarray, y: np.ndarray, init: Optional[np.ndarray] = None) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(Z.shape[1])
    x0 = _normalize_simplex(np.asarray(init, dtype=float) if init is not None else np.full(n, 1.0 / n))

    def obj(w: np.ndarray) -> float:
        return float(np.mean((y - Z.dot(np.asarray(w, dtype=float))) ** 2))

    try:
        res = minimize(
            obj,
            x0=x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"maxiter": 250, "ftol": 1e-10},
        )
        if res.success and np.isfinite(res.fun):
            return _normalize_simplex(res.x)
    except Exception:
        pass
    return _normalize_simplex(x0)


def _add_nonlinear_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    add_cols: Dict[str, np.ndarray] = {}
    for col in [
        "elo_diff_pre",
        "elo_neutral_diff_pre",
        "massey_diff",
        "offdef_net_diff",
        "offdef_margin_neutral",
        "ema_margin_diff",
        "mean_margin_diff",
        "oppadj_margin_diff",
        "conf_strength_diff",
        "trend_margin_last5_vs_season_diff",
        "trend_oppadj_last5_vs_season_diff",
        "consistency_ratio_diff",
    ]:
        if col not in out.columns:
            continue
        x = out[col].astype(float).to_numpy() / 50.0
        add_cols[f"{col}__sq"] = x * x
        add_cols[f"{col}__cu"] = x * x * x
        add_cols[f"{col}__abs"] = np.abs(x)
    for col, knots in [
        ("elo_diff_pre", [-150, -90, -45, 0, 45, 90, 150]),
        ("elo_neutral_diff_pre", [-150, -90, -45, 0, 45, 90, 150]),
        ("massey_diff", [-80, -40, -20, 0, 20, 40, 80]),
        ("offdef_net_diff", [-60, -30, -10, 0, 10, 30, 60]),
        ("trend_margin_last5_vs_season_diff", [-40, -20, -10, 0, 10, 20, 40]),
    ]:
        if col not in out.columns:
            continue
        x = out[col].astype(float).to_numpy()
        for k in knots:
            add_cols[f"{col}__hinge_gt_{k}"] = np.maximum(x - float(k), 0.0) / 50.0
            add_cols[f"{col}__hinge_lt_{k}"] = np.maximum(float(k) - x, 0.0) / 50.0
    if add_cols:
        out = pd.concat([out, pd.DataFrame(add_cols, index=out.index)], axis=1)
    return out


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in META_EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def _feature_profile_columns(all_cols: Sequence[str], profile: str) -> List[str]:
    cols = list(all_cols)
    if profile == "full_recency":
        return cols
    if profile == "compact_recency":
        keep = []
        for c in cols:
            if re.search(r"(ema_(margin|oppadj|winrate)_a20_|last5_|trend_|consistency_)", c):
                keep.append(c)
            elif re.search(r"(ema_(margin|oppadj|winrate)_a10_|ema_(margin|oppadj|winrate)_a35_|last3_|last8_)", c):
                continue
            else:
                keep.append(c)
        return sorted(dict.fromkeys(keep))
    if profile == "no_extra_recency":
        pat = re.compile(r"(ema_(margin|oppadj|winrate)_a\d+|last\d+_|trend_|consistency_)")
        return [c for c in cols if not pat.search(c)]
    raise ValueError(profile)


def _xy(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str = "HomeWinMargin") -> tuple[pd.DataFrame, np.ndarray]:
    X = df.reindex(columns=list(feature_cols), fill_value=0.0).astype(float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def _compute_time_decay_weights(dates: pd.Series, half_life_days: Optional[int]) -> np.ndarray:
    if half_life_days is None or int(half_life_days) <= 0:
        return np.ones(len(dates), dtype=float)
    d = pd.to_datetime(dates)
    ref = pd.Timestamp(d.max())
    ages = (ref - d).dt.days.astype(float).to_numpy()
    w = np.power(0.5, np.clip(ages, 0.0, None) / max(float(half_life_days), 1.0))
    return np.clip(w, 0.05, 1.0)


def _index_signature(index: pd.Index) -> tuple:
    arr = np.asarray(index.to_numpy(), dtype=np.int64) if len(index) else np.asarray([], dtype=np.int64)
    if arr.size == 0:
        return (0, -1, -1, 0)
    return (int(arr.size), int(arr[0]), int(arr[-1]), int(arr.sum()))


def _feature_signature(feature_cols: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(c) for c in feature_cols)


def _prep_cache_key(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    half_life_days: Optional[int],
) -> tuple:
    hl = None if half_life_days is None else int(half_life_days)
    return (_index_signature(train_df.index), _index_signature(val_df.index), _feature_signature(feature_cols), hl)


def _prepare_prediction_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    half_life_days: Optional[int],
) -> PreparedMatrixBundle:
    cfg = _runtime_cfg()
    key = _prep_cache_key(train_df, val_df, feature_cols, half_life_days)
    if cfg.enable_optimizations and key in _PREP_CACHE:
        return _PREP_CACHE[key]
    with _timed_phase("_prepare_prediction_matrices", detail="cache_miss"):
        X_train_df, y_train = _xy(train_df, feature_cols)
        X_val_df = val_df.reindex(columns=list(feature_cols), fill_value=0.0).astype(float)
        sw = _compute_time_decay_weights(train_df["Date"], half_life_days)
        bundle = PreparedMatrixBundle(
            X_train=np.asarray(X_train_df.to_numpy(dtype=float), dtype=float),
            y_train=np.asarray(y_train, dtype=float),
            X_val=np.asarray(X_val_df.to_numpy(dtype=float), dtype=float),
            sample_weight=np.asarray(sw, dtype=float),
        )
    if cfg.enable_optimizations:
        _PREP_CACHE[key] = bundle
    return bundle


def _make_pipeline_ridge(alpha: float, seed: int) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=float(alpha), random_state=seed))])


def _make_pipeline_huber(seed: int) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=2000))])


def _make_histgb(params: Mapping[str, Any], seed: int, quantile: Optional[float] = None):
    cfg = _runtime_cfg()
    max_iter = int(params["max_iter"])
    if cfg.fast_mode:
        max_iter = min(max_iter, int(cfg.histgb_max_iter_cap))
    base = dict(
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        max_iter=int(max_iter),
        early_stopping=False,
        random_state=seed,
    )
    if quantile is None:
        return HistGradientBoostingRegressor(loss="squared_error", **base)
    try:
        return HistGradientBoostingRegressor(loss="quantile", quantile=float(quantile), **base)
    except Exception:
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=float(quantile),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            n_estimators=min(max(int(max_iter), 100), 500),
            random_state=seed,
        )


def _fit_model(estimator, X: Any, y: np.ndarray, sample_weight: Optional[np.ndarray]):
    if sample_weight is None:
        return estimator.fit(X, y)
    if isinstance(estimator, Pipeline):
        return estimator.fit(X, y, model__sample_weight=sample_weight)
    return estimator.fit(X, y, sample_weight=sample_weight)


def _safe_constant_fallback(y_train: np.ndarray, n_out: int) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=float)
    c = float(np.nanmean(y_train)) if y_train.size else 0.0
    if not np.isfinite(c):
        c = 0.0
    return np.full(int(n_out), c, dtype=float)


def _budget_reserve_fit_or_false(family: str, n: int = 1) -> bool:
    b = _budget()
    if b is None:
        return True
    return bool(b.try_reserve_fits(n=n, family=family))


def _histgb_bag_seeds(seed: int, n_models: int) -> List[int]:
    offsets = [0, 11, 23, 37, 53, 71, 89]
    n = max(1, int(n_models))
    return [int(seed + offsets[i]) for i in range(min(n, len(offsets)))]


def _predict_families_bundle(
    families: Sequence[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    ridge_alpha: float,
    histgb_params: Mapping[str, Any],
    half_life_days: Optional[int],
    seed: int,
) -> Dict[str, np.ndarray]:
    with _timed_phase("_predict_family_bundle", detail="|".join(map(str, families))):
        prep = _prepare_prediction_matrices(train_df, val_df, feature_cols, half_life_days)
        X_train = prep.X_train
        y_train = prep.y_train
        X_val = prep.X_val
        sw = prep.sample_weight
        out: Dict[str, np.ndarray] = {}
        cfg = _runtime_cfg()
        b = _budget()
        fits_remaining = b.remaining_fits() if b is not None else 10**9
        fam_seed_offsets = {
            "ridge": 0,
            "huber": 10,
            "histgb": 20,
            "histgb_bag": 30,
            "histgb_q50": 40,
            "histgb_q20": 50,
            "histgb_q80": 60,
        }

        def fit_and_predict(one_family: str, est) -> np.ndarray:
            if not _budget_reserve_fit_or_false(one_family, 1):
                return _safe_constant_fallback(y_train, len(X_val))
            _record_model_fit_count(one_family)
            _fit_model(est, X_train, y_train, sw)
            return np.asarray(est.predict(X_val), dtype=float)

        for family in families:
            if family in out:
                continue
            with _timed_phase("_predict_family", detail=family):
                if _budget_total_exceeded() and len(out) > 0:
                    out[family] = _safe_constant_fallback(y_train, len(X_val))
                    continue
                fam_seed = int(seed + int(fam_seed_offsets.get(family, 0)))
                if family == "ridge":
                    out[family] = fit_and_predict(family, _make_pipeline_ridge(ridge_alpha, fam_seed))
                    continue
                if family == "huber":
                    out[family] = fit_and_predict(family, _make_pipeline_huber(fam_seed))
                    continue
                if family == "histgb":
                    out[family] = fit_and_predict(family, _make_histgb(histgb_params, fam_seed, None))
                    continue
                if family == "histgb_q50":
                    out[family] = fit_and_predict(family, _make_histgb(histgb_params, fam_seed, 0.5))
                    continue
                if family == "histgb_q20":
                    out[family] = fit_and_predict(family, _make_histgb(histgb_params, fam_seed, 0.2))
                    continue
                if family == "histgb_q80":
                    out[family] = fit_and_predict(family, _make_histgb(histgb_params, fam_seed, 0.8))
                    continue
                if family == "histgb_bag":
                    bag_n = int(cfg.histgb_bag_n_models)
                    if cfg.fast_mode:
                        bag_n = 1
                    if fits_remaining <= 1 or (b is not None and b.remaining_fits() <= 1):
                        bag_n = 1
                        if b is not None:
                            b.note("histgb_bag_auto_reduce", "Reduced HistGB bagging to 1 model due tight fit budget.")
                    seeds = _histgb_bag_seeds(fam_seed, bag_n)
                    preds = []
                    for s in seeds:
                        if not _budget_reserve_fit_or_false(family, 1):
                            break
                        _record_model_fit_count(family)
                        est = _make_histgb(histgb_params, int(s), None)
                        _fit_model(est, X_train, y_train, sw)
                        preds.append(np.asarray(est.predict(X_val), dtype=float))
                    if not preds:
                        out[family] = _safe_constant_fallback(y_train, len(X_val))
                    else:
                        out[family] = np.mean(np.column_stack(preds), axis=1)
                    continue
                raise ValueError(family)
        return out


def _predict_family(
    family: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    ridge_alpha: float,
    histgb_params: Mapping[str, Any],
    half_life_days: Optional[int],
    seed: int,
) -> np.ndarray:
    return _predict_families_bundle(
        [family],
        train_df,
        val_df,
        feature_cols,
        ridge_alpha=ridge_alpha,
        histgb_params=histgb_params,
        half_life_days=half_life_days,
        seed=seed,
    )[family]


def _resolve_input_paths(root: Path) -> Dict[str, Path]:
    cand_pred = [
        root / "Predictions.csv",
        root / "predictions.csv",
        root / "Submission.zip" / "Predictions.csv",
        root / "Submission.zip1" / "Predictions.csv",
    ]
    cand_rank = [
        root / "Rankings.xlsx",
        root / "rankings.xlsx",
        root / "Submission.zip" / "Rankings.xlsx",
        root / "Submission.zip1" / "Rankings.xlsx",
    ]
    train_path = root / "Train.csv"
    pred_path = next((p for p in cand_pred if p.exists()), None)
    rank_path = next((p for p in cand_rank if p.exists()), None)
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if pred_path is None:
        raise FileNotFoundError("Predictions.csv not found")
    if rank_path is None:
        raise FileNotFoundError("Rankings.xlsx not found")
    return {"train": train_path, "pred": pred_path, "rankings": rank_path}


def _load_inputs(paths: Mapping[str, Path]) -> Dict[str, pd.DataFrame]:
    return {"train": pd.read_csv(paths["train"]), "pred": pd.read_csv(paths["pred"]), "rankings": pd.read_excel(paths["rankings"])}


def _build_elo_variants() -> Dict[str, EloVariantConfig]:
    return {
        "elo_base_static": EloVariantConfig(name="elo_base_static", home_adv=55.0, k_factor=24.0),
        "elo_dynamic_k": EloVariantConfig(name="elo_dynamic_k", home_adv=55.0, k_factor=22.0, use_dynamic_k=True, k_early_multiplier=0.9, k_games_scale=16.0, k_uncertainty_multiplier=0.30),
        "elo_dynamic_k_teamha": EloVariantConfig(name="elo_dynamic_k_teamha", home_adv=50.0, k_factor=22.0, use_dynamic_k=True, k_early_multiplier=0.9, k_games_scale=18.0, k_uncertainty_multiplier=0.25, use_team_home_adv=True, team_home_adv_scale=72.0, team_home_adv_reg=7.0, team_home_adv_cap=45.0),
        "elo_dynamic_k_teamha_decay": EloVariantConfig(name="elo_dynamic_k_teamha_decay", home_adv=50.0, k_factor=21.0, use_dynamic_k=True, k_early_multiplier=0.95, k_games_scale=18.0, k_uncertainty_multiplier=0.35, use_team_home_adv=True, team_home_adv_scale=70.0, team_home_adv_reg=8.0, team_home_adv_cap=45.0, use_inactivity_decay=True, inactivity_tau_days=95.0),
    }


def _histgb_param_grid() -> List[dict]:
    return [
        {"learning_rate": 0.04, "max_depth": 3, "max_leaf_nodes": 31, "min_samples_leaf": 10, "l2_regularization": 0.3, "max_iter": 280},
        {"learning_rate": 0.05, "max_depth": 4, "max_leaf_nodes": 31, "min_samples_leaf": 8, "l2_regularization": 0.8, "max_iter": 360},
        {"learning_rate": 0.03, "max_depth": 4, "max_leaf_nodes": 63, "min_samples_leaf": 6, "l2_regularization": 1.2, "max_iter": 420},
    ]


def _build_split_tables(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    seq_train: pd.DataFrame,
    seq_val: pd.DataFrame,
    team_ids: Sequence[int],
    conf_values: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    with _timed_phase("_build_split_tables"):
        cfg = _runtime_cfg()
        variant_key = str(seq_train.attrs.get("_variant_name", "unknown"))
        cache_key = (
            variant_key,
            _index_signature(train_df.index),
            _index_signature(val_df.index),
            _index_signature(seq_train.index),
            _index_signature(seq_val.index),
            int(len(team_ids)),
            int(len(conf_values)),
        )
        if cfg.enable_optimizations and cache_key in _BUILD_SPLIT_TABLES_CACHE:
            return _BUILD_SPLIT_TABLES_CACHE[cache_key]

        static_models = fit_static_models_for_fold(train_df, team_ids=team_ids, conf_values=conf_values)
        train_static = apply_static_models_to_train_like_rows(train_df, static_models, neutral_site=False)
        val_static = apply_static_models_to_train_like_rows(val_df, static_models, neutral_site=False)
        train_tbl = assemble_model_table(train_df, seq_train, train_static)
        val_tbl = assemble_model_table(val_df, seq_val, val_static)
        train_tbl.index = train_df.index
        val_tbl.index = val_df.index
        if "Date" in train_tbl.columns:
            train_tbl["Date"] = pd.to_datetime(train_tbl["Date"])
        if "Date" in val_tbl.columns:
            val_tbl["Date"] = pd.to_datetime(val_tbl["Date"])
        out = (train_tbl, val_tbl, static_models)
        if cfg.enable_optimizations:
            _BUILD_SPLIT_TABLES_CACHE[cache_key] = out
        return out


def _prepare_variant_outer_data(
    train_df: pd.DataFrame,
    outer_fold: Mapping[str, Any],
    inner_folds: Sequence[Mapping[str, Any]],
    seq_variant_full: pd.DataFrame,
    *,
    team_ids: Sequence[int],
    conf_values: Sequence[str],
    variant_name: Optional[str] = None,
) -> dict:
    with _timed_phase("_prepare_variant_outer_data"):
        cfg = _runtime_cfg()
        outer_cache_key = (
            str(variant_name or seq_variant_full.attrs.get("_variant_name", "unknown")),
            int(outer_fold.get("fold", -1)),
            _index_signature(pd.Index(np.asarray(outer_fold["train_idx"], dtype=int))),
            _index_signature(pd.Index(np.asarray(outer_fold["val_idx"], dtype=int))),
            tuple(
                (
                    int(inner.get("fold", -1)),
                    _index_signature(pd.Index(np.asarray(inner["train_idx"], dtype=int))),
                    _index_signature(pd.Index(np.asarray(inner["val_idx"], dtype=int))),
                )
                for inner in inner_folds
            ),
        )
        if cfg.enable_optimizations and outer_cache_key in _PREPARE_VARIANT_CACHE:
            return _PREPARE_VARIANT_CACHE[outer_cache_key]
        outer_train_idx = np.asarray(outer_fold["train_idx"], dtype=int)
        outer_val_idx = np.asarray(outer_fold["val_idx"], dtype=int)
        outer_train_games = train_df.loc[outer_train_idx]
        outer_val_games = train_df.loc[outer_val_idx]
        seq_outer_train = seq_variant_full.loc[outer_train_idx]
        seq_outer_val = seq_variant_full.loc[outer_val_idx]
        seq_outer_train.attrs["_variant_name"] = str(variant_name or seq_variant_full.attrs.get("_variant_name", "unknown"))
        seq_outer_val.attrs["_variant_name"] = str(variant_name or seq_variant_full.attrs.get("_variant_name", "unknown"))

        outer_train_tbl, outer_val_tbl, static_models_outer = _build_split_tables(
            outer_train_games,
            outer_val_games,
            seq_train=seq_outer_train,
            seq_val=seq_outer_val,
            team_ids=team_ids,
            conf_values=conf_values,
        )
        outer_train_nl = _add_nonlinear_features(outer_train_tbl)
        outer_val_nl = _add_nonlinear_features(outer_val_tbl)
        all_cols = _select_feature_columns(outer_train_nl)
        feature_cols_by_profile = {
            "no_extra_recency": _feature_profile_columns(all_cols, "no_extra_recency"),
            "compact_recency": _feature_profile_columns(all_cols, "compact_recency"),
            "full_recency": _feature_profile_columns(all_cols, "full_recency"),
        }

        inner_splits = []
        for inner in inner_folds:
            if _budget_total_exceeded() and inner_splits:
                break
            tr_idx_local = np.asarray(inner["train_idx"], dtype=int)
            va_idx_local = np.asarray(inner["val_idx"], dtype=int)
            tr_games = outer_train_games.iloc[tr_idx_local]
            va_games = outer_train_games.iloc[va_idx_local]
            tr_seq = seq_outer_train.iloc[tr_idx_local]
            va_seq = seq_outer_train.iloc[va_idx_local]
            tr_seq.attrs["_variant_name"] = str(variant_name or seq_variant_full.attrs.get("_variant_name", "unknown"))
            va_seq.attrs["_variant_name"] = str(variant_name or seq_variant_full.attrs.get("_variant_name", "unknown"))
            tr_tbl, va_tbl, _ = _build_split_tables(
                tr_games, va_games, seq_train=tr_seq, seq_val=va_seq, team_ids=team_ids, conf_values=conf_values
            )
            inner_splits.append(
                {
                    "fold": int(inner["fold"]),
                    "train_tbl": tr_tbl,
                    "val_tbl": va_tbl,
                    "train_nl": _add_nonlinear_features(tr_tbl),
                    "val_nl": _add_nonlinear_features(va_tbl),
                }
            )

        out = {
            "outer_train_games": outer_train_games,
            "outer_val_games": outer_val_games,
            "outer_train_tbl": outer_train_tbl,
            "outer_val_tbl": outer_val_tbl,
            "outer_train_nl": outer_train_nl,
            "outer_val_nl": outer_val_nl,
            "feature_cols_by_profile": feature_cols_by_profile,
            "inner_splits": inner_splits,
            "outer_static_models": static_models_outer,
        }
        if cfg.enable_optimizations:
            _PREPARE_VARIANT_CACHE[outer_cache_key] = out
        return out


def _core_scan_score(
    candidate: CoreCandidate,
    variant_data: Mapping[str, Any],
    histgb_grid: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    split_limit: Optional[int] = None,
) -> dict:
    with _timed_phase("_core_scan_score"):
        feature_cols = variant_data["feature_cols_by_profile"][candidate.feature_profile]
        histgb_params = histgb_grid[int(candidate.histgb_idx)]
        rows = []
        for i_split, split in enumerate(variant_data["inner_splits"]):
            if split_limit is not None and i_split >= int(split_limit):
                break
            if _budget_total_exceeded() and rows:
                break
            tr = split["train_nl"]
            va = split["val_nl"]
            y = va["HomeWinMargin"].to_numpy(dtype=float)
            if _runtime_cfg().enable_optimizations:
                preds = _predict_families_bundle(
                    ["ridge", "histgb"],
                    tr,
                    va,
                    feature_cols,
                    ridge_alpha=candidate.ridge_alpha,
                    histgb_params=histgb_params,
                    half_life_days=candidate.half_life_days,
                    seed=seed + 1,
                )
                pr = preds["ridge"]
                ph = preds["histgb"]
            else:
                pr = _predict_family("ridge", tr, va, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 1)
                ph = _predict_family("histgb", tr, va, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 2)
            mr = _reg_metrics(y, pr)
            mh = _reg_metrics(y, ph)
            rows.append({"fold": int(split["fold"]), "ridge_rmse": mr["rmse"], "histgb_rmse": mh["rmse"], "ridge_mae": mr["mae"], "histgb_mae": mh["mae"], "histgb_pred_std": mh["pred_std"], "actual_std": mh["actual_std"]})
        if not rows:
            return {
                "candidate_key": candidate.key(),
                "elo_variant": candidate.elo_variant,
                "feature_profile": candidate.feature_profile,
                "half_life_days": np.nan if candidate.half_life_days is None else int(candidate.half_life_days),
                "ridge_alpha": float(candidate.ridge_alpha),
                "histgb_idx": int(candidate.histgb_idx),
                "scan_score": float("inf"),
                "scan_histgb_rmse": float("inf"),
                "scan_histgb_mae": float("inf"),
                "scan_ridge_rmse": float("inf"),
                "scan_ridge_mae": float("inf"),
                "scan_pred_std_gap_histgb": float("nan"),
                "scan_splits_used": 0,
            }
        fr = pd.DataFrame(rows)
        score = (
            0.55 * float(fr["histgb_rmse"].mean())
            + 0.35 * float(fr["ridge_rmse"].mean())
            + 0.05 * float(fr["histgb_rmse"].std(ddof=0))
            + 0.05 * abs(float(fr["histgb_pred_std"].mean() - fr["actual_std"].mean()))
        )
        return {
            "candidate_key": candidate.key(),
            "elo_variant": candidate.elo_variant,
            "feature_profile": candidate.feature_profile,
            "half_life_days": np.nan if candidate.half_life_days is None else int(candidate.half_life_days),
            "ridge_alpha": float(candidate.ridge_alpha),
            "histgb_idx": int(candidate.histgb_idx),
            "scan_score": float(score),
            "scan_histgb_rmse": float(fr["histgb_rmse"].mean()),
            "scan_histgb_mae": float(fr["histgb_mae"].mean()),
            "scan_ridge_rmse": float(fr["ridge_rmse"].mean()),
            "scan_ridge_mae": float(fr["ridge_mae"].mean()),
            "scan_pred_std_gap_histgb": float(fr["histgb_pred_std"].mean() - fr["actual_std"].mean()),
            "scan_splits_used": int(len(fr)),
        }


def _scan_and_select_core_candidates(
    outer_variant_data: Mapping[str, Any],
    histgb_grid: Sequence[Mapping[str, Any]],
    seed: int,
) -> tuple[List[CoreCandidate], pd.DataFrame]:
    with _timed_phase("_scan_and_select_core_candidates"):
        cfg = _runtime_cfg()
        scan_start_elapsed = now_seconds()
        all_candidates: List[CoreCandidate] = []
        for v in outer_variant_data.keys():
            for profile in ["no_extra_recency", "compact_recency", "full_recency"]:
                for hl in [None, 30, 60]:
                    for ra in [4.0, 16.0]:
                        for hidx in range(len(histgb_grid)):
                            all_candidates.append(CoreCandidate(v, profile, hl, ra, hidx))

        if not cfg.enable_optimizations:
            coarse_keep: List[CoreCandidate] = list(all_candidates)
            coarse_rows_df = pd.DataFrame()
        else:
            coarse_rows: List[dict] = []
            for i, cand in enumerate(all_candidates):
                if _budget_scan_exceeded(scan_start_elapsed) and coarse_rows:
                    break
                if _budget_total_exceeded() and coarse_rows:
                    break
                row = _core_scan_score(cand, outer_variant_data[cand.elo_variant], histgb_grid, seed=seed, split_limit=1)
                row["scan_stage"] = "coarse"
                coarse_rows.append(row)
                if cfg.fast_mode and len(coarse_rows) >= int(cfg.fast_scan_topn):
                    # In fast mode, still use broad generation logic but cap coarse pass count for guaranteed runtime.
                    break
            coarse_rows_df = pd.DataFrame(coarse_rows)
            if coarse_rows_df.empty:
                coarse_keep = [all_candidates[0]] if all_candidates else []
            else:
                n_keep = max(
                    1,
                    min(
                        int(cfg.scan_prune_keep_top_n),
                        int(math.ceil(float(cfg.scan_prune_keep_frac) * float(len(coarse_rows_df)))),
                    ),
                )
                if cfg.fast_mode:
                    n_keep = min(n_keep, int(cfg.fast_scan_topn))
                keep_keys = set(coarse_rows_df.sort_values(["scan_score", "scan_histgb_rmse"], kind="mergesort").head(n_keep)["candidate_key"].astype(str))
                coarse_keep = [c for c in all_candidates if c.key() in keep_keys]

        scan_rows = []
        for i, cand in enumerate(coarse_keep):
            if _budget_scan_exceeded(scan_start_elapsed) and scan_rows:
                break
            if _budget_total_exceeded() and scan_rows:
                break
            row = _core_scan_score(cand, outer_variant_data[cand.elo_variant], histgb_grid, seed=seed)
            row["scan_stage"] = "full"
            scan_rows.append(row)
            if cfg.fast_mode and len(scan_rows) >= int(cfg.fast_scan_topn):
                break
        if scan_rows:
            scan_df = pd.DataFrame(scan_rows)
        elif not coarse_rows_df.empty:
            scan_df = coarse_rows_df.copy()
        else:
            # Guaranteed fallback candidate if budgets stopped everything.
            fallback = all_candidates[0]
            scan_df = pd.DataFrame([_core_scan_score(fallback, outer_variant_data[fallback.elo_variant], histgb_grid, seed=seed, split_limit=1)])
        if not coarse_rows_df.empty and "scan_stage" in coarse_rows_df.columns:
            coarse_for_log = coarse_rows_df.copy()
            coarse_for_log["scan_stage"] = coarse_for_log.get("scan_stage", "coarse")
            scan_df = pd.concat([scan_df, coarse_for_log], ignore_index=True).drop_duplicates(subset=["candidate_key", "scan_stage"], keep="first")
        scan_df = scan_df.sort_values(["scan_score", "scan_histgb_rmse", "scan_ridge_rmse"], kind="mergesort").reset_index(drop=True)
        top_cands: List[CoreCandidate] = []
        scan_for_select = scan_df.copy()
        if "scan_stage" in scan_for_select.columns:
            scan_for_select = scan_for_select[scan_for_select["scan_stage"].astype(str) != "coarse"]
        for _, r in scan_for_select.head(int(cfg.scan_top_candidates_return)).iterrows():
            hl = None if pd.isna(r["half_life_days"]) else int(r["half_life_days"])
            top_cands.append(CoreCandidate(str(r["elo_variant"]), str(r["feature_profile"]), hl, float(r["ridge_alpha"]), int(r["histgb_idx"])))
        if not top_cands:
            for _, r in scan_df.head(int(cfg.scan_top_candidates_return)).iterrows():
                hl = None if pd.isna(r["half_life_days"]) else int(r["half_life_days"])
                top_cands.append(CoreCandidate(str(r["elo_variant"]), str(r["feature_profile"]), hl, float(r["ridge_alpha"]), int(r["histgb_idx"])))
        return top_cands, scan_df


def _regime_thresholds(df: pd.DataFrame) -> dict:
    vol_sum = (df["volatility_home"].astype(float) + df["volatility_away"].astype(float)).to_numpy()
    return {"elo_cuts": [35.0, 90.0], "vol_cut": float(np.nanmedian(vol_sum)) if len(vol_sum) else 0.0, "info_cut": 5.0, "info_hi_cut": 8.0}


def _assign_regimes(df: pd.DataFrame, thresholds: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    abs_elo = np.abs(out["elo_diff_pre"].astype(float).to_numpy())
    c1, c2 = [float(x) for x in thresholds["elo_cuts"]]
    elo_bin = np.where(abs_elo < c1, "close", np.where(abs_elo < c2, "moderate", "mismatch"))
    same = np.where(out.get("same_conf_flag", 0.0).astype(float).to_numpy() >= 0.5, "same", "cross")
    info_min = np.minimum(out["games_played_home"].astype(float), out["games_played_away"].astype(float))
    info_bin = np.where(info_min < float(thresholds["info_cut"]), "lowinfo", np.where(info_min >= float(thresholds["info_hi_cut"]), "highinfo", "midinfo"))
    vol_sum = out["volatility_home"].astype(float).to_numpy() + out["volatility_away"].astype(float).to_numpy()
    vol_bin = np.where(vol_sum >= float(thresholds["vol_cut"]), "highvol", "lowvol")
    out["_regime_full"] = [f"{a}|{b}|{c}|{d}" for a, b, c, d in zip(elo_bin, same, info_bin, vol_bin)]
    out["_regime_simple"] = [f"{a}|{b}|{c}" for a, b, c in zip(elo_bin, same, info_bin)]
    out["_regime_gap"] = elo_bin
    return out


def _fit_regime_simplex(prior: pd.DataFrame, pred_cols: Sequence[str], use_regime: bool) -> tuple[dict, pd.DataFrame]:
    pooled_w = _optimize_simplex_weights(prior[list(pred_cols)].to_numpy(dtype=float), prior["y_true"].to_numpy(dtype=float))
    fit = {"mode": "regime" if use_regime else "pooled", "pred_cols": list(pred_cols), "pooled": pooled_w, "full": {}, "simple": {}, "gap": {}}
    rows = [{"level": "pooled", "regime": "ALL", "n": len(prior), **{f"w_{c}": float(pooled_w[i]) for i, c in enumerate(pred_cols)}}]
    if not use_regime:
        return fit, pd.DataFrame(rows)
    for level, min_n in [("_regime_full", 45), ("_regime_simple", 35), ("_regime_gap", 28)]:
        dest = {"_regime_full": "full", "_regime_simple": "simple", "_regime_gap": "gap"}[level]
        for reg, g in prior.groupby(level, sort=False):
            if len(g) < min_n:
                continue
            w = _optimize_simplex_weights(g[list(pred_cols)].to_numpy(dtype=float), g["y_true"].to_numpy(dtype=float), init=pooled_w)
            fit[dest][str(reg)] = w
            rows.append({"level": dest, "regime": str(reg), "n": len(g), **{f"w_{c}": float(w[i]) for i, c in enumerate(pred_cols)}})
    return fit, pd.DataFrame(rows)


def _apply_regime_simplex(df: pd.DataFrame, fit: Mapping[str, Any]) -> np.ndarray:
    Z = df[list(fit["pred_cols"])].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    reg_full = df["_regime_full"].astype(str).to_numpy() if "_regime_full" in df.columns else np.array([""] * len(df), dtype=object)
    reg_simple = df["_regime_simple"].astype(str).to_numpy() if "_regime_simple" in df.columns else np.array([""] * len(df), dtype=object)
    reg_gap = df["_regime_gap"].astype(str).to_numpy() if "_regime_gap" in df.columns else np.array([""] * len(df), dtype=object)
    for i in range(len(df)):
        w = None
        if fit["mode"] == "regime":
            for level_name, k in [("full", reg_full[i]), ("simple", reg_simple[i]), ("gap", reg_gap[i])]:
                if k in fit[level_name]:
                    w = fit[level_name][k]
                    break
        if w is None:
            w = fit["pooled"]
        out[i] = float(np.dot(Z[i], np.asarray(w, dtype=float)))
    return out


def _fit_regime_calibration(prior: pd.DataFrame, raw_col: str, mode: str) -> tuple[dict, pd.DataFrame]:
    a, b = _fit_linear_calibration(prior[raw_col].to_numpy(dtype=float), prior["y_true"].to_numpy(dtype=float))
    fit = {"mode": mode, "pooled": (a, b), "full": {}, "simple": {}, "gap": {}}
    rows = [{"level": "pooled", "regime": "ALL", "n": len(prior), "intercept": a, "slope": b}]
    if mode != "regime":
        return fit, pd.DataFrame(rows)
    for level, min_n in [("_regime_full", 35), ("_regime_simple", 28), ("_regime_gap", 24)]:
        dest = {"_regime_full": "full", "_regime_simple": "simple", "_regime_gap": "gap"}[level]
        for reg, g in prior.groupby(level, sort=False):
            if len(g) < min_n:
                continue
            aa, bb = _fit_linear_calibration(g[raw_col].to_numpy(dtype=float), g["y_true"].to_numpy(dtype=float))
            fit[dest][str(reg)] = (aa, bb)
            rows.append({"level": dest, "regime": str(reg), "n": len(g), "intercept": aa, "slope": bb})
    return fit, pd.DataFrame(rows)


def _apply_regime_calibration(df: pd.DataFrame, raw_pred: np.ndarray, fit: Mapping[str, Any]) -> np.ndarray:
    raw_pred = np.asarray(raw_pred, dtype=float)
    out = np.zeros(len(df), dtype=float)
    reg_full = df["_regime_full"].astype(str).to_numpy() if "_regime_full" in df.columns else np.array([""] * len(df), dtype=object)
    reg_simple = df["_regime_simple"].astype(str).to_numpy() if "_regime_simple" in df.columns else np.array([""] * len(df), dtype=object)
    reg_gap = df["_regime_gap"].astype(str).to_numpy() if "_regime_gap" in df.columns else np.array([""] * len(df), dtype=object)
    for i in range(len(df)):
        params = None
        if fit["mode"] == "regime":
            for level_name, k in [("full", reg_full[i]), ("simple", reg_simple[i]), ("gap", reg_gap[i])]:
                if k in fit[level_name]:
                    params = fit[level_name][k]
                    break
        if params is None:
            params = fit["pooled"]
        a, b = params
        out[i] = float(a + b * raw_pred[i])
    return out


def _best_scale_factor(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    best_s = 1.0
    best_rmse = float("inf")
    for s in np.linspace(0.70, 1.65, 20):
        p = s * pred
        r = _rmse(y, p)
        if r < best_rmse:
            best_rmse = r
            best_s = float(s)
    return best_s


def _fit_regime_scale(prior: pd.DataFrame, cal_col: str, mode: str) -> tuple[dict, pd.DataFrame]:
    pooled_s = _best_scale_factor(prior["y_true"].to_numpy(dtype=float), prior[cal_col].to_numpy(dtype=float))
    fit = {"mode": mode, "center": 0.0, "pooled": pooled_s, "full": {}, "simple": {}, "gap": {}}
    rows = [{"level": "pooled", "regime": "ALL", "n": len(prior), "scale": pooled_s}]
    if mode != "regime":
        return fit, pd.DataFrame(rows)
    for level, min_n in [("_regime_full", 35), ("_regime_simple", 28), ("_regime_gap", 24)]:
        dest = {"_regime_full": "full", "_regime_simple": "simple", "_regime_gap": "gap"}[level]
        for reg, g in prior.groupby(level, sort=False):
            if len(g) < min_n:
                continue
            s = _best_scale_factor(g["y_true"].to_numpy(dtype=float), g[cal_col].to_numpy(dtype=float))
            fit[dest][str(reg)] = float(s)
            rows.append({"level": dest, "regime": str(reg), "n": len(g), "scale": float(s)})
    return fit, pd.DataFrame(rows)


def _apply_regime_scale(df: pd.DataFrame, cal_pred: np.ndarray, fit: Mapping[str, Any]) -> np.ndarray:
    cal_pred = np.asarray(cal_pred, dtype=float)
    out = np.zeros(len(df), dtype=float)
    reg_full = df["_regime_full"].astype(str).to_numpy() if "_regime_full" in df.columns else np.array([""] * len(df), dtype=object)
    reg_simple = df["_regime_simple"].astype(str).to_numpy() if "_regime_simple" in df.columns else np.array([""] * len(df), dtype=object)
    reg_gap = df["_regime_gap"].astype(str).to_numpy() if "_regime_gap" in df.columns else np.array([""] * len(df), dtype=object)
    for i in range(len(df)):
        s = None
        if fit["mode"] == "regime":
            for level_name, k in [("full", reg_full[i]), ("simple", reg_simple[i]), ("gap", reg_gap[i])]:
                if k in fit[level_name]:
                    s = float(fit[level_name][k])
                    break
        if s is None:
            s = float(fit["pooled"])
        out[i] = float(s * cal_pred[i])
    return out


def _fit_postprocess_on_prior(prior: pd.DataFrame, cand: PostprocessCandidate) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = _regime_thresholds(prior)
    prior_reg = _assign_regimes(prior, thresholds)
    stack_fit, stack_summary = _fit_regime_simplex(prior_reg, MEAN_MODEL_COLS, use_regime=bool(cand.regime_stack))
    raw_mean = _apply_regime_simplex(prior_reg, stack_fit)
    raw_blend = (1.0 - float(cand.q50_blend)) * raw_mean + float(cand.q50_blend) * prior_reg["pred_q50"].to_numpy(dtype=float)
    prior_reg = prior_reg.assign(_raw_blend=raw_blend)
    cal_fit, cal_summary = _fit_regime_calibration(prior_reg, "_raw_blend", cand.calibration_mode)
    cal_pred = _apply_regime_calibration(prior_reg, raw_blend, cal_fit)
    prior_reg = prior_reg.assign(_cal=cal_pred)
    scale_fit, scale_summary = _fit_regime_scale(prior_reg, "_cal", cand.scale_mode)
    y = prior_reg["y_true"].to_numpy(dtype=float)
    if float(cand.winsor_q) > 0:
        lo = float(np.quantile(y, float(cand.winsor_q)))
        hi = float(np.quantile(y, 1.0 - float(cand.winsor_q)))
    else:
        lo, hi = -np.inf, np.inf
    fit = {"thresholds": thresholds, "stack_fit": stack_fit, "cal_fit": cal_fit, "scale_fit": scale_fit, "winsor_bounds": (lo, hi)}
    return fit, stack_summary, cal_summary, scale_summary


def _apply_postprocess_fit(df: pd.DataFrame, fit: Mapping[str, Any], cand: PostprocessCandidate) -> Dict[str, np.ndarray]:
    dfr = _assign_regimes(df, fit["thresholds"])
    raw_mean = _apply_regime_simplex(dfr, fit["stack_fit"])
    raw_blend = (1.0 - float(cand.q50_blend)) * raw_mean + float(cand.q50_blend) * dfr["pred_q50"].to_numpy(dtype=float)
    cal = _apply_regime_calibration(dfr, raw_blend, fit["cal_fit"])
    scaled = cal.copy() if cand.scale_mode == "none" else _apply_regime_scale(dfr, cal, fit["scale_fit"])
    lo, hi = fit["winsor_bounds"]
    final = np.clip(scaled, lo, hi)
    return {"raw_mean": raw_mean, "raw_blend": raw_blend, "calibrated": cal, "scaled": scaled, "final": final}


def _fit_postprocess_model(inner_oof: pd.DataFrame, cand: PostprocessCandidate) -> PostprocessModel:
    fit, stack_summary, cal_summary, scale_summary = _fit_postprocess_on_prior(inner_oof.copy(), cand)
    return PostprocessModel(
        candidate=cand,
        thresholds=fit["thresholds"],
        stack_fit=fit["stack_fit"],
        cal_fit=fit["cal_fit"],
        scale_fit=fit["scale_fit"],
        winsor_bounds=fit["winsor_bounds"],
        regime_summary=stack_summary,
        calibration_summary=cal_summary,
        scale_summary=scale_summary,
    )


def _apply_postprocess_model(df: pd.DataFrame, model: PostprocessModel) -> Dict[str, np.ndarray]:
    fit = {
        "thresholds": model.thresholds,
        "stack_fit": model.stack_fit,
        "cal_fit": model.cal_fit,
        "scale_fit": model.scale_fit,
        "winsor_bounds": model.winsor_bounds,
    }
    return _apply_postprocess_fit(df, fit, model.candidate)


def _make_postprocess_grid() -> List[PostprocessCandidate]:
    # Curated grid to keep nested runtime tractable while preserving required comparisons:
    # pooled vs regime stack/calibration, no/pooled/regime scale, and mean-vs-median blending.
    out: List[PostprocessCandidate] = []
    templates = [
        (False, "pooled", "none"),
        (False, "pooled", "pooled"),
        (True, "pooled", "pooled"),
        (True, "regime", "pooled"),
        (True, "regime", "regime"),
    ]
    for q50_blend in [0.0, 0.25]:
        for winsor_q in [0.0, 0.01]:
            for regime_stack, cal_mode, scale_mode in templates:
                out.append(
                    PostprocessCandidate(
                        regime_stack=bool(regime_stack),
                        calibration_mode=str(cal_mode),
                        scale_mode=str(scale_mode),
                        q50_blend=float(q50_blend),
                        winsor_q=float(winsor_q),
                    )
                )
    return out


def _generate_inner_oof_and_outer_preds(
    candidate: CoreCandidate,
    variant_data: Mapping[str, Any],
    histgb_grid: Sequence[Mapping[str, Any]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with _timed_phase("_generate_inner_oof_and_outer_preds"):
        histgb_params = histgb_grid[int(candidate.histgb_idx)]
        feature_cols = variant_data["feature_cols_by_profile"][candidate.feature_profile]
        cfg = _runtime_cfg()
        inner_rows = []
        for split in variant_data["inner_splits"]:
            if _budget_total_exceeded() and inner_rows:
                break
            tr = split["train_nl"]
            va = split["val_nl"]
            y_val = va["HomeWinMargin"].to_numpy(dtype=float)
            base = va[
                [
                    "Date",
                    "elo_diff_pre",
                    "massey_diff",
                    "offdef_net_diff",
                    "volatility_home",
                    "volatility_away",
                    "games_played_home",
                    "games_played_away",
                    "same_conf_flag",
                    "cross_conf_flag",
                    "trend_margin_last5_vs_season_diff",
                    "trend_oppadj_last5_vs_season_diff",
                    "consistency_ratio_diff",
                ]
            ].copy()
            base["row_index"] = va.index.to_numpy()
            base["fold"] = int(split["fold"])
            base["y_true"] = y_val
            if cfg.enable_optimizations:
                pred_bundle = _predict_families_bundle(
                    ["ridge", "huber", "histgb", "histgb_bag", "histgb_q50"],
                    tr,
                    va,
                    feature_cols,
                    ridge_alpha=candidate.ridge_alpha,
                    histgb_params=histgb_params,
                    half_life_days=candidate.half_life_days,
                    seed=seed + 10,
                )
                base["pred_ridge"] = pred_bundle["ridge"]
                base["pred_huber"] = pred_bundle["huber"]
                base["pred_histgb"] = pred_bundle["histgb"]
                base["pred_histgb_bag"] = pred_bundle["histgb_bag"]
                base["pred_q50"] = pred_bundle["histgb_q50"]
            else:
                base["pred_ridge"] = _predict_family("ridge", tr, va, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 10)
                base["pred_huber"] = _predict_family("huber", tr, va, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 20)
                base["pred_histgb"] = _predict_family("histgb", tr, va, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 30)
                base["pred_histgb_bag"] = _predict_family("histgb_bag", tr, va, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 40)
                base["pred_q50"] = _predict_family("histgb_q50", tr, va, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 50)
            inner_rows.append(base)
        inner_oof = pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame()

        tr_full = variant_data["outer_train_nl"]
        va_outer = variant_data["outer_val_nl"]
        outer_pred_df = va_outer[
            [
                "Date",
                "elo_diff_pre",
                "massey_diff",
                "offdef_net_diff",
                "volatility_home",
                "volatility_away",
                "games_played_home",
                "games_played_away",
                "same_conf_flag",
                "cross_conf_flag",
                "trend_margin_last5_vs_season_diff",
                "trend_oppadj_last5_vs_season_diff",
                "consistency_ratio_diff",
            ]
        ].copy()
        outer_pred_df["row_index"] = va_outer.index.to_numpy()
        outer_pred_df["y_true"] = va_outer["HomeWinMargin"].to_numpy(dtype=float)
        if cfg.enable_optimizations:
            outer_bundle = _predict_families_bundle(
                ["ridge", "huber", "histgb", "histgb_bag", "histgb_q50"],
                tr_full,
                va_outer,
                feature_cols,
                ridge_alpha=candidate.ridge_alpha,
                histgb_params=histgb_params,
                half_life_days=candidate.half_life_days,
                seed=seed + 110,
            )
            outer_pred_df["pred_ridge"] = outer_bundle["ridge"]
            outer_pred_df["pred_huber"] = outer_bundle["huber"]
            outer_pred_df["pred_histgb"] = outer_bundle["histgb"]
            outer_pred_df["pred_histgb_bag"] = outer_bundle["histgb_bag"]
            outer_pred_df["pred_q50"] = outer_bundle["histgb_q50"]
        else:
            outer_pred_df["pred_ridge"] = _predict_family("ridge", tr_full, va_outer, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 110)
            outer_pred_df["pred_huber"] = _predict_family("huber", tr_full, va_outer, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 120)
            outer_pred_df["pred_histgb"] = _predict_family("histgb", tr_full, va_outer, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 130)
            outer_pred_df["pred_histgb_bag"] = _predict_family("histgb_bag", tr_full, va_outer, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 140)
            outer_pred_df["pred_q50"] = _predict_family("histgb_q50", tr_full, va_outer, feature_cols, ridge_alpha=candidate.ridge_alpha, histgb_params=histgb_params, half_life_days=candidate.half_life_days, seed=seed + 150)
        return inner_oof, outer_pred_df


def _evaluate_postprocess_progressive(inner_oof: pd.DataFrame, cand: PostprocessCandidate) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if inner_oof is None or inner_oof.empty:
        empty_metrics = {
            "candidate_key": cand.key(),
            "rmse_final": float("inf"),
            "mae_final": float("inf"),
            "bias_final": 0.0,
            "rmse_mean_stack": float("inf"),
            "rmse_q50": float("inf"),
            "rmse_blend_raw": float("inf"),
            "rmse_calibrated": float("inf"),
            "rmse_scaled": float("inf"),
            "pred_std_final": 0.0,
            "actual_std": 0.0,
            "dispersion_ratio": 1.0,
        }
        return empty_metrics, pd.DataFrame(), pd.DataFrame(columns=["row_index", "fold", "y_true", "pred_q50", "pred_raw_mean", "pred_raw_blend", "pred_calibrated", "pred_scaled", "pred_final"])
    df = inner_oof.sort_values(["fold", "row_index"], kind="mergesort").reset_index(drop=True)
    final_pred = np.zeros(len(df), dtype=float)
    raw_mean_all = np.zeros(len(df), dtype=float)
    raw_blend_all = np.zeros(len(df), dtype=float)
    cal_all = np.zeros(len(df), dtype=float)
    scaled_all = np.zeros(len(df), dtype=float)
    fold_rows: List[dict] = []

    for fold_id in sorted(df["fold"].unique().tolist()):
        val_mask = df["fold"].to_numpy(dtype=int) == int(fold_id)
        prior_mask = df["fold"].to_numpy(dtype=int) < int(fold_id)
        prior = df.loc[prior_mask].copy()
        val = df.loc[val_mask].copy()
        if len(prior) < 50:
            raw_mean = val[MEAN_MODEL_COLS].mean(axis=1).to_numpy(dtype=float)
            raw_blend = (1.0 - float(cand.q50_blend)) * raw_mean + float(cand.q50_blend) * val["pred_q50"].to_numpy(dtype=float)
            cal = raw_blend.copy()
            scaled = raw_blend.copy()
            final = raw_blend.copy()
            source = "warm_start"
        else:
            fit, _, _, _ = _fit_postprocess_on_prior(prior, cand)
            applied = _apply_postprocess_fit(val, fit, cand)
            raw_mean = applied["raw_mean"]
            raw_blend = applied["raw_blend"]
            cal = applied["calibrated"]
            scaled = applied["scaled"]
            final = applied["final"]
            source = "prior_folds"

        final_pred[val_mask] = final
        raw_mean_all[val_mask] = raw_mean
        raw_blend_all[val_mask] = raw_blend
        cal_all[val_mask] = cal
        scaled_all[val_mask] = scaled
        yv = val["y_true"].to_numpy(dtype=float)
        fold_rows.append(
            {
                "candidate_key": cand.key(),
                "fold": int(fold_id),
                "source": source,
                "n_val": int(len(val)),
                "rmse": _rmse(yv, final),
                "mae": _mae(yv, final),
                "bias": _bias(yv, final),
                "pred_std_final": float(np.std(final)),
                "pred_std_cal": float(np.std(cal)),
                "actual_std": float(np.std(yv)),
            }
        )

    y = df["y_true"].to_numpy(dtype=float)
    metrics = {
        "candidate_key": cand.key(),
        "rmse_final": _rmse(y, final_pred),
        "mae_final": _mae(y, final_pred),
        "bias_final": _bias(y, final_pred),
        "rmse_mean_stack": _rmse(y, raw_mean_all),
        "rmse_q50": _rmse(y, df["pred_q50"].to_numpy(dtype=float)),
        "rmse_blend_raw": _rmse(y, raw_blend_all),
        "rmse_calibrated": _rmse(y, cal_all),
        "rmse_scaled": _rmse(y, scaled_all),
        "pred_std_final": float(np.std(final_pred)),
        "actual_std": float(np.std(y)),
        "dispersion_ratio": float(np.std(final_pred) / max(np.std(y), 1e-9)),
    }
    pred_rows = df[["row_index", "fold", "y_true", "pred_q50"]].copy()
    pred_rows["pred_raw_mean"] = raw_mean_all
    pred_rows["pred_raw_blend"] = raw_blend_all
    pred_rows["pred_calibrated"] = cal_all
    pred_rows["pred_scaled"] = scaled_all
    pred_rows["pred_final"] = final_pred
    return metrics, pd.DataFrame(fold_rows), pred_rows


def _select_postprocess(inner_oof: pd.DataFrame, post_grid: Sequence[PostprocessCandidate]) -> tuple[PostprocessCandidate, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    fold_rows = []
    pred_map: Dict[str, pd.DataFrame] = {}
    for i, cand in enumerate(post_grid):
        if _budget_total_exceeded() and metric_rows:
            break
        m, f, p = _evaluate_postprocess_progressive(inner_oof, cand)
        metric_rows.append(m)
        fold_rows.append(f)
        pred_map[cand.key()] = p
        if _runtime_cfg().fast_mode and i + 1 >= int(_runtime_cfg().fast_post_grid_limit):
            break
    if not metric_rows:
        fallback = post_grid[0]
        m, f, p = _evaluate_postprocess_progressive(inner_oof, fallback)
        metric_rows.append(m)
        fold_rows.append(f)
        pred_map[fallback.key()] = p
    metric_df = pd.DataFrame(metric_rows)
    metric_df["selection_score"] = (
        metric_df["rmse_final"]
        + 0.18 * metric_df["mae_final"]
        + 0.12 * metric_df["bias_final"].abs()
        + 0.10 * (metric_df["dispersion_ratio"] - 1.0).abs()
    )
    metric_df = metric_df.sort_values(["selection_score", "rmse_final", "mae_final"], kind="mergesort").reset_index(drop=True)
    best_key = str(metric_df.iloc[0]["candidate_key"])
    best_cand = next(c for c in post_grid if c.key() == best_key)
    fold_diag = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    return best_cand, metric_df, fold_diag, pred_map[best_key].copy()


def _extract_recent_form_map(final_states: Mapping[int, Any]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for tid, st in final_states.items():
        try:
            ema_fast = float(st.ema_margin_by_alpha.get(0.35, getattr(st, "ema_margin", 0.0)))
            ema_mid = float(st.ema_margin_by_alpha.get(0.20, getattr(st, "ema_margin", 0.0)))
            last5 = float(np.mean(st.margin_history[-5:])) if getattr(st, "margin_history", None) else 0.0
            opp_last5 = float(np.mean(st.oppadj_history[-5:])) if getattr(st, "oppadj_history", None) else 0.0
            out[int(tid)] = 0.35 * ema_fast + 0.20 * ema_mid + 0.20 * last5 + 0.25 * opp_last5
        except Exception:
            out[int(tid)] = 0.0
    return out


def _zscore_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - float(s.mean())) / sd


def _compute_ranking_weights_from_oof(component_oof: pd.DataFrame) -> Dict[str, float]:
    cols = ["elo_diff_pre", "massey_diff", "offdef_net_diff", "recent_form_diff"]
    if component_oof is None or component_oof.empty or len(component_oof) < 10:
        return {"elo": 0.25, "massey": 0.25, "net": 0.25, "recent": 0.25}
    X = component_oof[cols].astype(float)
    y = component_oof["HomeWinMargin"].astype(float).to_numpy()
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=4.0, random_state=SEED_DEFAULT))])
    model.fit(X, y)
    coef = np.abs(np.asarray(model.named_steps["ridge"].coef_, dtype=float))
    if coef.sum() <= 0:
        coef = np.ones_like(coef)
    coef = coef / coef.sum()
    return {"elo": float(coef[0]), "massey": float(coef[1]), "net": float(coef[2]), "recent": float(coef[3])}


def _build_rankings_output_v2(
    rankings_df: pd.DataFrame,
    *,
    team_universe: pd.DataFrame,
    final_elo: Mapping[int, float],
    final_massey: Mapping[int, float],
    final_net: Mapping[int, float],
    recent_form: Mapping[int, float],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    out = rankings_df.copy()
    out["TeamID"] = out["TeamID"].astype(int)
    out = out.merge(team_universe.rename(columns={"Team": "Team_universe"}), on="TeamID", how="left")
    elo_s = out["TeamID"].map(final_elo).fillna(1500.0).astype(float)
    massey_s = out["TeamID"].map(final_massey).fillna(0.0).astype(float)
    net_s = out["TeamID"].map(final_net).fillna(0.0).astype(float)
    recent_s = out["TeamID"].map(recent_form).fillna(0.0).astype(float)
    out["_elo_z"] = _zscore_series(elo_s)
    out["_massey_z"] = _zscore_series(massey_s)
    out["_net_z"] = _zscore_series(net_s)
    out["_recent_z"] = _zscore_series(recent_s)
    out["_score"] = (
        float(weights.get("elo", 0.25)) * out["_elo_z"]
        + float(weights.get("massey", 0.25)) * out["_massey_z"]
        + float(weights.get("net", 0.25)) * out["_net_z"]
        + float(weights.get("recent", 0.25)) * out["_recent_z"]
    )
    out = out.sort_values(["_score", "TeamID"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1, dtype=int)
    out = out.sort_values("TeamID", kind="mergesort").reset_index(drop=True)
    out["Rank"] = out["Rank"].astype(int)
    return out.drop(columns=[c for c in out.columns if c.startswith("_") or c == "Team_universe"])


def _ks_psi(train_vals: np.ndarray, derby_vals: np.ndarray, n_bins: int = 8) -> tuple[float, float, float]:
    train_vals = np.asarray(train_vals, dtype=float)
    derby_vals = np.asarray(derby_vals, dtype=float)
    train_vals = train_vals[np.isfinite(train_vals)]
    derby_vals = derby_vals[np.isfinite(derby_vals)]
    if len(train_vals) == 0 or len(derby_vals) == 0:
        return float("nan"), float("nan"), float("nan")
    ks = stats.ks_2samp(train_vals, derby_vals)
    qs = np.unique(np.quantile(train_vals, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        return float(ks.statistic), float(ks.pvalue), 0.0
    bins = np.asarray(qs, dtype=float)
    bins[0] = -np.inf
    bins[-1] = np.inf
    th, _ = np.histogram(train_vals, bins=bins)
    dh, _ = np.histogram(derby_vals, bins=bins)
    tp = np.clip(th / max(th.sum(), 1), 1e-6, None)
    dp = np.clip(dh / max(dh.sum(), 1), 1e-6, None)
    psi = float(np.sum((dp - tp) * np.log(dp / tp)))
    return float(ks.statistic), float(ks.pvalue), psi


def _shift_diagnostics(train_tbl: pd.DataFrame, derby_tbl: pd.DataFrame) -> pd.DataFrame:
    recent_cut = pd.to_datetime(train_tbl["Date"]).quantile(0.75)
    train_recent = train_tbl[pd.to_datetime(train_tbl["Date"]) >= recent_cut].copy()
    rows = []
    for c in [
        "elo_diff_pre",
        "elo_neutral_diff_pre",
        "massey_diff",
        "offdef_net_diff",
        "offdef_margin_neutral",
        "volatility_sum",
        "same_conf_flag",
        "games_played_min",
        "trend_margin_last5_vs_season_diff",
        "consistency_ratio_diff",
    ]:
        if c not in train_recent.columns or c not in derby_tbl.columns:
            continue
        tr = train_recent[c].astype(float).to_numpy()
        de = derby_tbl[c].astype(float).to_numpy()
        ks_stat, ks_p, psi = _ks_psi(tr, de, n_bins=8)
        q1, q99 = np.quantile(tr, [0.01, 0.99]) if len(tr) else (np.nan, np.nan)
        out_of_range = float(np.mean((de < q1) | (de > q99))) if len(de) and np.isfinite(q1) and np.isfinite(q99) else np.nan
        rows.append(
            {
                "feature": c,
                "train_recent_mean": float(np.nanmean(tr)),
                "derby_mean": float(np.nanmean(de)),
                "train_recent_std": float(np.nanstd(tr)),
                "derby_std": float(np.nanstd(de)),
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
                "psi": psi,
                "derby_outside_train_1_99_frac": out_of_range,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ood_flag"] = (out["psi"].fillna(0.0) > 0.20) | (out["ks_stat"].fillna(0.0) > 0.30) | (out["derby_outside_train_1_99_frac"].fillna(0.0) > 0.20)
        out = out.sort_values(["ood_flag", "psi", "ks_stat"], ascending=[False, False, False], kind="mergesort").reset_index(drop=True)
    return out


def _format_df_for_text(df: pd.DataFrame, max_rows: int = 20, width: int = 140) -> str:
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
    fig.text(0.05, 0.97, title, fontsize=15, fontweight="bold", va="top")
    y = 0.94
    for line in lines:
        if y < 0.05:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            fig.text(0.05, 0.97, f"{title} (cont.)", fontsize=15, fontweight="bold", va="top")
            y = 0.94
        fig.text(0.05, y, line, fontsize=8.6, va="top", family="monospace")
        y -= 0.017
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
        tbl.set_fontsize(6.4)
        tbl.scale(1, 1.15)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _timing_phase_table(summary: Mapping[str, Any], focus_phases: Sequence[str]) -> pd.DataFrame:
    pt = dict(summary.get("phase_totals_sec", {}) or {})
    pc = dict(summary.get("phase_counts", {}) or {})
    rows = []
    for p in focus_phases:
        rows.append({"phase": p, "total_sec": float(pt.get(p, 0.0)), "calls": int(pc.get(p, 0))})
    return pd.DataFrame(rows).sort_values("total_sec", ascending=False, kind="mergesort").reset_index(drop=True)


def _timing_family_table(summary: Mapping[str, Any]) -> pd.DataFrame:
    pf = dict(summary.get("predict_family_totals_sec", {}) or {})
    pfc = dict(summary.get("predict_family_call_counts", {}) or {})
    fitc = dict(summary.get("model_fit_counts_by_family", {}) or {})
    fams = sorted(set(pf.keys()) | set(pfc.keys()) | set(fitc.keys()))
    rows = []
    for fam in fams:
        rows.append(
            {
                "family": fam,
                "predict_time_sec": float(pf.get(fam, 0.0)),
                "predict_calls": int(pfc.get(fam, 0)),
                "fit_count": int(fitc.get(fam, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["predict_time_sec", "fit_count"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def _load_json_if_exists(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _timing_compare_table(before_summary: Optional[Mapping[str, Any]], after_summary: Mapping[str, Any], focus_phases: Sequence[str]) -> pd.DataFrame:
    before_pt = dict((before_summary or {}).get("phase_totals_sec", {}) or {})
    before_fit = dict((before_summary or {}).get("model_fit_counts_by_family", {}) or {})
    after_pt = dict(after_summary.get("phase_totals_sec", {}) or {})
    after_fit = dict(after_summary.get("model_fit_counts_by_family", {}) or {})
    rows = []
    for p in focus_phases:
        b = float(before_pt.get(p, np.nan)) if before_summary is not None else np.nan
        a = float(after_pt.get(p, np.nan))
        rows.append({"kind": "phase", "name": p, "before_sec": b, "after_sec": a, "delta_sec": a - b if np.isfinite(b) and np.isfinite(a) else np.nan})
    for fam in sorted(set(before_fit.keys()) | set(after_fit.keys())):
        b = float(before_fit.get(fam, np.nan)) if before_summary is not None else np.nan
        a = float(after_fit.get(fam, np.nan))
        rows.append({"kind": "fit_count", "name": fam, "before_sec": b, "after_sec": a, "delta_sec": a - b if np.isfinite(b) and np.isfinite(a) else np.nan})
    return pd.DataFrame(rows)


def _write_run_report(
    path: Path,
    *,
    root: Path,
    inputs_used: Mapping[str, Path],
    runtime_cfg: RuntimeConfig,
    budget: Optional[BudgetController],
    timing_summary: Mapping[str, Any],
    selected_summary: Mapping[str, Any],
    outer_rmse: float,
    outer_mae: float,
    predictions_out: pd.DataFrame,
    rankings_out: pd.DataFrame,
) -> None:
    focus_phases = [
        "_scan_and_select_core_candidates",
        "_core_scan_score",
        "_generate_inner_oof_and_outer_preds",
        "_predict_family",
        "_build_split_tables",
        "_prepare_variant_outer_data",
    ]
    fast_before = _load_json_if_exists(root / "timing_summary_fast_baseline.json")
    fast_after = _load_json_if_exists(root / "timing_summary_fast.json")
    lines: List[str] = []
    lines.append("# AlgoSports23 Run Report")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {pd.Timestamp.utcnow().isoformat()}")
    lines.append(f"- Working directory: `{root}`")
    lines.append(f"- Thread limits: `OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')}`, `MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '')}`")
    lines.append(f"- Runtime config: fast_mode={int(runtime_cfg.fast_mode)}, max_total={runtime_cfg.max_total_seconds}s, max_scan={runtime_cfg.max_scan_seconds}s, max_fits={runtime_cfg.max_fits}, bag_n={runtime_cfg.histgb_bag_n_models}, optimizations={int(runtime_cfg.enable_optimizations)}")
    lines.append("")
    lines.append("## Files Used / Found")
    for k, p in inputs_used.items():
        lines.append(f"- {k}: `{p}` (exists={p.exists()})")
    lines.append("")
    lines.append("## Fast Profile Findings")
    if fast_after is None:
        lines.append("- `timing_summary_fast.json` not found in this run session.")
    else:
        lines.append("- Fast profile timing summary loaded from `timing_summary_fast.json`.")
        if fast_before is not None:
            lines.append("- Baseline fast profile timing summary loaded from `timing_summary_fast_baseline.json`.")
            cmp_df = _timing_compare_table(fast_before, fast_after, focus_phases)
            lines.append("")
            lines.append("### Fast Profile Before vs After (selected phases / fit counts)")
            lines.append("")
            lines.append("```text")
            lines.append(_format_df_for_text(cmp_df, max_rows=40, width=160))
            lines.append("```")
        else:
            lines.append("- Baseline fast profile file not found; reporting optimized fast summary only.")
        lines.append("")
        lines.append("### Optimized Fast Timing (focus phases)")
        lines.append("")
        lines.append("```text")
        lines.extend(_timing_summary_lines(fast_after, focus_phases=focus_phases, top_n=10))
        lines.append("```")
    lines.append("")
    lines.append("## Final Run Timing Summary")
    lines.append("")
    lines.append("```text")
    lines.extend(_timing_summary_lines(timing_summary, focus_phases=focus_phases, top_n=10))
    lines.append("```")
    lines.append("")
    lines.append("## Budgets / Safety Controls")
    if budget is None:
        lines.append("- No active budget controller recorded.")
    else:
        lines.append(f"- Elapsed wall time (pipeline): {budget.elapsed():.2f}s")
        lines.append(f"- Fit count used: {budget.fit_count}")
        lines.append(f"- Budgets triggered: {', '.join(sorted(budget.triggered.keys())) if budget.triggered else 'none'}")
        if budget.events:
            lines.append("- Budget events:")
            for msg in budget.events[:20]:
                lines.append(f"  - {msg}")
    lines.append("")
    lines.append("## Chosen Model / CV Metrics")
    lines.append(f"- Selected model family: {selected_summary.get('model_family')}")
    lines.append(f"- Selected Elo variant: {selected_summary.get('elo_variant')}")
    lines.append(f"- Feature profile / half-life: {selected_summary.get('feature_profile')} / {selected_summary.get('half_life_days')}")
    lines.append(f"- Calibration / scale / regime stack: {selected_summary.get('calibration_mode')} / {selected_summary.get('scale_mode')} / {selected_summary.get('use_regime_stack')}")
    lines.append(f"- Nested outer RMSE / MAE: {outer_rmse:.5f} / {outer_mae:.5f}")
    lines.append("")
    lines.append("## Sanity Checks")
    pred_num = pd.to_numeric(predictions_out.get('Team1_WinMargin'), errors='coerce') if "Team1_WinMargin" in predictions_out.columns else pd.Series(dtype=float)
    lines.append(f"- predictions.csv rows={len(predictions_out)}; Team1_WinMargin numeric+nonmissing={bool(len(pred_num)==len(predictions_out) and pred_num.notna().all())}")
    rank_vals = pd.to_numeric(rankings_out.get('Rank'), errors='coerce') if "Rank" in rankings_out.columns else pd.Series(dtype=float)
    rank_ok = bool(len(rankings_out) == 165 and len(rank_vals) == len(rankings_out) and rank_vals.notna().all() and set(rank_vals.astype(int).tolist()) == set(range(1, 166)))
    lines.append(f"- rankings.xlsx rows={len(rankings_out)}; Rank exactly 1..165={rank_ok}")
    lines.append("")
    lines.append("## Output File Paths")
    for out_name in ["predictions.csv", "rankings.xlsx", "final_report.pdf", "run_report.md"]:
        p = root / out_name
        size = p.stat().st_size if p.exists() else -1
        lines.append(f"- `{p}` (exists={p.exists()}, bytes={size})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_report(
    out_path: Path,
    *,
    train_df: pd.DataFrame,
    pred_template: pd.DataFrame,
    outer_folds: Sequence[Mapping[str, Any]],
    outer_summary: pd.DataFrame,
    outer_fold_metrics: pd.DataFrame,
    outer_pred_oof: pd.DataFrame,
    core_scan_summary: pd.DataFrame,
    elo_variant_compare: pd.DataFrame,
    recency_compare: pd.DataFrame,
    histgb_sweep_summary: pd.DataFrame,
    postprocess_compare: pd.DataFrame,
    scale_dispersion_table: pd.DataFrame,
    shift_table: pd.DataFrame,
    shift_mitigation_notes: List[str],
    regime_tables: Dict[str, pd.DataFrame],
    calibration_tables: Dict[str, pd.DataFrame],
    scale_tables: Dict[str, pd.DataFrame],
    final_derby_pred_float: np.ndarray,
    final_derby_pred_int: np.ndarray,
    final_derby_stage_df: pd.DataFrame,
    rankings_out: pd.DataFrame,
    prediction_head: pd.DataFrame,
    selected_summary: Mapping[str, Any],
    selected_hparams_table: pd.DataFrame,
    runtime_notes: Optional[List[str]] = None,
    runtime_phase_table: Optional[pd.DataFrame] = None,
    runtime_family_table: Optional[pd.DataFrame] = None,
    runtime_compare_table: Optional[pd.DataFrame] = None,
) -> None:
    with PdfPages(out_path) as pdf:
        outer_rmse = float(outer_fold_metrics["rmse"].mean()) if not outer_fold_metrics.empty else float("nan")
        outer_mae = float(outer_fold_metrics["mae"].mean()) if not outer_fold_metrics.empty else float("nan")
        _new_text_page(
            pdf,
            "1. Executive Summary",
            [
                "1. Executive summary",
                f"- Train games={len(train_df)} | Derby games={len(pred_template)}",
                f"- Nested CV={len(outer_folds)} outer folds x 3 inner folds (expanding time-aware)",
                f"- Selected model family: {selected_summary.get('model_family')}",
                f"- Selected Elo variant: {selected_summary.get('elo_variant')}",
                f"- Nested outer RMSE/MAE: {outer_rmse:.3f} / {outer_mae:.3f}",
                f"- Calibration={selected_summary.get('calibration_mode')} | Scale={selected_summary.get('scale_mode')} | Regime stack={selected_summary.get('use_regime_stack')}",
                f"- Runtime safety: fast_mode={selected_summary.get('fast_mode')} | budget_triggered={selected_summary.get('budget_triggered')}",
            ],
        )
        if runtime_notes:
            _new_text_page(pdf, "1B. Runtime Bottleneck And Fixes", ["Runtime bottleneck and fixes"] + [f"- {x}" for x in runtime_notes])
        if runtime_compare_table is not None and not runtime_compare_table.empty:
            _page_with_table(pdf, "1C. Fast Profiling Before vs After (timing / fits)", runtime_compare_table, max_rows_per_page=24)
        if runtime_phase_table is not None and not runtime_phase_table.empty:
            _page_with_table(pdf, "1D. Runtime Focus Phase Timing Summary", runtime_phase_table, max_rows_per_page=24)
        if runtime_family_table is not None and not runtime_family_table.empty:
            _page_with_table(pdf, "1E. _predict_family Aggregated Timing / Fit Counts", runtime_family_table, max_rows_per_page=24)

        lines = [
            "2. What changed vs previous pipeline",
            "- Nested tuning for post-model layers and key hyperparameters (Ridge alpha, HistGB grid, recency half-life).",
            "- Regime-specific simplex stacking + regime-specific calibration + regime/pool scale correction with fallback.",
            "- Recency features: EMA multi-alpha, last3/5/8, trend, consistency ratio.",
            "- Upgraded Elo variants: dynamic-K, team-specific regularized home effect, inactivity decay.",
            "- HistGB seed bagging and quantile HistGB (median, plus q20/q80 for diagnostics).",
            "- Shift diagnostics (KS + PSI + out-of-range fraction) and OOD-triggered mitigation.",
            "",
            "3. Data constraints and leakage controls",
            "- All features are pregame-only and built sequentially.",
            "- Static rating models refit inside each train split; no outer-fold leakage in postprocessing.",
            "- Derby predictions are neutral-site (home effects zeroed for derby features).",
            "",
            "4. Nested CV design (outer/inner schematic)",
        ]
        for f in outer_folds:
            lines.append(
                f" outer_fold={f['fold']} train_n={len(f['train_idx'])} val_n={len(f['val_idx'])} "
                f"train_end={pd.Timestamp(f['train_end_date']).date()} "
                f"val={pd.Timestamp(f['val_start_date']).date()}..{pd.Timestamp(f['val_end_date']).date()}"
            )
        _new_text_page(pdf, "2-4. Changes / Leakage / Nested CV", lines)

        _page_with_table(pdf, "5. Outer-fold selections and metrics", outer_summary, max_rows_per_page=18)
        _page_with_table(pdf, "5. Outer-fold RMSE/MAE table", outer_fold_metrics, max_rows_per_page=18)
        _page_with_table(pdf, "6. Feature additions and recency ablations (core scan)", recency_compare, max_rows_per_page=24)
        _page_with_table(pdf, "6. Core scan results (top rows)", core_scan_summary.head(80), max_rows_per_page=24)
        _page_with_table(pdf, "7. Elo variant comparisons", elo_variant_compare, max_rows_per_page=24)
        _page_with_table(pdf, "7. HistGB hyperparameter sweep summary", histgb_sweep_summary, max_rows_per_page=24)
        _page_with_table(pdf, "8. Regime-specific stacking / calibration results", postprocess_compare, max_rows_per_page=24)
        _page_with_table(pdf, "9. Dispersion and scale correction analysis", scale_dispersion_table, max_rows_per_page=24)
        _page_with_table(pdf, "10. Shift diagnostics: Train recent vs Derby", shift_table, max_rows_per_page=24)
        _new_text_page(pdf, "10. Shift Mitigation Notes", ["10. Shift mitigation rules"] + [f"- {x}" for x in shift_mitigation_notes])

        for name, tbl in regime_tables.items():
            _page_with_table(pdf, f"8. Regime weights ({name})", tbl, max_rows_per_page=24)
        for name, tbl in calibration_tables.items():
            _page_with_table(pdf, f"8. Calibration params ({name})", tbl, max_rows_per_page=24)
        for name, tbl in scale_tables.items():
            _page_with_table(pdf, f"9. Scale params ({name})", tbl, max_rows_per_page=24)

        oof = outer_pred_oof.copy()
        if oof.empty:
            oof = pd.DataFrame({"y_true": np.array([0.0]), "pred_final": np.array([0.0]), "pred_raw_mean": np.array([0.0]), "pred_q50": np.array([0.0]), "pred_raw_blend": np.array([0.0]), "pred_calibrated": np.array([0.0]), "elo_diff_pre": np.array([0.0])})
        y = oof["y_true"].to_numpy(dtype=float)
        p = oof["pred_final"].to_numpy(dtype=float)
        resid = y - p
        tail = oof.copy()
        tail["abs_error"] = np.abs(tail["y_true"] - tail["pred_final"])
        q90 = float(np.quantile(tail["abs_error"], 0.90)) if len(tail) else np.nan
        tail_top = tail[tail["abs_error"] >= q90].sort_values("abs_error", ascending=False, kind="mergesort").head(40)

        _new_text_page(
            pdf,
            "11. Residual Diagnostics (Summary)",
            [
                "11. Residual diagnostics",
                f"- OOF final RMSE={_rmse(y, p):.3f}, MAE={_mae(y, p):.3f}, Bias={_bias(y, p):.3f}",
                f"- Residual std={np.std(resid):.3f}, skew={stats.skew(resid, bias=False):.3f}, kurtosis={stats.kurtosis(resid, fisher=True, bias=False):.3f}",
                f"- Mean-stack RMSE={_rmse(y, oof['pred_raw_mean'].to_numpy(dtype=float)):.3f}",
                f"- Quantile median (q50) RMSE={_rmse(y, oof['pred_q50'].to_numpy(dtype=float)):.3f}",
                f"- Mean+median blend RMSE={_rmse(y, oof['pred_raw_blend'].to_numpy(dtype=float)):.3f}",
                f"- Calibrated RMSE={_rmse(y, oof['pred_calibrated'].to_numpy(dtype=float)):.3f}",
                f"- Tail error threshold (top 10%)={q90:.3f}",
            ],
        )
        _page_with_table(pdf, "11. Tail error diagnostics (top 10% abs-error)", tail_top, max_rows_per_page=22)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("11. Residual diagnostics plots", fontsize=14, fontweight="bold")
        axes[0, 0].hist(resid, bins=30, color="#4C78A8", edgecolor="white")
        axes[0, 0].set_title("Residual histogram")
        axes[0, 1].scatter(p, resid, s=14, alpha=0.5)
        axes[0, 1].axhline(0, color="black", lw=1)
        axes[0, 1].set_title("Residual vs fitted")
        axes[0, 1].set_xlabel("Fitted")
        axes[0, 1].set_ylabel("Residual")
        (osm, osr), (slope, intercept, rqq) = stats.probplot(resid, dist="norm")
        axes[1, 0].scatter(osm, osr, s=14, alpha=0.55)
        axes[1, 0].plot(osm, slope * np.asarray(osm) + intercept, color="#E45756", lw=2)
        axes[1, 0].set_title(f"QQ-style residual (r={rqq:.3f})")
        axes[1, 1].scatter(oof["elo_diff_pre"].to_numpy(dtype=float), np.abs(resid), s=14, alpha=0.4)
        axes[1, 1].set_title("|Residual| vs elo_diff_pre")
        axes[1, 1].set_xlabel("elo_diff_pre")
        axes[1, 1].set_ylabel("|Residual|")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        desc = pd.Series(final_derby_pred_float).describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
        _new_text_page(
            pdf,
            "12-14. Final Decisions / Limitations / Next Steps",
            [
                "12. Final derby prediction distribution",
                f"- count={len(final_derby_pred_float)}, mean={np.mean(final_derby_pred_float):.3f}, std={np.std(final_derby_pred_float):.3f}",
                f"- min={np.min(final_derby_pred_float):.3f}, max={np.max(final_derby_pred_float):.3f}",
                f"- quantiles 1/5/25/50/75/95/99 = {desc.get('1%', np.nan):.3f}, {desc.get('5%', np.nan):.3f}, {desc.get('25%', np.nan):.3f}, {desc.get('50%', np.nan):.3f}, {desc.get('75%', np.nan):.3f}, {desc.get('95%', np.nan):.3f}, {desc.get('99%', np.nan):.3f}",
                "",
                "13. Final decisions and rationale",
                f"- Final core candidate: {selected_summary.get('core_candidate_key')}",
                f"- Final postprocess candidate: {selected_summary.get('postprocess_key')}",
                f"- Recency profile={selected_summary.get('feature_profile')} | half_life={selected_summary.get('half_life_days')}",
                f"- Shift mitigation applied={selected_summary.get('shift_mitigation_applied')}",
                f"- Dispersion guard applied={selected_summary.get('dispersion_guard_applied')}",
                "",
                "14. Limitations + next steps",
                "- Derby labels hidden: final selection relies on nested train-only proxies.",
                "- No injuries/lineups/travel features available.",
                "- Future work: monotone piecewise regime calibration and richer Bayesian uncertainty.",
            ],
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle("12. Final derby prediction distribution", fontsize=13, fontweight="bold")
        axes[0].hist(final_derby_pred_float, bins=20, color="#72B7B2", edgecolor="white")
        axes[0].set_title("Float predictions")
        axes[1].hist(final_derby_pred_int, bins=20, color="#F58518", edgecolor="white")
        axes[1].set_title("Rounded predictions")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _page_with_table(pdf, "Appendix A. First 10 rows of predictions.csv", prediction_head, max_rows_per_page=15)
        _page_with_table(pdf, "Appendix B. Top 20 rankings by Rank", rankings_out.sort_values("Rank", kind="mergesort").head(20), max_rows_per_page=24)
        _page_with_table(pdf, "Appendix C. Selected hyperparameters", selected_hparams_table, max_rows_per_page=24)
        if final_derby_stage_df is not None and not final_derby_stage_df.empty:
            _page_with_table(pdf, "Appendix D. Final derby stage outputs (head)", final_derby_stage_df.head(20), max_rows_per_page=24)


def _prepare_full_derby_predictions(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    seq_build,
    team_ids: Sequence[int],
    conf_values: Sequence[str],
    core_candidate: CoreCandidate,
    histgb_params: Mapping[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, dict]:
    static_models_full = fit_static_models_for_fold(train_df, team_ids=team_ids, conf_values=conf_values)
    train_static = apply_static_models_to_train_like_rows(train_df, static_models_full, neutral_site=False)
    pred_static = apply_static_models_to_derby_rows(pred_df, static_models_full)
    seq_train = seq_build.features
    seq_derby = build_derby_sequential_features(pred_df, seq_build.final_states, seq_build.final_elo)
    train_tbl = assemble_model_table(train_df, seq_train, train_static)
    derby_tbl = assemble_model_table(pred_df, seq_derby, pred_static)
    train_tbl.index = train_df.index
    derby_tbl.index = pred_df.index
    train_tbl["Date"] = pd.to_datetime(train_tbl["Date"])
    derby_tbl["Date"] = pd.to_datetime(derby_tbl["Date"])
    train_nl = _add_nonlinear_features(train_tbl)
    derby_nl = _add_nonlinear_features(derby_tbl)
    all_cols = _select_feature_columns(train_nl)
    feature_cols = _feature_profile_columns(all_cols, core_candidate.feature_profile)

    derby_pred_df = derby_nl[
        [
            "Date",
            "elo_diff_pre",
            "massey_diff",
            "offdef_net_diff",
            "volatility_home",
            "volatility_away",
            "games_played_home",
            "games_played_away",
            "same_conf_flag",
            "cross_conf_flag",
            "trend_margin_last5_vs_season_diff",
            "trend_oppadj_last5_vs_season_diff",
            "consistency_ratio_diff",
        ]
    ].copy()
    derby_pred_df["row_index"] = derby_nl.index.to_numpy()
    family_spec = [
        ("ridge", "pred_ridge", 10),
        ("huber", "pred_huber", 20),
        ("histgb", "pred_histgb", 30),
        ("histgb_bag", "pred_histgb_bag", 40),
        ("histgb_q50", "pred_q50", 50),
    ]
    cfg = _runtime_cfg()
    if not (cfg.fast_mode and cfg.fast_disable_q20_q80):
        family_spec.extend([("histgb_q20", "pred_q20", 60), ("histgb_q80", "pred_q80", 70)])
    if cfg.enable_optimizations:
        fams = [fam for fam, _, _ in family_spec]
        preds = _predict_families_bundle(
            fams,
            train_nl,
            derby_nl,
            feature_cols,
            ridge_alpha=core_candidate.ridge_alpha,
            histgb_params=histgb_params,
            half_life_days=core_candidate.half_life_days,
            seed=seed + 10,
        )
        for fam, col, _ in family_spec:
            derby_pred_df[col] = preds[fam]
    else:
        for fam, col, sdelta in family_spec:
            derby_pred_df[col] = _predict_family(
                fam,
                train_nl,
                derby_nl,
                feature_cols,
                ridge_alpha=core_candidate.ridge_alpha,
                histgb_params=histgb_params,
                half_life_days=core_candidate.half_life_days,
                seed=seed + sdelta,
            )
    return train_nl, derby_nl, static_models_full, {"derby_pred_df": derby_pred_df}


def run_nextgen_pipeline(root: Path, seed: int = SEED_DEFAULT) -> dict:
    global _ACTIVE_PROFILER, _ACTIVE_RUNTIME_CFG, _ACTIVE_BUDGET, _PIPELINE_START_PERF
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    _set_deterministic(seed)
    runtime_cfg = _load_runtime_config_from_env()
    prev_profiler = _ACTIVE_PROFILER
    prev_runtime_cfg = _ACTIVE_RUNTIME_CFG
    prev_budget = _ACTIVE_BUDGET
    prev_pipeline_start = _PIPELINE_START_PERF
    _ACTIVE_PROFILER = RuntimeProfiler()
    _ACTIVE_RUNTIME_CFG = runtime_cfg
    _ACTIVE_BUDGET = BudgetController(cfg=runtime_cfg)
    _PIPELINE_START_PERF = time.perf_counter()
    _PREP_CACHE.clear()
    _BUILD_SPLIT_TABLES_CACHE.clear()
    _PREPARE_VARIANT_CACHE.clear()
    pipeline_t0 = time.perf_counter()

    paths = _resolve_input_paths(root)
    inputs = _load_inputs(paths)
    train_raw = inputs["train"]
    pred_raw = inputs["pred"]
    rankings_raw = inputs["rankings"]

    train_df = parse_and_sort_train(train_raw)
    pred_df = parse_predictions(pred_raw)
    team_universe = build_team_universe(rankings_raw)
    team_ids = team_universe["TeamID"].astype(int).tolist()
    conf_values = sorted(set(train_df["HomeConf"].astype(str)) | set(train_df["AwayConf"].astype(str)))

    elo_variants = _build_elo_variants()
    seq_builds: Dict[str, Any] = {}
    seq_by_variant_train: Dict[str, pd.DataFrame] = {}
    for name, cfg in elo_variants.items():
        sb = build_train_sequential_features(
            train_df,
            home_adv=float(cfg.home_adv),
            team_ids=team_ids,
            elo_k=float(cfg.k_factor),
            elo_variant=cfg,
        )
        seq_builds[name] = sb
        seq_copy = sb.features.copy()
        seq_copy.attrs["_variant_name"] = str(name)
        seq_by_variant_train[name] = seq_copy

    histgb_grid = _histgb_param_grid()
    outer_n_folds = int(runtime_cfg.fast_outer_folds) if runtime_cfg.fast_mode else 4
    outer_folds = make_expanding_time_folds(train_df, n_folds=outer_n_folds)

    outer_results = []
    outer_oof_rows = []
    outer_scan_rows = []
    outer_disp_rows = []
    regime_tables: Dict[str, pd.DataFrame] = {}
    calibration_tables: Dict[str, pd.DataFrame] = {}
    scale_tables: Dict[str, pd.DataFrame] = {}
    postprocess_metric_rows = []

    for ofold in outer_folds:
        if _budget_total_exceeded() and outer_results:
            break
        outer_train_df_local = train_df.iloc[np.asarray(ofold["train_idx"], dtype=int)].copy().reset_index(drop=True)
        inner_n_folds = int(runtime_cfg.fast_inner_folds) if runtime_cfg.fast_mode else 3
        inner_folds = make_expanding_time_folds(outer_train_df_local, n_folds=inner_n_folds)
        outer_variant_data = {
            vname: _prepare_variant_outer_data(
                train_df=train_df,
                outer_fold=ofold,
                inner_folds=inner_folds,
                seq_variant_full=seq_df,
                team_ids=team_ids,
                conf_values=conf_values,
                variant_name=vname,
            )
            for vname, seq_df in seq_by_variant_train.items()
        }

        top_core, scan_df = _scan_and_select_core_candidates(outer_variant_data, histgb_grid, seed=seed + 100 * int(ofold["fold"]))
        scan_df["outer_fold"] = int(ofold["fold"])
        outer_scan_rows.append(scan_df)

        post_grid = _make_postprocess_grid()
        bundle_lookup: Dict[str, dict] = {}
        eval_rows = []

        for i, cand in enumerate(top_core):
            if _budget_total_exceeded() and eval_rows:
                break
            vdata = outer_variant_data[cand.elo_variant]
            inner_oof, outer_val_base = _generate_inner_oof_and_outer_preds(
                cand,
                vdata,
                histgb_grid,
                seed=seed + 1000 + 200 * int(ofold["fold"]) + 17 * i,
            )
            if inner_oof.empty:
                continue
            best_post, post_metric_df, post_fold_diag, best_post_pred_rows = _select_postprocess(inner_oof, post_grid)
            post_model = _fit_postprocess_model(inner_oof, best_post)
            outer_applied = _apply_postprocess_model(outer_val_base, post_model)
            y_outer = outer_val_base["y_true"].to_numpy(dtype=float)
            outer_metrics = _reg_metrics(y_outer, outer_applied["final"])

            pooled_comp = PostprocessCandidate(
                regime_stack=False,
                calibration_mode="pooled",
                scale_mode="pooled" if best_post.scale_mode != "none" else "none",
                q50_blend=best_post.q50_blend,
                winsor_q=best_post.winsor_q,
            )
            pooled_model = _fit_postprocess_model(inner_oof, pooled_comp)
            pooled_outer = _apply_postprocess_model(outer_val_base, pooled_model)
            pooled_rmse = _rmse(y_outer, pooled_outer["final"])

            row = {
                "outer_fold": int(ofold["fold"]),
                "core_candidate_key": cand.key(),
                "elo_variant": cand.elo_variant,
                "feature_profile": cand.feature_profile,
                "half_life_days": np.nan if cand.half_life_days is None else cand.half_life_days,
                "ridge_alpha": cand.ridge_alpha,
                "histgb_idx": cand.histgb_idx,
                "inner_best_post_key": best_post.key(),
                "inner_post_rmse": float(post_metric_df.iloc[0]["rmse_final"]),
                "inner_post_mae": float(post_metric_df.iloc[0]["mae_final"]),
                "inner_mean_stack_rmse": float(post_metric_df.iloc[0]["rmse_mean_stack"]),
                "inner_q50_rmse": float(post_metric_df.iloc[0]["rmse_q50"]),
                "outer_rmse": outer_metrics["rmse"],
                "outer_mae": outer_metrics["mae"],
                "outer_bias": outer_metrics["bias"],
                "outer_pred_std": outer_metrics["pred_std"],
                "outer_actual_std": outer_metrics["actual_std"],
                "outer_dispersion_ratio": outer_metrics["pred_std"] / max(outer_metrics["actual_std"], 1e-9),
                "outer_pooled_comp_rmse": float(pooled_rmse),
                "outer_regime_gain_vs_pooled": float(pooled_rmse - outer_metrics["rmse"]),
                "use_regime_stack": bool(best_post.regime_stack),
                "calibration_mode": best_post.calibration_mode,
                "scale_mode": best_post.scale_mode,
                "q50_blend": float(best_post.q50_blend),
                "winsor_q": float(best_post.winsor_q),
            }
            eval_rows.append(row)
            postprocess_metric_rows.append(post_metric_df.assign(outer_fold=int(ofold["fold"]), core_candidate_key=cand.key()))

            bundle_lookup[cand.key()] = {
                "candidate": cand,
                "inner_oof": inner_oof,
                "post_metric_df": post_metric_df,
                "post_fold_diag": post_fold_diag,
                "best_post_pred_rows": best_post_pred_rows,
                "best_post": best_post,
                "post_model": post_model,
                "outer_val_base": outer_val_base,
                "outer_applied": outer_applied,
                "pooled_outer": pooled_outer,
            }

        if not eval_rows:
            continue
        eval_df = pd.DataFrame(eval_rows).sort_values(["inner_post_rmse", "inner_post_mae", "outer_rmse"], kind="mergesort").reset_index(drop=True)
        outer_results.append(eval_df)
        chosen_key = str(eval_df.iloc[0]["core_candidate_key"])
        chosen = bundle_lookup[chosen_key]

        outer_val_base = chosen["outer_val_base"]
        applied = chosen["outer_applied"]
        oof_chunk = outer_val_base.copy()
        oof_chunk["outer_fold"] = int(ofold["fold"])
        oof_chunk["pred_raw_mean"] = applied["raw_mean"]
        oof_chunk["pred_raw_blend"] = applied["raw_blend"]
        oof_chunk["pred_calibrated"] = applied["calibrated"]
        oof_chunk["pred_scaled"] = applied["scaled"]
        oof_chunk["pred_final"] = applied["final"]
        oof_chunk["pred_q50"] = outer_val_base["pred_q50"].to_numpy(dtype=float)
        oof_chunk["core_candidate_key"] = chosen_key
        oof_chunk["postprocess_key"] = chosen["best_post"].key()
        outer_oof_rows.append(oof_chunk)

        y_outer = outer_val_base["y_true"].to_numpy(dtype=float)
        outer_disp_rows.append(
            pd.DataFrame(
                [
                    {"outer_fold": int(ofold["fold"]), "stage": "raw_mean", "pred_std": float(np.std(applied["raw_mean"])), "actual_std": float(np.std(y_outer)), "rmse": _rmse(y_outer, applied["raw_mean"])},
                    {"outer_fold": int(ofold["fold"]), "stage": "calibrated", "pred_std": float(np.std(applied["calibrated"])), "actual_std": float(np.std(y_outer)), "rmse": _rmse(y_outer, applied["calibrated"])},
                    {"outer_fold": int(ofold["fold"]), "stage": "scaled", "pred_std": float(np.std(applied["scaled"])), "actual_std": float(np.std(y_outer)), "rmse": _rmse(y_outer, applied["scaled"])},
                    {"outer_fold": int(ofold["fold"]), "stage": "final", "pred_std": float(np.std(applied["final"])), "actual_std": float(np.std(y_outer)), "rmse": _rmse(y_outer, applied["final"])},
                ]
            )
        )

        regime_tables[f"outer_fold_{int(ofold['fold'])}_selected"] = chosen["post_model"].regime_summary.copy()
        calibration_tables[f"outer_fold_{int(ofold['fold'])}_selected"] = chosen["post_model"].calibration_summary.copy()
        scale_tables[f"outer_fold_{int(ofold['fold'])}_selected"] = chosen["post_model"].scale_summary.copy()

    outer_results_df = pd.concat(outer_results, ignore_index=True) if outer_results else pd.DataFrame()
    outer_oof = pd.concat(outer_oof_rows, ignore_index=True) if outer_oof_rows else pd.DataFrame()
    core_scan_all = pd.concat(outer_scan_rows, ignore_index=True) if outer_scan_rows else pd.DataFrame()
    scale_dispersion_table = pd.concat(outer_disp_rows, ignore_index=True) if outer_disp_rows else pd.DataFrame()
    post_metric_all = pd.concat(postprocess_metric_rows, ignore_index=True) if postprocess_metric_rows else pd.DataFrame()

    outer_fold_metric_rows = []
    if not outer_oof.empty:
        for ofid, g in outer_oof.groupby("outer_fold", sort=True):
            m = _reg_metrics(g["y_true"].to_numpy(dtype=float), g["pred_final"].to_numpy(dtype=float))
            outer_fold_metric_rows.append({"outer_fold": int(ofid), **m})
    outer_fold_metrics = pd.DataFrame(outer_fold_metric_rows)
    outer_summary = (
        outer_results_df.sort_values(["outer_fold", "inner_post_rmse", "inner_post_mae"], kind="mergesort")
        .groupby("outer_fold", as_index=False)
        .head(1)
        .sort_values("outer_fold", kind="mergesort")
        .reset_index(drop=True)
        if not outer_results_df.empty
        else pd.DataFrame()
    )

    elo_variant_compare = (
        core_scan_all.groupby("elo_variant", as_index=False)
        .agg(scan_score_mean=("scan_score", "mean"), scan_histgb_rmse_mean=("scan_histgb_rmse", "mean"), scan_ridge_rmse_mean=("scan_ridge_rmse", "mean"), n_configs=("candidate_key", "count"))
        .sort_values(["scan_score_mean", "scan_histgb_rmse_mean"], kind="mergesort")
        .reset_index(drop=True)
        if not core_scan_all.empty
        else pd.DataFrame()
    )
    recency_compare = (
        core_scan_all.groupby(["feature_profile", "half_life_days"], dropna=False, as_index=False)
        .agg(scan_score_mean=("scan_score", "mean"), scan_histgb_rmse_mean=("scan_histgb_rmse", "mean"), scan_ridge_rmse_mean=("scan_ridge_rmse", "mean"), n_configs=("candidate_key", "count"))
        .sort_values(["scan_score_mean", "scan_histgb_rmse_mean"], kind="mergesort")
        .reset_index(drop=True)
        if not core_scan_all.empty
        else pd.DataFrame()
    )
    histgb_sweep_summary = (
        core_scan_all.groupby("histgb_idx", as_index=False)
        .agg(scan_score_mean=("scan_score", "mean"), scan_histgb_rmse_mean=("scan_histgb_rmse", "mean"), n=("candidate_key", "count"))
        .merge(pd.DataFrame({"histgb_idx": list(range(len(histgb_grid))), "params": [str(p) for p in histgb_grid]}), on="histgb_idx", how="left")
        .sort_values(["scan_histgb_rmse_mean", "scan_score_mean"], kind="mergesort")
        .reset_index(drop=True)
        if not core_scan_all.empty
        else pd.DataFrame()
    )
    postprocess_compare = (
        outer_results_df.groupby(["use_regime_stack", "calibration_mode", "scale_mode"], as_index=False)
        .agg(outer_rmse_mean=("outer_rmse", "mean"), outer_mae_mean=("outer_mae", "mean"), outer_dispersion_ratio_mean=("outer_dispersion_ratio", "mean"), regime_gain_vs_pooled_mean=("outer_regime_gain_vs_pooled", "mean"), n=("outer_fold", "count"))
        .sort_values(["outer_rmse_mean", "outer_mae_mean"], kind="mergesort")
        .reset_index(drop=True)
        if not outer_results_df.empty
        else pd.DataFrame()
    )

    # Final selection on full train using time-aware inner folds (capped in fast mode).
    full_idx = np.arange(len(train_df), dtype=int)
    pseudo_outer = {
        "fold": 0,
        "train_idx": full_idx,
        "val_idx": full_idx,
        "train_start_date": pd.to_datetime(train_df["Date"]).min(),
        "train_end_date": pd.to_datetime(train_df["Date"]).max(),
        "val_start_date": pd.to_datetime(train_df["Date"]).min(),
        "val_end_date": pd.to_datetime(train_df["Date"]).max(),
    }
    final_inner_n_folds = int(runtime_cfg.fast_inner_folds) if runtime_cfg.fast_mode else 3
    final_inner_folds = make_expanding_time_folds(train_df.copy().reset_index(drop=True), n_folds=final_inner_n_folds)
    final_variant_data = {
        vname: _prepare_variant_outer_data(
            train_df=train_df,
            outer_fold=pseudo_outer,
            inner_folds=final_inner_folds,
            seq_variant_full=seq_df,
            team_ids=team_ids,
            conf_values=conf_values,
            variant_name=vname,
        )
        for vname, seq_df in seq_by_variant_train.items()
    }
    final_top_core, final_scan_df = _scan_and_select_core_candidates(final_variant_data, histgb_grid, seed=seed + 7000)
    final_post_grid = _make_postprocess_grid()
    final_bundle_lookup: Dict[str, dict] = {}
    final_eval_rows = []
    for i, cand in enumerate(final_top_core):
        if _budget_total_exceeded() and final_eval_rows:
            break
        inner_oof, _ = _generate_inner_oof_and_outer_preds(cand, final_variant_data[cand.elo_variant], histgb_grid, seed=seed + 8000 + 50 * i)
        if inner_oof.empty:
            continue
        best_post, post_metric_df, post_fold_diag, best_post_pred_rows = _select_postprocess(inner_oof, final_post_grid)
        post_model = _fit_postprocess_model(inner_oof, best_post)
        final_eval_rows.append(
            {
                "core_candidate_key": cand.key(),
                "elo_variant": cand.elo_variant,
                "feature_profile": cand.feature_profile,
                "half_life_days": np.nan if cand.half_life_days is None else cand.half_life_days,
                "ridge_alpha": cand.ridge_alpha,
                "histgb_idx": cand.histgb_idx,
                "postprocess_key": best_post.key(),
                "inner_rmse_final": float(post_metric_df.iloc[0]["rmse_final"]),
                "inner_mae_final": float(post_metric_df.iloc[0]["mae_final"]),
                "inner_rmse_q50": float(post_metric_df.iloc[0]["rmse_q50"]),
                "inner_rmse_mean_stack": float(post_metric_df.iloc[0]["rmse_mean_stack"]),
                "use_regime_stack": bool(best_post.regime_stack),
                "calibration_mode": best_post.calibration_mode,
                "scale_mode": best_post.scale_mode,
                "q50_blend": best_post.q50_blend,
                "winsor_q": best_post.winsor_q,
            }
        )
        final_bundle_lookup[cand.key()] = {
            "candidate": cand,
            "inner_oof": inner_oof,
            "best_post": best_post,
            "post_model": post_model,
            "best_post_pred_rows": best_post_pred_rows,
            "post_metric_df": post_metric_df,
            "post_fold_diag": post_fold_diag,
        }
    if not final_eval_rows:
        # Hard fallback under severe budget pressure: use first scanned core candidate with a simple pooled postprocess.
        fallback_core = final_top_core[0]
        inner_oof, _ = _generate_inner_oof_and_outer_preds(fallback_core, final_variant_data[fallback_core.elo_variant], histgb_grid, seed=seed + 8000)
        fallback_post = PostprocessCandidate(regime_stack=False, calibration_mode="pooled", scale_mode="none", q50_blend=0.0, winsor_q=0.0)
        post_model = _fit_postprocess_model(inner_oof, fallback_post) if not inner_oof.empty else PostprocessModel(
            candidate=fallback_post,
            thresholds={"elo_cuts": [35.0, 90.0], "vol_cut": 0.0, "info_cut": 5.0, "info_hi_cut": 8.0},
            stack_fit={"mode": "pooled", "pred_cols": MEAN_MODEL_COLS, "pooled": np.array([0.25, 0.25, 0.25, 0.25]), "full": {}, "simple": {}, "gap": {}},
            cal_fit={"mode": "pooled", "pooled": (0.0, 1.0), "full": {}, "simple": {}, "gap": {}},
            scale_fit={"mode": "none", "center": 0.0, "pooled": 1.0, "full": {}, "simple": {}, "gap": {}},
            winsor_bounds=(-np.inf, np.inf),
            regime_summary=pd.DataFrame(),
            calibration_summary=pd.DataFrame(),
            scale_summary=pd.DataFrame(),
        )
        final_eval_rows.append(
            {
                "core_candidate_key": fallback_core.key(),
                "elo_variant": fallback_core.elo_variant,
                "feature_profile": fallback_core.feature_profile,
                "half_life_days": np.nan if fallback_core.half_life_days is None else fallback_core.half_life_days,
                "ridge_alpha": fallback_core.ridge_alpha,
                "histgb_idx": fallback_core.histgb_idx,
                "postprocess_key": fallback_post.key(),
                "inner_rmse_final": float("inf"),
                "inner_mae_final": float("inf"),
                "inner_rmse_q50": float("inf"),
                "inner_rmse_mean_stack": float("inf"),
                "use_regime_stack": bool(fallback_post.regime_stack),
                "calibration_mode": fallback_post.calibration_mode,
                "scale_mode": fallback_post.scale_mode,
                "q50_blend": fallback_post.q50_blend,
                "winsor_q": fallback_post.winsor_q,
            }
        )
        final_bundle_lookup[fallback_core.key()] = {
            "candidate": fallback_core,
            "inner_oof": inner_oof,
            "best_post": fallback_post,
            "post_model": post_model,
            "best_post_pred_rows": pd.DataFrame({"fold": [], "row_index": [], "pred_final": []}),
            "post_metric_df": pd.DataFrame(final_eval_rows),
            "post_fold_diag": pd.DataFrame(),
        }
    final_eval_df = pd.DataFrame(final_eval_rows).sort_values(["inner_rmse_final", "inner_mae_final"], kind="mergesort").reset_index(drop=True)
    final_core_key = str(final_eval_df.iloc[0]["core_candidate_key"])
    final_sel = final_bundle_lookup[final_core_key]
    final_core_candidate: CoreCandidate = final_sel["candidate"]
    final_post_model: PostprocessModel = final_sel["post_model"]
    final_histgb_params = histgb_grid[int(final_core_candidate.histgb_idx)]

    # Fit full-train -> derby with selected configuration.
    train_nl_selected, derby_nl_selected, static_models_full, derby_pred_bundle = _prepare_full_derby_predictions(
        train_df=train_df,
        pred_df=pred_df,
        seq_build=seq_builds[final_core_candidate.elo_variant],
        team_ids=team_ids,
        conf_values=conf_values,
        core_candidate=final_core_candidate,
        histgb_params=final_histgb_params,
        seed=seed + 9000,
    )
    derby_pred_df = derby_pred_bundle["derby_pred_df"]

    shift_table = _shift_diagnostics(train_nl_selected, derby_nl_selected)
    shift_mitigation_notes: List[str] = []
    shift_mitigation_applied = False
    ood_count = int(shift_table["ood_flag"].sum()) if (not shift_table.empty and "ood_flag" in shift_table.columns) else 0
    if ood_count >= 2:
        shift_mitigation_applied = True
        shift_mitigation_notes.append(f"Detected {ood_count} OOD-like feature shifts (KS/PSI/out-of-range checks).")
        if "pooled" in final_post_model.scale_fit and np.isfinite(float(final_post_model.scale_fit["pooled"])):
            final_post_model.scale_fit["pooled"] = float(1.0 + 0.5 * (float(final_post_model.scale_fit["pooled"]) - 1.0))
        for lvl in ["full", "simple", "gap"]:
            if lvl in final_post_model.scale_fit and isinstance(final_post_model.scale_fit[lvl], dict):
                for k, v in list(final_post_model.scale_fit[lvl].items()):
                    final_post_model.scale_fit[lvl][k] = float(1.0 + 0.5 * (float(v) - 1.0))
        y_train = train_df["HomeWinMargin"].to_numpy(dtype=float)
        final_post_model.winsor_bounds = (float(np.quantile(y_train, 0.0025)), float(np.quantile(y_train, 0.9975)))
        shift_mitigation_notes.append("Applied softer scaling (blend toward 1.0) and wider winsor bounds under OOD risk.")
    else:
        shift_mitigation_notes.append(f"OOD-like shift count below threshold (count={ood_count}); no extra mitigation applied.")

    derby_applied = _apply_postprocess_model(derby_pred_df, final_post_model)
    final_derby_float = np.asarray(derby_applied["final"], dtype=float)

    # Anti-overcompression guard.
    final_inner_oof_best = final_sel["best_post_pred_rows"].copy().sort_values(["fold", "row_index"], kind="mergesort").reset_index(drop=True)
    oof_std_ref = float(np.std(final_inner_oof_best["pred_final"].to_numpy(dtype=float)))
    derby_std_pre = float(np.std(final_derby_float))
    dispersion_guard_applied = False
    dispersion_guard_factor = 1.0
    if np.isfinite(oof_std_ref) and oof_std_ref > 1e-9 and derby_std_pre < 0.75 * oof_std_ref:
        target = 0.90 * oof_std_ref
        if derby_std_pre > 1e-9:
            dispersion_guard_factor = float(np.clip(target / derby_std_pre, 1.0, 1.35))
            final_derby_float = np.clip(dispersion_guard_factor * final_derby_float, *final_post_model.winsor_bounds)
            dispersion_guard_applied = True
            shift_mitigation_notes.append(
                f"Applied anti-compression guard scale {dispersion_guard_factor:.3f} (derby std {derby_std_pre:.2f} vs ref {oof_std_ref:.2f})."
            )
    else:
        shift_mitigation_notes.append(f"No anti-compression guard applied (derby std {derby_std_pre:.2f}, ref {oof_std_ref:.2f}).")

    final_derby_int = np.rint(final_derby_float).astype(int)
    pred_out = pred_raw.copy()
    pred_out["Team1_WinMargin"] = final_derby_int
    pred_out.to_csv(root / "predictions.csv", index=False)

    selected_seq_build = seq_builds[final_core_candidate.elo_variant]
    recent_form_map = _extract_recent_form_map(selected_seq_build.final_states)
    if outer_oof.empty:
        component_oof = pd.DataFrame(columns=["elo_diff_pre", "massey_diff", "offdef_net_diff", "HomeWinMargin", "recent_form_diff"])
    else:
        component_oof = outer_oof[["elo_diff_pre", "massey_diff", "offdef_net_diff", "y_true"]].copy()
        component_oof = component_oof.rename(columns={"y_true": "HomeWinMargin"})
        component_oof["recent_form_diff"] = outer_oof["trend_margin_last5_vs_season_diff"].astype(float).to_numpy()
    ranking_weights = _compute_ranking_weights_from_oof(component_oof)
    final_net_map = static_models_full.offdef.net_rating_map()
    rankings_out = _build_rankings_output_v2(
        rankings_df=rankings_raw,
        team_universe=team_universe,
        final_elo=selected_seq_build.final_elo,
        final_massey=static_models_full.massey.team_rating,
        final_net=final_net_map,
        recent_form=recent_form_map,
        weights=ranking_weights,
    )
    rankings_out.to_excel(root / "rankings.xlsx", index=False)

    # Build report inputs.
    prediction_head = pred_out.head(10).copy()
    final_derby_stage_df = derby_pred_df.copy()
    for k, v in derby_applied.items():
        final_derby_stage_df[f"stage_{k}"] = np.asarray(v, dtype=float)
    if "pred_q20" in derby_pred_df.columns and "pred_q80" in derby_pred_df.columns:
        final_derby_stage_df["quantile_spread_q80_q20"] = derby_pred_df["pred_q80"].astype(float) - derby_pred_df["pred_q20"].astype(float)

    selected_hparams_table = pd.DataFrame(
        [
            {"name": "elo_variant", "value": final_core_candidate.elo_variant},
            {"name": "feature_profile", "value": final_core_candidate.feature_profile},
            {"name": "half_life_days", "value": "none" if final_core_candidate.half_life_days is None else final_core_candidate.half_life_days},
            {"name": "ridge_alpha", "value": final_core_candidate.ridge_alpha},
            {"name": "histgb_idx", "value": final_core_candidate.histgb_idx},
            {"name": "histgb_params", "value": str(final_histgb_params)},
            {"name": "postprocess", "value": final_sel["best_post"].key()},
            {"name": "winsor_bounds", "value": str(tuple(round(float(x), 4) for x in final_post_model.winsor_bounds))},
            {"name": "dispersion_guard_factor", "value": dispersion_guard_factor},
            {"name": "seed", "value": int(seed)},
            {"name": "OMP_NUM_THREADS", "value": os.environ.get("OMP_NUM_THREADS", "")},
            {"name": "MKL_NUM_THREADS", "value": os.environ.get("MKL_NUM_THREADS", "")},
        ]
    )

    focus_phases = [
        "_scan_and_select_core_candidates",
        "_core_scan_score",
        "_generate_inner_oof_and_outer_preds",
        "_predict_family",
        "_build_split_tables",
        "_prepare_variant_outer_data",
    ]
    timing_summary_pre_report = _timing_summary_dict(_ACTIVE_PROFILER, top_n=10)
    timing_summary_pre_report["focus_phase_totals_sec"] = {k: float(timing_summary_pre_report["phase_totals_sec"].get(k, 0.0)) for k in focus_phases}
    timing_summary_pre_report["focus_phase_counts"] = {k: int(timing_summary_pre_report["phase_counts"].get(k, 0)) for k in focus_phases}
    runtime_phase_table = _timing_phase_table(timing_summary_pre_report, focus_phases)
    runtime_family_table = _timing_family_table(timing_summary_pre_report)
    fast_before_summary = _load_json_if_exists(root / "timing_summary_fast_baseline.json")
    fast_after_summary = _load_json_if_exists(root / "timing_summary_fast.json")
    runtime_compare_table = _timing_compare_table(fast_before_summary, fast_after_summary, focus_phases) if fast_after_summary is not None else pd.DataFrame()
    b = _budget()
    runtime_notes = [
        "Profile-first instrumentation added for hotspot phase totals, per-family _predict_family timing, model fit counts, and top slow events.",
        "Shared split-level matrix/weight cache keyed by train/val index signatures + feature signature + half-life to eliminate repeated _xy/time-weight prep.",
        "Bundled family prediction path prepares X/y/w once per split and fits Ridge/Huber/HistGB/HistGB-bag/q50 sequentially.",
        f"HistGB bagging reduced to {runtime_cfg.histgb_bag_n_models} models by default (fast mode uses 1), with deterministic seed schedule and budget-aware auto-reduction.",
        "Candidate scan uses conservative early pruning: coarse score on first inner split, then full evaluation on top candidates only.",
        "Budget enforcement: total time, scan time, and fit cap stop further scanning/training and continue with best-so-far.",
        f"Budgets triggered: {', '.join(sorted(b.triggered.keys())) if (b is not None and b.triggered) else 'none'}",
        f"Fast mode={runtime_cfg.fast_mode}; optimization toggles enabled={runtime_cfg.enable_optimizations}.",
        f"Reproducibility: seed={seed}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')}, MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '')}.",
        f"Artifact validation: predictions rows={len(pred_out)} (expected 75), rankings rows={len(rankings_out)} (expected 165), ranks unique 1..165={set(pd.to_numeric(rankings_out['Rank'], errors='coerce').fillna(-1).astype(int).tolist()) == set(range(1,166)) if 'Rank' in rankings_out.columns else False}.",
    ]

    _build_report(
        root / "final_report.pdf",
        train_df=train_df,
        pred_template=pred_raw.copy(),
        outer_folds=outer_folds,
        outer_summary=outer_summary.copy(),
        outer_fold_metrics=outer_fold_metrics.copy(),
        outer_pred_oof=outer_oof.sort_values(["outer_fold", "row_index"], kind="mergesort").reset_index(drop=True),
        core_scan_summary=core_scan_all.sort_values(["scan_score", "scan_histgb_rmse"], kind="mergesort").reset_index(drop=True),
        elo_variant_compare=elo_variant_compare.copy(),
        recency_compare=recency_compare.copy(),
        histgb_sweep_summary=histgb_sweep_summary.copy(),
        postprocess_compare=postprocess_compare.copy(),
        scale_dispersion_table=scale_dispersion_table.copy(),
        shift_table=shift_table.copy(),
        shift_mitigation_notes=shift_mitigation_notes,
        regime_tables=regime_tables,
        calibration_tables=calibration_tables,
        scale_tables=scale_tables,
        final_derby_pred_float=final_derby_float,
        final_derby_pred_int=final_derby_int,
        final_derby_stage_df=final_derby_stage_df.copy(),
        rankings_out=rankings_out.copy(),
        prediction_head=prediction_head.copy(),
        selected_summary={
            "model_family": "Regime-aware simplex stack (Ridge/Huber/HistGB/HistGB-bag) + q50 blend",
            "elo_variant": final_core_candidate.elo_variant,
            "feature_profile": final_core_candidate.feature_profile,
            "half_life_days": final_core_candidate.half_life_days,
            "calibration_mode": final_sel["best_post"].calibration_mode,
            "scale_mode": final_sel["best_post"].scale_mode,
            "use_regime_stack": final_sel["best_post"].regime_stack,
            "core_candidate_key": final_core_candidate.key(),
            "postprocess_key": final_sel["best_post"].key(),
            "shift_mitigation_applied": shift_mitigation_applied,
            "dispersion_guard_applied": dispersion_guard_applied,
            "fast_mode": bool(runtime_cfg.fast_mode),
            "budget_triggered": ", ".join(sorted(b.triggered.keys())) if (b is not None and b.triggered) else "none",
        },
        selected_hparams_table=selected_hparams_table.copy(),
        runtime_notes=runtime_notes,
        runtime_phase_table=runtime_phase_table,
        runtime_family_table=runtime_family_table,
        runtime_compare_table=runtime_compare_table,
    )

    outer_rmse = float(_rmse(outer_oof["y_true"].to_numpy(dtype=float), outer_oof["pred_final"].to_numpy(dtype=float))) if not outer_oof.empty else float("nan")
    outer_mae = float(_mae(outer_oof["y_true"].to_numpy(dtype=float), outer_oof["pred_final"].to_numpy(dtype=float))) if not outer_oof.empty else float("nan")
    profiler = _ACTIVE_PROFILER
    if profiler is not None:
        profiler.add("run_nextgen_pipeline_total", time.perf_counter() - pipeline_t0, detail="total")
    timing_summary = _timing_summary_dict(profiler, top_n=10)
    timing_summary["focus_phase_totals_sec"] = {k: float(timing_summary["phase_totals_sec"].get(k, 0.0)) for k in focus_phases}
    timing_summary["focus_phase_counts"] = {k: int(timing_summary["phase_counts"].get(k, 0)) for k in focus_phases}
    timing_summary["seed"] = int(seed)
    timing_summary["generated_at_utc"] = pd.Timestamp.utcnow().isoformat()
    timing_summary["runtime_config"] = {
        "max_total_seconds": int(runtime_cfg.max_total_seconds),
        "max_scan_seconds": int(runtime_cfg.max_scan_seconds),
        "max_fits": int(runtime_cfg.max_fits),
        "fast_mode": bool(runtime_cfg.fast_mode),
        "enable_optimizations": bool(runtime_cfg.enable_optimizations),
        "histgb_bag_n_models": int(runtime_cfg.histgb_bag_n_models),
    }
    if _ACTIVE_BUDGET is not None:
        timing_summary["budget"] = {
            "elapsed_sec": float(_ACTIVE_BUDGET.elapsed()),
            "fit_count": int(_ACTIVE_BUDGET.fit_count),
            "triggered": dict(_ACTIVE_BUDGET.triggered),
            "events": list(_ACTIVE_BUDGET.events),
        }
    timing_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(os.environ.get("ALGOSPORTS_TIMING_TAG", "latest"))).strip("_") or "latest"
    timing_json_path = root / f"timing_summary_{timing_tag}.json"
    _write_json(timing_json_path, timing_summary)
    _write_run_report(
        root / "run_report.md",
        root=root,
        inputs_used=paths,
        runtime_cfg=runtime_cfg,
        budget=_ACTIVE_BUDGET,
        timing_summary=timing_summary,
        selected_summary={
            "model_family": "Regime-aware simplex stack (Ridge/Huber/HistGB/HistGB-bag) + q50 blend",
            "elo_variant": final_core_candidate.elo_variant,
            "feature_profile": final_core_candidate.feature_profile,
            "half_life_days": final_core_candidate.half_life_days,
            "calibration_mode": final_sel["best_post"].calibration_mode,
            "scale_mode": final_sel["best_post"].scale_mode,
            "use_regime_stack": final_sel["best_post"].regime_stack,
        },
        outer_rmse=outer_rmse,
        outer_mae=outer_mae,
        predictions_out=pred_out,
        rankings_out=rankings_out,
    )
    print("\n" + "\n".join(_timing_summary_lines(timing_summary, focus_phases=focus_phases, top_n=10)))
    _ACTIVE_PROFILER = prev_profiler
    _ACTIVE_RUNTIME_CFG = prev_runtime_cfg
    _ACTIVE_BUDGET = prev_budget
    _PIPELINE_START_PERF = prev_pipeline_start
    return {
        "selected_model_family": "Regime-aware simplex stack + q50 blend + nested calibration/scale",
        "selected_elo_variant": final_core_candidate.elo_variant,
        "nested_outer_rmse": outer_rmse,
        "nested_outer_mae": outer_mae,
        "calibration_type": final_sel["best_post"].calibration_mode,
        "scale_correction_type": final_sel["best_post"].scale_mode,
        "regime_specific_stack_used": bool(final_sel["best_post"].regime_stack),
        "predictions_head": prediction_head.copy(),
        "rankings_top10": rankings_out.sort_values("Rank", kind="mergesort").head(10).copy(),
        "dispersion_guard_applied": dispersion_guard_applied,
        "shift_mitigation_applied": shift_mitigation_applied,
        "final_eval_df": final_eval_df.copy(),
        "outer_summary": outer_summary.copy(),
        "outer_fold_metrics": outer_fold_metrics.copy(),
        "timing_summary": timing_summary,
        "timing_summary_path": str(timing_json_path),
        "run_report_path": str(root / "run_report.md"),
    }
