from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap
from typing import Dict, List, Mapping, Sequence

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.features import (
    FoldFeatureArtifacts,
    build_derby_table,
    build_full_train_table,
    build_fold_feature_tables,
    build_team_universe,
    parse_and_sort_train,
    parse_predictions,
    make_expanding_time_folds,
)
from src.models import (
    F1_FEATURES,
    F2_FEATURES,
    ModelSpec,
    RatingSpec,
    _get_xy,
    aggregate_config_metrics,
    apply_postprocessor,
    apply_postprocessor_with_cv_guard,
    build_inner_oof,
    feature_cols_for_rating_spec,
    fit_postprocessor,
    generate_model_specs,
    generate_rating_specs,
    make_estimator,
    regression_metrics,
    select_with_1se_and_stability,
    simplicity_rank,
)


SEED = 23
ROOT = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


_CPU_COUNT = os.cpu_count() or 8
N_JOBS = max(1, _env_int("ALGOSPORTS_N_JOBS", min(8, max(1, _CPU_COUNT - 2))))
MAX_MODEL_FITS = max(1, _env_int("ALGOSPORTS_MAX_MODEL_FITS", 800))
MAX_TOTAL_SECONDS = max(60, _env_int("ALGOSPORTS_MAX_TOTAL_SECONDS", 900))
LAMBDA_TAIL_GRID = [0.10, 0.20]
LAMBDA_XTAIL_GRID = [0.05, 0.10]
LAMBDA_DISP_GRID = [0.05, 0.10, 0.20]
LAMBDA_TDISP_GRID = [0.05, 0.10]
LAMBDA_STABILITY_GRID = [0.10, 0.20]
LAMBDA_CAP_HIT_GRID = [0.0, 0.05, 0.10]
LAMBDA_TBIAS_GRID = [0.0, 0.002, 0.005]
POSTPROCESS_MODULES = [
    "none",
    "expand_affine_global",
    "expand_regime_affine_v2",
    "expand_regime_affine_v2_heterosk",
    "expand_piecewise_tail_v2",
]
BASELINE_DISPERSION_RATIO_REFERENCE = 43.62 / 26.15


def _resolve_input_path(candidates: Sequence[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    joined = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Missing {label}. Tried: {joined}")


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _json_dump(path: Path, obj: Mapping) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    sd = float(x.std(ddof=0))
    if sd <= 1e-12 or not np.isfinite(sd):
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - float(x.mean())) / sd


def _wrap_lines(text: str, width: int = 120) -> List[str]:
    lines: List[str] = []
    for line in text.splitlines():
        if len(line) <= width:
            lines.append(line)
        else:
            lines.extend(wrap(line, width=width, break_long_words=False, replace_whitespace=False))
    return lines


def _add_text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.04, 0.97, title, fontsize=13, fontweight="bold", va="top")
    y = 0.94
    for line in lines:
        if y < 0.04:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            fig.text(0.04, 0.97, f"{title} (cont.)", fontsize=13, fontweight="bold", va="top")
            y = 0.94
        fig.text(0.04, y, line, fontsize=8.5, family="monospace", va="top")
        y -= 0.017
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _df_text(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "(empty)"
    use = df.head(max_rows).copy()
    txt = use.to_string(index=False)
    if len(df) > max_rows:
        txt += f"\n... ({len(df) - max_rows} more rows)"
    return txt


def _time_exceeded(run_start_ts: float) -> bool:
    return (time.time() - float(run_start_ts)) > float(MAX_TOTAL_SECONDS)


def _evaluate_rating_fold(train_df: pd.DataFrame, fold: Mapping, rating_spec: RatingSpec, team_ids: Sequence[int]) -> dict:
    feature_set_name, feature_cols = feature_cols_for_rating_spec(rating_spec)
    train_rows = train_df.loc[np.asarray(fold["train_idx"], dtype=int)].copy()
    val_rows = train_df.loc[np.asarray(fold["val_idx"], dtype=int)].copy()
    art: FoldFeatureArtifacts = build_fold_feature_tables(
        train_rows,
        val_rows,
        team_ids=team_ids,
        massey_alpha=rating_spec.massey_alpha,
        use_elo=rating_spec.use_elo,
        elo_home_adv=rating_spec.elo_home_adv,
        elo_k=rating_spec.elo_k,
        elo_decay_a=rating_spec.elo_decay_a,
        elo_decay_g=rating_spec.elo_decay_g,
    )
    fixed_model = ModelSpec("ridge", {"alpha": 10.0})
    X_tr, y_tr = _get_xy(art.train_table, feature_cols=feature_cols)
    X_va, y_va = _get_xy(art.val_table, feature_cols=feature_cols)
    est = make_estimator(fixed_model, seed=SEED)
    est.fit(X_tr, y_tr)
    p_base = np.asarray(est.predict(X_va), dtype=float)

    inner_oof = build_inner_oof(
        train_rows,
        team_ids=team_ids,
        rating_spec=rating_spec,
        model_spec=fixed_model,
        feature_cols=feature_cols,
    )
    post_fit = fit_postprocessor("expand_affine_global", inner_oof)
    gp_min = np.minimum(
        art.val_table["games_played_home"].to_numpy(dtype=float),
        art.val_table["games_played_away"].to_numpy(dtype=float),
    )
    mismatch_key = np.abs(art.val_table["d_key"].to_numpy(dtype=float))
    info_key = gp_min.copy()
    if bool(post_fit.get("invalid", False)):
        p_final = p_base.copy()
        invalid_post = True
        cv_disp_ratio = float(np.std(p_final, ddof=0) / max(np.std(y_va, ddof=0), 1e-9))
        cv_disp_guard_reject = False
    else:
        post_applied = apply_postprocessor_with_cv_guard(
            post_fit,
            y_pred_base=p_base,
            d_key=art.val_table["d_key"].to_numpy(dtype=float),
            gp_min=gp_min,
            y_true=y_va,
            max_disp_ratio=None,
        )
        p_final = np.asarray(post_applied["y_pred_final"], dtype=float)
        cv_disp_ratio = float(post_applied.get("cv_dispersion_ratio", np.nan))
        cv_disp_guard_reject = bool(post_applied.get("cv_dispersion_guard_reject", False))
        invalid_post = bool(cv_disp_guard_reject)

    met = regression_metrics(y_va, p_final)
    aff = post_fit.get("affine", {})
    return {
        "config_id": rating_spec.key,
        "fold": int(fold["fold"]),
        "rmse": met["rmse"],
        "mae": met["mae"],
        "tail_rmse": met["tail_rmse"],
        "xtail_rmse": met["xtail_rmse"],
        "dispersion_ratio": met["dispersion_ratio"],
        "tail_dispersion_ratio": met["tail_dispersion_ratio"],
        "bias": met["bias"],
        "tail_bias": met["tail_bias"],
        "max_abs_pred": float(np.max(np.abs(p_final))) if len(p_final) else 0.0,
        "simplicity_rank": 1 if rating_spec.use_elo else 0,
        "use_elo": int(rating_spec.use_elo),
        "feature_set": feature_set_name,
        "massey_alpha": float(rating_spec.massey_alpha),
        "elo_home_adv": float(rating_spec.elo_home_adv),
        "elo_decay_a": float(rating_spec.elo_decay_a),
        "elo_decay_g": float(rating_spec.elo_decay_g),
        "invalid_postprocess": int(invalid_post),
        "r_base": float(aff.get("r_base", np.nan)),
        "naive_r": float(aff.get("naive_r", np.nan)),
        "s_star": float(aff.get("s_star", np.nan)),
        "chosen_s": float(aff.get("s", np.nan)),
        "eta": float(aff.get("eta")) if aff.get("eta") is not None else np.nan,
        "s_cap_hit": int(bool(aff.get("s_cap_hit", False))),
        "cv_dispersion_ratio": cv_disp_ratio,
        "cv_disp_guard_reject": int(cv_disp_guard_reject),
        "mismatch_key_mean": float(np.mean(mismatch_key)),
        "info_key_mean": float(np.mean(info_key)),
    }


def _evaluate_model_fold(
    train_df: pd.DataFrame,
    fold: Mapping,
    rating_spec: RatingSpec,
    model_spec: ModelSpec,
    module_name: str,
    team_ids: Sequence[int],
) -> tuple[dict, pd.DataFrame]:
    feature_set_name, feature_cols = feature_cols_for_rating_spec(rating_spec)
    train_rows = train_df.loc[np.asarray(fold["train_idx"], dtype=int)].copy()
    val_rows = train_df.loc[np.asarray(fold["val_idx"], dtype=int)].copy()

    art = build_fold_feature_tables(
        train_rows,
        val_rows,
        team_ids=team_ids,
        massey_alpha=rating_spec.massey_alpha,
        use_elo=rating_spec.use_elo,
        elo_home_adv=rating_spec.elo_home_adv,
        elo_k=rating_spec.elo_k,
        elo_decay_a=rating_spec.elo_decay_a,
        elo_decay_g=rating_spec.elo_decay_g,
    )

    X_tr, y_tr = _get_xy(art.train_table, feature_cols=feature_cols)
    X_va, y_va = _get_xy(art.val_table, feature_cols=feature_cols)
    est = make_estimator(model_spec, seed=SEED)
    est.fit(X_tr, y_tr)
    p_base = np.asarray(est.predict(X_va), dtype=float)

    inner_oof = build_inner_oof(train_rows, team_ids=team_ids, rating_spec=rating_spec, model_spec=model_spec, feature_cols=feature_cols)
    post_fit = fit_postprocessor(module_name, inner_oof)

    gp_min = np.minimum(
        art.val_table["games_played_home"].to_numpy(dtype=float),
        art.val_table["games_played_away"].to_numpy(dtype=float),
    )
    mismatch_key = np.abs(art.val_table["d_key"].to_numpy(dtype=float))
    info_key = gp_min.copy()
    if bool(post_fit.get("invalid", False)):
        p_final = p_base.copy()
        regime_ids = np.full(len(p_base), "invalid", dtype=object)
        s_used = np.full(len(p_base), np.nan, dtype=float)
        t_used = np.full(len(p_base), np.nan, dtype=float)
        k_used = np.full(len(p_base), np.nan, dtype=float)
        eta_used = np.full(len(p_base), np.nan, dtype=float)
        invalid_postprocess = True
        cv_disp_ratio = float(np.std(p_final, ddof=0) / max(np.std(y_va, ddof=0), 1e-9))
        cv_disp_guard_reject = False
    else:
        post_applied = apply_postprocessor_with_cv_guard(
            post_fit,
            y_pred_base=p_base,
            d_key=art.val_table["d_key"].to_numpy(dtype=float),
            gp_min=gp_min,
            y_true=y_va,
            max_disp_ratio=None,
        )
        p_final = np.asarray(post_applied["y_pred_final"], dtype=float)
        regime_ids = np.asarray(post_applied["regime_id"], dtype=object)
        s_used = np.asarray(post_applied["postprocess_s_used"], dtype=float)
        t_used = np.asarray(post_applied["postprocess_t_used"], dtype=float)
        k_used = np.asarray(post_applied["postprocess_k_used"], dtype=float)
        eta_used = np.asarray(post_applied["postprocess_eta_used"], dtype=float)
        cv_disp_ratio = float(post_applied.get("cv_dispersion_ratio", np.nan))
        cv_disp_guard_reject = bool(post_applied.get("cv_dispersion_guard_reject", False))
        invalid_postprocess = bool(cv_disp_guard_reject)

    met_base = regression_metrics(y_va, p_base)
    met_final = regression_metrics(y_va, p_final)
    aff = post_fit.get("affine", {})
    par = post_fit.get("params", {})
    reg = par.get("regimes", {})
    blow_s = [float(v.get("s", np.nan)) for k, v in reg.items() if str(k).startswith("blowout|")] if isinstance(reg, dict) else []
    blowout_mean_s = float(np.mean(blow_s)) if len(blow_s) else np.nan

    metric_row = {
        "config_id": f"{rating_spec.key}||{model_spec.key}||module={module_name}",
        "rating_config_id": rating_spec.key,
        "model_config_id": model_spec.key,
        "module": module_name,
        "feature_set": feature_set_name,
        "fold": int(fold["fold"]),
        "rmse": met_final["rmse"],
        "mae": met_final["mae"],
        "tail_rmse": met_final["tail_rmse"],
        "xtail_rmse": met_final["xtail_rmse"],
        "dispersion_ratio": met_final["dispersion_ratio"],
        "tail_dispersion_ratio": met_final["tail_dispersion_ratio"],
        "bias": met_final["bias"],
        "tail_bias": met_final["tail_bias"],
        "max_abs_pred": float(np.max(np.abs(p_final))) if len(p_final) else 0.0,
        "base_rmse": met_base["rmse"],
        "base_tail_rmse": met_base["tail_rmse"],
        "base_xtail_rmse": met_base["xtail_rmse"],
        "base_tail_dispersion_ratio": met_base["tail_dispersion_ratio"],
        "simplicity_rank": simplicity_rank(rating_spec.use_elo, model_spec.model_name, module_name),
        "model_name": model_spec.model_name,
        "massey_alpha": rating_spec.massey_alpha,
        "use_elo": int(rating_spec.use_elo),
        "elo_home_adv": rating_spec.elo_home_adv,
        "elo_decay_a": rating_spec.elo_decay_a,
        "elo_decay_g": rating_spec.elo_decay_g,
        "affine_a": float(aff.get("a", 0.0)),
        "affine_s": float(aff.get("s", 1.0)),
        "affine_naive_r": float(aff.get("naive_r", np.nan)),
        "affine_r_base": float(aff.get("r_base", np.nan)),
        "affine_s_star": float(aff.get("s_star", np.nan)),
        "affine_eta": float(aff.get("eta")) if aff.get("eta") is not None else np.nan,
        "s_cap_hit": int(bool(aff.get("s_cap_hit", False))),
        "cv_dispersion_ratio": cv_disp_ratio,
        "cv_disp_guard_reject": int(cv_disp_guard_reject),
        "blowout_mean_s": blowout_mean_s,
        "tail_t": float(par.get("tail_t", np.nan)),
        "tail_k": float(par.get("tail_k", np.nan)),
        "tail_q": float(par.get("tail_q", np.nan)),
        "invalid_postprocess": int(invalid_postprocess),
        "invalid_reason": "cv_dispersion_guard_fail" if cv_disp_guard_reject else str(post_fit.get("invalid_reason", "")),
    }

    fold_pred = art.val_table[["GameID", "Date", "HomeID", "AwayID", "HomeWinMargin"]].copy()
    fold_pred = fold_pred.rename(columns={"HomeWinMargin": "y_true"})
    fold_pred["y_pred_base"] = p_base
    fold_pred["y_pred_final"] = p_final
    fold_pred["fold_id"] = int(fold["fold"])
    fold_pred["regime_id"] = pd.Series(regime_ids, index=fold_pred.index).astype(str)
    fold_pred["mismatch_key"] = mismatch_key
    fold_pred["info_key"] = info_key
    fold_pred["postprocess_s_used"] = s_used
    fold_pred["postprocess_t_used"] = t_used
    fold_pred["postprocess_k_used"] = k_used
    fold_pred["postprocess_eta_used"] = eta_used
    return metric_row, fold_pred


def _select_rating_config(
    train_df: pd.DataFrame,
    folds: Sequence[Mapping],
    team_ids: Sequence[int],
    *,
    run_start_ts: float,
) -> tuple[RatingSpec, pd.DataFrame, dict]:
    rating_specs = generate_rating_specs()
    registry = {spec.key: spec for spec in rating_specs}
    fold_rows: List[dict] = []

    for i, spec in enumerate(rating_specs, start=1):
        if _time_exceeded(run_start_ts):
            print(f"[rating-stage] time budget reached at {i-1}/{len(rating_specs)} specs")
            break
        rows: List[dict] = []
        for fold in folds:
            r = _evaluate_rating_fold(train_df, fold, spec, team_ids)
            rows.append(r)
            if float(r.get("rmse", np.nan)) > 60.0:
                break
        fold_rows.extend(rows)
        if i % 50 == 0 or i == len(rating_specs):
            print(f"[rating-stage] evaluated {i}/{len(rating_specs)} rating configs")

    fold_df = pd.DataFrame(fold_rows)
    summary = aggregate_config_metrics(fold_df)
    selection = select_with_1se_and_stability(
        summary,
        lambda_tail_values=LAMBDA_TAIL_GRID,
        lambda_xtail_values=LAMBDA_XTAIL_GRID,
        lambda_disp_values=LAMBDA_DISP_GRID,
        lambda_cap_hit_values=LAMBDA_CAP_HIT_GRID,
        lambda_tbias_values=LAMBDA_TBIAS_GRID,
        lambda_tdisp_values=LAMBDA_TDISP_GRID,
        lambda_stability_values=LAMBDA_STABILITY_GRID,
    )
    selected_id = str(selection["selected_config_id"])
    selected_spec = registry[selected_id]
    return selected_spec, fold_df, selection


def _evaluate_model_stage(
    train_df: pd.DataFrame,
    folds: Sequence[Mapping],
    team_ids: Sequence[int],
    rating_spec: RatingSpec,
    *,
    run_start_ts: float,
) -> tuple[pd.DataFrame, dict, dict]:
    model_specs = generate_model_specs()
    feature_set_name, _ = feature_cols_for_rating_spec(rating_spec)
    registry: Dict[str, dict] = {}
    metric_rows: List[dict] = []
    model_fit_count = 0
    stopped_on_time = False
    stopped_on_fits = False

    for i, model_spec in enumerate(model_specs, start=1):
        if _time_exceeded(run_start_ts):
            stopped_on_time = True
            break
        for module in POSTPROCESS_MODULES:
            if _time_exceeded(run_start_ts):
                stopped_on_time = True
                break
            if model_fit_count >= MAX_MODEL_FITS:
                stopped_on_fits = True
                break
            config_id = f"{rating_spec.key}||{model_spec.key}||module={module}"
            registry[config_id] = {
                "rating_spec": asdict(rating_spec),
                "model_spec": {"model_name": model_spec.model_name, "params": dict(model_spec.params)},
                "module": module,
                "feature_set": feature_set_name,
            }
            for fold in folds:
                met, _pred = _evaluate_model_fold(train_df, fold, rating_spec, model_spec, module, team_ids)
                metric_rows.append(met)
                model_fit_count += 1
                if float(met.get("rmse", np.nan)) > 60.0:
                    break
                if model_fit_count >= MAX_MODEL_FITS or _time_exceeded(run_start_ts):
                    break
        if stopped_on_time or stopped_on_fits:
            break
        print(f"[model-stage] completed {i}/{len(model_specs)} model families (with module grid)")

    metrics_df = pd.DataFrame(metric_rows)
    budget_info = {
        "model_fit_count": int(model_fit_count),
        "max_model_fits": int(MAX_MODEL_FITS),
        "max_total_seconds": int(MAX_TOTAL_SECONDS),
        "stopped_on_time": bool(stopped_on_time),
        "stopped_on_model_fits": bool(stopped_on_fits),
        "elapsed_seconds": float(time.time() - float(run_start_ts)),
    }
    return metrics_df, registry, budget_info


def _rerun_selected_for_oof(
    train_df: pd.DataFrame,
    folds: Sequence[Mapping],
    team_ids: Sequence[int],
    rating_spec: RatingSpec,
    model_spec: ModelSpec,
    module_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_results = Parallel(n_jobs=min(N_JOBS, len(folds)), prefer="processes")(
        delayed(_evaluate_model_fold)(train_df, fold, rating_spec, model_spec, module_name, team_ids) for fold in folds
    )
    met_rows = [x[0] for x in fold_results]
    pred_rows = [x[1] for x in fold_results]
    metrics_df = pd.DataFrame(met_rows)
    oof_df = pd.concat(pred_rows, axis=0, ignore_index=True)
    oof_df = oof_df.sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True)
    return metrics_df, oof_df


def _late_fold_diagnostics(metrics_df: pd.DataFrame, oof_df: pd.DataFrame) -> dict:
    folds = sorted(metrics_df["fold"].astype(int).unique().tolist()) if not metrics_df.empty else []
    late_folds = folds[-2:] if len(folds) >= 2 else folds
    if not late_folds:
        return {
            "late_folds": [],
            "late_n_games": 0,
            "late_rmse": float("nan"),
            "late_mae": float("nan"),
            "late_tail_rmse": float("nan"),
            "late_dispersion_ratio": float("nan"),
            "late_tail_dispersion_ratio": float("nan"),
            "late_bias": float("nan"),
            "late_tail_bias": float("nan"),
        }
    late_oof = oof_df[oof_df["fold_id"].astype(int).isin(late_folds)].copy()
    met = regression_metrics(
        late_oof["y_true"].to_numpy(dtype=float),
        late_oof["y_pred_final"].to_numpy(dtype=float),
    )
    return {
        "late_folds": [int(x) for x in late_folds],
        "late_n_games": int(len(late_oof)),
        "late_rmse": float(met["rmse"]),
        "late_mae": float(met["mae"]),
        "late_tail_rmse": float(met["tail_rmse"]),
        "late_dispersion_ratio": float(met["dispersion_ratio"]),
        "late_tail_dispersion_ratio": float(met["tail_dispersion_ratio"]),
        "late_bias": float(met["bias"]),
        "late_tail_bias": float(met["tail_bias"]),
    }


def _parse_selected_config(registry: Mapping[str, dict], selected_config_id: str) -> tuple[RatingSpec, ModelSpec, str]:
    cfg = registry[selected_config_id]
    rs = cfg["rating_spec"]
    ms = cfg["model_spec"]
    rating_spec = RatingSpec(
        massey_alpha=float(rs["massey_alpha"]),
        use_elo=bool(rs["use_elo"]),
        elo_home_adv=float(rs["elo_home_adv"]),
        elo_k=float(rs["elo_k"]),
        elo_decay_a=float(rs["elo_decay_a"]),
        elo_decay_g=float(rs["elo_decay_g"]),
    )
    model_spec = ModelSpec(model_name=str(ms["model_name"]), params={k: float(v) for k, v in ms["params"].items()})
    module = str(cfg["module"])
    return rating_spec, model_spec, module


def _enforce_massey_only_gate(model_fold_metrics: pd.DataFrame, model_summary: pd.DataFrame, selected_config_id: str) -> tuple[str, dict]:
    gate_info = {
        "applied": False,
        "reason": "",
        "all_fold_winners_massey_only": False,
    }
    if model_fold_metrics.empty or model_summary.empty:
        return selected_config_id, gate_info

    fold_winners = (
        model_fold_metrics.sort_values(["fold", "rmse", "tail_rmse"], kind="mergesort")
        .groupby("fold", as_index=False)
        .first()
    )
    all_massey_only = bool((fold_winners["use_elo"].astype(int) == 0).all())
    gate_info["all_fold_winners_massey_only"] = all_massey_only
    if not all_massey_only:
        return selected_config_id, gate_info

    sel_rows = model_summary[model_summary["config_id"] == selected_config_id]
    if sel_rows.empty:
        return selected_config_id, gate_info
    selected_use_elo = int(sel_rows.iloc[0]["use_elo"])
    if selected_use_elo == 0:
        return selected_config_id, gate_info

    candidates = model_summary[model_summary["use_elo"].astype(int) == 0].copy()
    if candidates.empty:
        return selected_config_id, gate_info
    candidates = candidates.sort_values(
        ["mean_rmse", "mean_tail_rmse", "std_rmse", "mean_mae", "simplicity_rank"],
        kind="mergesort",
    ).reset_index(drop=True)
    override_id = str(candidates.iloc[0]["config_id"])
    gate_info["applied"] = True
    gate_info["reason"] = "All per-fold winners were massey-only; overriding to best massey-only config."
    return override_id, gate_info


def _build_rankings_output(rankings_input: pd.DataFrame, massey_map: Mapping[int, float], elo_map: Mapping[int, float], use_elo: bool) -> pd.DataFrame:
    out = rankings_input[["TeamID", "Team"]].copy()
    out["TeamID"] = out["TeamID"].astype(int)
    out["Team"] = out["Team"].astype(str)
    out["massey"] = out["TeamID"].map(massey_map).fillna(0.0).astype(float)
    out["elo"] = out["TeamID"].map(elo_map).fillna(1500.0).astype(float)
    out["massey_z"] = _zscore(out["massey"])
    out["elo_z"] = _zscore(out["elo"])
    out["score"] = out["massey_z"] + (0.20 if use_elo else 0.0) * out["elo_z"]
    out = out.sort_values(["score", "TeamID"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1, dtype=int)
    out = out[["TeamID", "Team", "Rank"]].sort_values("TeamID", kind="mergesort").reset_index(drop=True)
    return out


def _build_pdf_report(
    out_path: Path,
    *,
    git_hash: str,
    timestamp_utc: str,
    selected_config_digest: str,
    selected_config: Mapping,
    selection_lock_info: Mapping,
    folds: Sequence[Mapping],
    rating_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    selected_outer_metrics: pd.DataFrame,
    oof_df: pd.DataFrame,
    run_metadata_path: Path,
    selected_config_path: Path,
) -> None:
    with PdfPages(out_path) as pdf:
        _add_text_page(
            pdf,
            "Rocketball Derby Final Report",
            [
                f"Timestamp (UTC): {timestamp_utc}",
                f"Git hash: {git_hash}",
                f"Run metadata: {run_metadata_path}",
                f"Selected config path: {selected_config_path}",
                f"Selected config sha256: {selected_config_digest}",
                f"Rating summary rows: {len(rating_summary)}",
                f"Model summary rows: {len(model_summary)}",
                f"Selected outer-metric rows: {len(selected_outer_metrics)}",
                "",
                "Selection lock",
                "- Outer CV determines winning pipeline; full-train performs refit-only of locked choices.",
                "- No re-selection at full train stage.",
                "- Selection score: mean_rmse_w + lambda_tail*(mean_tail_rmse_w/mean_rmse_w) + lambda_xtail*(mean_xtail_rmse_w/mean_rmse_w) + lambda_disp*|log(disp_ratio_w)| + lambda_cap_hit*s_cap_hit_rate + lambda_tbias*|tail_bias_w| + lambda_tdisp*|log(tail_disp_ratio_w)| + lambda_stability*std_rmse.",
                "",
                "Selected config (locked)",
            ]
            + _wrap_lines(json.dumps(selected_config, indent=2), width=100),
        )

        diagram = [
            "Nested expanding CV (leakage-safe)",
            "",
            "Outer fold k:",
            "  train(earlier dates) -----------------> val(later dates)",
            "  1) Fit fold-local ratings/features on outer-train only.",
            "  2) Train base regressor on outer-train.",
            "  3) Build inner expanding splits within outer-train.",
            "  4) Generate inner OOF base predictions.",
            "  5) Fit postprocessor on inner OOF only.",
            "  6) Apply postprocessor to outer-val predictions.",
            "",
            "No game uses future outcomes for its own features/prediction.",
            "",
            "Outer folds:",
        ]
        for f in folds:
            diagram.append(
                f"fold {f['fold']}: train {pd.Timestamp(f['train_start_date']).date()}..{pd.Timestamp(f['train_end_date']).date()} | "
                f"val {pd.Timestamp(f['val_start_date']).date()}..{pd.Timestamp(f['val_end_date']).date()}"
            )
        _add_text_page(pdf, "Validation Design", diagram)

        derivation_lines = [
            "Affine dispersion correction derivation",
            "Given y_hat = a + s p, minimize E[(y - (a + s p))^2].",
            "Set partial derivatives to zero:",
            "  d/da: E[y - a - s p] = 0 -> a = E[y] - s E[p]",
            "  d/ds: E[p(y - a - s p)] = 0",
            "Substitute a and solve:",
            "  s* = Cov(p, y) / Var(p)",
            "  a* = mean(y) - s* mean(p)",
            "Fallback when covariance/variance unstable: s ~= std(y)/std(p).",
            f"Illustrative prior baseline ratio std(y)/std(p): {BASELINE_DISPERSION_RATIO_REFERENCE:.3f} (43.62/26.15).",
            "",
            "Expansion slope rule (anti-compression under under-dispersion):",
            "  r_base = std(p_oof_train) / std(y_train)",
            "  if r_base < 0.90: choose s = max(1.0, eta * naive_r), eta in {0.55,0.60,0.65,0.70}",
            "  where naive_r = std(y_train) / std(p_oof_train)",
            "  else: s = clip(s_star, 0.8, 1.2)",
            "  then recompute a = mean(y) - s*mean(p) with selected s",
            "",
            "Rating methods",
            "- Massey ridge anchor: margin = r_home - r_away + h + eps; neutral inference uses h=0.",
            "- EloK is secondary feature only (pregame diff, optional linear K-decay).",
            "- EloK is never allowed as standalone model family.",
        ]
        _add_text_page(pdf, "Dispersion and Ratings", derivation_lines)

        failure_lines = [
            "Failure analysis of 308 run",
            "- Early fold instability produced fold-1 RMSE blowups (observed near ~70).",
            "- Affine slope expansion could overshoot in small-OOF folds (observed s near ~2.6).",
            "- Over-dispersion appeared in selected configs (dispersion ratio > 1.05), hurting generalization.",
            "- 1SE + simplicity-first tie-break could pick brittle configs over better-scoring alternatives.",
        ]
        _add_text_page(pdf, "Failure Analysis", failure_lines)

        guardrail_lines = [
            "Guardrails added",
            "- Stability lambda enforced: lambda_stability in {0.10, 0.20} (no zero).",
            "- Selection uses weighted fold metrics (0.10,0.15,0.20,0.25,0.30) to emphasize late season.",
            "- Eta search restricted to {0.55,0.60,0.65,0.70}; higher eta values (0.8/0.9/1.0) removed.",
            "- Cap-hit penalty in score and cap-hit rejection gate to avoid selecting cap-bound calibrations.",
            "- Hard gates: max_fold_rmse, fold1_rmse, mean_dispersion_ratio upper bound, tail-dispersion, under-dispersion expansion checks.",
            "- Score-first 1SE tie-break: sort by score first, simplicity only as tie-break.",
            "- Massey alpha grid hardened: removed alpha=0.1; using {1.0, 10.0, 50.0}.",
            "- Expansion slope caps by inner OOF size: n<120=>1.25, <200=>1.35, <300=>1.45, else 1.60.",
            "- CV-only over-dispersion correction: if fold disp>1.10 (or 1.15 for n_oof>=300), one-step slope shrink; reject if still above threshold.",
            "- Piecewise tail eligibility gate: allowed only when max_fold_rmse<=45 and mean_disp in [0.75,1.00].",
        ]
        _add_text_page(pdf, "Guardrails Added", guardrail_lines)

        candidate_cols = [
            c
            for c in [
                "config_id",
                "mean_rmse",
                "std_rmse",
                "max_fold_rmse",
                "fold1_rmse",
                "mean_tail_rmse",
                "mean_dispersion_ratio",
                "mean_dispersion_ratio_w",
                "mean_affine_s",
                "mean_blowout_s",
                "s_cap_hit_rate",
                "cv_disp_guard_fail_rate",
                "invalid_postprocess_rate",
            ]
            if c in model_summary.columns
        ]
        _add_text_page(
            pdf,
            f"Candidate Stability Summary (rows={len(model_summary)})",
            _wrap_lines(_df_text(model_summary[candidate_cols], max_rows=50), width=120),
        )

        cfg_cols = [
            c
            for c in [
                "config_id",
                "rating_config_id",
                "model_config_id",
                "module",
                "massey_alpha",
                "use_elo",
                "elo_home_adv",
                "elo_decay_a",
                "elo_decay_g",
                "model_name",
            ]
            if c in model_summary.columns
        ]
        metric_cols = [
            c
            for c in [
                "config_id",
                "mean_rmse",
                "mean_rmse_w",
                "std_rmse",
                "mean_mae",
                "mean_mae_w",
                "mean_tail_rmse",
                "mean_tail_rmse_w",
                "mean_xtail_rmse",
                "mean_xtail_rmse_w",
                "mean_dispersion_ratio",
                "mean_dispersion_ratio_w",
                "mean_tail_dispersion_ratio",
                "mean_tail_dispersion_ratio_w",
                "mean_bias",
                "mean_bias_w",
                "mean_tail_bias",
                "mean_tail_bias_w",
                "n_folds",
            ]
            if c in model_summary.columns
        ]
        post_cols = [
            c
            for c in [
                "config_id",
                "mean_base_rmse",
                "mean_base_tail_rmse",
                "mean_affine_s",
                "mean_affine_eta",
                "mean_affine_r_base",
                "mean_affine_naive_r",
                "s_cap_hit_rate",
                "mean_tail_t",
                "mean_tail_k",
                "mean_tail_q",
                "invalid_postprocess_rate",
                "simplicity_rank",
            ]
            if c in model_summary.columns
        ]
        _add_text_page(
            pdf,
            f"Model Benchmark Table Group 1 (Config, rows={len(model_summary)})",
            _wrap_lines(_df_text(model_summary[cfg_cols], max_rows=50), width=120),
        )
        _add_text_page(
            pdf,
            f"Model Benchmark Table Group 2 (Metrics, rows={len(model_summary)})",
            _wrap_lines(_df_text(model_summary[metric_cols], max_rows=50), width=120),
        )
        _add_text_page(
            pdf,
            f"Model Benchmark Table Group 3 (Postprocessing, rows={len(model_summary)})",
            _wrap_lines(_df_text(model_summary[post_cols], max_rows=50), width=120),
        )

        tail_improve = float(selected_outer_metrics["base_tail_rmse"].mean() - selected_outer_metrics["tail_rmse"].mean())
        disp_before = float(np.std(oof_df["y_pred_base"].to_numpy(dtype=float), ddof=0) / max(np.std(oof_df["y_true"].to_numpy(dtype=float), ddof=0), 1e-9))
        disp_after = float(np.std(oof_df["y_pred_final"].to_numpy(dtype=float), ddof=0) / max(np.std(oof_df["y_true"].to_numpy(dtype=float), ddof=0), 1e-9))
        aff = selected_config.get("postprocessor_fit", {}).get("affine", {})
        par = selected_config.get("postprocessor_fit", {}).get("params", {})
        derby_n = int(selected_config.get("derby_pred_distribution", {}).get("n", -1))
        if derby_n != 75:
            raise ValueError(f"Derby prediction count assertion failed in report build: expected 75, got {derby_n}")
        summary_lines = [
            "Global vs regime scaling comparison",
            "- Candidate postprocessors included: none, expand_affine_global, expand_regime_affine_v2, expand_regime_affine_v2_heterosk, expand_piecewise_tail_v2.",
            f"- Locked module: {selected_config['model_stage']['module']}",
            "",
            "Tail and dispersion diagnostics (outer OOF for selected config)",
            f"- Tail RMSE improvement vs base: {tail_improve:.4f}",
            f"- Extreme Tail RMSE improvement vs base: {(selected_outer_metrics['base_xtail_rmse'].mean() - selected_outer_metrics['xtail_rmse'].mean()):.4f}",
            f"- Dispersion ratio base: {disp_before:.4f}",
            f"- Dispersion ratio final: {disp_after:.4f}",
            f"- Dispersion moved toward 1.0: {abs(1.0 - disp_after) <= abs(1.0 - disp_before)}",
            f"- Tail dispersion ratio final: {float(selected_outer_metrics['tail_dispersion_ratio'].mean()):.4f}",
            "",
            "Selected dispersion expansion parameters",
            f"- r_base={aff.get('r_base')}, naive_r={aff.get('naive_r')}",
            f"- s_star={aff.get('s_star')}, chosen_s={aff.get('s')}, eta={aff.get('eta')}",
            f"- tail_q={par.get('tail_q')}, tail_t={par.get('tail_t')}, tail_k={par.get('tail_k')}",
            "",
            "Derby prediction distribution summary",
            f"- n={derby_n} (asserted)",
            f"- mean={selected_config['derby_pred_distribution']['mean']:.4f}",
            f"- std={selected_config['derby_pred_distribution']['std']:.4f}",
            f"- p05={selected_config['derby_pred_distribution']['p05']:.4f}",
            f"- p50={selected_config['derby_pred_distribution']['p50']:.4f}",
            f"- p95={selected_config['derby_pred_distribution']['p95']:.4f}",
            "",
            "Selection lock details",
            f"- Rating-stage lambda_tail chosen: {selection_lock_info.get('rating_lambda_tail')}",
            f"- Rating-stage lambda_xtail chosen: {selection_lock_info.get('rating_lambda_xtail')}",
            f"- Rating-stage lambda chosen: {selection_lock_info.get('rating_lambda')}",
            f"- Rating-stage lambda_disp chosen: {selection_lock_info.get('rating_lambda_disp')}",
            f"- Rating-stage lambda_cap_hit chosen: {selection_lock_info.get('rating_lambda_cap_hit')}",
            f"- Rating-stage lambda_tbias chosen: {selection_lock_info.get('rating_lambda_tbias')}",
            f"- Rating-stage lambda_tdisp chosen: {selection_lock_info.get('rating_lambda_tdisp')}",
            f"- Model-stage lambda_tail chosen: {selection_lock_info.get('model_lambda_tail')}",
            f"- Model-stage lambda_xtail chosen: {selection_lock_info.get('model_lambda_xtail')}",
            f"- Model-stage lambda chosen: {selection_lock_info.get('model_lambda')}",
            f"- Model-stage lambda_disp chosen: {selection_lock_info.get('model_lambda_disp')}",
            f"- Model-stage lambda_cap_hit chosen: {selection_lock_info.get('model_lambda_cap_hit')}",
            f"- Model-stage lambda_tbias chosen: {selection_lock_info.get('model_lambda_tbias')}",
            f"- Model-stage lambda_tdisp chosen: {selection_lock_info.get('model_lambda_tdisp')}",
            f"- Selection gate rejections count: {selection_lock_info.get('gate_rejections_count')}",
            f"- Massey-only gate applied: {selection_lock_info.get('massey_gate_applied')}",
            f"- Massey-only gate reason: {selection_lock_info.get('massey_gate_reason')}",
        ]
        _add_text_page(pdf, "Results Summary", summary_lines)

        reg = par.get("regimes")
        if isinstance(reg, dict) and len(reg) > 0:
            reg_rows = []
            for rid, rv in reg.items():
                reg_rows.append(
                    {
                        "regime_id": rid,
                        "n": rv.get("n"),
                        "s": rv.get("s"),
                        "a": rv.get("a"),
                        "eta": rv.get("eta"),
                        "r_base": rv.get("r_base"),
                    }
                )
            reg_df = pd.DataFrame(reg_rows).sort_values("regime_id", kind="mergesort").reset_index(drop=True)
            _add_text_page(
                pdf,
                f"Per-Regime Expansion Table (rows={len(reg_df)})",
                _wrap_lines(_df_text(reg_df, max_rows=30), width=120),
            )

        y_true = oof_df["y_true"].to_numpy(dtype=float)
        y_base = oof_df["y_pred_base"].to_numpy(dtype=float)
        y_final = oof_df["y_pred_final"].to_numpy(dtype=float)
        resid_final = y_true - y_final

        fig1, ax1 = plt.subplots(figsize=(8.5, 5.5))
        ax1.scatter(y_final, resid_final, s=12, alpha=0.6, edgecolors="none")
        ax1.axhline(0.0, color="black", linewidth=1.0)
        ax1.set_title("Residual vs Fitted (OOF Final)")
        ax1.set_xlabel("Fitted (y_pred_final)")
        ax1.set_ylabel("Residual (y_true - y_pred_final)")
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
        bins = 28
        ax2.hist(y_true, bins=bins, alpha=0.55, label="y_true")
        ax2.hist(y_final, bins=bins, alpha=0.55, label="y_pred_final")
        ax2.set_title("Histogram: OOF Actual vs Final Predictions")
        ax2.legend()
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        tail_thr = float(np.quantile(np.abs(y_true), 0.80))
        tail_mask = np.abs(y_true) >= tail_thr
        fig3, ax3 = plt.subplots(figsize=(8.5, 5.5))
        ax3.scatter(y_true[tail_mask], y_base[tail_mask], s=14, alpha=0.5, label="base", edgecolors="none")
        ax3.scatter(y_true[tail_mask], y_final[tail_mask], s=14, alpha=0.5, label="final", edgecolors="none")
        lo = float(min(np.min(y_true[tail_mask]), np.min(y_final[tail_mask]), np.min(y_base[tail_mask])))
        hi = float(max(np.max(y_true[tail_mask]), np.max(y_final[tail_mask]), np.max(y_base[tail_mask])))
        ax3.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        ax3.set_title("Tail Scatter (|y_true| top 20%)")
        ax3.set_xlabel("y_true")
        ax3.set_ylabel("prediction")
        ax3.legend()
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(8.5, 5.5))
        ax4.scatter(y_true, y_final, s=12, alpha=0.6, edgecolors="none")
        lo_all = float(min(np.min(y_true), np.min(y_final)))
        hi_all = float(max(np.max(y_true), np.max(y_final)))
        ax4.plot([lo_all, hi_all], [lo_all, hi_all], color="black", linewidth=1.0)
        ax4.set_title("Prediction vs Actual (OOF Final)")
        ax4.set_xlabel("y_true")
        ax4.set_ylabel("y_pred_final")
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

        fold_disp = (
            oof_df.groupby("fold_id", as_index=False)
            .agg(
                y_sd=("y_true", lambda s: float(np.std(np.asarray(s, dtype=float), ddof=0))),
                base_sd=("y_pred_base", lambda s: float(np.std(np.asarray(s, dtype=float), ddof=0))),
                final_sd=("y_pred_final", lambda s: float(np.std(np.asarray(s, dtype=float), ddof=0))),
            )
        )
        fold_disp["base_ratio"] = fold_disp["base_sd"] / np.maximum(fold_disp["y_sd"], 1e-9)
        fold_disp["final_ratio"] = fold_disp["final_sd"] / np.maximum(fold_disp["y_sd"], 1e-9)
        x = np.arange(len(fold_disp), dtype=float)
        fig5, ax5 = plt.subplots(figsize=(8.5, 5.5))
        ax5.plot(x, fold_disp["base_ratio"], marker="o", label="base ratio")
        ax5.plot(x, fold_disp["final_ratio"], marker="o", label="final ratio")
        ax5.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax5.set_xticks(x)
        ax5.set_xticklabels([str(int(f)) for f in fold_disp["fold_id"]])
        ax5.set_title("Dispersion Ratio by Fold (std(pred)/std(y))")
        ax5.set_xlabel("fold_id")
        ax5.set_ylabel("dispersion ratio")
        ax5.legend()
        pdf.savefig(fig5, bbox_inches="tight")
        plt.close(fig5)


def _write_run_report_md(
    out_path: Path,
    *,
    git_hash: str,
    timestamp_utc: str,
    selected_config_path: Path,
    selected_config_digest: str,
    oof_csv_path: Path,
    oof_parquet_path: Path,
    runner_up_path: Path,
    rating_selection: Mapping,
    model_selection: Mapping,
    selected_config: Mapping,
    selected_outer_metrics: pd.DataFrame,
    invalid_reason_counts: Mapping[str, int],
    validations: Mapping[str, str],
) -> None:
    lines = []
    lines.append("# Run Report")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {timestamp_utc}")
    lines.append(f"- Git hash: `{git_hash}`")
    lines.append(f"- selected_config.json: `{selected_config_path}`")
    lines.append(f"- selected_config digest (sha256): `{selected_config_digest}`")
    lines.append(f"- oof_predictions.csv: `{oof_csv_path}`")
    lines.append(f"- oof_predictions.parquet: `{oof_parquet_path}`")
    lines.append(f"- runner_up_configs.json: `{runner_up_path}`")
    lines.append("")
    lines.append("## Selection Lock Proof")
    lines.append(f"- Rating-stage selected config id: `{rating_selection['selected_config_id']}`")
    lines.append(f"- Rating-stage tail lambda: `{rating_selection['lambda_tail']}`")
    lines.append(f"- Rating-stage xtail lambda: `{rating_selection['lambda_xtail']}`")
    lines.append(f"- Rating-stage stability lambda: `{rating_selection['lambda_stability']}`")
    lines.append(f"- Rating-stage dispersion lambda: `{rating_selection['lambda_disp']}`")
    lines.append(f"- Rating-stage cap-hit lambda: `{rating_selection.get('lambda_cap_hit')}`")
    lines.append(f"- Rating-stage tail-bias lambda: `{rating_selection.get('lambda_tbias')}`")
    lines.append(f"- Rating-stage tail-dispersion lambda: `{rating_selection['lambda_tdisp']}`")
    lines.append(f"- Model-stage selected config id: `{model_selection['selected_config_id']}`")
    lines.append(f"- Model-stage tail lambda: `{model_selection['lambda_tail']}`")
    lines.append(f"- Model-stage xtail lambda: `{model_selection['lambda_xtail']}`")
    lines.append(f"- Model-stage stability lambda: `{model_selection['lambda_stability']}`")
    lines.append(f"- Model-stage dispersion lambda: `{model_selection['lambda_disp']}`")
    lines.append(f"- Model-stage cap-hit lambda: `{model_selection.get('lambda_cap_hit')}`")
    lines.append(f"- Model-stage tail-bias lambda: `{model_selection.get('lambda_tbias')}`")
    lines.append(f"- Model-stage tail-dispersion lambda: `{model_selection['lambda_tdisp']}`")
    lines.append(f"- Model-stage best-of-three applied: `{model_selection.get('best_of_three_applied', False)}`")
    lines.append(f"- Model-stage best-of-three feasible count: `{model_selection.get('best_of_three_feasible_count', 0)}`")
    lines.append(f"- Selection gate rejections count: `{model_selection.get('gate_rejections_count', 0)}`")
    lines.append(f"- Massey-only gate applied: `{model_selection.get('massey_gate_applied', False)}`")
    if str(model_selection.get("massey_gate_reason", "")):
        lines.append(f"- Massey-only gate reason: `{model_selection.get('massey_gate_reason')}`")
    lines.append("- Full-train stage performed refit-only on locked selected configuration.")
    lines.append("")
    lines.append("## Outer Fold Table (Selected Config)")
    fold_cols = [c for c in ["fold", "rmse", "mae", "tail_rmse", "dispersion_ratio", "max_abs_pred"] if c in selected_outer_metrics.columns]
    lines.append("")
    lines.append("```text")
    lines.append(selected_outer_metrics[fold_cols].sort_values("fold", kind="mergesort").to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Summary Stats")
    lines.append(f"- mean_rmse: {selected_outer_metrics['rmse'].mean():.6f}")
    lines.append(f"- std_rmse: {selected_outer_metrics['rmse'].std(ddof=0):.6f}")
    lines.append(f"- max_fold_rmse: {selected_outer_metrics['rmse'].max():.6f}")
    lines.append(f"- mean_disp_ratio: {selected_outer_metrics['dispersion_ratio'].mean():.6f}")
    lines.append(f"- mean_tailrmse: {selected_outer_metrics['tail_rmse'].mean():.6f}")
    lines.append("")
    lines.append("## Diagnostics")
    lines.append(f"- Mean MAE: {selected_outer_metrics['mae'].mean():.6f}")
    lines.append(f"- Mean Extreme Tail RMSE: {selected_outer_metrics['xtail_rmse'].mean():.6f}")
    lines.append(f"- Mean tail dispersion ratio: {selected_outer_metrics['tail_dispersion_ratio'].mean():.6f}")
    lines.append(f"- Mean bias: {selected_outer_metrics['bias'].mean():.6f}")
    lines.append(f"- Mean tail bias: {selected_outer_metrics['tail_bias'].mean():.6f}")
    gate_table = model_selection.get("gate_table")
    if isinstance(gate_table, pd.DataFrame) and not gate_table.empty:
        sel_id = str(model_selection.get("selected_config_id"))
        gsel = gate_table[gate_table["config_id"].astype(str) == sel_id]
        if not gsel.empty:
            g = gsel.iloc[0]
            if "mean_rmse_w" in gsel.columns:
                lines.append(f"- Weighted mean RMSE: {float(g['mean_rmse_w']):.6f}")
            if "mean_tail_rmse_w" in gsel.columns:
                lines.append(f"- Weighted mean Tail RMSE: {float(g['mean_tail_rmse_w']):.6f}")
            if "mean_dispersion_ratio_w" in gsel.columns:
                lines.append(f"- Weighted mean dispersion ratio: {float(g['mean_dispersion_ratio_w']):.6f}")
            if "mean_tail_bias_w" in gsel.columns:
                lines.append(f"- Weighted mean tail bias: {float(g['mean_tail_bias_w']):.6f}")
            if "s_cap_hit_rate" in gsel.columns:
                lines.append(f"- Cap-hit rate: {float(g['s_cap_hit_rate']):.6f}")
    aff = selected_config.get("postprocessor_fit", {}).get("affine", {})
    par = selected_config.get("postprocessor_fit", {}).get("params", {})
    lines.append(f"- Selected expansion affine r_base: {aff.get('r_base')}")
    lines.append(f"- Selected expansion affine naive_r: {aff.get('naive_r')}")
    lines.append(f"- Selected expansion affine s_star: {aff.get('s_star')}")
    lines.append(f"- Selected expansion affine chosen_s: {aff.get('s')}")
    lines.append(f"- Selected expansion affine eta: {aff.get('eta')}")
    lines.append(f"- Selected piecewise t: {par.get('tail_t')}")
    lines.append(f"- Selected piecewise k: {par.get('tail_k')}")
    lines.append(f"- Selected piecewise q: {par.get('tail_q')}")
    if isinstance(gate_table, pd.DataFrame) and not gate_table.empty and "rejected" in gate_table.columns:
        rej = gate_table[gate_table["rejected"]].copy()
        lines.append(f"- Gate rejected configs: {len(rej)} / {len(gate_table)}")
        if not rej.empty and "reject_reason" in rej.columns:
            counts = rej["reject_reason"].fillna("").astype(str).value_counts()
            for reason, cnt in counts.items():
                lines.append(f"- Gate reason `{reason}`: {int(cnt)}")
    if invalid_reason_counts:
        for reason, cnt in invalid_reason_counts.items():
            lines.append(f"- Invalid postprocess reason `{reason}`: {int(cnt)}")
    lines.append("")
    lines.append("## Validation Proofs")
    for k, v in validations.items():
        lines.append(f"- {k}: {v}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    np.random.seed(SEED)
    run_start_ts = time.time()

    train_path = _resolve_input_path([ROOT / "Train.csv"], "Train.csv")
    pred_path = _resolve_input_path(
        [
            ROOT / "Predictions.csv",
            ROOT / "predictions.csv",
            ROOT / "Submission.zip" / "Predictions.csv",
            ROOT / "Submission.zip1" / "Predictions.csv",
        ],
        "Predictions.csv",
    )
    rank_path = _resolve_input_path(
        [
            ROOT / "Rankings.xlsx",
            ROOT / "rankings.xlsx",
            ROOT / "Submission.zip" / "Rankings.xlsx",
            ROOT / "Submission.zip1" / "Rankings.xlsx",
        ],
        "Rankings.xlsx",
    )

    train_raw = pd.read_csv(train_path)
    pred_raw = pd.read_csv(pred_path)
    rankings_raw = pd.read_excel(rank_path)

    train_df = parse_and_sort_train(train_raw)
    pred_df = parse_predictions(pred_raw)
    team_universe = build_team_universe(rankings_raw)
    team_ids = team_universe["TeamID"].astype(int).tolist()

    folds = make_expanding_time_folds(train_df, n_folds=5)
    fold_count_used = 5
    print("[folds] using 5 expanding folds globally")

    git_hash = _git_hash()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    print("[start] selecting rating config (outer-CV, lockable)")
    rating_spec, rating_fold_df, rating_selection = _select_rating_config(
        train_df,
        folds,
        team_ids,
        run_start_ts=run_start_ts,
    )
    rating_summary = aggregate_config_metrics(rating_fold_df)

    print("[start] benchmarking model families + dispersion modules on locked rating spec")
    model_fold_metrics, registry, budget_info = _evaluate_model_stage(
        train_df,
        folds,
        team_ids,
        rating_spec,
        run_start_ts=run_start_ts,
    )

    model_summary = aggregate_config_metrics(model_fold_metrics)
    model_summary = model_summary.merge(
        model_fold_metrics.groupby("config_id", as_index=False)
        .agg(
            mean_base_rmse=("base_rmse", "mean"),
            mean_base_tail_rmse=("base_tail_rmse", "mean"),
            mean_base_xtail_rmse=("base_xtail_rmse", "mean"),
            mean_base_tail_dispersion_ratio=("base_tail_dispersion_ratio", "mean"),
            mean_affine_s=("affine_s", "mean"),
            mean_affine_eta=("affine_eta", "mean"),
            mean_affine_r_base=("affine_r_base", "mean"),
            mean_affine_naive_r=("affine_naive_r", "mean"),
            mean_max_abs_pred=("max_abs_pred", "mean"),
            max_max_abs_pred=("max_abs_pred", "max"),
            cv_disp_guard_fail_rate=("cv_disp_guard_reject", "mean"),
            mean_cv_dispersion_ratio=("cv_dispersion_ratio", "mean"),
            mean_blowout_s=("blowout_mean_s", "mean"),
            mean_tail_t=("tail_t", "mean"),
            mean_tail_k=("tail_k", "mean"),
            mean_tail_q=("tail_q", "mean"),
            invalid_postprocess_rate=("invalid_postprocess", "mean"),
            module=("module", "first"),
            feature_set=("feature_set", "first"),
            rating_config_id=("rating_config_id", "first"),
            model_config_id=("model_config_id", "first"),
            model_name=("model_name", "first"),
            massey_alpha=("massey_alpha", "first"),
            use_elo=("use_elo", "first"),
            elo_home_adv=("elo_home_adv", "first"),
            elo_decay_a=("elo_decay_a", "first"),
            elo_decay_g=("elo_decay_g", "first"),
        ),
        on="config_id",
        how="left",
    )

    model_selection = select_with_1se_and_stability(
        model_summary,
        lambda_tail_values=LAMBDA_TAIL_GRID,
        lambda_xtail_values=LAMBDA_XTAIL_GRID,
        lambda_disp_values=LAMBDA_DISP_GRID,
        lambda_cap_hit_values=LAMBDA_CAP_HIT_GRID,
        lambda_tbias_values=LAMBDA_TBIAS_GRID,
        lambda_tdisp_values=LAMBDA_TDISP_GRID,
        lambda_stability_values=LAMBDA_STABILITY_GRID,
    )
    selected_config_id = str(model_selection["selected_config_id"])
    selected_config_id, massey_gate = _enforce_massey_only_gate(model_fold_metrics, model_summary, selected_config_id)
    if massey_gate["applied"]:
        pick_df = model_selection.get("lambda_picks")
        if isinstance(pick_df, pd.DataFrame) and not pick_df.empty:
            alt = pick_df[pick_df["config_id"] == selected_config_id].copy()
            if not alt.empty:
                alt = alt.sort_values(["score", "mean_rmse", "mean_tail_rmse"], kind="mergesort").iloc[0]
                model_selection["lambda_tail"] = float(alt["lambda_tail"])
                model_selection["lambda_xtail"] = float(alt["lambda_xtail"])
                model_selection["lambda_stability"] = float(alt["lambda_stability"])
                model_selection["lambda_disp"] = float(alt["lambda_disp"])
                model_selection["lambda_cap_hit"] = float(alt["lambda_cap_hit"])
                model_selection["lambda_tbias"] = float(alt["lambda_tbias"])
                model_selection["lambda_tdisp"] = float(alt["lambda_tdisp"])
        model_selection["selected_config_id"] = selected_config_id
        model_selection["massey_gate_applied"] = True
        model_selection["massey_gate_reason"] = massey_gate["reason"]
        print(f"[gate] {massey_gate['reason']}")
    else:
        model_selection["massey_gate_applied"] = False
        model_selection["massey_gate_reason"] = ""
    # Best-of-three safety selection: top-3 by weighted score, then late-fold diagnostics.
    gate_tbl = model_selection.get("gate_table")
    if isinstance(gate_tbl, pd.DataFrame) and not gate_tbl.empty:
        cand = gate_tbl[~gate_tbl["rejected"]].copy() if "rejected" in gate_tbl.columns else gate_tbl.copy()
        if cand.empty:
            cand = gate_tbl.copy()
        rmse_sort_col = "mean_rmse_w" if "mean_rmse_w" in cand.columns else "mean_rmse"
        tail_rmse_sort_col = "mean_tail_rmse_w" if "mean_tail_rmse_w" in cand.columns else "mean_tail_rmse"
        cand = cand.sort_values(["score", rmse_sort_col, tail_rmse_sort_col, "simplicity_rank"], kind="mergesort").reset_index(drop=True)
        top3 = cand.head(3).copy()
    else:
        top3 = pd.DataFrame([{"config_id": selected_config_id}])

    rerun_cache: Dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    runner_rows: List[dict] = []
    for row in top3.to_dict(orient="records"):
        cid = str(row.get("config_id", selected_config_id))
        rs, ms, md = _parse_selected_config(registry, cid)
        met_df, oof_df = _rerun_selected_for_oof(train_df, folds, team_ids, rs, ms, md)
        rerun_cache[cid] = (met_df, oof_df)
        late = _late_fold_diagnostics(met_df, oof_df)
        runner_rows.append(
            {
                "config_id": cid,
                "module": str(row.get("module", md)),
                "score": float(row.get("score", np.nan)),
                "mean_dispersion_ratio_w": float(row.get("mean_dispersion_ratio_w", row.get("mean_dispersion_ratio", np.nan))),
                "mean_tail_bias_w": float(row.get("mean_tail_bias_w", row.get("mean_tail_bias", np.nan))),
                **late,
            }
        )

    runner_df = pd.DataFrame(runner_rows).sort_values(["late_rmse", "score"], kind="mergesort").reset_index(drop=True)
    feasible = runner_df[
        (runner_df["mean_dispersion_ratio_w"] >= 0.65)
        & (runner_df["mean_dispersion_ratio_w"] <= 0.95)
        & (runner_df["mean_tail_bias_w"].abs() <= 10.0)
    ].copy()
    choose_df = feasible if not feasible.empty else runner_df.copy()
    choose_df["is_regime"] = choose_df["module"].astype(str).isin(["expand_regime_affine_v2", "expand_regime_affine_v2_heterosk"]).astype(int)
    choose_df = choose_df.sort_values(["late_rmse", "is_regime", "score"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    selected_config_id = str(choose_df.iloc[0]["config_id"])
    model_selection["selected_config_id"] = selected_config_id
    model_selection["best_of_three_applied"] = True
    model_selection["best_of_three_feasible_count"] = int(len(feasible))
    model_selection["best_of_three_pool_count"] = int(len(runner_df))

    sel_rating_spec, sel_model_spec, sel_module = _parse_selected_config(registry, selected_config_id)

    print("[start] rerun selected config to produce locked outer OOF predictions artifact")
    if selected_config_id in rerun_cache:
        selected_outer_metrics, oof_predictions = rerun_cache[selected_config_id]
    else:
        selected_outer_metrics, oof_predictions = _rerun_selected_for_oof(
            train_df,
            folds,
            team_ids,
            sel_rating_spec,
            sel_model_spec,
            sel_module,
        )

    oof_out_csv = ROOT / "oof_predictions.csv"
    oof_predictions.to_csv(oof_out_csv, index=False)
    oof_out_parquet: Path | None = ROOT / "oof_predictions.parquet"
    try:
        oof_predictions.to_parquet(oof_out_parquet, index=False)
    except Exception:
        oof_out_parquet = None

    print("[start] full-train refit with locked config only")
    full_train_tbl, massey_model, elo_model, history_states = build_full_train_table(
        train_df,
        team_ids,
        massey_alpha=sel_rating_spec.massey_alpha,
        use_elo=sel_rating_spec.use_elo,
        elo_home_adv=sel_rating_spec.elo_home_adv,
        elo_k=sel_rating_spec.elo_k,
        elo_decay_a=sel_rating_spec.elo_decay_a,
        elo_decay_g=sel_rating_spec.elo_decay_g,
    )
    selected_feature_set, selected_feature_cols = feature_cols_for_rating_spec(sel_rating_spec)

    X_full, y_full = _get_xy(full_train_tbl, feature_cols=selected_feature_cols)
    base_est = make_estimator(sel_model_spec, seed=SEED)
    base_est.fit(X_full, y_full)

    inner_oof_full = build_inner_oof(
        train_df,
        team_ids=team_ids,
        rating_spec=sel_rating_spec,
        model_spec=sel_model_spec,
        feature_cols=selected_feature_cols,
    )
    post_full = fit_postprocessor(sel_module, inner_oof_full)

    derby_tbl = build_derby_table(pred_df, massey_model, elo_model, history_states, use_elo=sel_rating_spec.use_elo)
    X_derby = derby_tbl.reindex(columns=selected_feature_cols, fill_value=0.0).astype(float)
    pred_base_derby = np.asarray(base_est.predict(X_derby), dtype=float)
    derby_gp_min = np.minimum(derby_tbl["games_played_home"].to_numpy(dtype=float), derby_tbl["games_played_away"].to_numpy(dtype=float))
    derby_post = apply_postprocessor(post_full, pred_base_derby, derby_tbl["d_key"].to_numpy(dtype=float), derby_gp_min)
    pred_final_derby = np.asarray(derby_post["y_pred_final"], dtype=float)
    if len(pred_final_derby) != 75:
        raise ValueError(f"Derby prediction count mismatch: expected 75, got {len(pred_final_derby)}")
    pred_final_int = np.rint(pred_final_derby).astype(int)

    predictions_out = pred_df.copy()
    predictions_out["Team1_WinMargin"] = pred_final_int.astype(int)
    predictions_path = ROOT / "predictions.csv"
    predictions_out.to_csv(predictions_path, index=False)

    rankings_out = _build_rankings_output(rankings_raw, massey_model.team_rating, elo_model.ratings, use_elo=sel_rating_spec.use_elo)
    rankings_path = ROOT / "rankings.xlsx"
    rankings_out.to_excel(rankings_path, index=False)

    selected_config = {
        "selection_lock": True,
        "elok_secondary_only": True,
        "selected_at": timestamp_utc,
        "git_hash": git_hash,
        "outer_fold_count": fold_count_used,
        "rating_stage": {
            "config_id": rating_selection["selected_config_id"],
            "lambda_tail": rating_selection["lambda_tail"],
            "lambda_xtail": rating_selection["lambda_xtail"],
            "lambda_stability": rating_selection["lambda_stability"],
            "lambda_disp": rating_selection["lambda_disp"],
            "lambda_cap_hit": rating_selection.get("lambda_cap_hit"),
            "lambda_tbias": rating_selection.get("lambda_tbias"),
            "lambda_tdisp": rating_selection["lambda_tdisp"],
            "spec": asdict(sel_rating_spec),
        },
        "model_stage": {
            "config_id": selected_config_id,
            "lambda_tail": model_selection["lambda_tail"],
            "lambda_xtail": model_selection["lambda_xtail"],
            "lambda_stability": model_selection["lambda_stability"],
            "lambda_disp": model_selection["lambda_disp"],
            "lambda_cap_hit": model_selection.get("lambda_cap_hit"),
            "lambda_tbias": model_selection.get("lambda_tbias"),
            "lambda_tdisp": model_selection["lambda_tdisp"],
            "model_spec": {"model_name": sel_model_spec.model_name, "params": sel_model_spec.params},
            "module": sel_module,
            "gate_rejections_count": int(model_selection.get("gate_rejections_count", 0)),
            "massey_gate_applied": bool(model_selection.get("massey_gate_applied", False)),
            "massey_gate_reason": str(model_selection.get("massey_gate_reason", "")),
            "best_of_three_applied": bool(model_selection.get("best_of_three_applied", False)),
            "best_of_three_feasible_count": int(model_selection.get("best_of_three_feasible_count", 0)),
            "best_of_three_pool_count": int(model_selection.get("best_of_three_pool_count", 0)),
        },
        "oof_paths": {
            "csv": str(oof_out_csv),
            "parquet": str(oof_out_parquet) if oof_out_parquet is not None else "",
        },
        "feature_set": selected_feature_set,
        "features": selected_feature_cols,
        "selection_score": "mean_rmse_w + lambda_tail*(mean_tail_rmse_w/mean_rmse_w) + lambda_xtail*(mean_xtail_rmse_w/mean_rmse_w) + lambda_disp*abs(log(mean_dispersion_ratio_w)) + lambda_cap_hit*s_cap_hit_rate + lambda_tbias*abs(mean_tail_bias_w) + lambda_tdisp*abs(log(mean_tail_dispersion_ratio_w)) + lambda_stability*std_rmse",
        "postprocessor_fit": post_full,
        "derby_pred_distribution": {
            "n": int(len(pred_final_derby)),
            "mean": float(np.mean(pred_final_derby)),
            "std": float(np.std(pred_final_derby, ddof=0)),
            "p05": float(np.quantile(pred_final_derby, 0.05)),
            "p50": float(np.quantile(pred_final_derby, 0.50)),
            "p95": float(np.quantile(pred_final_derby, 0.95)),
        },
    }

    runner_up_path = ROOT / "runner_up_configs.json"
    runner_rows_json = json.loads(runner_df.to_json(orient="records"))
    _json_dump(
        runner_up_path,
        {
            "timestamp_utc": timestamp_utc,
            "chosen_config_id": selected_config_id,
            "top_candidates": runner_rows_json,
        },
    )
    selected_config["runner_up_configs_path"] = str(runner_up_path)

    selected_config_path = ROOT / "selected_config.json"
    _json_dump(selected_config_path, selected_config)
    selected_digest = _sha256(selected_config_path)

    run_metadata = {
        "timestamp_utc": timestamp_utc,
        "git_hash": git_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cwd": str(ROOT),
        "input_paths": {"train": str(train_path), "predictions_seed": str(pred_path), "rankings_seed": str(rank_path)},
        "thread_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "search_env": {
            "ALGOSPORTS_N_JOBS": str(os.environ.get("ALGOSPORTS_N_JOBS", N_JOBS)),
            "ALGOSPORTS_MAX_MODEL_FITS": str(os.environ.get("ALGOSPORTS_MAX_MODEL_FITS", MAX_MODEL_FITS)),
            "ALGOSPORTS_MAX_TOTAL_SECONDS": str(os.environ.get("ALGOSPORTS_MAX_TOTAL_SECONDS", MAX_TOTAL_SECONDS)),
        },
        "n_jobs": min(N_JOBS, len(folds)),
        "max_model_fits": int(MAX_MODEL_FITS),
        "max_total_seconds": int(MAX_TOTAL_SECONDS),
        "budget_info": budget_info,
        "outer_fold_count": fold_count_used,
    }
    run_metadata_path = ROOT / "run_metadata.json"
    _json_dump(run_metadata_path, run_metadata)

    final_report_path = ROOT / "final_report.pdf"
    _build_pdf_report(
        final_report_path,
        git_hash=git_hash,
        timestamp_utc=timestamp_utc,
        selected_config_digest=selected_digest,
        selected_config=selected_config,
        selection_lock_info={
            "rating_lambda_tail": rating_selection["lambda_tail"],
            "rating_lambda_xtail": rating_selection["lambda_xtail"],
            "rating_lambda": rating_selection["lambda_stability"],
            "rating_lambda_disp": rating_selection["lambda_disp"],
            "rating_lambda_cap_hit": rating_selection.get("lambda_cap_hit"),
            "rating_lambda_tbias": rating_selection.get("lambda_tbias"),
            "rating_lambda_tdisp": rating_selection["lambda_tdisp"],
            "model_lambda_tail": model_selection["lambda_tail"],
            "model_lambda_xtail": model_selection["lambda_xtail"],
            "model_lambda": model_selection["lambda_stability"],
            "model_lambda_disp": model_selection["lambda_disp"],
            "model_lambda_cap_hit": model_selection.get("lambda_cap_hit"),
            "model_lambda_tbias": model_selection.get("lambda_tbias"),
            "model_lambda_tdisp": model_selection["lambda_tdisp"],
            "gate_rejections_count": model_selection.get("gate_rejections_count", 0),
            "massey_gate_applied": model_selection.get("massey_gate_applied", False),
            "massey_gate_reason": model_selection.get("massey_gate_reason", ""),
        },
        folds=folds,
        rating_summary=rating_summary,
        model_summary=model_summary,
        selected_outer_metrics=selected_outer_metrics,
        oof_df=oof_predictions,
        run_metadata_path=run_metadata_path,
        selected_config_path=selected_config_path,
    )

    validations = {}
    pred_check = pd.read_csv(predictions_path)
    validations["predictions.csv exists"] = str(predictions_path.exists())
    validations["predictions rows"] = str(len(pred_check))
    validations["predictions Team1_WinMargin int"] = str(pd.api.types.is_integer_dtype(pred_check["Team1_WinMargin"]))
    validations["predictions Team1_WinMargin missing"] = str(int(pred_check["Team1_WinMargin"].isna().sum()))

    rank_check = pd.read_excel(rankings_path)
    validations["rankings.xlsx exists"] = str(rankings_path.exists())
    validations["rankings rows"] = str(len(rank_check))
    rank_set_ok = set(rank_check["Rank"].astype(int).tolist()) == set(range(1, len(rank_check) + 1))
    validations["rank permutation 1..165"] = str(rank_set_ok)

    validations["final_report.pdf exists"] = str(final_report_path.exists())
    validations["final_report.pdf size_bytes"] = str(final_report_path.stat().st_size if final_report_path.exists() else 0)
    validations["oof_predictions.csv exists"] = str(oof_out_csv.exists())
    validations["oof_predictions.parquet exists"] = str(oof_out_parquet is not None and oof_out_parquet.exists())
    validations["runner_up_configs.json exists"] = str(runner_up_path.exists())
    required_oof_cols = {
        "GameID",
        "Date",
        "HomeID",
        "AwayID",
        "y_true",
        "y_pred_base",
        "y_pred_final",
        "fold_id",
        "regime_id",
        "mismatch_key",
        "info_key",
        "postprocess_s_used",
        "postprocess_t_used",
        "postprocess_k_used",
        "postprocess_eta_used",
    }
    oof_check = pd.read_csv(oof_out_csv)
    validations["oof_predictions required cols"] = str(required_oof_cols.issubset(set(oof_check.columns)))

    run_report_path = ROOT / "run_report.md"
    invalid_reason_counts = (
        model_fold_metrics.loc[model_fold_metrics["invalid_postprocess"].astype(int) == 1, "invalid_reason"]
        .fillna("")
        .astype(str)
        .loc[lambda s: s != ""]
        .value_counts()
        .to_dict()
    )
    _write_run_report_md(
        run_report_path,
        git_hash=git_hash,
        timestamp_utc=timestamp_utc,
        selected_config_path=selected_config_path,
        selected_config_digest=selected_digest,
        oof_csv_path=oof_out_csv,
        oof_parquet_path=oof_out_parquet if oof_out_parquet is not None else Path(""),
        runner_up_path=runner_up_path,
        rating_selection=rating_selection,
        model_selection=model_selection,
        selected_config=selected_config,
        selected_outer_metrics=selected_outer_metrics,
        invalid_reason_counts=invalid_reason_counts,
        validations=validations,
    )

    print("\n[hard-stop validations]")
    print(f"predictions.csv exists={predictions_path.exists()} rows={len(pred_check)} missing={int(pred_check['Team1_WinMargin'].isna().sum())}")
    print(f"rankings.xlsx exists={rankings_path.exists()} rows={len(rank_check)} rank_perm_1..{len(rank_check)}={rank_set_ok}")
    print(f"final_report.pdf exists={final_report_path.exists()} size={final_report_path.stat().st_size if final_report_path.exists() else 0}")
    print(f"oof_predictions.csv exists={oof_out_csv.exists()} required_cols_ok={required_oof_cols.issubset(set(oof_check.columns))}")
    print(f"runner_up_configs.json exists={runner_up_path.exists()}")
    print(f"selected_config.json digest={selected_digest}")
    print("\npredictions head(10):")
    print(pred_check.head(10).to_string(index=False))
    print("\nrankings top10 by rank:")
    print(rank_check.sort_values("Rank", kind="mergesort").head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
