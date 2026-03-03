from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def _next_model_number(root: Path) -> int:
    nums: List[int] = []
    pat = re.compile(r"^Submission\.zip(\d+)(?:\.zip)?$", re.IGNORECASE)
    for p in root.iterdir():
        m = pat.match(p.name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass
    legacy = root / "legacy_reports"
    if legacy.exists():
        for p in legacy.glob("final_report_model*.pdf"):
            m = re.search(r"model(\d+)", p.name)
            if m:
                try:
                    nums.append(int(m.group(1)))
                except Exception:
                    pass
    return (max(nums) + 1) if nums else 1


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else float("nan")


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p))) if len(y) else float("nan")


def _bias(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(p - y)) if len(y) else float("nan")


def _corr(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    sy = float(np.std(y))
    sp = float(np.std(p))
    if sy <= 1e-12 or sp <= 1e-12:
        return 0.0
    c = float(np.corrcoef(y, p)[0, 1])
    return c if np.isfinite(c) else 0.0


def _tail_metrics(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    q80 = float(np.quantile(np.abs(y), 0.80))
    m = np.abs(y) >= q80
    if not np.any(m):
        return _rmse(y, p), _bias(y, p), 1.0
    yt = y[m]
    pt = p[m]
    disp = float(np.std(pt) / max(np.std(yt), 1e-9))
    return _rmse(yt, pt), _bias(yt, pt), disp


def _fold_metrics_from_oof(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "outer_fold" not in df.columns:
        return pd.DataFrame()
    out: List[Dict[str, Any]] = []
    for fold, g in df.groupby("outer_fold", sort=True):
        y = g["y_true"].to_numpy(dtype=float)
        p = g["y_pred_final"].to_numpy(dtype=float) if "y_pred_final" in g.columns else g["pred_final"].to_numpy(dtype=float)
        tail_rmse, tail_bias, tail_disp = _tail_metrics(y, p)
        out.append(
            {
                "outer_fold": int(fold),
                "rmse": _rmse(y, p),
                "mae": _mae(y, p),
                "bias": _bias(y, p),
                "tail_rmse": tail_rmse,
                "tail_bias": tail_bias,
                "tail_dispersion_ratio": tail_disp,
                "corr": _corr(y, p),
            }
        )
    return pd.DataFrame(out).sort_values("outer_fold", kind="mergesort").reset_index(drop=True)


def _top3_all_global(root: Path) -> bool:
    p = root / "runner_up_configs.json"
    if not p.exists():
        return True
    obj = _read_json(p)
    rows = obj.get("top_candidates", [])
    if len(rows) < 3:
        return True
    mods = [str(r.get("module_name", r.get("module", ""))) for r in rows[:3]]
    return len(set(mods)) == 1 and mods[0] == "expand_affine_global"


def _hard_gates(metrics: Dict[str, float]) -> bool:
    return (
        float(metrics.get("mean_rmse", float("inf"))) <= 35.2
        and float(metrics.get("corr", float("-inf"))) >= 0.60
        and float(metrics.get("mean_tail_dispersion", float("-inf"))) >= 0.58
        and float(metrics.get("mean_tail_bias", float("-inf"))) >= -6.0
        and float(metrics.get("fold2_rmse", float("inf"))) <= 38.0
    )


def _run1_gate(metrics: Dict[str, float]) -> bool:
    return float(metrics.get("corr", float("-inf"))) >= 0.58 and float(metrics.get("mean_rmse", float("inf"))) <= 35.8


def _run_pipeline(run_label: str, attempt: int, env_extra: Dict[str, str]) -> Dict[str, Any]:
    tag = f"phase1_{run_label}_a{attempt}"
    env = os.environ.copy()
    env.update(
        {
            "ALGOSPORTS_PIPELINE_ENGINE": "nextgen",
            "ALGOSPORTS_FAST_MODE": "0",
            "ALGOSPORTS_ENABLE_OPTIMIZATIONS": "1",
            "ALGOSPORTS_ABORT_ON_BUDGET_HIT": "1",
            "ALGOSPORTS_MAX_MODEL_FITS": "8000",
            "ALGOSPORTS_MAX_FITS": "8000",
            "ALGOSPORTS_MAX_TOTAL_SECONDS": "3600",
            "ALGOSPORTS_MAX_SCAN_SECONDS": "1200",
            "ALGOSPORTS_PHASE1_RUN": run_label,
            "ALGOSPORTS_TIMING_TAG": tag,
            "ALGOSPORTS_ETA_GRID": "0.55,0.60,0.65,0.70,0.75,0.80",
        }
    )
    env.update({k: str(v) for k, v in env_extra.items()})

    proc = subprocess.run(
        [sys.executable, "run_pipeline.py"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    log_path = ROOT / f"{tag}.log"
    log_path.write_text(proc.stdout + "\n\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")

    metadata = _read_json(ROOT / "run_metadata.json")
    selected = _read_json(ROOT / "selected_config.json")
    runner = _read_json(ROOT / "runner_up_configs.json")
    fold_df = _fold_metrics_from_oof(ROOT / "oof_predictions.csv")
    phase_metrics = dict(metadata.get("phase_metrics", {}))
    budget_triggered = [str(x) for x in metadata.get("budget_triggered", [])]
    fit_count = int(metadata.get("model_fit_count", 0))
    fallback_flag = bool(metadata.get("fallback_triggered", True))
    no_shortcut = bool(metadata.get("no_regression_to_mean_shortcut", False))

    return {
        "run_label": run_label,
        "attempt": int(attempt),
        "env": dict(env_extra),
        "returncode": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "log_path": str(log_path),
        "phase_metrics": phase_metrics,
        "fit_count": fit_count,
        "budget_triggered": budget_triggered,
        "fallback_triggered": fallback_flag,
        "no_regression_to_mean_shortcut": no_shortcut,
        "fold_metrics": fold_df,
        "selected_config": selected,
        "runner_up": runner,
    }


def _is_clean_run(res: Dict[str, Any]) -> bool:
    if int(res.get("returncode", 1)) != 0:
        return False
    if bool(res.get("fallback_triggered", True)):
        return False
    if not bool(res.get("no_regression_to_mean_shortcut", False)):
        return False
    bt = set(res.get("budget_triggered", []))
    if "max_fits" in bt:
        return False
    return True


def _delta_row(prev: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    pm = prev.get("phase_metrics", {})
    cm = cur.get("phase_metrics", {})
    return {
        "run": cur.get("run_label"),
        "attempt": cur.get("attempt"),
        "d_mean_rmse": float(cm.get("mean_rmse", np.nan)) - float(pm.get("mean_rmse", np.nan)),
        "d_corr": float(cm.get("corr", np.nan)) - float(pm.get("corr", np.nan)),
        "d_tail_disp": float(cm.get("mean_tail_dispersion", np.nan)) - float(pm.get("mean_tail_dispersion", np.nan)),
        "d_tail_bias": float(cm.get("mean_tail_bias", np.nan)) - float(pm.get("mean_tail_bias", np.nan)),
        "d_fold2_rmse": float(cm.get("fold2_rmse", np.nan)) - float(pm.get("fold2_rmse", np.nan)),
    }


def _print_run_summary(res: Dict[str, Any]) -> None:
    m = res.get("phase_metrics", {})
    print(
        json.dumps(
            {
                "run": res.get("run_label"),
                "attempt": res.get("attempt"),
                "returncode": res.get("returncode"),
                "mean_rmse": m.get("mean_rmse"),
                "corr": m.get("corr"),
                "mean_tail_dispersion": m.get("mean_tail_dispersion"),
                "mean_tail_bias": m.get("mean_tail_bias"),
                "fold2_rmse": m.get("fold2_rmse"),
                "fit_count": res.get("fit_count"),
                "budget_triggered": res.get("budget_triggered"),
                "fallback_triggered": res.get("fallback_triggered"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _ensure_submission_rotation(root: Path, model_num: int) -> None:
    sub = root / "Submission.zip"
    if sub.exists():
        dst = root / f"Submission.zip{model_num}"
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(sub), str(dst))

    tmp = root / "_submission_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "Predictions.csv", tmp / "Predictions.csv")
    shutil.copy2(root / "Rankings.xlsx", tmp / "Rankings.xlsx")

    out_zip = root / "Submission.zip"
    if out_zip.exists():
        out_zip.unlink()
    archive = shutil.make_archive(str(root / "Submission"), "zip", root_dir=str(tmp))
    if Path(archive) != out_zip:
        shutil.move(str(archive), str(out_zip))
    shutil.rmtree(tmp, ignore_errors=True)


def _archive_prior_reports(root: Path, model_num: int, prior_files: Dict[str, Path]) -> None:
    legacy = root / "legacy_reports"
    legacy.mkdir(exist_ok=True)
    for name, path in prior_files.items():
        if not path.exists():
            continue
        ext = path.suffix
        stem = "final_report" if "final_report" in name else "run_report"
        dst = legacy / f"{stem}_model{model_num}{ext}"
        if dst.exists():
            dst.unlink()
        shutil.move(str(path), str(dst))


def _prestage_prior_reports(root: Path) -> Dict[str, Path]:
    tmp = root / ".phase1_preexisting_reports"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    moved: Dict[str, Path] = {}
    for fname in ["final_report.pdf", "run_report.md"]:
        src = root / fname
        if src.exists():
            dst = tmp / fname
            if dst.exists():
                dst.unlink()
            shutil.copy2(src, dst)
            moved[fname] = dst
    return moved


def _print_evidence(run1: Dict[str, Any], run2: Dict[str, Any]) -> None:
    s2 = run2.get("selected_config", {})
    f2 = run2.get("fold_metrics", pd.DataFrame())

    print("\n=== FINAL EVIDENCE BLOCK ===")
    print("Run 1 Metric Table:")
    print(json.dumps(run1.get("phase_metrics", {}), indent=2, sort_keys=True))
    print("Run 2 Metric Table:")
    print(json.dumps(run2.get("phase_metrics", {}), indent=2, sort_keys=True))

    print("\nFold-level RMSE / Tail / Corr (Run 2):")
    if isinstance(f2, pd.DataFrame) and not f2.empty:
        print(f2[["outer_fold", "rmse", "tail_rmse", "tail_dispersion_ratio", "tail_bias", "corr"]].to_string(index=False))
    else:
        print("(empty)")

    print("\nTail RMSE:", run2.get("phase_metrics", {}).get("mean_tail_rmse"))
    print("Tail Dispersion:", run2.get("phase_metrics", {}).get("mean_tail_dispersion"))
    print("Tail Bias:", run2.get("phase_metrics", {}).get("mean_tail_bias"))
    print("Correlation:", run2.get("phase_metrics", {}).get("corr"))
    print("Cap-hit rate:", run2.get("phase_metrics", {}).get("cap_hit_rate"))
    print("Fit count used:", run2.get("fit_count"))

    print("\nModel family selected:")
    print(s2.get("model_family"))
    print("\nFinal feature list:")
    print(", ".join(s2.get("features", [])))
    print("\nCore hyperparameters:")
    print(json.dumps(s2.get("core_candidate", {}), indent=2, sort_keys=True))
    print("\nPostprocess hyperparameters:")
    print(json.dumps(s2.get("postprocess", {}), indent=2, sort_keys=True))

    no_fallback = _is_clean_run(run2)
    print("\nNo fallback occurred:", bool(no_fallback))
    print("No regression-to-mean shortcut used:", bool(run2.get("no_regression_to_mean_shortcut", False)))
    print("Regime gate logic confirmation:", s2.get("gate_confirmation"))


def main() -> int:
    model_num = _next_model_number(ROOT)
    prior_reports = _prestage_prior_reports(ROOT)

    run_history: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []

    run1_plans: List[Dict[str, str]] = [
        {
            "ALGOSPORTS_POST_MODULES": "expand_affine_global",
            "ALGOSPORTS_SCAN_RETURN_TOP": "5",
            "ALGOSPORTS_SCAN_PRUNE_TOPN": "40",
            "ALGOSPORTS_SCAN_PRUNE_FRAC": "0.45",
            "ALGOSPORTS_HISTGB_BAG_N_MODELS": "2",
            "ALGOSPORTS_HISTGB_GRID_LEVEL": "1",
            "ALGOSPORTS_FEATURE_PROFILES": "compact_recency",
            "ALGOSPORTS_HALF_LIFE_GRID": "none,30",
            "ALGOSPORTS_RIDGE_ALPHA_GRID": "1,4",
            "ALGOSPORTS_ELO_VARIANTS": "elo_base_static,elo_dynamic_k",
            "ALGOSPORTS_ENABLE_INTERACTION_FEATURES": "1",
            "ALGOSPORTS_HUBER_EPSILON": "1.35",
        },
        {
            "ALGOSPORTS_POST_MODULES": "expand_affine_global",
            "ALGOSPORTS_SCAN_RETURN_TOP": "6",
            "ALGOSPORTS_SCAN_PRUNE_TOPN": "55",
            "ALGOSPORTS_SCAN_PRUNE_FRAC": "0.55",
            "ALGOSPORTS_HISTGB_BAG_N_MODELS": "2",
            "ALGOSPORTS_HISTGB_GRID_LEVEL": "3",
            "ALGOSPORTS_FEATURE_PROFILES": "compact_recency",
            "ALGOSPORTS_HALF_LIFE_GRID": "none,20,40",
            "ALGOSPORTS_RIDGE_ALPHA_GRID": "0.5,1,2,4",
            "ALGOSPORTS_ELO_VARIANTS": "elo_base_static,elo_dynamic_k",
            "ALGOSPORTS_ENABLE_INTERACTION_FEATURES": "1",
            "ALGOSPORTS_HUBER_EPSILON": "1.20",
        },
        {
            "ALGOSPORTS_POST_MODULES": "expand_affine_global",
            "ALGOSPORTS_SCAN_RETURN_TOP": "7",
            "ALGOSPORTS_SCAN_PRUNE_TOPN": "65",
            "ALGOSPORTS_SCAN_PRUNE_FRAC": "0.62",
            "ALGOSPORTS_HISTGB_BAG_N_MODELS": "3",
            "ALGOSPORTS_HISTGB_GRID_LEVEL": "2",
            "ALGOSPORTS_FEATURE_PROFILES": "compact_recency,full_recency",
            "ALGOSPORTS_HALF_LIFE_GRID": "15,30",
            "ALGOSPORTS_RIDGE_ALPHA_GRID": "0.5,1,2,4",
            "ALGOSPORTS_ELO_VARIANTS": "elo_base_static,elo_dynamic_k,elo_dynamic_k_teamha",
            "ALGOSPORTS_ENABLE_INTERACTION_FEATURES": "1",
            "ALGOSPORTS_HUBER_EPSILON": "1.10",
            "ALGOSPORTS_HUBER_ALPHA": "0.0005",
        },
        {
            "ALGOSPORTS_POST_MODULES": "expand_affine_global",
            "ALGOSPORTS_SCAN_RETURN_TOP": "8",
            "ALGOSPORTS_SCAN_PRUNE_TOPN": "75",
            "ALGOSPORTS_SCAN_PRUNE_FRAC": "0.68",
            "ALGOSPORTS_HISTGB_BAG_N_MODELS": "3",
            "ALGOSPORTS_HISTGB_GRID_LEVEL": "2",
            "ALGOSPORTS_FEATURE_PROFILES": "full_recency",
            "ALGOSPORTS_HALF_LIFE_GRID": "10,20,30,45",
            "ALGOSPORTS_RIDGE_ALPHA_GRID": "0.25,0.5,1,2,4",
            "ALGOSPORTS_ELO_VARIANTS": "elo_base_static,elo_dynamic_k,elo_dynamic_k_teamha",
            "ALGOSPORTS_ENABLE_INTERACTION_FEATURES": "1",
            "ALGOSPORTS_HUBER_EPSILON": "1.05",
            "ALGOSPORTS_HUBER_ALPHA": "0.0003",
        },
    ]

    run1_result: Dict[str, Any] | None = None
    for i, plan in enumerate(run1_plans, start=1):
        res = _run_pipeline("run1", i, plan)
        _print_run_summary(res)
        if run_history:
            deltas.append(_delta_row(run_history[-1], res))
        run_history.append(res)

        if "max_fits" in set(res.get("budget_triggered", [])) or (
            int(res.get("returncode", 1)) != 0 and "Fit budget exhausted" in (res.get("stderr_tail", "") + "\n" + res.get("stdout_tail", ""))
        ):
            # retry with higher fit cap per contract if hard cap hit
            plan = dict(plan)
            plan["ALGOSPORTS_MAX_MODEL_FITS"] = "10000"
            plan["ALGOSPORTS_MAX_FITS"] = "10000"
            retry = _run_pipeline("run1", i + 100, plan)
            _print_run_summary(retry)
            deltas.append(_delta_row(res, retry))
            run_history.append(retry)
            res = retry

        clean = _is_clean_run(res)
        fit_ok = int(res.get("fit_count", 0)) >= 1200
        if clean and fit_ok and _run1_gate(res.get("phase_metrics", {})):
            run1_result = res
            break

    if run1_result is None:
        print("\nDelta table across attempts:")
        if deltas:
            print(pd.DataFrame(deltas).to_string(index=False))
        raise RuntimeError("Run 1 gate not met (corr>=0.58 and mean_rmse<=35.8) with clean, non-starved execution.")

    run2_plans: List[Dict[str, str]] = [
        {
            "ALGOSPORTS_POST_MODULES": "expand_affine_global,expand_regime_affine_v2,expand_regime_affine_v2_heterosk",
            "ALGOSPORTS_SCAN_RETURN_TOP": "7",
            "ALGOSPORTS_SCAN_PRUNE_TOPN": "70",
            "ALGOSPORTS_SCAN_PRUNE_FRAC": "0.62",
            "ALGOSPORTS_HISTGB_BAG_N_MODELS": "3",
            "ALGOSPORTS_HISTGB_GRID_LEVEL": "2",
            "ALGOSPORTS_FEATURE_PROFILES": "compact_recency,full_recency",
            "ALGOSPORTS_HALF_LIFE_GRID": "15,30,45",
            "ALGOSPORTS_RIDGE_ALPHA_GRID": "0.5,1,2,4",
            "ALGOSPORTS_ELO_VARIANTS": "elo_base_static,elo_dynamic_k,elo_dynamic_k_teamha",
            "ALGOSPORTS_ENABLE_INTERACTION_FEATURES": "1",
            "ALGOSPORTS_HUBER_EPSILON": "1.10",
        },
        {
            "ALGOSPORTS_POST_MODULES": "expand_affine_global,expand_regime_affine_v2,expand_regime_affine_v2_heterosk",
            "ALGOSPORTS_SCAN_RETURN_TOP": "8",
            "ALGOSPORTS_SCAN_PRUNE_TOPN": "85",
            "ALGOSPORTS_SCAN_PRUNE_FRAC": "0.70",
            "ALGOSPORTS_HISTGB_BAG_N_MODELS": "3",
            "ALGOSPORTS_HISTGB_GRID_LEVEL": "3",
            "ALGOSPORTS_FEATURE_PROFILES": "full_recency",
            "ALGOSPORTS_HALF_LIFE_GRID": "10,20,30,45",
            "ALGOSPORTS_RIDGE_ALPHA_GRID": "0.25,0.5,1,2,4",
            "ALGOSPORTS_ELO_VARIANTS": "elo_base_static,elo_dynamic_k,elo_dynamic_k_teamha,elo_dynamic_k_teamha_decay",
            "ALGOSPORTS_ENABLE_INTERACTION_FEATURES": "1",
            "ALGOSPORTS_HUBER_EPSILON": "1.05",
            "ALGOSPORTS_ETA_GRID": "0.55,0.60,0.65,0.70,0.75,0.80,0.85",
        },
        {
            "ALGOSPORTS_POST_MODULES": "expand_affine_global,expand_regime_affine_v2,expand_regime_affine_v2_heterosk",
            "ALGOSPORTS_SCAN_RETURN_TOP": "9",
            "ALGOSPORTS_SCAN_PRUNE_TOPN": "95",
            "ALGOSPORTS_SCAN_PRUNE_FRAC": "0.76",
            "ALGOSPORTS_HISTGB_BAG_N_MODELS": "4",
            "ALGOSPORTS_HISTGB_GRID_LEVEL": "3",
            "ALGOSPORTS_FEATURE_PROFILES": "compact_recency,full_recency",
            "ALGOSPORTS_HALF_LIFE_GRID": "10,15,20,30,45",
            "ALGOSPORTS_RIDGE_ALPHA_GRID": "0.25,0.5,1,2,4",
            "ALGOSPORTS_ELO_VARIANTS": "elo_base_static,elo_dynamic_k,elo_dynamic_k_teamha,elo_dynamic_k_teamha_decay",
            "ALGOSPORTS_ENABLE_INTERACTION_FEATURES": "1",
            "ALGOSPORTS_HUBER_EPSILON": "1.0",
            "ALGOSPORTS_HUBER_ALPHA": "0.0003",
            "ALGOSPORTS_ETA_GRID": "0.55,0.60,0.65,0.70,0.75,0.80,0.85",
        },
    ]

    run2_result: Dict[str, Any] | None = None
    for i, plan in enumerate(run2_plans, start=1):
        res = _run_pipeline("run2", i, plan)
        _print_run_summary(res)
        deltas.append(_delta_row(run_history[-1], res))
        run_history.append(res)

        if "max_fits" in set(res.get("budget_triggered", [])) or (
            int(res.get("returncode", 1)) != 0 and "Fit budget exhausted" in (res.get("stderr_tail", "") + "\n" + res.get("stdout_tail", ""))
        ):
            plan = dict(plan)
            plan["ALGOSPORTS_MAX_MODEL_FITS"] = "10000"
            plan["ALGOSPORTS_MAX_FITS"] = "10000"
            retry = _run_pipeline("run2", i + 100, plan)
            _print_run_summary(retry)
            deltas.append(_delta_row(res, retry))
            run_history.append(retry)
            res = retry

        clean = _is_clean_run(res)
        fit_ok = int(res.get("fit_count", 0)) >= 1200
        diversity_ok = not _top3_all_global(ROOT)
        gates_ok = _hard_gates(res.get("phase_metrics", {}))
        if clean and fit_ok and diversity_ok and gates_ok:
            run2_result = res
            break

    print("\nDelta table across attempts:")
    if deltas:
        print(pd.DataFrame(deltas).to_string(index=False))
    else:
        print("(empty)")

    if run2_result is None:
        raise RuntimeError("Run 2 hard gates not met with clean non-starved execution.")

    _archive_prior_reports(ROOT, model_num, prior_reports)
    _ensure_submission_rotation(ROOT, model_num)
    _print_evidence(run1_result, run2_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
