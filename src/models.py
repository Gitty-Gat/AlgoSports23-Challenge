from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import build_fold_feature_tables, make_inner_expanding_splits


SEED = 23
F1_FEATURES = ["massey_diff", "games_played_diff", "volatility_diff"]
F2_FEATURES = ["massey_diff", "elok_diff", "games_played_diff", "volatility_diff"]
UNDERDISP_THRESHOLD = 0.90
ETA_GRID = [0.55, 0.60, 0.65, 0.70]
FOLD_WEIGHTS = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.25, 5: 0.30}


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    params: Dict[str, float]

    @property
    def key(self) -> str:
        bits = [self.model_name]
        for k in sorted(self.params):
            bits.append(f"{k}={self.params[k]}")
        return "|".join(bits)


@dataclass(frozen=True)
class RatingSpec:
    massey_alpha: float
    use_elo: bool
    elo_home_adv: float
    elo_k: float
    elo_decay_a: float
    elo_decay_g: float

    @property
    def key(self) -> str:
        return (
            f"massey_alpha={self.massey_alpha}|use_elo={int(self.use_elo)}|"
            f"elo_home_adv={self.elo_home_adv}|elo_k={self.elo_k}|"
            f"elo_decay_a={self.elo_decay_a}|elo_decay_g={self.elo_decay_g}"
        )


def feature_cols_for_rating_spec(rating_spec: RatingSpec) -> tuple[str, List[str]]:
    return ("F2", list(F2_FEATURES)) if rating_spec.use_elo else ("F1", list(F1_FEATURES))


def generate_model_specs() -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for alpha in [20.0, 50.0, 100.0]:
        specs.append(ModelSpec("ridge", {"alpha": float(alpha)}))
    for eps in [1.2, 1.35]:
        specs.append(ModelSpec("huber", {"epsilon": float(eps)}))
    for l1_ratio in [0.08, 0.10, 0.12]:
        for alpha in [0.7, 1.0, 1.5]:
            specs.append(ModelSpec("elasticnet", {"l1_ratio": float(l1_ratio), "alpha": float(alpha)}))
    for lr in [0.05]:
        specs.append(ModelSpec("histgb", {"max_depth": 3.0, "learning_rate": float(lr), "max_iter": 200.0}))
    return specs


def generate_rating_specs() -> List[RatingSpec]:
    out: List[RatingSpec] = []
    for massey_alpha in [1.0, 10.0, 50.0]:
        out.append(
            RatingSpec(
                massey_alpha=float(massey_alpha),
                use_elo=False,
                elo_home_adv=0.0,
                elo_k=24.0,
                elo_decay_a=0.0,
                elo_decay_g=100.0,
            )
        )
        for A in [0.0, 0.25]:
            for G in [50.0, 100.0]:
                for home_adv in [40.0, 60.0]:
                    out.append(
                        RatingSpec(
                            massey_alpha=float(massey_alpha),
                            use_elo=True,
                            elo_home_adv=float(home_adv),
                            elo_k=24.0,
                            elo_decay_a=float(A),
                            elo_decay_g=float(G),
                        )
                    )
    return out


def make_estimator(spec: ModelSpec, seed: int = SEED):
    name = spec.model_name
    p = spec.params
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=float(p["alpha"]), random_state=seed))])
    if name == "huber":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(alpha=0.0001, epsilon=float(p["epsilon"]), max_iter=2000)),
            ]
        )
    if name == "elasticnet":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=float(p["alpha"]),
                        l1_ratio=float(p["l1_ratio"]),
                        max_iter=30000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "histgb":
        return HistGradientBoostingRegressor(
            random_state=seed,
            max_depth=int(p["max_depth"]),
            learning_rate=float(p["learning_rate"]),
            max_iter=int(p["max_iter"]),
            loss="squared_error",
        )
    raise ValueError(f"Unknown model: {name}")


def _get_xy(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str = "HomeWinMargin") -> tuple[pd.DataFrame, np.ndarray]:
    X = df.reindex(columns=list(feature_cols), fill_value=0.0).astype(float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def _tail_masks(y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    q80 = float(np.quantile(np.abs(y), 0.80))
    q90 = float(np.quantile(np.abs(y), 0.90))
    return np.abs(y) >= q80, np.abs(y) >= q90


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    err = p - y
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))

    tail_mask, xtail_mask = _tail_masks(y)
    tail_rmse = float(np.sqrt(np.mean((y[tail_mask] - p[tail_mask]) ** 2))) if np.any(tail_mask) else rmse
    xtail_rmse = float(np.sqrt(np.mean((y[xtail_mask] - p[xtail_mask]) ** 2))) if np.any(xtail_mask) else tail_rmse

    y_sd = float(np.std(y, ddof=0))
    p_sd = float(np.std(p, ddof=0))
    disp_ratio = float(p_sd / y_sd) if y_sd > 1e-12 else 1.0

    y_tail_sd = float(np.std(y[tail_mask], ddof=0)) if np.any(tail_mask) else y_sd
    p_tail_sd = float(np.std(p[tail_mask], ddof=0)) if np.any(tail_mask) else p_sd
    tail_disp_ratio = float(p_tail_sd / y_tail_sd) if y_tail_sd > 1e-12 else 1.0

    tail_bias = float(np.mean(err[tail_mask])) if np.any(tail_mask) else float(np.mean(err))

    return {
        "rmse": rmse,
        "mae": mae,
        "tail_rmse": tail_rmse,
        "xtail_rmse": xtail_rmse,
        "dispersion_ratio": disp_ratio,
        "tail_dispersion_ratio": tail_disp_ratio,
        "bias": float(np.mean(err)),
        "tail_bias": tail_bias,
    }


def _inner_tail_score(y: np.ndarray, pred: np.ndarray) -> float:
    met = regression_metrics(y, pred)
    return (
        float(met["tail_rmse"])
        + 0.25 * float(met["xtail_rmse"])
        + 0.10 * float(met["rmse"])
        + 0.10 * abs(float(np.log(max(met["tail_dispersion_ratio"], 1e-9))))
    )


def _fit_affine_core(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    var_p = float(np.var(p, ddof=0))
    cov_py = float(np.cov(p, y, ddof=0)[0, 1]) if len(p) > 1 else 0.0
    p_sd = float(np.std(p, ddof=0))
    y_sd = float(np.std(y, ddof=0))
    naive_r = float(y_sd / max(p_sd, 1e-9))
    r_base = float(p_sd / y_sd) if y_sd > 1e-12 else 1.0

    fallback = False
    if not np.isfinite(cov_py) or var_p <= 1e-9:
        s_star = float(naive_r)
        fallback = True
    else:
        s_star = float(cov_py / var_p)
    if not np.isfinite(s_star):
        s_star = float(naive_r)
        fallback = True
    a_star = float(np.mean(y) - s_star * np.mean(p))
    if not np.isfinite(a_star):
        a_star = 0.0
        fallback = True

    return {
        "s_star": float(s_star),
        "a_star": float(a_star),
        "naive_r": float(naive_r),
        "r_base": float(r_base),
        "underdispersed": bool(r_base < UNDERDISP_THRESHOLD),
        "fallback": bool(fallback),
    }


def _apply_affine(p: np.ndarray, a: float, s: float) -> np.ndarray:
    return float(a) + float(s) * np.asarray(p, dtype=float)


def _fit_expand_affine(y: np.ndarray, p: np.ndarray, eta_grid: Sequence[float]) -> Dict[str, float]:
    core = _fit_affine_core(y, p)
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    n_oof = int(len(y_arr))
    if n_oof < 120:
        s_cap = 1.25
    elif n_oof < 200:
        s_cap = 1.35
    elif n_oof < 300:
        s_cap = 1.45
    else:
        s_cap = 1.60

    if core["underdispersed"]:
        best = None
        for eta in eta_grid:
            s = max(1.0, float(eta) * float(core["naive_r"]))
            a = float(np.mean(y_arr) - s * np.mean(p_arr))
            pred = _apply_affine(p_arr, a, s)
            obj = _inner_tail_score(y_arr, pred)
            cand = {"eta": float(eta), "s": float(s), "a": float(a), "obj": float(obj)}
            if best is None or cand["obj"] < best["obj"]:
                best = cand
        s_uncapped = float(best["s"])
        s = float(min(s_uncapped, s_cap))
        a = float(np.mean(y_arr) - s * np.mean(p_arr))
        eta = float(best["eta"])
    else:
        s_uncapped = float(np.clip(core["s_star"], 0.8, 1.2))
        s = float(min(s_uncapped, s_cap))
        a = float(np.mean(y_arr) - s * np.mean(p_arr))
        eta = None

    return {
        "a": float(a),
        "s": float(s),
        "s_uncapped": float(s_uncapped),
        "s_cap": float(s_cap),
        "s_cap_hit": bool(s_uncapped > s + 1e-12),
        "n_oof": int(n_oof),
        "eta": eta,
        "s_star": float(core["s_star"]),
        "a_star": float(core["a_star"]),
        "naive_r": float(core["naive_r"]),
        "r_base": float(core["r_base"]),
        "underdispersed": bool(core["underdispersed"]),
        "fallback": bool(core["fallback"]),
    }


def _make_regime_labels(abs_d: np.ndarray, info_key: np.ndarray, q1: float, q2: float, info_med: float) -> np.ndarray:
    mismatch = np.where(abs_d <= q1, "close", np.where(abs_d <= q2, "mid", "blowout"))
    info = np.where(info_key <= info_med, "low", "high")
    return np.char.add(np.char.add(mismatch.astype(str), "|"), info.astype(str)).astype(object)


def _apply_regime_predictions(p: np.ndarray, regime_ids: np.ndarray, reg_map: Dict[str, dict], fallback_a: float, fallback_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    pred = np.zeros(len(p), dtype=float)
    s_used = np.zeros(len(p), dtype=float)
    eta_used = np.full(len(p), np.nan, dtype=float)
    for i, rid in enumerate(regime_ids.tolist()):
        rp = reg_map.get(str(rid))
        if rp is None:
            s_val = float(fallback_s)
            a_val = float(fallback_a)
            eta_val = np.nan
        else:
            s_val = float(rp["s"])
            a_val = float(rp["a"])
            eta_val = float(rp["eta"]) if rp.get("eta") is not None else np.nan
        pred[i] = a_val + s_val * p[i]
        s_used[i] = s_val
        eta_used[i] = eta_val
    return pred, s_used, eta_used


def fit_postprocessor(module_name: str, oof_df: pd.DataFrame) -> dict:
    y = oof_df["y_true"].to_numpy(dtype=float)
    p = oof_df["y_pred_base"].to_numpy(dtype=float)
    d = oof_df["d_key"].to_numpy(dtype=float)
    info_key = oof_df["gp_min"].to_numpy(dtype=float)
    abs_d = np.abs(d)

    eta_grid = list(ETA_GRID)
    global_aff = _fit_expand_affine(y, p, eta_grid=eta_grid)
    p_global = _apply_affine(p, global_aff["a"], global_aff["s"])

    if module_name == "none":
        invalid = bool(global_aff["underdispersed"])
        return {
            "module": "none",
            "affine": global_aff,
            "params": {},
            "invalid": invalid,
            "invalid_reason": "module_none_for_underdispersed_base" if invalid else "",
        }

    if module_name == "expand_affine_global":
        invalid = bool(global_aff["underdispersed"] and global_aff["s"] <= 1.0)
        return {
            "module": "expand_affine_global",
            "affine": global_aff,
            "params": {},
            "invalid": invalid,
            "invalid_reason": "underdispersed_but_no_expansion" if invalid else "",
        }

    if module_name in {"expand_regime_affine_v2", "expand_regime_affine_v2_heterosk"}:
        q1 = float(np.quantile(abs_d, 1.0 / 3.0))
        q2 = float(np.quantile(abs_d, 2.0 / 3.0))
        info_med = float(np.quantile(info_key, 0.5))
        regime_ids = _make_regime_labels(abs_d, info_key, q1=q1, q2=q2, info_med=info_med)

        reg_map: Dict[str, dict] = {}
        for rid in sorted(np.unique(regime_ids).tolist()):
            mask = regime_ids == rid
            n_r = int(np.sum(mask))
            if n_r < 8:
                reg_map[str(rid)] = {
                    "a": float(global_aff["a"]),
                    "s": float(global_aff["s"]),
                    "eta": global_aff["eta"],
                    "n": n_r,
                    "r_base": float(global_aff["r_base"]),
                }
                continue
            local = _fit_expand_affine(y[mask], p[mask], eta_grid=eta_grid)
            w = float(n_r / (n_r + 40.0))
            s_r = w * float(local["s"]) + (1.0 - w) * float(global_aff["s"])
            if local["underdispersed"]:
                s_r = max(1.0, s_r)
            else:
                s_r = float(np.clip(s_r, 0.8, 1.2))
            a_r = float(np.mean(y[mask]) - s_r * np.mean(p[mask]))
            reg_map[str(rid)] = {
                "a": float(a_r),
                "s": float(s_r),
                "eta": local["eta"],
                "n": n_r,
                "r_base": float(local["r_base"]),
            }

        pred_reg, s_used_reg, eta_used_reg = _apply_regime_predictions(
            p,
            regime_ids,
            reg_map,
            fallback_a=float(global_aff["a"]),
            fallback_s=float(global_aff["s"]),
        )

        invalid = bool(global_aff["underdispersed"] and global_aff["r_base"] < 0.7)
        reason = ""

        blow_mask = np.array([str(x).startswith("blowout|") for x in regime_ids], dtype=bool)
        if np.any(blow_mask):
            yb = y[blow_mask]
            pb = p[blow_mask]
            pr = pred_reg[blow_mask]
            met_b = regression_metrics(yb, pb)
            met_r = regression_metrics(yb, pr)
            tail_improve = float(met_b["tail_rmse"] - met_r["tail_rmse"])
            if float(met_r["tail_dispersion_ratio"]) < 0.85 and tail_improve < 2.0:
                invalid = True
                reason = "blowout_taildisp_gate"
            if float(np.std(pb, ddof=0) / max(np.std(yb, ddof=0), 1e-9)) < 0.7:
                blow_s = [float(reg_map[k]["s"]) for k in reg_map if k.startswith("blowout|")]
                if len(blow_s) > 0 and float(np.max(blow_s)) <= 1.0:
                    invalid = True
                    reason = "blowout_underdisp_no_expansion"

        out = {
            "module": "expand_regime_affine_v2",
            "affine": global_aff,
            "params": {
                "q_low": q1,
                "q_high": q2,
                "info_median": info_med,
                "regimes": reg_map,
            },
            "invalid": bool(invalid),
            "invalid_reason": reason,
        }

        if module_name == "expand_regime_affine_v2_heterosk":
            base_obj = _inner_tail_score(y, pred_reg)
            best = None
            a0 = float(np.mean(y))
            for gamma in [0.01, 0.02, 0.03, 0.05]:
                scale = 1.0 + float(gamma) * np.log1p(abs_d)
                pred_h = a0 + (pred_reg - a0) * scale
                obj = _inner_tail_score(y, pred_h)
                cand = {
                    "gamma": float(gamma),
                    "obj": float(obj),
                }
                if best is None or cand["obj"] < best["obj"]:
                    best = cand
            out["module"] = "expand_regime_affine_v2_heterosk"
            if best is None or float(best["obj"]) >= float(base_obj) - 1e-12:
                out["params"]["gamma"] = 0.0
                out["invalid"] = True
                out["invalid_reason"] = "heterosk_nonzero_no_gain"
            else:
                out["params"]["gamma"] = float(best["gamma"])
        return out

    if module_name == "expand_piecewise_tail_v2":
        best = None
        for q in [0.70, 0.80, 0.90]:
            t = float(np.quantile(abs_d, q))
            for k in [0.10, 0.25, 0.50, 0.75, 1.00]:
                pred = p_global + float(k) * np.maximum(abs_d - t, 0.0) * np.sign(d)
                obj = _inner_tail_score(y, pred)
                cand = {"q": float(q), "t": float(t), "k": float(k), "obj": float(obj)}
                if best is None or cand["obj"] < best["obj"]:
                    best = cand

        invalid = bool(global_aff["underdispersed"] and global_aff["s"] <= 1.0)
        return {
            "module": "expand_piecewise_tail_v2",
            "affine": global_aff,
            "params": {
                "tail_q": float(best["q"]),
                "tail_t": float(best["t"]),
                "tail_k": float(best["k"]),
            },
            "invalid": invalid,
            "invalid_reason": "underdispersed_but_no_expansion" if invalid else "",
        }

    raise ValueError(f"Unknown module: {module_name}")


def apply_postprocessor(fit_obj: dict, y_pred_base: np.ndarray, d_key: np.ndarray, gp_min: np.ndarray) -> dict:
    module = str(fit_obj["module"])
    p = np.asarray(y_pred_base, dtype=float)
    d = np.asarray(d_key, dtype=float)
    info_key = np.asarray(gp_min, dtype=float)
    mismatch_key = np.abs(d)

    regime_id = np.full(len(p), "global", dtype=object)
    s_used = np.full(len(p), np.nan, dtype=float)
    k_used = np.zeros(len(p), dtype=float)
    t_used = np.full(len(p), np.nan, dtype=float)
    eta_used = np.full(len(p), np.nan, dtype=float)

    if module == "none":
        return {
            "y_pred_final": p.copy(),
            "regime_id": np.full(len(p), "none", dtype=object),
            "mismatch_key": mismatch_key,
            "info_key": info_key,
            "postprocess_s_used": np.ones(len(p), dtype=float),
            "postprocess_t_used": t_used,
            "postprocess_k_used": k_used,
            "postprocess_eta_used": eta_used,
        }

    aff = fit_obj["affine"]
    p_global = _apply_affine(p, float(aff["a"]), float(aff["s"]))

    if module == "expand_affine_global":
        s_used[:] = float(aff["s"])
        eta_used[:] = float(aff["eta"]) if aff.get("eta") is not None else np.nan
        return {
            "y_pred_final": p_global,
            "regime_id": regime_id,
            "mismatch_key": mismatch_key,
            "info_key": info_key,
            "postprocess_s_used": s_used,
            "postprocess_t_used": t_used,
            "postprocess_k_used": k_used,
            "postprocess_eta_used": eta_used,
        }

    if module in {"expand_regime_affine_v2", "expand_regime_affine_v2_heterosk"}:
        par = fit_obj["params"]
        q1 = float(par["q_low"])
        q2 = float(par["q_high"])
        info_med = float(par["info_median"])
        regime_id = _make_regime_labels(mismatch_key, info_key, q1=q1, q2=q2, info_med=info_med)
        reg_map = par["regimes"]
        pred_reg, s_used, eta_used = _apply_regime_predictions(
            p,
            regime_id,
            reg_map,
            fallback_a=float(aff["a"]),
            fallback_s=float(aff["s"]),
        )

        if module == "expand_regime_affine_v2_heterosk":
            gamma = float(par.get("gamma", 0.0))
            scale = 1.0 + gamma * np.log1p(mismatch_key)
            a0 = float(np.mean(pred_reg))
            pred_reg = a0 + (pred_reg - a0) * scale
            s_used = s_used * scale

        return {
            "y_pred_final": pred_reg,
            "regime_id": regime_id,
            "mismatch_key": mismatch_key,
            "info_key": info_key,
            "postprocess_s_used": s_used,
            "postprocess_t_used": t_used,
            "postprocess_k_used": k_used,
            "postprocess_eta_used": eta_used,
        }

    if module == "expand_piecewise_tail_v2":
        par = fit_obj["params"]
        t = float(par["tail_t"])
        k = float(par["tail_k"])
        tail_amt = np.maximum(mismatch_key - t, 0.0)
        pred = p_global + k * tail_amt * np.sign(d)
        s_used[:] = float(aff["s"])
        t_used[:] = float(t)
        k_used[:] = float(k)
        k_used[tail_amt <= 0] = 0.0
        eta_used[:] = float(aff["eta"]) if aff.get("eta") is not None else np.nan
        return {
            "y_pred_final": pred,
            "regime_id": np.where(tail_amt > 0, "tail", "base").astype(object),
            "mismatch_key": mismatch_key,
            "info_key": info_key,
            "postprocess_s_used": s_used,
            "postprocess_t_used": t_used,
            "postprocess_k_used": k_used,
            "postprocess_eta_used": eta_used,
        }

    raise ValueError(f"Unknown module: {module}")


def model_is_nonlinear(model_name: str) -> bool:
    return model_name in {"histgb"}


def simplicity_rank(use_elo: bool, model_name: str, module_name: str) -> int:
    if module_name == "expand_regime_affine_v2" and not use_elo and not model_is_nonlinear(model_name):
        return 0
    if module_name == "expand_regime_affine_v2" and use_elo and not model_is_nonlinear(model_name):
        return 1
    if module_name == "expand_regime_affine_v2_heterosk" and not model_is_nonlinear(model_name):
        return 2
    if module_name == "expand_piecewise_tail_v2" and not model_is_nonlinear(model_name):
        return 3
    if model_is_nonlinear(model_name):
        return 4
    if module_name == "expand_affine_global":
        return 5
    return 99


def _shrink_expansion_step(x: float) -> float:
    return 1.0 + 0.7 * (float(x) - 1.0)


def _shrink_postprocessor_for_cv_guard(fit_obj: dict) -> dict:
    out = deepcopy(fit_obj)
    module = str(out.get("module", ""))
    aff = out.get("affine")
    if isinstance(aff, dict) and "s" in aff:
        aff["s"] = float(_shrink_expansion_step(float(aff["s"])))

    params = out.get("params", {})
    if module in {"expand_regime_affine_v2", "expand_regime_affine_v2_heterosk"}:
        regimes = params.get("regimes", {})
        if isinstance(regimes, dict):
            for rid, rv in regimes.items():
                if isinstance(rv, dict) and "s" in rv:
                    rv["s"] = float(_shrink_expansion_step(float(rv["s"])))
                    regimes[rid] = rv
        if module == "expand_regime_affine_v2_heterosk" and "gamma" in params:
            params["gamma"] = float(0.7 * float(params["gamma"]))
    if module == "expand_piecewise_tail_v2" and "tail_k" in params:
        params["tail_k"] = float(0.7 * float(params["tail_k"]))
    out["params"] = params
    return out


def apply_postprocessor_with_cv_guard(
    fit_obj: dict,
    y_pred_base: np.ndarray,
    d_key: np.ndarray,
    gp_min: np.ndarray,
    y_true: np.ndarray,
    max_disp_ratio: float | None = None,
) -> dict:
    if max_disp_ratio is None:
        n_oof = int(fit_obj.get("affine", {}).get("n_oof", 0))
        max_disp_ratio = 1.15 if n_oof >= 300 else 1.10

    y = np.asarray(y_true, dtype=float)
    out0 = apply_postprocessor(fit_obj, y_pred_base=y_pred_base, d_key=d_key, gp_min=gp_min)
    p0 = np.asarray(out0["y_pred_final"], dtype=float)
    y_sd = float(np.std(y, ddof=0))
    p0_sd = float(np.std(p0, ddof=0))
    disp0 = float(p0_sd / max(y_sd, 1e-9))

    if disp0 <= float(max_disp_ratio):
        out0["cv_dispersion_ratio_initial"] = float(disp0)
        out0["cv_dispersion_ratio"] = float(disp0)
        out0["cv_dispersion_guard_applied"] = False
        out0["cv_dispersion_guard_reject"] = False
        return out0

    shrunk_fit = _shrink_postprocessor_for_cv_guard(fit_obj)
    out1 = apply_postprocessor(shrunk_fit, y_pred_base=y_pred_base, d_key=d_key, gp_min=gp_min)
    p1 = np.asarray(out1["y_pred_final"], dtype=float)
    p1_sd = float(np.std(p1, ddof=0))
    disp1 = float(p1_sd / max(y_sd, 1e-9))

    out1["cv_dispersion_ratio_initial"] = float(disp0)
    out1["cv_dispersion_ratio"] = float(disp1)
    out1["cv_dispersion_guard_applied"] = True
    out1["cv_dispersion_guard_reject"] = bool(disp1 > float(max_disp_ratio))
    return out1


def build_inner_oof(
    outer_train_rows: pd.DataFrame,
    team_ids: Sequence[int],
    rating_spec: RatingSpec,
    model_spec: ModelSpec,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    splits = make_inner_expanding_splits(outer_train_rows, n_splits=3)
    out_rows: List[dict] = []

    for split in splits:
        inner_train = outer_train_rows.iloc[np.asarray(split["train_idx"], dtype=int)].copy()
        inner_val = outer_train_rows.iloc[np.asarray(split["val_idx"], dtype=int)].copy()

        art = build_fold_feature_tables(
            inner_train,
            inner_val,
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
        p_va = np.asarray(est.predict(X_va), dtype=float)

        gp_min = np.minimum(
            art.val_table["games_played_home"].to_numpy(dtype=float),
            art.val_table["games_played_away"].to_numpy(dtype=float),
        )
        for idx, y_i, p_i, d_i, gp_i in zip(
            inner_val.index.tolist(),
            y_va.tolist(),
            p_va.tolist(),
            art.val_table["d_key"].tolist(),
            gp_min.tolist(),
        ):
            out_rows.append(
                {
                    "row_index": int(idx),
                    "y_true": float(y_i),
                    "y_pred_base": float(p_i),
                    "d_key": float(d_i),
                    "gp_min": float(gp_i),
                }
            )

    oof = pd.DataFrame(out_rows)
    if oof.empty:
        raise RuntimeError("Inner OOF could not be generated")
    oof = oof.sort_values(["row_index"], kind="mergesort").drop_duplicates("row_index", keep="last").reset_index(drop=True)
    return oof


def aggregate_config_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    agg_dict = {
        "mean_rmse": ("rmse", "mean"),
        "std_rmse": ("rmse", "std"),
        "mean_mae": ("mae", "mean"),
        "mean_tail_rmse": ("tail_rmse", "mean"),
        "mean_xtail_rmse": ("xtail_rmse", "mean"),
        "mean_dispersion_ratio": ("dispersion_ratio", "mean"),
        "mean_tail_dispersion_ratio": ("tail_dispersion_ratio", "mean"),
        "mean_bias": ("bias", "mean"),
        "mean_tail_bias": ("tail_bias", "mean"),
        "n_folds": ("fold", "count"),
        "simplicity_rank": ("simplicity_rank", "max"),
    }
    if "invalid_postprocess" in fold_metrics.columns:
        agg_dict["invalid_rate"] = ("invalid_postprocess", "mean")

    summary = (
        fold_metrics.groupby("config_id", as_index=False)
        .agg(**agg_dict)
        .sort_values(["mean_rmse", "mean_tail_rmse", "std_rmse"], kind="mergesort")
        .reset_index(drop=True)
    )
    weighted_rows: List[dict] = []
    for cfg, grp in fold_metrics.groupby("config_id"):
        g = grp.copy()
        w = g["fold"].map(FOLD_WEIGHTS).fillna(0.0).to_numpy(dtype=float)
        if float(np.sum(w)) <= 0:
            w = np.ones(len(g), dtype=float)
        w = w / float(np.sum(w))

        def _wmean(col: str) -> float:
            if col not in g.columns:
                return float("nan")
            vals = g[col].to_numpy(dtype=float)
            return float(np.sum(w * vals))

        weighted_rows.append(
            {
                "config_id": str(cfg),
                "mean_rmse_w": _wmean("rmse"),
                "mean_mae_w": _wmean("mae"),
                "mean_tail_rmse_w": _wmean("tail_rmse"),
                "mean_xtail_rmse_w": _wmean("xtail_rmse"),
                "mean_dispersion_ratio_w": _wmean("dispersion_ratio"),
                "mean_tail_dispersion_ratio_w": _wmean("tail_dispersion_ratio"),
                "mean_bias_w": _wmean("bias"),
                "mean_tail_bias_w": _wmean("tail_bias"),
                "s_cap_hit_rate": _wmean("s_cap_hit"),
            }
        )
    summary = summary.merge(pd.DataFrame(weighted_rows), on="config_id", how="left")
    summary = summary.merge(
        fold_metrics.groupby("config_id", as_index=False).agg(max_fold_rmse=("rmse", "max")),
        on="config_id",
        how="left",
    )
    fold1 = fold_metrics[fold_metrics["fold"] == 1].groupby("config_id", as_index=False).agg(fold1_rmse=("rmse", "mean"))
    summary = summary.merge(fold1, on="config_id", how="left")
    if "invalid_rate" not in summary.columns:
        summary["invalid_rate"] = 0.0
    return summary


def _apply_selection_gates(work: pd.DataFrame) -> pd.DataFrame:
    w = work.copy()
    w["rejected"] = False
    w["reject_reason"] = ""
    rmse_col = "mean_rmse_w" if "mean_rmse_w" in w.columns else "mean_rmse"
    tail_rmse_col = "mean_tail_rmse_w" if "mean_tail_rmse_w" in w.columns else "mean_tail_rmse"
    disp_col = "mean_dispersion_ratio_w" if "mean_dispersion_ratio_w" in w.columns else "mean_dispersion_ratio"
    tail_disp_col = "mean_tail_dispersion_ratio_w" if "mean_tail_dispersion_ratio_w" in w.columns else "mean_tail_dispersion_ratio"

    def _mark(mask: pd.Series, reason: str) -> None:
        m = mask.fillna(False)
        if not bool(np.any(m.to_numpy(dtype=bool))):
            return
        prev = w.loc[m, "reject_reason"].astype(str)
        w.loc[m, "reject_reason"] = np.where(prev == "", reason, prev + ";" + reason)
        w.loc[m, "rejected"] = True

    _mark(w["invalid_rate"] > 0.0, "invalid_postprocess")

    if "max_fold_rmse" in w.columns:
        feasible = w[w["max_fold_rmse"] <= 55.0]
        if not feasible.empty:
            best_rmse = float(feasible[rmse_col].min())
            _mark((w["max_fold_rmse"] > 55.0) & ((best_rmse - w[rmse_col]) < 1.0), "max_fold_rmse_gate")

    if disp_col in w.columns:
        feasible = w[w[disp_col] <= 1.05]
        if not feasible.empty:
            best_rmse = float(feasible[rmse_col].min())
            _mark((w[disp_col] > 1.05) & ((best_rmse - w[rmse_col]) < 0.5), "dispersion_upper_gate")

    if "fold1_rmse" in w.columns:
        feasible = w[w["fold1_rmse"] <= 55.0]
        if not feasible.empty:
            best_rmse = float(feasible[rmse_col].min())
            _mark((w["fold1_rmse"] > 55.0) & ((best_rmse - w[rmse_col]) < 1.0), "fold1_rmse_gate")

    if tail_disp_col in w.columns:
        feasible_tail = w[w[tail_disp_col] >= 0.75]
        if not feasible_tail.empty:
            best_tail_rmse = float(feasible_tail[tail_rmse_col].min())
            _mark(
                (w[tail_disp_col] < 0.75) & ((best_tail_rmse - w[tail_rmse_col]) < 2.0),
                "tail_dispersion_gate",
            )

    if "s_cap_hit_rate" in w.columns:
        feasible = w[w["s_cap_hit_rate"] <= 0.40]
        if not feasible.empty:
            best_rmse = float(feasible[rmse_col].min())
            _mark((w["s_cap_hit_rate"] > 0.40) & ((best_rmse - w[rmse_col]) < 0.40), "cap_hit_gate")

    if "mean_affine_r_base" in w.columns:
        if "mean_blowout_s" in w.columns:
            blow_s = w["mean_blowout_s"].copy()
            if "mean_affine_s" in w.columns:
                blow_s = blow_s.fillna(w["mean_affine_s"])
            _mark((w["mean_affine_r_base"] < 0.70) & (blow_s <= 1.0), "underdisp_no_blowout_expansion_gate")

    if "module" in w.columns and disp_col in w.columns and "max_fold_rmse" in w.columns:
        piece_mask = w["module"].astype(str).eq("expand_piecewise_tail_v2")
        piece_bad = piece_mask & ((w["max_fold_rmse"] > 45.0) | (w[disp_col] < 0.75) | (w[disp_col] > 1.00))
        _mark(piece_bad, "piecewise_eligibility_gate")

    return w


def select_with_1se_and_stability(
    summary: pd.DataFrame,
    lambda_tail_values: Sequence[float] = (0.10, 0.20),
    lambda_xtail_values: Sequence[float] = (0.05, 0.10),
    lambda_disp_values: Sequence[float] = (0.05, 0.10, 0.20),
    lambda_cap_hit_values: Sequence[float] = (0.0, 0.05, 0.10),
    lambda_tbias_values: Sequence[float] = (0.0, 0.002, 0.005),
    lambda_tdisp_values: Sequence[float] = (0.05, 0.10),
    lambda_stability_values: Sequence[float] = (0.10, 0.20),
) -> dict:
    if summary.empty:
        raise RuntimeError("No configs to select from")

    s = summary.copy()
    s["std_rmse"] = s["std_rmse"].fillna(0.0)
    rmse_col = "mean_rmse_w" if "mean_rmse_w" in s.columns else "mean_rmse"
    tail_rmse_col = "mean_tail_rmse_w" if "mean_tail_rmse_w" in s.columns else "mean_tail_rmse"
    xtail_rmse_col = "mean_xtail_rmse_w" if "mean_xtail_rmse_w" in s.columns else "mean_xtail_rmse"
    disp_col = "mean_dispersion_ratio_w" if "mean_dispersion_ratio_w" in s.columns else "mean_dispersion_ratio"
    tail_disp_col = "mean_tail_dispersion_ratio_w" if "mean_tail_dispersion_ratio_w" in s.columns else "mean_tail_dispersion_ratio"
    tail_bias_col = "mean_tail_bias_w" if "mean_tail_bias_w" in s.columns else "mean_tail_bias"

    s[disp_col] = s[disp_col].clip(lower=1e-9)
    s[tail_disp_col] = s[tail_disp_col].clip(lower=1e-9)
    s["tail_over_rmse"] = s[tail_rmse_col] / np.maximum(s[rmse_col], 1e-9)
    s["xtail_over_rmse"] = s[xtail_rmse_col] / np.maximum(s[rmse_col], 1e-9)
    s["disp_pen"] = np.abs(np.log(s[disp_col]))
    s["tail_disp_pen"] = np.abs(np.log(s[tail_disp_col]))
    s["tail_bias_abs"] = np.abs(s[tail_bias_col])
    if "s_cap_hit_rate" not in s.columns:
        s["s_cap_hit_rate"] = 0.0

    picks: List[dict] = []

    for l_tail in lambda_tail_values:
        for l_xtail in lambda_xtail_values:
            for l_disp in lambda_disp_values:
                for l_cap in lambda_cap_hit_values:
                    for l_tbias in lambda_tbias_values:
                        for l_tdisp in lambda_tdisp_values:
                            for l_stab in lambda_stability_values:
                                work = s.copy()
                                work["lambda_tail"] = float(l_tail)
                                work["lambda_xtail"] = float(l_xtail)
                                work["lambda_disp"] = float(l_disp)
                                work["lambda_cap_hit"] = float(l_cap)
                                work["lambda_tbias"] = float(l_tbias)
                                work["lambda_tdisp"] = float(l_tdisp)
                                work["lambda_stability"] = float(l_stab)
                                work["score"] = (
                                    work[rmse_col]
                                    + float(l_tail) * work["tail_over_rmse"]
                                    + float(l_xtail) * work["xtail_over_rmse"]
                                    + float(l_disp) * work["disp_pen"]
                                    + float(l_cap) * work["s_cap_hit_rate"]
                                    + float(l_tbias) * work["tail_bias_abs"]
                                    + float(l_tdisp) * work["tail_disp_pen"]
                                    + float(l_stab) * work["std_rmse"]
                                )

                                gated = _apply_selection_gates(work)
                                valid = gated[~gated["rejected"]].copy()
                                if valid.empty:
                                    valid = gated.copy()
                                valid = valid.sort_values(["score", rmse_col, tail_rmse_col], kind="mergesort").reset_index(drop=True)
                                best = valid.iloc[0]
                                score_se = float(best["std_rmse"]) / max(np.sqrt(float(best["n_folds"])), 1.0)
                                elig = valid[valid["score"] <= float(best["score"]) + score_se + 1e-12].copy()
                                if elig.empty:
                                    elig = valid.copy()
                                elig = elig.sort_values(
                                    ["score", "simplicity_rank", rmse_col, tail_rmse_col, "mean_mae"],
                                    kind="mergesort",
                                ).reset_index(drop=True)

                                top = elig.iloc[0]
                                module_pref_applied = False
                                if str(top.get("module", "")) == "expand_affine_global":
                                    reg_elig = elig[elig["module"].astype(str).isin(["expand_regime_affine_v2", "expand_regime_affine_v2_heterosk"])]
                                    if not reg_elig.empty:
                                        top = reg_elig.sort_values(
                                            ["score", "simplicity_rank", rmse_col, tail_rmse_col],
                                            kind="mergesort",
                                        ).iloc[0]
                                        module_pref_applied = True

                                pick = top.to_dict()
                                pick["best_score"] = float(best["score"])
                                pick["best_score_se"] = float(score_se)
                                pick["gate_rejections_count"] = int(gated["rejected"].sum())
                                pick["module_preference_applied"] = bool(module_pref_applied)
                                picks.append(pick)

    pick_df = pd.DataFrame(picks).sort_values(
        [
            "score",
            "simplicity_rank",
            rmse_col,
            tail_rmse_col,
            "mean_xtail_rmse",
            "lambda_tail",
            "lambda_xtail",
            "lambda_disp",
            "lambda_cap_hit",
            "lambda_tbias",
            "lambda_tdisp",
            "lambda_stability",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    selected = pick_df.iloc[0].to_dict()

    # gate table for selected lambda values
    gate_table = s.copy()
    gate_table["lambda_tail"] = float(selected["lambda_tail"])
    gate_table["lambda_xtail"] = float(selected["lambda_xtail"])
    gate_table["lambda_disp"] = float(selected["lambda_disp"])
    gate_table["lambda_cap_hit"] = float(selected["lambda_cap_hit"])
    gate_table["lambda_tbias"] = float(selected["lambda_tbias"])
    gate_table["lambda_tdisp"] = float(selected["lambda_tdisp"])
    gate_table["lambda_stability"] = float(selected["lambda_stability"])
    gate_table["score"] = (
        gate_table[rmse_col]
        + gate_table["lambda_tail"] * gate_table["tail_over_rmse"]
        + gate_table["lambda_xtail"] * gate_table["xtail_over_rmse"]
        + gate_table["lambda_disp"] * gate_table["disp_pen"]
        + gate_table["lambda_cap_hit"] * gate_table["s_cap_hit_rate"]
        + gate_table["lambda_tbias"] * gate_table["tail_bias_abs"]
        + gate_table["lambda_tdisp"] * gate_table["tail_disp_pen"]
        + gate_table["lambda_stability"] * gate_table["std_rmse"]
    )
    gate_table = _apply_selection_gates(gate_table)

    return {
        "selected_config_id": str(selected["config_id"]),
        "lambda_tail": float(selected["lambda_tail"]),
        "lambda_xtail": float(selected["lambda_xtail"]),
        "lambda_disp": float(selected["lambda_disp"]),
        "lambda_cap_hit": float(selected["lambda_cap_hit"]),
        "lambda_tbias": float(selected["lambda_tbias"]),
        "lambda_tdisp": float(selected["lambda_tdisp"]),
        "lambda_stability": float(selected["lambda_stability"]),
        "best_score": float(selected["best_score"]),
        "best_score_se": float(selected["best_score_se"]),
        "module_preference_applied": bool(selected.get("module_preference_applied", False)),
        "selection_table": s,
        "lambda_picks": pick_df,
        "gate_table": gate_table,
        "gate_rejections_count": int(gate_table["rejected"].sum()),
    }
