from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.ratings import (
    EloModel,
    EloVariantConfig,
    MasseyModel,
    apply_elo_features,
    apply_massey_features,
    fit_massey_ridge,
    run_elo_over_games,
)


PRIOR_VOL_SD = 45.0
PRIOR_VOL_VAR = PRIOR_VOL_SD**2
PRIOR_VOL_N = 3.0


@dataclass
class TeamHistoryState:
    games_played: int = 0
    margin_sum: float = 0.0
    margin_sq_sum: float = 0.0


@dataclass
class FoldFeatureArtifacts:
    train_table: pd.DataFrame
    val_table: pd.DataFrame
    massey_model: MasseyModel
    elo_model: EloModel
    history_states: Dict[int, TeamHistoryState]


def parse_and_sort_train(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True)


def parse_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True)


def build_team_universe(rankings_df: pd.DataFrame) -> pd.DataFrame:
    out = rankings_df[["TeamID", "Team"]].copy()
    out["TeamID"] = out["TeamID"].astype(int)
    out["Team"] = out["Team"].astype(str)
    return out.drop_duplicates("TeamID").sort_values("TeamID", kind="mergesort").reset_index(drop=True)


def make_expanding_time_folds(train_df: pd.DataFrame, n_folds: int = 5) -> List[dict]:
    unique_dates = np.array(sorted(pd.to_datetime(train_df["Date"]).unique()))
    if len(unique_dates) < n_folds + 1:
        raise ValueError("Insufficient unique dates for expanding folds")
    buckets = np.array_split(unique_dates, n_folds + 1)
    folds: List[dict] = []
    for i in range(1, n_folds + 1):
        train_dates = np.concatenate(buckets[:i])
        val_dates = buckets[i]
        train_mask = train_df["Date"].isin(train_dates).to_numpy()
        val_mask = train_df["Date"].isin(val_dates).to_numpy()
        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_start_date": pd.Timestamp(train_df.loc[train_idx[0], "Date"]),
                "train_end_date": pd.Timestamp(train_df.loc[train_idx[-1], "Date"]),
                "val_start_date": pd.Timestamp(train_df.loc[val_idx[0], "Date"]),
                "val_end_date": pd.Timestamp(train_df.loc[val_idx[-1], "Date"]),
            }
        )
    if len(folds) != n_folds:
        raise ValueError(f"Constructed {len(folds)} folds, expected {n_folds}")
    return folds


def make_inner_expanding_splits(train_rows: pd.DataFrame, n_splits: int = 3) -> List[dict]:
    idx = np.arange(len(train_rows), dtype=int)
    unique_dates = np.array(sorted(pd.to_datetime(train_rows["Date"]).unique()))
    if len(unique_dates) < n_splits + 1:
        n_splits = max(1, len(unique_dates) - 1)
    buckets = np.array_split(unique_dates, n_splits + 1)
    splits: List[dict] = []
    for i in range(1, n_splits + 1):
        tr_dates = np.concatenate(buckets[:i])
        va_dates = buckets[i]
        tr_mask = train_rows["Date"].isin(tr_dates).to_numpy()
        va_mask = train_rows["Date"].isin(va_dates).to_numpy()
        tr_idx = idx[tr_mask]
        va_idx = idx[va_mask]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        splits.append({"split": len(splits) + 1, "train_idx": tr_idx, "val_idx": va_idx})
    if not splits:
        raise ValueError("No valid inner expanding splits")
    return splits


def _state_snapshot(state: TeamHistoryState) -> tuple[float, float]:
    gp = float(state.games_played)
    if state.games_played > 1:
        sample_var = (state.margin_sq_sum - (state.margin_sum**2) / gp) / max(gp - 1.0, 1.0)
        sample_var = max(sample_var, 0.0)
    else:
        sample_var = PRIOR_VOL_VAR
    n = max(gp - 1.0, 0.0)
    shrunk_var = (n * sample_var + PRIOR_VOL_N * PRIOR_VOL_VAR) / max(n + PRIOR_VOL_N, 1.0)
    vol = float(np.sqrt(max(shrunk_var, 1e-9)))
    return gp, vol


def _build_history_train_features(train_rows: pd.DataFrame) -> tuple[pd.DataFrame, Dict[int, TeamHistoryState]]:
    states: Dict[int, TeamHistoryState] = {}
    rows = []
    for row in train_rows.itertuples(index=False):
        home_id = int(row.HomeID)
        away_id = int(row.AwayID)
        st_h = states.get(home_id, TeamHistoryState())
        st_a = states.get(away_id, TeamHistoryState())
        gp_h, vol_h = _state_snapshot(st_h)
        gp_a, vol_a = _state_snapshot(st_a)
        rows.append(
            {
                "games_played_home": gp_h,
                "games_played_away": gp_a,
                "games_played_diff": gp_h - gp_a,
                "volatility_home": vol_h,
                "volatility_away": vol_a,
                "volatility_diff": vol_h - vol_a,
            }
        )

        margin = float(row.HomeWinMargin)
        st_h.games_played += 1
        st_h.margin_sum += margin
        st_h.margin_sq_sum += margin * margin
        st_a.games_played += 1
        st_a.margin_sum -= margin
        st_a.margin_sq_sum += margin * margin
        states[home_id] = st_h
        states[away_id] = st_a

    return pd.DataFrame(rows, index=train_rows.index), states


def _build_history_static_features(rows_df: pd.DataFrame, states: Mapping[int, TeamHistoryState], home_col: str, away_col: str) -> pd.DataFrame:
    out = []
    for row in rows_df.itertuples(index=False):
        home_id = int(getattr(row, home_col))
        away_id = int(getattr(row, away_col))
        st_h = states.get(home_id, TeamHistoryState())
        st_a = states.get(away_id, TeamHistoryState())
        gp_h, vol_h = _state_snapshot(st_h)
        gp_a, vol_a = _state_snapshot(st_a)
        out.append(
            {
                "games_played_home": gp_h,
                "games_played_away": gp_a,
                "games_played_diff": gp_h - gp_a,
                "volatility_home": vol_h,
                "volatility_away": vol_a,
                "volatility_diff": vol_h - vol_a,
            }
        )
    return pd.DataFrame(out, index=rows_df.index)


def _assemble_table(base_rows: pd.DataFrame, massey_df: pd.DataFrame, elo_df: pd.DataFrame, hist_df: pd.DataFrame, use_elo: bool) -> pd.DataFrame:
    tbl = pd.concat([base_rows.reset_index(drop=True), massey_df.reset_index(drop=True), elo_df.reset_index(drop=True), hist_df.reset_index(drop=True)], axis=1)
    if not use_elo:
        tbl["elo_diff_pre"] = 0.0
    tbl["elok_diff"] = tbl["elo_diff_pre"].astype(float)
    tbl["d_key"] = tbl["massey_diff"].astype(float)
    return tbl


def build_fold_feature_tables(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    team_ids: Sequence[int],
    *,
    massey_alpha: float,
    use_elo: bool,
    elo_home_adv: float,
    elo_k: float,
    elo_decay_a: float,
    elo_decay_g: float,
) -> FoldFeatureArtifacts:
    massey_model = fit_massey_ridge(train_rows, team_ids=team_ids, alpha=float(massey_alpha))

    elo_train_df, elo_model = run_elo_over_games(
        train_rows,
        home_id_col="HomeID",
        away_id_col="AwayID",
        margin_col="HomeWinMargin",
        home_adv=float(elo_home_adv if use_elo else 0.0),
        k_factor=float(elo_k),
        decay_a=float(elo_decay_a if use_elo else 0.0),
        decay_g=float(elo_decay_g),
        update_ratings=True,
    )
    elo_val_df = apply_elo_features(val_rows, elo_model, "HomeID", "AwayID", neutral_site=False)

    hist_train_df, history_states = _build_history_train_features(train_rows)
    hist_val_df = _build_history_static_features(val_rows, history_states, "HomeID", "AwayID")

    massey_train = apply_massey_features(train_rows, massey_model, "HomeID", "AwayID", neutral_site=False)
    massey_val = apply_massey_features(val_rows, massey_model, "HomeID", "AwayID", neutral_site=False)

    train_table = _assemble_table(train_rows, massey_train, elo_train_df, hist_train_df, use_elo=use_elo)
    val_table = _assemble_table(val_rows, massey_val, elo_val_df, hist_val_df, use_elo=use_elo)
    train_table.index = train_rows.index
    val_table.index = val_rows.index
    return FoldFeatureArtifacts(
        train_table=train_table,
        val_table=val_table,
        massey_model=massey_model,
        elo_model=elo_model,
        history_states=history_states,
    )


def build_full_train_table(
    train_df: pd.DataFrame,
    team_ids: Sequence[int],
    *,
    massey_alpha: float,
    use_elo: bool,
    elo_home_adv: float,
    elo_k: float,
    elo_decay_a: float,
    elo_decay_g: float,
) -> tuple[pd.DataFrame, MasseyModel, EloModel, Dict[int, TeamHistoryState]]:
    massey_model = fit_massey_ridge(train_df, team_ids=team_ids, alpha=float(massey_alpha))
    elo_train_df, elo_model = run_elo_over_games(
        train_df,
        home_id_col="HomeID",
        away_id_col="AwayID",
        margin_col="HomeWinMargin",
        home_adv=float(elo_home_adv if use_elo else 0.0),
        k_factor=float(elo_k),
        decay_a=float(elo_decay_a if use_elo else 0.0),
        decay_g=float(elo_decay_g),
        update_ratings=True,
    )
    hist_train_df, history_states = _build_history_train_features(train_df)
    massey_train = apply_massey_features(train_df, massey_model, "HomeID", "AwayID", neutral_site=False)
    train_table = _assemble_table(train_df, massey_train, elo_train_df, hist_train_df, use_elo=use_elo)
    train_table.index = train_df.index
    return train_table, massey_model, elo_model, history_states


def build_derby_table(
    pred_df: pd.DataFrame,
    massey_model: MasseyModel,
    elo_model: EloModel,
    history_states: Mapping[int, TeamHistoryState],
    *,
    use_elo: bool,
) -> pd.DataFrame:
    massey_df = apply_massey_features(pred_df, massey_model, "Team1_ID", "Team2_ID", neutral_site=True)
    elo_df = apply_elo_features(pred_df, elo_model, "Team1_ID", "Team2_ID", neutral_site=True)
    hist_df = _build_history_static_features(pred_df, history_states, "Team1_ID", "Team2_ID")
    derby_tbl = _assemble_table(pred_df, massey_df, elo_df, hist_df, use_elo=use_elo)
    derby_tbl.index = pred_df.index
    return derby_tbl


@dataclass
class OffDefModel:
    offense: Dict[int, float]
    defense: Dict[int, float]
    net: Dict[int, float]

    def net_rating_map(self) -> Dict[int, float]:
        return dict(self.net)


@dataclass
class StaticModelBundle:
    massey: MasseyModel
    offdef: OffDefModel
    conf_strength: Dict[str, float]


@dataclass
class TeamSequentialState:
    games_played: int = 0
    margin_history: List[float] = field(default_factory=list)
    oppadj_history: List[float] = field(default_factory=list)
    win_history: List[float] = field(default_factory=list)
    ema_margin_by_alpha: Dict[float, float] = field(default_factory=lambda: {0.10: 0.0, 0.20: 0.0, 0.35: 0.0})
    ema_oppadj_by_alpha: Dict[float, float] = field(default_factory=lambda: {0.10: 0.0, 0.20: 0.0, 0.35: 0.0})
    ema_winrate_by_alpha: Dict[float, float] = field(default_factory=lambda: {0.10: 0.5, 0.20: 0.5, 0.35: 0.5})
    ema_margin: float = 0.0
    home_games: int = 0
    away_games: int = 0
    home_margin_sum: float = 0.0
    away_margin_sum: float = 0.0
    last_date: Optional[pd.Timestamp] = None


@dataclass
class TrainSequentialBuild:
    features: pd.DataFrame
    final_states: Dict[int, TeamSequentialState]
    final_elo: Dict[int, float]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _safe_mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(vals, dtype=float)))


def _safe_std(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return float(PRIOR_VOL_SD)
    v = float(np.std(np.asarray(vals, dtype=float), ddof=0))
    if not np.isfinite(v):
        return float(PRIOR_VOL_SD)
    return float(max(v, 1e-6))


def _last_n_mean(vals: Sequence[float], n: int) -> float:
    if not vals:
        return 0.0
    arr = vals[-int(max(n, 1)) :]
    return float(np.mean(np.asarray(arr, dtype=float)))


def _ema_update(prev: float, value: float, alpha: float) -> float:
    return (1.0 - float(alpha)) * float(prev) + float(alpha) * float(value)


def _elo_expected(diff: float) -> float:
    return float(1.0 / (1.0 + np.power(10.0, -float(diff) / 400.0)))


def _apply_inactivity_decay(rating: float, days_inactive: float, tau_days: float) -> float:
    if not np.isfinite(days_inactive) or days_inactive <= 0.0 or tau_days <= 0.0:
        return float(rating)
    w = float(np.exp(-float(days_inactive) / float(max(tau_days, 1e-6))))
    return float(1500.0 + w * (float(rating) - 1500.0))


def _team_home_adjustment(state: TeamSequentialState, cfg: EloVariantConfig) -> float:
    if not bool(getattr(cfg, "use_team_home_adv", False)):
        return 0.0
    if state.home_games <= 0 or state.away_games <= 0:
        return 0.0
    home_avg = float(state.home_margin_sum) / float(max(state.home_games, 1))
    away_avg = float(state.away_margin_sum) / float(max(state.away_games, 1))
    raw = home_avg - away_avg
    reg = float(max(getattr(cfg, "team_home_adv_reg", 6.0), 1e-6))
    shrink = float(state.home_games + state.away_games) / float(state.home_games + state.away_games + reg)
    scale = float(getattr(cfg, "team_home_adv_scale", 70.0))
    cap = float(getattr(cfg, "team_home_adv_cap", 45.0))
    # Margin-to-Elo conversion: 20 points ~= 1 std game spread in this dataset.
    adj = float(np.clip(shrink * raw * (scale / 20.0), -cap, cap))
    return adj


def _elo_k_effective(st_h: TeamSequentialState, st_a: TeamSequentialState, cfg: EloVariantConfig, base_k: float) -> float:
    k = float(base_k)
    if bool(getattr(cfg, "use_dynamic_k", False)):
        gp_pair = 0.5 * float(st_h.games_played + st_a.games_played)
        games_scale = float(max(getattr(cfg, "k_games_scale", 16.0), 1.0))
        early_mult = float(max(getattr(cfg, "k_early_multiplier", 1.0), 0.1))
        decay = float(np.exp(-gp_pair / games_scale))
        early_factor = 1.0 + (early_mult - 1.0) * decay
        vol_h = _safe_std(st_h.margin_history)
        vol_a = _safe_std(st_a.margin_history)
        unc = float(getattr(cfg, "k_uncertainty_multiplier", 0.0)) * ((vol_h + vol_a) / (2.0 * PRIOR_VOL_SD))
        k *= float(max(early_factor * (1.0 + unc), 0.2))
    return float(np.clip(k, 4.0, 80.0))


def _ensure_state(states: Dict[int, TeamSequentialState], team_id: int) -> TeamSequentialState:
    tid = int(team_id)
    if tid not in states:
        states[tid] = TeamSequentialState()
    return states[tid]


def _snapshot_state_features(state: TeamSequentialState) -> dict:
    margin_mean = _safe_mean(state.margin_history)
    opp_mean = _safe_mean(state.oppadj_history)
    gp = float(state.games_played)
    vol = _safe_std(state.margin_history)
    last3 = _last_n_mean(state.margin_history, 3)
    last5 = _last_n_mean(state.margin_history, 5)
    last8 = _last_n_mean(state.margin_history, 8)
    opp_last5 = _last_n_mean(state.oppadj_history, 5)
    win_last5 = _last_n_mean(state.win_history, 5)
    trend_margin = float(last5 - margin_mean)
    trend_oppadj = float(opp_last5 - opp_mean)
    consistency = float(abs(margin_mean) / (vol + 1.0))
    return {
        "games_played": gp,
        "volatility": vol,
        "mean_margin": margin_mean,
        "oppadj_margin": opp_mean,
        "ema_margin_a10": float(state.ema_margin_by_alpha.get(0.10, 0.0)),
        "ema_margin_a20": float(state.ema_margin_by_alpha.get(0.20, 0.0)),
        "ema_margin_a35": float(state.ema_margin_by_alpha.get(0.35, 0.0)),
        "ema_oppadj_a10": float(state.ema_oppadj_by_alpha.get(0.10, 0.0)),
        "ema_oppadj_a20": float(state.ema_oppadj_by_alpha.get(0.20, 0.0)),
        "ema_oppadj_a35": float(state.ema_oppadj_by_alpha.get(0.35, 0.0)),
        "ema_winrate_a10": float(state.ema_winrate_by_alpha.get(0.10, 0.5)),
        "ema_winrate_a20": float(state.ema_winrate_by_alpha.get(0.20, 0.5)),
        "ema_winrate_a35": float(state.ema_winrate_by_alpha.get(0.35, 0.5)),
        "last3_margin": last3,
        "last5_margin": last5,
        "last8_margin": last8,
        "last5_oppadj": opp_last5,
        "last5_winrate": win_last5,
        "trend_margin_last5_vs_season": trend_margin,
        "trend_oppadj_last5_vs_season": trend_oppadj,
        "consistency_ratio": consistency,
    }


def _update_state(state: TeamSequentialState, *, margin: float, oppadj: float, win: float, date: pd.Timestamp, is_home: bool) -> None:
    state.games_played += 1
    state.margin_history.append(float(margin))
    state.oppadj_history.append(float(oppadj))
    state.win_history.append(float(win))
    state.margin_history = state.margin_history[-80:]
    state.oppadj_history = state.oppadj_history[-80:]
    state.win_history = state.win_history[-80:]
    for a in [0.10, 0.20, 0.35]:
        state.ema_margin_by_alpha[a] = _ema_update(state.ema_margin_by_alpha.get(a, 0.0), float(margin), a)
        state.ema_oppadj_by_alpha[a] = _ema_update(state.ema_oppadj_by_alpha.get(a, 0.0), float(oppadj), a)
        state.ema_winrate_by_alpha[a] = _ema_update(state.ema_winrate_by_alpha.get(a, 0.5), float(win), a)
    state.ema_margin = float(state.ema_margin_by_alpha.get(0.20, 0.0))
    if bool(is_home):
        state.home_games += 1
        state.home_margin_sum += float(margin)
    else:
        state.away_games += 1
        state.away_margin_sum += float(margin)
    state.last_date = pd.Timestamp(date)


def fit_static_models_for_fold(
    train_df: pd.DataFrame,
    *,
    team_ids: Sequence[int],
    conf_values: Sequence[str],
) -> StaticModelBundle:
    use = train_df.copy()
    use["Date"] = pd.to_datetime(use["Date"])
    use = use.sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True)
    massey = fit_massey_ridge(use, team_ids=team_ids, alpha=8.0)

    scored: Dict[int, float] = {}
    allowed: Dict[int, float] = {}
    games: Dict[int, int] = {}
    conf_sum: Dict[str, float] = {str(c): 0.0 for c in conf_values}
    conf_n: Dict[str, int] = {str(c): 0 for c in conf_values}
    for row in use.itertuples(index=False):
        hid = int(row.HomeID)
        aid = int(row.AwayID)
        hp = _safe_float(row.HomePts)
        ap = _safe_float(row.AwayPts)
        margin = _safe_float(row.HomeWinMargin)
        scored[hid] = scored.get(hid, 0.0) + hp
        allowed[hid] = allowed.get(hid, 0.0) + ap
        games[hid] = games.get(hid, 0) + 1
        scored[aid] = scored.get(aid, 0.0) + ap
        allowed[aid] = allowed.get(aid, 0.0) + hp
        games[aid] = games.get(aid, 0) + 1
        hc = str(row.HomeConf)
        ac = str(row.AwayConf)
        conf_sum[hc] = conf_sum.get(hc, 0.0) + margin
        conf_n[hc] = conf_n.get(hc, 0) + 1
        conf_sum[ac] = conf_sum.get(ac, 0.0) - margin
        conf_n[ac] = conf_n.get(ac, 0) + 1

    league_pts = float(np.mean(np.r_[use["HomePts"].astype(float).to_numpy(), use["AwayPts"].astype(float).to_numpy()]))
    offense: Dict[int, float] = {}
    defense: Dict[int, float] = {}
    net: Dict[int, float] = {}
    for tid in set([int(t) for t in team_ids] + [int(t) for t in games.keys()]):
        g = int(games.get(int(tid), 0))
        if g <= 0:
            offense[int(tid)] = 0.0
            defense[int(tid)] = 0.0
            net[int(tid)] = 0.0
            continue
        shrink = float(g / (g + 6.0))
        off = shrink * ((float(scored.get(int(tid), 0.0)) / g) - league_pts)
        de = shrink * (league_pts - (float(allowed.get(int(tid), 0.0)) / g))
        offense[int(tid)] = float(off)
        defense[int(tid)] = float(de)
        net[int(tid)] = float(off + de)

    conf_strength = {}
    for c in conf_sum.keys():
        n = int(conf_n.get(c, 0))
        conf_strength[str(c)] = float(conf_sum[c] / n) if n > 0 else 0.0
    return StaticModelBundle(massey=massey, offdef=OffDefModel(offense=offense, defense=defense, net=net), conf_strength=conf_strength)


def _static_row_features(
    rows: pd.DataFrame,
    *,
    home_id_col: str,
    away_id_col: str,
    home_conf_col: str,
    away_conf_col: str,
    models: StaticModelBundle,
    neutral_site: bool,
) -> pd.DataFrame:
    massey = apply_massey_features(rows, models.massey, home_id_col, away_id_col, neutral_site=bool(neutral_site))
    hid = rows[home_id_col].astype(int)
    aid = rows[away_id_col].astype(int)
    hc = rows[home_conf_col].astype(str)
    ac = rows[away_conf_col].astype(str)

    def _map_float(ids: pd.Series, mp: Mapping[int, float]) -> pd.Series:
        return ids.map({int(k): float(v) for k, v in mp.items()}).fillna(0.0).astype(float)

    off_h = _map_float(hid, models.offdef.offense)
    off_a = _map_float(aid, models.offdef.offense)
    def_h = _map_float(hid, models.offdef.defense)
    def_a = _map_float(aid, models.offdef.defense)
    net_h = _map_float(hid, models.offdef.net)
    net_a = _map_float(aid, models.offdef.net)

    conf_h = hc.map(models.conf_strength).fillna(0.0).astype(float)
    conf_a = ac.map(models.conf_strength).fillna(0.0).astype(float)
    return pd.DataFrame(
        {
            "massey_home_rating": massey["massey_home_rating"].astype(float).to_numpy(),
            "massey_away_rating": massey["massey_away_rating"].astype(float).to_numpy(),
            "massey_diff": massey["massey_diff"].astype(float).to_numpy(),
            "offdef_offense_home": off_h.to_numpy(dtype=float),
            "offdef_offense_away": off_a.to_numpy(dtype=float),
            "offdef_defense_home": def_h.to_numpy(dtype=float),
            "offdef_defense_away": def_a.to_numpy(dtype=float),
            "offdef_net_home": net_h.to_numpy(dtype=float),
            "offdef_net_away": net_a.to_numpy(dtype=float),
            "offdef_net_diff": (net_h - net_a).to_numpy(dtype=float),
            "offdef_margin_neutral": (net_h - net_a).to_numpy(dtype=float),
            "conf_strength_home": conf_h.to_numpy(dtype=float),
            "conf_strength_away": conf_a.to_numpy(dtype=float),
            "conf_strength_diff": (conf_h - conf_a).to_numpy(dtype=float),
        },
        index=rows.index,
    )


def apply_static_models_to_train_like_rows(rows: pd.DataFrame, models: StaticModelBundle, neutral_site: bool = False) -> pd.DataFrame:
    return _static_row_features(
        rows,
        home_id_col="HomeID",
        away_id_col="AwayID",
        home_conf_col="HomeConf",
        away_conf_col="AwayConf",
        models=models,
        neutral_site=bool(neutral_site),
    )


def apply_static_models_to_derby_rows(rows: pd.DataFrame, models: StaticModelBundle) -> pd.DataFrame:
    return _static_row_features(
        rows,
        home_id_col="Team1_ID",
        away_id_col="Team2_ID",
        home_conf_col="Team1_Conf",
        away_conf_col="Team2_Conf",
        models=models,
        neutral_site=True,
    )


def build_train_sequential_features(
    train_df: pd.DataFrame,
    *,
    home_adv: float,
    team_ids: Sequence[int],
    elo_k: float,
    elo_variant: Optional[EloVariantConfig] = None,
) -> TrainSequentialBuild:
    cfg = elo_variant if elo_variant is not None else EloVariantConfig(name="default", home_adv=float(home_adv), k_factor=float(elo_k))
    use = train_df.copy()
    use["Date"] = pd.to_datetime(use["Date"])
    use = use.sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True)

    states: Dict[int, TeamSequentialState] = {int(t): TeamSequentialState() for t in team_ids}
    elo: Dict[int, float] = {int(t): 1500.0 for t in team_ids}
    conf_sum: Dict[str, float] = {}
    conf_n: Dict[str, int] = {}

    rows: List[dict] = []
    for row in use.itertuples(index=False):
        dt = pd.Timestamp(row.Date)
        hid = int(row.HomeID)
        aid = int(row.AwayID)
        hconf = str(row.HomeConf)
        aconf = str(row.AwayConf)
        st_h = _ensure_state(states, hid)
        st_a = _ensure_state(states, aid)
        r_h = float(elo.get(hid, 1500.0))
        r_a = float(elo.get(aid, 1500.0))

        if bool(getattr(cfg, "use_inactivity_decay", False)):
            tau = float(max(getattr(cfg, "inactivity_tau_days", 95.0), 1.0))
            if st_h.last_date is not None:
                r_h = _apply_inactivity_decay(r_h, float((dt - pd.Timestamp(st_h.last_date)).days), tau)
            if st_a.last_date is not None:
                r_a = _apply_inactivity_decay(r_a, float((dt - pd.Timestamp(st_a.last_date)).days), tau)

        neutral_diff = float(r_h - r_a)
        home_edge = float(home_adv) + _team_home_adjustment(st_h, cfg)
        elo_diff = float(neutral_diff + home_edge)
        p_home = _elo_expected(elo_diff)
        k_eff = _elo_k_effective(st_h, st_a, cfg, base_k=float(getattr(cfg, "k_factor", elo_k)))

        sh = _snapshot_state_features(st_h)
        sa = _snapshot_state_features(st_a)
        conf_h = float(conf_sum.get(hconf, 0.0) / max(conf_n.get(hconf, 0), 1)) if conf_n.get(hconf, 0) > 0 else 0.0
        conf_a = float(conf_sum.get(aconf, 0.0) / max(conf_n.get(aconf, 0), 1)) if conf_n.get(aconf, 0) > 0 else 0.0

        row_out = {
            "elo_home_pre": r_h,
            "elo_away_pre": r_a,
            "elo_diff_pre": elo_diff,
            "elo_neutral_diff_pre": neutral_diff,
            "elo_prob_home_pre": p_home,
            "elo_k_eff_pre": k_eff,
            "games_played_home": sh["games_played"],
            "games_played_away": sa["games_played"],
            "games_played_diff": sh["games_played"] - sa["games_played"],
            "games_played_min": min(sh["games_played"], sa["games_played"]),
            "volatility_home": sh["volatility"],
            "volatility_away": sa["volatility"],
            "volatility_diff": sh["volatility"] - sa["volatility"],
            "volatility_sum": sh["volatility"] + sa["volatility"],
            "mean_margin_home": sh["mean_margin"],
            "mean_margin_away": sa["mean_margin"],
            "mean_margin_diff": sh["mean_margin"] - sa["mean_margin"],
            "oppadj_margin_home": sh["oppadj_margin"],
            "oppadj_margin_away": sa["oppadj_margin"],
            "oppadj_margin_diff": sh["oppadj_margin"] - sa["oppadj_margin"],
            "ema_margin_a10_home": sh["ema_margin_a10"],
            "ema_margin_a10_away": sa["ema_margin_a10"],
            "ema_margin_a10_diff": sh["ema_margin_a10"] - sa["ema_margin_a10"],
            "ema_margin_a20_home": sh["ema_margin_a20"],
            "ema_margin_a20_away": sa["ema_margin_a20"],
            "ema_margin_a20_diff": sh["ema_margin_a20"] - sa["ema_margin_a20"],
            "ema_margin_a35_home": sh["ema_margin_a35"],
            "ema_margin_a35_away": sa["ema_margin_a35"],
            "ema_margin_a35_diff": sh["ema_margin_a35"] - sa["ema_margin_a35"],
            "ema_margin_diff": sh["ema_margin_a20"] - sa["ema_margin_a20"],
            "ema_oppadj_a10_home": sh["ema_oppadj_a10"],
            "ema_oppadj_a10_away": sa["ema_oppadj_a10"],
            "ema_oppadj_a10_diff": sh["ema_oppadj_a10"] - sa["ema_oppadj_a10"],
            "ema_oppadj_a20_home": sh["ema_oppadj_a20"],
            "ema_oppadj_a20_away": sa["ema_oppadj_a20"],
            "ema_oppadj_a20_diff": sh["ema_oppadj_a20"] - sa["ema_oppadj_a20"],
            "ema_oppadj_a35_home": sh["ema_oppadj_a35"],
            "ema_oppadj_a35_away": sa["ema_oppadj_a35"],
            "ema_oppadj_a35_diff": sh["ema_oppadj_a35"] - sa["ema_oppadj_a35"],
            "ema_winrate_a10_home": sh["ema_winrate_a10"],
            "ema_winrate_a10_away": sa["ema_winrate_a10"],
            "ema_winrate_a10_diff": sh["ema_winrate_a10"] - sa["ema_winrate_a10"],
            "ema_winrate_a20_home": sh["ema_winrate_a20"],
            "ema_winrate_a20_away": sa["ema_winrate_a20"],
            "ema_winrate_a20_diff": sh["ema_winrate_a20"] - sa["ema_winrate_a20"],
            "ema_winrate_a35_home": sh["ema_winrate_a35"],
            "ema_winrate_a35_away": sa["ema_winrate_a35"],
            "ema_winrate_a35_diff": sh["ema_winrate_a35"] - sa["ema_winrate_a35"],
            "last3_margin_home": sh["last3_margin"],
            "last3_margin_away": sa["last3_margin"],
            "last3_margin_diff": sh["last3_margin"] - sa["last3_margin"],
            "last5_margin_home": sh["last5_margin"],
            "last5_margin_away": sa["last5_margin"],
            "last5_margin_diff": sh["last5_margin"] - sa["last5_margin"],
            "last8_margin_home": sh["last8_margin"],
            "last8_margin_away": sa["last8_margin"],
            "last8_margin_diff": sh["last8_margin"] - sa["last8_margin"],
            "last5_oppadj_home": sh["last5_oppadj"],
            "last5_oppadj_away": sa["last5_oppadj"],
            "last5_oppadj_diff": sh["last5_oppadj"] - sa["last5_oppadj"],
            "trend_margin_last5_vs_season_home": sh["trend_margin_last5_vs_season"],
            "trend_margin_last5_vs_season_away": sa["trend_margin_last5_vs_season"],
            "trend_margin_last5_vs_season_diff": sh["trend_margin_last5_vs_season"] - sa["trend_margin_last5_vs_season"],
            "trend_oppadj_last5_vs_season_home": sh["trend_oppadj_last5_vs_season"],
            "trend_oppadj_last5_vs_season_away": sa["trend_oppadj_last5_vs_season"],
            "trend_oppadj_last5_vs_season_diff": sh["trend_oppadj_last5_vs_season"] - sa["trend_oppadj_last5_vs_season"],
            "consistency_ratio_home": sh["consistency_ratio"],
            "consistency_ratio_away": sa["consistency_ratio"],
            "consistency_ratio_diff": sh["consistency_ratio"] - sa["consistency_ratio"],
            "conf_strength_home": conf_h,
            "conf_strength_away": conf_a,
            "conf_strength_diff": conf_h - conf_a,
            "same_conf_flag": 1.0 if hconf == aconf else 0.0,
            "cross_conf_flag": 0.0 if hconf == aconf else 1.0,
        }
        rows.append(row_out)

        margin = _safe_float(row.HomeWinMargin)
        score_h = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        score_a = 1.0 - score_h if margin != 0 else 0.5
        delta = float(k_eff * (score_h - p_home))
        elo[hid] = float(r_h + delta)
        elo[aid] = float(r_a - delta)

        # Opp-adjusted margin removes expected strength gap so blowouts by favorites are discounted.
        oppadj_h = float(margin - neutral_diff / 25.0)
        oppadj_a = float(-margin + neutral_diff / 25.0)
        _update_state(st_h, margin=margin, oppadj=oppadj_h, win=score_h, date=dt, is_home=True)
        _update_state(st_a, margin=-margin, oppadj=oppadj_a, win=score_a, date=dt, is_home=False)

        conf_sum[hconf] = conf_sum.get(hconf, 0.0) + margin
        conf_n[hconf] = conf_n.get(hconf, 0) + 1
        conf_sum[aconf] = conf_sum.get(aconf, 0.0) - margin
        conf_n[aconf] = conf_n.get(aconf, 0) + 1

    feat = pd.DataFrame(rows, index=use.index)
    return TrainSequentialBuild(features=feat, final_states=states, final_elo={int(k): float(v) for k, v in elo.items()})


def build_derby_sequential_features(
    pred_df: pd.DataFrame,
    final_states: Mapping[int, TeamSequentialState],
    final_elo: Mapping[int, float],
) -> pd.DataFrame:
    use = pred_df.copy()
    use["Date"] = pd.to_datetime(use["Date"])
    use = use.sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True)
    states = {int(k): v for k, v in final_states.items()}
    elo = {int(k): _safe_float(v, 1500.0) for k, v in final_elo.items()}

    rows: List[dict] = []
    for row in use.itertuples(index=False):
        t1 = int(row.Team1_ID)
        t2 = int(row.Team2_ID)
        c1 = str(row.Team1_Conf)
        c2 = str(row.Team2_Conf)
        st1 = states.get(t1, TeamSequentialState())
        st2 = states.get(t2, TeamSequentialState())
        s1 = _snapshot_state_features(st1)
        s2 = _snapshot_state_features(st2)
        r1 = float(elo.get(t1, 1500.0))
        r2 = float(elo.get(t2, 1500.0))
        neutral_diff = float(r1 - r2)
        conf1 = s1["oppadj_margin"]
        conf2 = s2["oppadj_margin"]
        rows.append(
            {
                "elo_home_pre": r1,
                "elo_away_pre": r2,
                "elo_diff_pre": neutral_diff,
                "elo_neutral_diff_pre": neutral_diff,
                "elo_prob_home_pre": _elo_expected(neutral_diff),
                "elo_k_eff_pre": 0.0,
                "games_played_home": s1["games_played"],
                "games_played_away": s2["games_played"],
                "games_played_diff": s1["games_played"] - s2["games_played"],
                "games_played_min": min(s1["games_played"], s2["games_played"]),
                "volatility_home": s1["volatility"],
                "volatility_away": s2["volatility"],
                "volatility_diff": s1["volatility"] - s2["volatility"],
                "volatility_sum": s1["volatility"] + s2["volatility"],
                "mean_margin_home": s1["mean_margin"],
                "mean_margin_away": s2["mean_margin"],
                "mean_margin_diff": s1["mean_margin"] - s2["mean_margin"],
                "oppadj_margin_home": s1["oppadj_margin"],
                "oppadj_margin_away": s2["oppadj_margin"],
                "oppadj_margin_diff": s1["oppadj_margin"] - s2["oppadj_margin"],
                "ema_margin_a10_home": s1["ema_margin_a10"],
                "ema_margin_a10_away": s2["ema_margin_a10"],
                "ema_margin_a10_diff": s1["ema_margin_a10"] - s2["ema_margin_a10"],
                "ema_margin_a20_home": s1["ema_margin_a20"],
                "ema_margin_a20_away": s2["ema_margin_a20"],
                "ema_margin_a20_diff": s1["ema_margin_a20"] - s2["ema_margin_a20"],
                "ema_margin_a35_home": s1["ema_margin_a35"],
                "ema_margin_a35_away": s2["ema_margin_a35"],
                "ema_margin_a35_diff": s1["ema_margin_a35"] - s2["ema_margin_a35"],
                "ema_margin_diff": s1["ema_margin_a20"] - s2["ema_margin_a20"],
                "ema_oppadj_a10_home": s1["ema_oppadj_a10"],
                "ema_oppadj_a10_away": s2["ema_oppadj_a10"],
                "ema_oppadj_a10_diff": s1["ema_oppadj_a10"] - s2["ema_oppadj_a10"],
                "ema_oppadj_a20_home": s1["ema_oppadj_a20"],
                "ema_oppadj_a20_away": s2["ema_oppadj_a20"],
                "ema_oppadj_a20_diff": s1["ema_oppadj_a20"] - s2["ema_oppadj_a20"],
                "ema_oppadj_a35_home": s1["ema_oppadj_a35"],
                "ema_oppadj_a35_away": s2["ema_oppadj_a35"],
                "ema_oppadj_a35_diff": s1["ema_oppadj_a35"] - s2["ema_oppadj_a35"],
                "ema_winrate_a10_home": s1["ema_winrate_a10"],
                "ema_winrate_a10_away": s2["ema_winrate_a10"],
                "ema_winrate_a10_diff": s1["ema_winrate_a10"] - s2["ema_winrate_a10"],
                "ema_winrate_a20_home": s1["ema_winrate_a20"],
                "ema_winrate_a20_away": s2["ema_winrate_a20"],
                "ema_winrate_a20_diff": s1["ema_winrate_a20"] - s2["ema_winrate_a20"],
                "ema_winrate_a35_home": s1["ema_winrate_a35"],
                "ema_winrate_a35_away": s2["ema_winrate_a35"],
                "ema_winrate_a35_diff": s1["ema_winrate_a35"] - s2["ema_winrate_a35"],
                "last3_margin_home": s1["last3_margin"],
                "last3_margin_away": s2["last3_margin"],
                "last3_margin_diff": s1["last3_margin"] - s2["last3_margin"],
                "last5_margin_home": s1["last5_margin"],
                "last5_margin_away": s2["last5_margin"],
                "last5_margin_diff": s1["last5_margin"] - s2["last5_margin"],
                "last8_margin_home": s1["last8_margin"],
                "last8_margin_away": s2["last8_margin"],
                "last8_margin_diff": s1["last8_margin"] - s2["last8_margin"],
                "last5_oppadj_home": s1["last5_oppadj"],
                "last5_oppadj_away": s2["last5_oppadj"],
                "last5_oppadj_diff": s1["last5_oppadj"] - s2["last5_oppadj"],
                "trend_margin_last5_vs_season_home": s1["trend_margin_last5_vs_season"],
                "trend_margin_last5_vs_season_away": s2["trend_margin_last5_vs_season"],
                "trend_margin_last5_vs_season_diff": s1["trend_margin_last5_vs_season"] - s2["trend_margin_last5_vs_season"],
                "trend_oppadj_last5_vs_season_home": s1["trend_oppadj_last5_vs_season"],
                "trend_oppadj_last5_vs_season_away": s2["trend_oppadj_last5_vs_season"],
                "trend_oppadj_last5_vs_season_diff": s1["trend_oppadj_last5_vs_season"] - s2["trend_oppadj_last5_vs_season"],
                "consistency_ratio_home": s1["consistency_ratio"],
                "consistency_ratio_away": s2["consistency_ratio"],
                "consistency_ratio_diff": s1["consistency_ratio"] - s2["consistency_ratio"],
                "conf_strength_home": conf1,
                "conf_strength_away": conf2,
                "conf_strength_diff": conf1 - conf2,
                "same_conf_flag": 1.0 if c1 == c2 else 0.0,
                "cross_conf_flag": 0.0 if c1 == c2 else 1.0,
            }
        )
    return pd.DataFrame(rows, index=use.index)


def assemble_model_table(base_rows: pd.DataFrame, seq_features: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    base = base_rows.reset_index(drop=True).copy()
    seq = seq_features.reset_index(drop=True).copy()
    sta = static_features.reset_index(drop=True).copy()
    tbl = pd.concat([base, seq, sta], axis=1)
    if tbl.columns.duplicated().any():
        tbl = tbl.loc[:, ~tbl.columns.duplicated(keep="last")].copy()
    if "Date" in tbl.columns:
        tbl["Date"] = pd.to_datetime(tbl["Date"])
    if "elo_diff_pre" in tbl.columns:
        tbl["elok_diff"] = tbl["elo_diff_pre"].astype(float)
    if "games_played_home" in tbl.columns and "games_played_away" in tbl.columns:
        tbl["games_played_diff"] = tbl["games_played_home"].astype(float) - tbl["games_played_away"].astype(float)
        tbl["games_played_min"] = np.minimum(tbl["games_played_home"].astype(float), tbl["games_played_away"].astype(float))
    if "volatility_home" in tbl.columns and "volatility_away" in tbl.columns:
        tbl["volatility_diff"] = tbl["volatility_home"].astype(float) - tbl["volatility_away"].astype(float)
        tbl["volatility_sum"] = tbl["volatility_home"].astype(float) + tbl["volatility_away"].astype(float)
    if "same_conf_flag" not in tbl.columns:
        if "HomeConf" in tbl.columns and "AwayConf" in tbl.columns:
            same = (tbl["HomeConf"].astype(str) == tbl["AwayConf"].astype(str)).astype(float)
        elif "Team1_Conf" in tbl.columns and "Team2_Conf" in tbl.columns:
            same = (tbl["Team1_Conf"].astype(str) == tbl["Team2_Conf"].astype(str)).astype(float)
        else:
            same = pd.Series(np.zeros(len(tbl)), index=tbl.index, dtype=float)
        tbl["same_conf_flag"] = same
        tbl["cross_conf_flag"] = 1.0 - same
    if "massey_diff" in tbl.columns:
        tbl["d_key"] = tbl["massey_diff"].astype(float)
    # Keep numeric columns finite to avoid train-time instability.
    for c in tbl.columns:
        if pd.api.types.is_numeric_dtype(tbl[c]):
            s = pd.to_numeric(tbl[c], errors="coerce").fillna(0.0).astype(float)
            s = s.replace([np.inf, -np.inf], 0.0)
            tbl[c] = s
    return tbl
