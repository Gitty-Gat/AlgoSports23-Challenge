from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.ratings import (
    ConferenceStrengthResult,
    EloResult,
    EloVariantConfig,
    OffDefRidgeResult,
    RidgeRatingsResult,
    apply_conference_features,
    apply_massey_features,
    apply_offdef_features,
    run_elo_over_games,
    fit_conference_strength_ridge,
    fit_massey_ridge,
    fit_offense_defense_ridge,
    run_elo_over_games_advanced,
)


DEFAULT_REST_DAYS = 7.0
PRIOR_VOL_SD = 45.0
EMA_ALPHA = 0.2
EMA_ALPHAS = (0.10, 0.20, 0.35)
LAST_KS = (3, 5, 8)


@dataclass
class TeamRollingState:
    games_played: int = 0
    wins: int = 0
    margin_sum: float = 0.0
    margin_sq_sum: float = 0.0
    ema_margin: float = 0.0
    oppadj_margin_sum: float = 0.0
    opp_elo_sum: float = 0.0
    last_date: Optional[pd.Timestamp] = None
    ema_margin_by_alpha: Dict[float, float] = field(default_factory=dict)
    ema_oppadj_by_alpha: Dict[float, float] = field(default_factory=dict)
    ema_winrate_by_alpha: Dict[float, float] = field(default_factory=dict)
    margin_history: List[float] = field(default_factory=list)
    oppadj_history: List[float] = field(default_factory=list)
    win_history: List[float] = field(default_factory=list)


@dataclass
class SequentialFeatureBuild:
    features: pd.DataFrame
    final_states: Dict[int, TeamRollingState]
    final_elo: Dict[int, float]


@dataclass
class StaticModelsBundle:
    massey: RidgeRatingsResult
    offdef: OffDefRidgeResult
    conf: ConferenceStrengthResult


def parse_and_sort_train(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True)
    date_game = df[["Date", "GameID"]].copy()
    if not date_game.equals(date_game.sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True)):
        raise ValueError("Train rows are not non-decreasing by Date then GameID after sort.")
    return df


def parse_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "GameID"], kind="mergesort").reset_index(drop=True)
    return df


def build_team_universe(rankings_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["TeamID", "Team"]
    missing = [c for c in cols if c not in rankings_df.columns]
    if missing:
        raise ValueError(f"Rankings.xlsx missing columns: {missing}")
    uni = rankings_df[cols].copy()
    uni["TeamID"] = uni["TeamID"].astype(int)
    uni["Team"] = uni["Team"].astype(str)
    return uni.drop_duplicates("TeamID").reset_index(drop=True)


def make_expanding_time_folds(train_df: pd.DataFrame, n_folds: int = 5) -> List[dict]:
    if "Date" not in train_df.columns:
        raise ValueError("Train df must have parsed Date column")
    unique_dates = np.array(sorted(pd.to_datetime(train_df["Date"]).unique()))
    if len(unique_dates) < n_folds + 5:
        raise ValueError("Too few unique dates for requested folds.")

    # Use roughly equal contiguous date chunks; validation starts after an expanding training window.
    buckets = np.array_split(unique_dates, n_folds + 1)
    folds: List[dict] = []
    for fold_num in range(1, n_folds + 1):
        train_dates = np.concatenate(buckets[:fold_num])
        val_dates = buckets[fold_num]
        train_mask = train_df["Date"].isin(train_dates).to_numpy()
        val_mask = train_df["Date"].isin(val_dates).to_numpy()
        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        if train_idx.max() >= val_idx.min():
            # Overlap in dates is acceptable only if rows are strictly partitioned; masks already partition by dates.
            pass
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
    if len(folds) < n_folds:
        raise ValueError(f"Constructed only {len(folds)} folds (requested {n_folds}).")
    return folds


def _default_state() -> TeamRollingState:
    st = TeamRollingState()
    st.ema_margin_by_alpha = {float(a): 0.0 for a in EMA_ALPHAS}
    st.ema_oppadj_by_alpha = {float(a): 0.0 for a in EMA_ALPHAS}
    st.ema_winrate_by_alpha = {float(a): 0.5 for a in EMA_ALPHAS}
    return st


def _recent_mean(values: Sequence[float], k: int, default: float = 0.0) -> float:
    if not values:
        return float(default)
    tail = np.asarray(values[-int(k) :], dtype=float)
    if tail.size == 0:
        return float(default)
    return float(np.mean(tail))


def _recent_sd(values: Sequence[float], k: int, default: float) -> float:
    if not values:
        return float(default)
    tail = np.asarray(values[-int(k) :], dtype=float)
    if tail.size <= 1:
        return float(default)
    return float(np.std(tail, ddof=0))


def _pregame_team_snapshot(state: TeamRollingState, game_date: pd.Timestamp) -> dict:
    gp = state.games_played
    mean_margin = state.margin_sum / gp if gp > 0 else 0.0
    ema_margin = state.ema_margin if gp > 0 else 0.0
    winrate = (state.wins + 1.0) / (gp + 2.0)  # Laplace-smoothed
    oppadj = state.oppadj_margin_sum / gp if gp > 0 else 0.0
    sos_elo = state.opp_elo_sum / gp if gp > 0 else 1500.0

    if gp > 1:
        sample_var = (state.margin_sq_sum - (state.margin_sum**2) / gp) / max(gp - 1, 1)
        sample_var = max(sample_var, 0.0)
    else:
        sample_var = PRIOR_VOL_SD**2
    prior_n = 3.0
    shrunk_var = (sample_var * max(gp - 1, 0) + prior_n * (PRIOR_VOL_SD**2)) / max(max(gp - 1, 0) + prior_n, 1.0)
    vol_sd = float(np.sqrt(max(shrunk_var, 1e-8)))

    rest_days = DEFAULT_REST_DAYS
    if state.last_date is not None:
        rest_days = float((game_date - state.last_date).days)
        if rest_days < 0:
            rest_days = 0.0

    ema_margin_vals = {a: float(state.ema_margin_by_alpha.get(a, ema_margin)) for a in EMA_ALPHAS}
    ema_oppadj_vals = {a: float(state.ema_oppadj_by_alpha.get(a, oppadj)) for a in EMA_ALPHAS}
    ema_winrate_vals = {a: float(state.ema_winrate_by_alpha.get(a, winrate)) for a in EMA_ALPHAS}

    out = {
        "games_played": float(gp),
        "winrate": float(winrate),
        "mean_margin": float(mean_margin),
        "ema_margin": float(ema_margin),
        "oppadj_margin": float(oppadj),
        "sos_elo": float(sos_elo),
        "volatility": float(vol_sd),
        "rest_days": float(rest_days),
    }
    for a in EMA_ALPHAS:
        tag = str(int(round(a * 100)))
        out[f"ema_margin_a{tag}"] = ema_margin_vals[a]
        out[f"ema_oppadj_a{tag}"] = ema_oppadj_vals[a]
        out[f"ema_winrate_a{tag}"] = ema_winrate_vals[a]

    for k in LAST_KS:
        out[f"last{k}_margin"] = _recent_mean(state.margin_history, k, default=mean_margin)
        out[f"last{k}_oppadj"] = _recent_mean(state.oppadj_history, k, default=oppadj)
        out[f"last{k}_winrate"] = _recent_mean(state.win_history, k, default=winrate)
        out[f"last{k}_volatility"] = _recent_sd(state.margin_history, k, default=vol_sd)
    out["consistency_ratio_last5"] = float(out["last5_volatility"] / max(vol_sd, 1e-6))
    out["consistency_ratio_last8"] = float(out["last8_volatility"] / max(vol_sd, 1e-6))
    return out


def _update_team_state(
    state: TeamRollingState,
    margin_for_team: float,
    opp_elo_pre: float,
    game_date: pd.Timestamp,
) -> None:
    margin_val = float(margin_for_team)
    oppadj_val = margin_val - (float(opp_elo_pre) - 1500.0) / 25.0
    win_val = 1.0 if margin_val > 0 else 0.0 if margin_val < 0 else 0.5
    state.games_played += 1
    state.wins += 1 if margin_val > 0 else 0
    state.margin_sum += margin_val
    state.margin_sq_sum += margin_val**2
    if state.games_played == 1:
        state.ema_margin = margin_val
    else:
        state.ema_margin = EMA_ALPHA * margin_val + (1.0 - EMA_ALPHA) * state.ema_margin
    for a in EMA_ALPHAS:
        a = float(a)
        if state.games_played == 1:
            state.ema_margin_by_alpha[a] = margin_val
            state.ema_oppadj_by_alpha[a] = oppadj_val
            state.ema_winrate_by_alpha[a] = win_val
        else:
            state.ema_margin_by_alpha[a] = a * margin_val + (1.0 - a) * float(state.ema_margin_by_alpha.get(a, 0.0))
            state.ema_oppadj_by_alpha[a] = a * oppadj_val + (1.0 - a) * float(state.ema_oppadj_by_alpha.get(a, 0.0))
            state.ema_winrate_by_alpha[a] = a * win_val + (1.0 - a) * float(state.ema_winrate_by_alpha.get(a, 0.5))
    # Opponent-adjusted margin in points; 25 Elo ~= 1 point heuristic (kept modest and stable).
    state.oppadj_margin_sum += oppadj_val
    state.opp_elo_sum += float(opp_elo_pre)
    state.margin_history.append(margin_val)
    state.oppadj_history.append(oppadj_val)
    state.win_history.append(win_val)
    state.last_date = pd.Timestamp(game_date)


def build_train_sequential_features(
    train_df: pd.DataFrame,
    home_adv: float,
    team_ids: Sequence[int],
    elo_k: float = 24.0,
    elo_variant: Optional[EloVariantConfig] = None,
) -> SequentialFeatureBuild:
    if elo_variant is None:
        elo_res: EloResult = run_elo_over_games(
            train_df,
            home_id_col="HomeID",
            away_id_col="AwayID",
            margin_col="HomeWinMargin",
            home_adv=float(home_adv),
            k_factor=elo_k,
            update_ratings=True,
        )
    else:
        elo_res = run_elo_over_games_advanced(
            train_df,
            home_id_col="HomeID",
            away_id_col="AwayID",
            margin_col="HomeWinMargin",
            config=elo_variant,
            date_col="Date",
            update_ratings=True,
        )
    elo_feat = elo_res.game_features

    states: Dict[int, TeamRollingState] = {int(t): _default_state() for t in team_ids}
    rows: List[dict] = []
    for idx, row in train_df.iterrows():
        date = pd.Timestamp(row["Date"])
        home_id = int(row["HomeID"])
        away_id = int(row["AwayID"])
        st_h = states.setdefault(home_id, _default_state())
        st_a = states.setdefault(away_id, _default_state())

        snap_h = _pregame_team_snapshot(st_h, date)
        snap_a = _pregame_team_snapshot(st_a, date)
        home_elo_pre = float(elo_feat.loc[idx, "elo_home_pre"])
        away_elo_pre = float(elo_feat.loc[idx, "elo_away_pre"])
        elo_row = elo_feat.loc[idx]
        feat_row = {
            "elo_home_pre": home_elo_pre,
            "elo_away_pre": away_elo_pre,
            "elo_diff_pre": float(elo_row.get("elo_diff_pre", home_elo_pre - away_elo_pre)),
            "elo_neutral_diff_pre": float(elo_row.get("elo_neutral_diff_pre", home_elo_pre - away_elo_pre)),
            "elo_prob_home_pre": float(elo_row.get("elo_prob_home_pre", 0.5)),
            "elo_home_edge_pre": float(elo_row.get("elo_home_edge_pre", home_adv)),
            "elo_home_team_dev_pre": float(elo_row.get("elo_home_team_dev_pre", 0.0)),
            "elo_k_eff_pre": float(elo_row.get("elo_k_eff_pre", elo_k)),
            "elo_gap_home_days": float(elo_row.get("elo_gap_home_days", 0.0)),
            "elo_gap_away_days": float(elo_row.get("elo_gap_away_days", 0.0)),
            "elo_gap_days_diff": float(elo_row.get("elo_gap_days_diff", 0.0)),
            "mean_margin_home": snap_h["mean_margin"],
            "mean_margin_away": snap_a["mean_margin"],
            "mean_margin_diff": snap_h["mean_margin"] - snap_a["mean_margin"],
            "ema_margin_home": snap_h["ema_margin"],
            "ema_margin_away": snap_a["ema_margin"],
            "ema_margin_diff": snap_h["ema_margin"] - snap_a["ema_margin"],
            "oppadj_margin_home": snap_h["oppadj_margin"],
            "oppadj_margin_away": snap_a["oppadj_margin"],
            "oppadj_margin_diff": snap_h["oppadj_margin"] - snap_a["oppadj_margin"],
            "sos_elo_home": snap_h["sos_elo"],
            "sos_elo_away": snap_a["sos_elo"],
            "sos_elo_diff": snap_h["sos_elo"] - snap_a["sos_elo"],
            "winrate_home": snap_h["winrate"],
            "winrate_away": snap_a["winrate"],
            "winrate_diff": snap_h["winrate"] - snap_a["winrate"],
            "volatility_home": snap_h["volatility"],
            "volatility_away": snap_a["volatility"],
            "volatility_diff": snap_h["volatility"] - snap_a["volatility"],
            "games_played_home": snap_h["games_played"],
            "games_played_away": snap_a["games_played"],
            "games_played_diff": snap_h["games_played"] - snap_a["games_played"],
            "rest_days_home": snap_h["rest_days"],
            "rest_days_away": snap_a["rest_days"],
            "rest_days_diff": snap_h["rest_days"] - snap_a["rest_days"],
        }
        for a in EMA_ALPHAS:
            tag = str(int(round(a * 100)))
            for stem in ["ema_margin", "ema_oppadj", "ema_winrate"]:
                hk = f"{stem}_a{tag}"
                feat_row[f"{hk}_home"] = float(snap_h[hk])
                feat_row[f"{hk}_away"] = float(snap_a[hk])
                feat_row[f"{hk}_diff"] = float(snap_h[hk] - snap_a[hk])
        for k in LAST_KS:
            for stem in ["margin", "oppadj", "winrate", "volatility"]:
                hk = f"last{k}_{stem}"
                feat_row[f"{hk}_home"] = float(snap_h[hk])
                feat_row[f"{hk}_away"] = float(snap_a[hk])
                feat_row[f"{hk}_diff"] = float(snap_h[hk] - snap_a[hk])
        feat_row["trend_margin_last3_vs_season_home"] = float(snap_h["last3_margin"] - snap_h["mean_margin"])
        feat_row["trend_margin_last3_vs_season_away"] = float(snap_a["last3_margin"] - snap_a["mean_margin"])
        feat_row["trend_margin_last3_vs_season_diff"] = float(
            feat_row["trend_margin_last3_vs_season_home"] - feat_row["trend_margin_last3_vs_season_away"]
        )
        feat_row["trend_margin_last5_vs_season_home"] = float(snap_h["last5_margin"] - snap_h["mean_margin"])
        feat_row["trend_margin_last5_vs_season_away"] = float(snap_a["last5_margin"] - snap_a["mean_margin"])
        feat_row["trend_margin_last5_vs_season_diff"] = float(
            feat_row["trend_margin_last5_vs_season_home"] - feat_row["trend_margin_last5_vs_season_away"]
        )
        feat_row["trend_oppadj_last5_vs_season_home"] = float(snap_h["last5_oppadj"] - snap_h["oppadj_margin"])
        feat_row["trend_oppadj_last5_vs_season_away"] = float(snap_a["last5_oppadj"] - snap_a["oppadj_margin"])
        feat_row["trend_oppadj_last5_vs_season_diff"] = float(
            feat_row["trend_oppadj_last5_vs_season_home"] - feat_row["trend_oppadj_last5_vs_season_away"]
        )
        feat_row["consistency_ratio_home"] = float(snap_h["consistency_ratio_last5"])
        feat_row["consistency_ratio_away"] = float(snap_a["consistency_ratio_last5"])
        feat_row["consistency_ratio_diff"] = float(snap_h["consistency_ratio_last5"] - snap_a["consistency_ratio_last5"])
        feat_row["low_info_matchup_flag"] = float(min(snap_h["games_played"], snap_a["games_played"]) < 5.0)
        rows.append(feat_row)

        margin = float(row["HomeWinMargin"])
        _update_team_state(st_h, margin_for_team=margin, opp_elo_pre=away_elo_pre, game_date=date)
        _update_team_state(st_a, margin_for_team=-margin, opp_elo_pre=home_elo_pre, game_date=date)

    seq_df = pd.DataFrame(rows, index=train_df.index)
    return SequentialFeatureBuild(features=seq_df, final_states=states, final_elo=elo_res.final_ratings)


def build_derby_sequential_features(
    pred_df: pd.DataFrame,
    final_states: Mapping[int, TeamRollingState],
    final_elo: Mapping[int, float],
) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in pred_df.iterrows():
        date = pd.Timestamp(row["Date"])
        t1 = int(row["Team1_ID"])
        t2 = int(row["Team2_ID"])
        st1 = final_states.get(t1, _default_state())
        st2 = final_states.get(t2, _default_state())
        snap1 = _pregame_team_snapshot(st1, date)
        snap2 = _pregame_team_snapshot(st2, date)
        elo1 = float(final_elo.get(t1, 1500.0))
        elo2 = float(final_elo.get(t2, 1500.0))
        elo_diff = elo1 - elo2  # neutral site: no home advantage
        # Elo probability with neutral site.
        elo_prob = 1.0 / (1.0 + np.power(10.0, -elo_diff / 400.0))

        feat_row = {
            "elo_home_pre": elo1,
            "elo_away_pre": elo2,
            "elo_diff_pre": elo_diff,
            "elo_neutral_diff_pre": elo_diff,
            "elo_prob_home_pre": float(elo_prob),
            "elo_home_edge_pre": 0.0,
            "elo_home_team_dev_pre": 0.0,
            "elo_k_eff_pre": 0.0,
            "elo_gap_home_days": 0.0,
            "elo_gap_away_days": 0.0,
            "elo_gap_days_diff": 0.0,
            "mean_margin_home": snap1["mean_margin"],
            "mean_margin_away": snap2["mean_margin"],
            "mean_margin_diff": snap1["mean_margin"] - snap2["mean_margin"],
            "ema_margin_home": snap1["ema_margin"],
            "ema_margin_away": snap2["ema_margin"],
            "ema_margin_diff": snap1["ema_margin"] - snap2["ema_margin"],
            "oppadj_margin_home": snap1["oppadj_margin"],
            "oppadj_margin_away": snap2["oppadj_margin"],
            "oppadj_margin_diff": snap1["oppadj_margin"] - snap2["oppadj_margin"],
            "sos_elo_home": snap1["sos_elo"],
            "sos_elo_away": snap2["sos_elo"],
            "sos_elo_diff": snap1["sos_elo"] - snap2["sos_elo"],
            "winrate_home": snap1["winrate"],
            "winrate_away": snap2["winrate"],
            "winrate_diff": snap1["winrate"] - snap2["winrate"],
            "volatility_home": snap1["volatility"],
            "volatility_away": snap2["volatility"],
            "volatility_diff": snap1["volatility"] - snap2["volatility"],
            "games_played_home": snap1["games_played"],
            "games_played_away": snap2["games_played"],
            "games_played_diff": snap1["games_played"] - snap2["games_played"],
            "rest_days_home": snap1["rest_days"],
            "rest_days_away": snap2["rest_days"],
            "rest_days_diff": snap1["rest_days"] - snap2["rest_days"],
        }
        for a in EMA_ALPHAS:
            tag = str(int(round(a * 100)))
            for stem in ["ema_margin", "ema_oppadj", "ema_winrate"]:
                hk = f"{stem}_a{tag}"
                feat_row[f"{hk}_home"] = float(snap1[hk])
                feat_row[f"{hk}_away"] = float(snap2[hk])
                feat_row[f"{hk}_diff"] = float(snap1[hk] - snap2[hk])
        for k in LAST_KS:
            for stem in ["margin", "oppadj", "winrate", "volatility"]:
                hk = f"last{k}_{stem}"
                feat_row[f"{hk}_home"] = float(snap1[hk])
                feat_row[f"{hk}_away"] = float(snap2[hk])
                feat_row[f"{hk}_diff"] = float(snap1[hk] - snap2[hk])
        feat_row["trend_margin_last3_vs_season_home"] = float(snap1["last3_margin"] - snap1["mean_margin"])
        feat_row["trend_margin_last3_vs_season_away"] = float(snap2["last3_margin"] - snap2["mean_margin"])
        feat_row["trend_margin_last3_vs_season_diff"] = float(
            feat_row["trend_margin_last3_vs_season_home"] - feat_row["trend_margin_last3_vs_season_away"]
        )
        feat_row["trend_margin_last5_vs_season_home"] = float(snap1["last5_margin"] - snap1["mean_margin"])
        feat_row["trend_margin_last5_vs_season_away"] = float(snap2["last5_margin"] - snap2["mean_margin"])
        feat_row["trend_margin_last5_vs_season_diff"] = float(
            feat_row["trend_margin_last5_vs_season_home"] - feat_row["trend_margin_last5_vs_season_away"]
        )
        feat_row["trend_oppadj_last5_vs_season_home"] = float(snap1["last5_oppadj"] - snap1["oppadj_margin"])
        feat_row["trend_oppadj_last5_vs_season_away"] = float(snap2["last5_oppadj"] - snap2["oppadj_margin"])
        feat_row["trend_oppadj_last5_vs_season_diff"] = float(
            feat_row["trend_oppadj_last5_vs_season_home"] - feat_row["trend_oppadj_last5_vs_season_away"]
        )
        feat_row["consistency_ratio_home"] = float(snap1["consistency_ratio_last5"])
        feat_row["consistency_ratio_away"] = float(snap2["consistency_ratio_last5"])
        feat_row["consistency_ratio_diff"] = float(snap1["consistency_ratio_last5"] - snap2["consistency_ratio_last5"])
        feat_row["low_info_matchup_flag"] = float(min(snap1["games_played"], snap2["games_played"]) < 5.0)
        rows.append(feat_row)
    return pd.DataFrame(rows, index=pred_df.index)


def fit_static_models_for_fold(
    fit_games: pd.DataFrame,
    team_ids: Sequence[int],
    conf_values: Sequence[str],
    massey_alpha: float = 30.0,
    offdef_alpha: float = 20.0,
    conf_alpha: float = 10.0,
) -> StaticModelsBundle:
    massey = fit_massey_ridge(fit_games, team_ids=team_ids, alpha=massey_alpha)
    offdef = fit_offense_defense_ridge(fit_games, team_ids=team_ids, alpha=offdef_alpha)
    conf = fit_conference_strength_ridge(fit_games, conf_values=conf_values, alpha=conf_alpha)
    return StaticModelsBundle(massey=massey, offdef=offdef, conf=conf)


def apply_static_models_to_train_like_rows(
    rows: pd.DataFrame,
    models: StaticModelsBundle,
    home_id_col: str = "HomeID",
    away_id_col: str = "AwayID",
    home_conf_col: str = "HomeConf",
    away_conf_col: str = "AwayConf",
    neutral_site: bool = False,
) -> pd.DataFrame:
    massey_df = apply_massey_features(rows, models.massey, home_id_col, away_id_col, neutral_site=neutral_site)
    offdef_df = apply_offdef_features(rows, models.offdef, home_id_col, away_id_col, neutral_site=neutral_site)
    conf_df = apply_conference_features(rows, models.conf, home_conf_col, away_conf_col, neutral_site=neutral_site)
    return pd.concat([massey_df, offdef_df, conf_df], axis=1)


def apply_static_models_to_derby_rows(
    rows: pd.DataFrame,
    models: StaticModelsBundle,
) -> pd.DataFrame:
    massey_df = apply_massey_features(rows, models.massey, "Team1_ID", "Team2_ID", neutral_site=True)
    offdef_df = apply_offdef_features(rows, models.offdef, "Team1_ID", "Team2_ID", neutral_site=True)
    conf_df = apply_conference_features(rows, models.conf, "Team1_Conf", "Team2_Conf", neutral_site=True)
    return pd.concat([massey_df, offdef_df, conf_df], axis=1)


def assemble_model_table(base_rows: pd.DataFrame, sequential_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat(
        [
            base_rows.reset_index(drop=True),
            sequential_df.reset_index(drop=True),
            static_df.reset_index(drop=True),
        ],
        axis=1,
    )
    # Safe conf representation: one-hot per side (no ordinal encoding).
    conf_cols = []
    if "HomeConf" in base_rows.columns and "AwayConf" in base_rows.columns:
        conf_home_d = pd.get_dummies(base_rows["HomeConf"].astype(str), prefix="conf_home")
        conf_away_d = pd.get_dummies(base_rows["AwayConf"].astype(str), prefix="conf_away")
        conf_cols = [conf_home_d, conf_away_d]
    elif "Team1_Conf" in base_rows.columns and "Team2_Conf" in base_rows.columns:
        conf_home_d = pd.get_dummies(base_rows["Team1_Conf"].astype(str), prefix="conf_home")
        conf_away_d = pd.get_dummies(base_rows["Team2_Conf"].astype(str), prefix="conf_away")
        conf_cols = [conf_home_d, conf_away_d]
    if conf_cols:
        conf_df = pd.concat(conf_cols, axis=1).reset_index(drop=True)
        df = pd.concat([df, conf_df], axis=1)
    # Pregame-only interaction features for regime-aware and nonlinear models.
    if "HomeConf" in base_rows.columns and "AwayConf" in base_rows.columns:
        same_conf = (base_rows["HomeConf"].astype(str).to_numpy() == base_rows["AwayConf"].astype(str).to_numpy()).astype(float)
        df["same_conf_flag"] = same_conf
        df["cross_conf_flag"] = 1.0 - same_conf
    elif "Team1_Conf" in base_rows.columns and "Team2_Conf" in base_rows.columns:
        same_conf = (base_rows["Team1_Conf"].astype(str).to_numpy() == base_rows["Team2_Conf"].astype(str).to_numpy()).astype(float)
        df["same_conf_flag"] = same_conf
        df["cross_conf_flag"] = 1.0 - same_conf
    if "elo_diff_pre" in df.columns:
        abs_elo = np.abs(df["elo_diff_pre"].astype(float).to_numpy())
        df["elo_abs_pre"] = abs_elo
        df["elo_close_flag"] = (abs_elo < 35.0).astype(float)
        df["elo_moderate_flag"] = ((abs_elo >= 35.0) & (abs_elo < 90.0)).astype(float)
        df["elo_mismatch_flag"] = (abs_elo >= 90.0).astype(float)
    if "volatility_home" in df.columns and "volatility_away" in df.columns:
        vh = df["volatility_home"].astype(float).to_numpy()
        va = df["volatility_away"].astype(float).to_numpy()
        vs = vh + va
        df["volatility_sum"] = vs
        df["volatility_ratio_home_away"] = vh / np.maximum(va, 1e-6)
        df["volatility_ratio_away_home"] = va / np.maximum(vh, 1e-6)
    if "elo_diff_pre" in df.columns and "volatility_diff" in df.columns:
        df["vol_x_elo_diff"] = (df["volatility_diff"].astype(float) * df["elo_diff_pre"].astype(float)) / 50.0
    if "same_conf_flag" in df.columns and "elo_diff_pre" in df.columns:
        df["same_conf_x_elo_diff"] = df["same_conf_flag"].astype(float) * df["elo_diff_pre"].astype(float)
        df["cross_conf_x_elo_diff"] = df["cross_conf_flag"].astype(float) * df["elo_diff_pre"].astype(float)
    if "games_played_home" in df.columns and "games_played_away" in df.columns:
        gp_min = np.minimum(df["games_played_home"].astype(float), df["games_played_away"].astype(float))
        gp_max = np.maximum(df["games_played_home"].astype(float), df["games_played_away"].astype(float))
        df["games_played_min"] = gp_min
        df["games_played_max"] = gp_max
        df["low_info_matchup_flag2"] = (gp_min < 5.0).astype(float)
        df["high_info_matchup_flag"] = (gp_min >= 8.0).astype(float)
    if "offdef_margin_with_side" in df.columns and "offdef_margin_neutral" in df.columns:
        df["offdef_home_effect_component"] = (
            df["offdef_margin_with_side"].astype(float) - df["offdef_margin_neutral"].astype(float)
        )
    if "massey_home_rating" in df.columns and "massey_away_rating" in df.columns:
        df["massey_neutral_component"] = (
            df["massey_home_rating"].astype(float) - df["massey_away_rating"].astype(float)
        )
    return df

