from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.ratings import EloModel, MasseyModel, apply_elo_features, apply_massey_features, fit_massey_ridge, run_elo_over_games


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
