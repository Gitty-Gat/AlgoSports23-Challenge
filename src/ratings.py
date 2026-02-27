from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ELO_SCALE = 400.0
ELO_BASE = 10.0


def elo_expected_score(elo_diff: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.power(ELO_BASE, -np.asarray(elo_diff, dtype=float) / ELO_SCALE))


@dataclass(frozen=True)
class MasseyModel:
    team_rating: Dict[int, float]
    home_adv: float
    intercept: float
    alpha: float


@dataclass(frozen=True)
class EloModel:
    ratings: Dict[int, float]
    games_played: Dict[int, int]
    home_adv: float
    k_factor: float
    decay_a: float
    decay_g: float


def fit_massey_ridge(games: pd.DataFrame, team_ids: Sequence[int], alpha: float = 10.0) -> MasseyModel:
    team_ids = [int(t) for t in team_ids]
    team_to_pos = {tid: i for i, tid in enumerate(team_ids)}
    n_games = len(games)
    n_teams = len(team_ids)
    X = np.zeros((n_games, n_teams + 2), dtype=float)
    y = games["HomeWinMargin"].to_numpy(dtype=float)
    for i, row in enumerate(games.itertuples(index=False)):
        X[i, team_to_pos[int(row.HomeID)]] = 1.0
        X[i, team_to_pos[int(row.AwayID)]] = -1.0
        X[i, n_teams] = 1.0
        X[i, n_teams + 1] = 1.0
    model = Ridge(alpha=float(alpha), fit_intercept=False, random_state=23)
    model.fit(X, y)
    coef = np.asarray(model.coef_, dtype=float)
    ratings = coef[:n_teams].copy()
    if n_teams > 0:
        ratings -= float(np.mean(ratings))
    return MasseyModel(
        team_rating={tid: float(ratings[pos]) for tid, pos in team_to_pos.items()},
        home_adv=float(coef[n_teams]),
        intercept=float(coef[n_teams + 1]),
        alpha=float(alpha),
    )


def apply_massey_features(
    rows: pd.DataFrame,
    model: MasseyModel,
    home_id_col: str,
    away_id_col: str,
    neutral_site: bool = False,
) -> pd.DataFrame:
    home_r = rows[home_id_col].astype(int).map(model.team_rating).fillna(0.0).astype(float)
    away_r = rows[away_id_col].astype(int).map(model.team_rating).fillna(0.0).astype(float)
    h = 0.0 if neutral_site else float(model.home_adv)
    diff = home_r - away_r + h + float(model.intercept)
    return pd.DataFrame(
        {
            "massey_home_rating": home_r.to_numpy(),
            "massey_away_rating": away_r.to_numpy(),
            "massey_diff": diff.to_numpy(),
        },
        index=rows.index,
    )


def _elo_k_eff(k_factor: float, decay_a: float, decay_g: float, gp_home: int, gp_away: int) -> float:
    base = float(k_factor)
    if decay_a <= 0 or decay_g <= 0:
        return max(base, 1e-6)
    gp_pair = 0.5 * (float(gp_home) + float(gp_away))
    frac = max(0.0, 1.0 - min(gp_pair, float(decay_g)) / float(decay_g))
    return max(base * (1.0 + float(decay_a) * frac), 1e-6)


def run_elo_over_games(
    games: pd.DataFrame,
    home_id_col: str,
    away_id_col: str,
    margin_col: str,
    home_adv: float,
    k_factor: float = 24.0,
    initial_rating: float = 1500.0,
    decay_a: float = 0.0,
    decay_g: float = 80.0,
    starting_ratings: Optional[Mapping[int, float]] = None,
    starting_games_played: Optional[Mapping[int, int]] = None,
    update_ratings: bool = True,
) -> tuple[pd.DataFrame, EloModel]:
    ratings: Dict[int, float] = dict(starting_ratings or {})
    games_played: Dict[int, int] = {int(k): int(v) for k, v in dict(starting_games_played or {}).items()}

    pre_home = []
    pre_away = []
    diff_pre = []
    diff_neutral = []
    prob_home = []
    k_eff_vals = []

    for row in games.itertuples(index=False):
        home_id = int(getattr(row, home_id_col))
        away_id = int(getattr(row, away_id_col))

        h_elo = float(ratings.get(home_id, initial_rating))
        a_elo = float(ratings.get(away_id, initial_rating))
        gp_h = int(games_played.get(home_id, 0))
        gp_a = int(games_played.get(away_id, 0))

        d_neutral = h_elo - a_elo
        d = d_neutral + float(home_adv)
        p_home = float(elo_expected_score(d))
        k_eff = _elo_k_eff(k_factor=float(k_factor), decay_a=float(decay_a), decay_g=float(decay_g), gp_home=gp_h, gp_away=gp_a)

        pre_home.append(h_elo)
        pre_away.append(a_elo)
        diff_pre.append(d)
        diff_neutral.append(d_neutral)
        prob_home.append(p_home)
        k_eff_vals.append(k_eff)

        if update_ratings:
            margin = float(getattr(row, margin_col))
            score = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            delta = k_eff * (score - p_home)
            ratings[home_id] = h_elo + delta
            ratings[away_id] = a_elo - delta
            games_played[home_id] = gp_h + 1
            games_played[away_id] = gp_a + 1

    features = pd.DataFrame(
        {
            "elo_home_pre": pre_home,
            "elo_away_pre": pre_away,
            "elo_diff_pre": diff_pre,
            "elo_neutral_diff_pre": diff_neutral,
            "elo_prob_home_pre": prob_home,
            "elo_k_eff_pre": k_eff_vals,
        },
        index=games.index,
    )
    model = EloModel(
        ratings={int(k): float(v) for k, v in ratings.items()},
        games_played={int(k): int(v) for k, v in games_played.items()},
        home_adv=float(home_adv),
        k_factor=float(k_factor),
        decay_a=float(decay_a),
        decay_g=float(decay_g),
    )
    return features, model


def apply_elo_features(
    rows: pd.DataFrame,
    model: EloModel,
    home_id_col: str,
    away_id_col: str,
    neutral_site: bool = False,
    initial_rating: float = 1500.0,
) -> pd.DataFrame:
    h = rows[home_id_col].astype(int).map(model.ratings).fillna(float(initial_rating)).astype(float)
    a = rows[away_id_col].astype(int).map(model.ratings).fillna(float(initial_rating)).astype(float)
    d_neutral = h - a
    d = d_neutral if neutral_site else d_neutral + float(model.home_adv)
    p_home = elo_expected_score(d)
    return pd.DataFrame(
        {
            "elo_home_pre": h.to_numpy(),
            "elo_away_pre": a.to_numpy(),
            "elo_diff_pre": d.to_numpy(),
            "elo_neutral_diff_pre": d_neutral.to_numpy(),
            "elo_prob_home_pre": np.asarray(p_home, dtype=float),
            "elo_k_eff_pre": np.full(len(rows), float(model.k_factor), dtype=float),
        },
        index=rows.index,
    )


def tune_home_advantage_elo(
    train_df: pd.DataFrame,
    folds: Iterable[Mapping[str, np.ndarray]],
    candidate_values: Iterable[int],
    k_factor: float = 24.0,
) -> tuple[int, pd.DataFrame]:
    y = (train_df["HomeWinMargin"].to_numpy(dtype=float) > 0.0).astype(float)
    rows = []
    for ha in candidate_values:
        feat, _ = run_elo_over_games(
            train_df,
            home_id_col="HomeID",
            away_id_col="AwayID",
            margin_col="HomeWinMargin",
            home_adv=float(ha),
            k_factor=float(k_factor),
            update_ratings=True,
        )
        p = feat["elo_prob_home_pre"].to_numpy(dtype=float)
        fold_scores = []
        for fold in folds:
            val_idx = np.asarray(fold["val_idx"], dtype=int)
            pb = p[val_idx]
            yb = y[val_idx]
            brier = float(np.mean((pb - yb) ** 2))
            fold_scores.append(brier)
            rows.append({"home_adv": int(ha), "fold": int(fold["fold"]), "brier": brier})
        rows.append({"home_adv": int(ha), "fold": -1, "brier": float(np.mean(fold_scores))})
    detail = pd.DataFrame(rows)
    summary = (
        detail[detail["fold"] >= 0]
        .groupby("home_adv", as_index=False)
        .agg(brier_mean=("brier", "mean"), brier_std=("brier", "std"))
        .sort_values(["brier_mean", "brier_std", "home_adv"], kind="mergesort")
        .reset_index(drop=True)
    )
    return int(summary.iloc[0]["home_adv"]), detail
