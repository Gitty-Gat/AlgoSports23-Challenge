from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ELO_SCALE = 400.0
ELO_BASE = 10.0


def elo_expected_score(elo_diff: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.power(ELO_BASE, -np.asarray(elo_diff) / ELO_SCALE))


def elo_mov_multiplier(margin: float, elo_diff_with_ha: float) -> float:
    # Smooth bounded multiplier; boosts decisive wins without exploding on outliers.
    mult = ((abs(float(margin)) + 3.0) ** 0.8) / (7.5 + 0.006 * abs(float(elo_diff_with_ha)))
    return float(np.clip(mult, 0.6, 2.4))


@dataclass
class EloResult:
    game_features: pd.DataFrame
    final_ratings: Dict[int, float]


def run_elo_over_games(
    games: pd.DataFrame,
    home_id_col: str,
    away_id_col: str,
    margin_col: str,
    home_adv: float,
    k_factor: float = 24.0,
    initial_rating: float = 1500.0,
    starting_ratings: Optional[Mapping[int, float]] = None,
    update_ratings: bool = True,
) -> EloResult:
    ratings: Dict[int, float] = dict(starting_ratings or {})
    pre_home = []
    pre_away = []
    elo_diff_pre = []
    win_prob_home = []

    for row in games.itertuples(index=False):
        home_id = int(getattr(row, home_id_col))
        away_id = int(getattr(row, away_id_col))
        home_elo = float(ratings.get(home_id, initial_rating))
        away_elo = float(ratings.get(away_id, initial_rating))
        diff_ha = home_elo - away_elo + float(home_adv)
        p_home = float(elo_expected_score(diff_ha))

        pre_home.append(home_elo)
        pre_away.append(away_elo)
        elo_diff_pre.append(diff_ha)
        win_prob_home.append(p_home)

        if update_ratings:
            margin = float(getattr(row, margin_col))
            score_home = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            mult = elo_mov_multiplier(margin, diff_ha)
            delta = k_factor * mult * (score_home - p_home)
            ratings[home_id] = home_elo + delta
            ratings[away_id] = away_elo - delta

    features = pd.DataFrame(
        {
            "elo_home_pre": pre_home,
            "elo_away_pre": pre_away,
            "elo_diff_pre": elo_diff_pre,
            "elo_prob_home_pre": win_prob_home,
        },
        index=games.index,
    )
    return EloResult(game_features=features, final_ratings=ratings)


def tune_home_advantage_elo(
    train_df: pd.DataFrame,
    folds: Sequence[Mapping[str, np.ndarray]],
    candidate_values: Iterable[int],
    k_factor: float = 24.0,
) -> Tuple[int, pd.DataFrame]:
    y = (train_df["HomeWinMargin"].to_numpy() > 0).astype(float)
    records: List[dict] = []

    for ha in candidate_values:
        elo_res = run_elo_over_games(
            train_df,
            home_id_col="HomeID",
            away_id_col="AwayID",
            margin_col="HomeWinMargin",
            home_adv=float(ha),
            k_factor=k_factor,
            update_ratings=True,
        )
        p = elo_res.game_features["elo_prob_home_pre"].to_numpy()
        fold_briers = []
        for fold in folds:
            val_idx = fold["val_idx"]
            pb = p[val_idx]
            yb = y[val_idx]
            fold_brier = float(np.mean((pb - yb) ** 2))
            eps = 1e-12
            fold_logloss = float(-np.mean(yb * np.log(np.clip(pb, eps, 1 - eps)) + (1 - yb) * np.log(np.clip(1 - pb, eps, 1 - eps))))
            fold_briers.append(fold_brier)
            records.append(
                {
                    "home_adv": int(ha),
                    "fold": int(fold["fold"]),
                    "brier": fold_brier,
                    "logloss": fold_logloss,
                }
            )
        records.append(
            {
                "home_adv": int(ha),
                "fold": -1,
                "brier": float(np.mean(fold_briers)),
                "logloss": np.nan,
            }
        )
    score_df = pd.DataFrame(records)
    summary = (
        score_df[score_df["fold"] >= 0]
        .groupby("home_adv", as_index=False)
        .agg(brier_mean=("brier", "mean"), brier_std=("brier", "std"), logloss_mean=("logloss", "mean"))
        .sort_values(["brier_mean", "brier_std", "home_adv"], kind="mergesort")
        .reset_index(drop=True)
    )
    best_ha = int(summary.iloc[0]["home_adv"])
    detail = score_df.merge(summary, on="home_adv", how="left")
    return best_ha, detail


@dataclass
class RidgeRatingsResult:
    team_rating: Dict[int, float]
    home_adv: float
    intercept: float


def _fit_sparse_like_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=alpha, fit_intercept=False, random_state=0)
    model.fit(X, y)
    return np.asarray(model.coef_, dtype=float)


def fit_massey_ridge(
    games: pd.DataFrame,
    team_ids: Sequence[int],
    alpha: float = 30.0,
) -> RidgeRatingsResult:
    team_ids = list(team_ids)
    team_to_pos = {int(t): i for i, t in enumerate(team_ids)}
    n_games = len(games)
    n_teams = len(team_ids)
    X = np.zeros((n_games, n_teams + 2), dtype=float)  # team ratings + home_adv + intercept
    y = games["HomeWinMargin"].to_numpy(dtype=float)
    for i, row in enumerate(games.itertuples(index=False)):
        X[i, team_to_pos[int(row.HomeID)]] = 1.0
        X[i, team_to_pos[int(row.AwayID)]] = -1.0
        X[i, n_teams] = 1.0  # home advantage
        X[i, n_teams + 1] = 1.0  # intercept
    coef = _fit_sparse_like_ridge(X, y, alpha=alpha)
    ratings = coef[:n_teams].copy()
    ratings -= ratings.mean() if n_teams else 0.0
    team_rating = {tid: float(ratings[pos]) for tid, pos in team_to_pos.items()}
    return RidgeRatingsResult(
        team_rating=team_rating,
        home_adv=float(coef[n_teams]),
        intercept=float(coef[n_teams + 1]),
    )


def apply_massey_features(
    rows: pd.DataFrame,
    fit_result: RidgeRatingsResult,
    home_id_col: str,
    away_id_col: str,
    neutral_site: bool = False,
) -> pd.DataFrame:
    home_r = rows[home_id_col].map(fit_result.team_rating).fillna(0.0).astype(float)
    away_r = rows[away_id_col].map(fit_result.team_rating).fillna(0.0).astype(float)
    h = 0.0 if neutral_site else fit_result.home_adv
    diff = home_r - away_r + h + fit_result.intercept
    return pd.DataFrame(
        {
            "massey_home_rating": home_r.to_numpy(),
            "massey_away_rating": away_r.to_numpy(),
            "massey_diff": diff.to_numpy(),
        },
        index=rows.index,
    )


@dataclass
class OffDefRidgeResult:
    offense: Dict[int, float]
    defense: Dict[int, float]
    home_adv_points: float
    base_points: float

    def net_rating_map(self) -> Dict[int, float]:
        keys = set(self.offense) | set(self.defense)
        return {k: float(self.offense.get(k, 0.0) - self.defense.get(k, 0.0)) for k in keys}


def fit_offense_defense_ridge(
    games: pd.DataFrame,
    team_ids: Sequence[int],
    alpha: float = 20.0,
) -> OffDefRidgeResult:
    team_ids = list(team_ids)
    team_to_pos = {int(t): i for i, t in enumerate(team_ids)}
    n_teams = len(team_ids)
    n_rows = len(games) * 2
    # columns: offense[n], defense[n], home_indicator, intercept
    X = np.zeros((n_rows, 2 * n_teams + 2), dtype=float)
    y = np.zeros(n_rows, dtype=float)

    r = 0
    for row in games.itertuples(index=False):
        h = team_to_pos[int(row.HomeID)]
        a = team_to_pos[int(row.AwayID)]

        # Home points row
        X[r, h] = 1.0
        X[r, n_teams + a] = -1.0
        X[r, 2 * n_teams] = 1.0  # home scoring boost
        X[r, 2 * n_teams + 1] = 1.0
        y[r] = float(row.HomePts)
        r += 1

        # Away points row
        X[r, a] = 1.0
        X[r, n_teams + h] = -1.0
        X[r, 2 * n_teams] = 0.0
        X[r, 2 * n_teams + 1] = 1.0
        y[r] = float(row.AwayPts)
        r += 1

    coef = _fit_sparse_like_ridge(X, y, alpha=alpha)
    offense = coef[:n_teams].copy()
    defense = coef[n_teams : 2 * n_teams].copy()
    offense -= offense.mean() if n_teams else 0.0
    defense -= defense.mean() if n_teams else 0.0

    off_map = {tid: float(offense[pos]) for tid, pos in team_to_pos.items()}
    def_map = {tid: float(defense[pos]) for tid, pos in team_to_pos.items()}
    return OffDefRidgeResult(
        offense=off_map,
        defense=def_map,
        home_adv_points=float(coef[2 * n_teams]),
        base_points=float(coef[2 * n_teams + 1]),
    )


def apply_offdef_features(
    rows: pd.DataFrame,
    fit_result: OffDefRidgeResult,
    home_id_col: str,
    away_id_col: str,
    neutral_site: bool = False,
) -> pd.DataFrame:
    h_ids = rows[home_id_col].astype(int)
    a_ids = rows[away_id_col].astype(int)
    h_off = h_ids.map(fit_result.offense).fillna(0.0).astype(float)
    a_off = a_ids.map(fit_result.offense).fillna(0.0).astype(float)
    h_def = h_ids.map(fit_result.defense).fillna(0.0).astype(float)
    a_def = a_ids.map(fit_result.defense).fillna(0.0).astype(float)
    home_pts_neutral = fit_result.base_points + h_off - a_def
    away_pts_neutral = fit_result.base_points + a_off - h_def
    pred_margin_neutral = home_pts_neutral - away_pts_neutral
    pred_margin_train_side = pred_margin_neutral + (0.0 if neutral_site else fit_result.home_adv_points)
    net_h = h_off - h_def
    net_a = a_off - a_def

    return pd.DataFrame(
        {
            "offdef_home_offense": h_off.to_numpy(),
            "offdef_away_offense": a_off.to_numpy(),
            "offdef_home_defense": h_def.to_numpy(),
            "offdef_away_defense": a_def.to_numpy(),
            "offdef_net_home": net_h.to_numpy(),
            "offdef_net_away": net_a.to_numpy(),
            "offdef_net_diff": (net_h - net_a).to_numpy(),
            "offdef_margin_neutral": pred_margin_neutral.to_numpy(),
            "offdef_margin_with_side": pred_margin_train_side.to_numpy(),
        },
        index=rows.index,
    )


@dataclass
class ConferenceStrengthResult:
    conf_strength: Dict[str, float]
    home_adv: float
    intercept: float


def fit_conference_strength_ridge(
    games: pd.DataFrame,
    conf_values: Sequence[str],
    alpha: float = 10.0,
) -> ConferenceStrengthResult:
    conf_values = list(conf_values)
    conf_to_pos = {c: i for i, c in enumerate(conf_values)}
    n_conf = len(conf_values)
    X = np.zeros((len(games), n_conf + 2), dtype=float)
    y = games["HomeWinMargin"].to_numpy(dtype=float)
    for i, row in enumerate(games.itertuples(index=False)):
        X[i, conf_to_pos[str(row.HomeConf)]] = 1.0
        X[i, conf_to_pos[str(row.AwayConf)]] = -1.0
        X[i, n_conf] = 1.0
        X[i, n_conf + 1] = 1.0
    coef = _fit_sparse_like_ridge(X, y, alpha=alpha)
    c = coef[:n_conf].copy()
    c -= c.mean() if n_conf else 0.0
    return ConferenceStrengthResult(
        conf_strength={conf: float(c[pos]) for conf, pos in conf_to_pos.items()},
        home_adv=float(coef[n_conf]),
        intercept=float(coef[n_conf + 1]),
    )


def apply_conference_features(
    rows: pd.DataFrame,
    fit_result: ConferenceStrengthResult,
    home_conf_col: str,
    away_conf_col: str,
    neutral_site: bool = False,
) -> pd.DataFrame:
    hc = rows[home_conf_col].astype(str).map(fit_result.conf_strength).fillna(0.0).astype(float)
    ac = rows[away_conf_col].astype(str).map(fit_result.conf_strength).fillna(0.0).astype(float)
    h = 0.0 if neutral_site else fit_result.home_adv
    conf_margin = hc - ac + h + fit_result.intercept
    return pd.DataFrame(
        {
            "conf_strength_home": hc.to_numpy(),
            "conf_strength_away": ac.to_numpy(),
            "conf_strength_diff": conf_margin.to_numpy(),
        },
        index=rows.index,
    )
