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


@dataclass(frozen=True)
class EloVariantConfig:
    name: str = "base"
    home_adv: float = 0.0
    k_factor: float = 24.0
    initial_rating: float = 1500.0
    use_dynamic_k: bool = False
    k_early_multiplier: float = 0.75
    k_games_scale: float = 18.0
    k_uncertainty_multiplier: float = 0.35
    use_team_home_adv: bool = False
    team_home_adv_scale: float = 70.0
    team_home_adv_reg: float = 6.0
    team_home_adv_cap: float = 50.0
    use_inactivity_decay: bool = False
    inactivity_tau_days: float = 120.0
    use_mov_multiplier: bool = True


def _decay_rating_toward_mean(rating: float, gap_days: float, tau_days: float, mean: float = 1500.0) -> float:
    if tau_days <= 0 or gap_days <= 0:
        return float(rating)
    w = float(np.exp(-float(gap_days) / float(tau_days)))
    return float(mean + w * (float(rating) - mean))


def _team_home_dev_elo(
    team_id: int,
    resid_sum: Mapping[int, float],
    resid_n: Mapping[int, int],
    scale: float,
    reg: float,
    cap: float,
) -> float:
    n = float(resid_n.get(int(team_id), 0))
    s = float(resid_sum.get(int(team_id), 0.0))
    shrunk = s / max(n + float(reg), 1.0)
    dev = float(scale) * shrunk
    return float(np.clip(dev, -abs(float(cap)), abs(float(cap))))


def run_elo_over_games_advanced(
    games: pd.DataFrame,
    home_id_col: str,
    away_id_col: str,
    margin_col: str,
    *,
    config: EloVariantConfig,
    date_col: str = "Date",
    starting_ratings: Optional[Mapping[int, float]] = None,
    update_ratings: bool = True,
) -> EloResult:
    ratings: Dict[int, float] = dict(starting_ratings or {})
    games_played: Dict[int, int] = {}
    last_seen: Dict[int, pd.Timestamp] = {}
    home_resid_sum: Dict[int, float] = {}
    home_resid_n: Dict[int, int] = {}

    pre_home = []
    pre_away = []
    elo_diff_pre = []
    elo_diff_neutral = []
    elo_prob_home = []
    elo_home_edge = []
    elo_home_dev = []
    elo_k_eff = []
    gap_home_days = []
    gap_away_days = []

    for row in games.itertuples(index=False):
        home_id = int(getattr(row, home_id_col))
        away_id = int(getattr(row, away_id_col))
        dt_raw = getattr(row, date_col, None)
        game_date = pd.Timestamp(dt_raw) if dt_raw is not None else pd.NaT

        home_gap = 0.0
        away_gap = 0.0
        if pd.notna(game_date):
            if config.use_inactivity_decay and config.inactivity_tau_days > 0:
                if home_id in last_seen:
                    home_gap = float(max((game_date - last_seen[home_id]).days, 0))
                    ratings[home_id] = _decay_rating_toward_mean(
                        ratings.get(home_id, config.initial_rating),
                        gap_days=home_gap,
                        tau_days=config.inactivity_tau_days,
                        mean=config.initial_rating,
                    )
                if away_id in last_seen:
                    away_gap = float(max((game_date - last_seen[away_id]).days, 0))
                    ratings[away_id] = _decay_rating_toward_mean(
                        ratings.get(away_id, config.initial_rating),
                        gap_days=away_gap,
                        tau_days=config.inactivity_tau_days,
                        mean=config.initial_rating,
                    )
            else:
                if home_id in last_seen:
                    home_gap = float(max((game_date - last_seen[home_id]).days, 0))
                if away_id in last_seen:
                    away_gap = float(max((game_date - last_seen[away_id]).days, 0))

        home_elo = float(ratings.get(home_id, config.initial_rating))
        away_elo = float(ratings.get(away_id, config.initial_rating))
        diff_neutral = float(home_elo - away_elo)

        team_dev = 0.0
        if config.use_team_home_adv:
            team_dev = _team_home_dev_elo(
                home_id,
                resid_sum=home_resid_sum,
                resid_n=home_resid_n,
                scale=config.team_home_adv_scale,
                reg=config.team_home_adv_reg,
                cap=config.team_home_adv_cap,
            )
        home_edge = float(config.home_adv + team_dev)
        diff_ha = float(diff_neutral + home_edge)
        p_home = float(elo_expected_score(diff_ha))

        k_eff = float(config.k_factor)
        if config.use_dynamic_k:
            gp_home = float(games_played.get(home_id, 0))
            gp_away = float(games_played.get(away_id, 0))
            gp_pair = 0.5 * (gp_home + gp_away)
            early_mult = 1.0 + float(config.k_early_multiplier) * float(
                np.exp(-gp_pair / max(float(config.k_games_scale), 1e-6))
            )
            uncertainty = 4.0 * p_home * (1.0 - p_home)  # [0,1], highest on coin flips
            unc_mult = 1.0 + float(config.k_uncertainty_multiplier) * float(uncertainty)
            k_eff *= early_mult * unc_mult
        k_eff = float(np.clip(k_eff, 4.0, 80.0))

        pre_home.append(home_elo)
        pre_away.append(away_elo)
        elo_diff_pre.append(diff_ha)
        elo_diff_neutral.append(diff_neutral)
        elo_prob_home.append(p_home)
        elo_home_edge.append(home_edge)
        elo_home_dev.append(team_dev)
        elo_k_eff.append(k_eff)
        gap_home_days.append(home_gap)
        gap_away_days.append(away_gap)

        if update_ratings:
            margin = float(getattr(row, margin_col))
            score_home = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            mult = elo_mov_multiplier(margin, diff_ha) if config.use_mov_multiplier else 1.0
            delta = k_eff * float(mult) * (score_home - p_home)
            ratings[home_id] = home_elo + delta
            ratings[away_id] = away_elo - delta

            if config.use_team_home_adv:
                # Estimate a team-specific home bump from binary surprise at home, shrunk in `_team_home_dev_elo`.
                p_neutral = float(elo_expected_score(diff_neutral))
                home_resid_sum[home_id] = float(home_resid_sum.get(home_id, 0.0) + (score_home - p_neutral))
                home_resid_n[home_id] = int(home_resid_n.get(home_id, 0) + 1)

            games_played[home_id] = int(games_played.get(home_id, 0) + 1)
            games_played[away_id] = int(games_played.get(away_id, 0) + 1)
            if pd.notna(game_date):
                last_seen[home_id] = game_date
                last_seen[away_id] = game_date

    features = pd.DataFrame(
        {
            "elo_home_pre": pre_home,
            "elo_away_pre": pre_away,
            "elo_diff_pre": elo_diff_pre,
            "elo_neutral_diff_pre": elo_diff_neutral,
            "elo_prob_home_pre": elo_prob_home,
            "elo_home_edge_pre": elo_home_edge,
            "elo_home_team_dev_pre": elo_home_dev,
            "elo_k_eff_pre": elo_k_eff,
            "elo_gap_home_days": gap_home_days,
            "elo_gap_away_days": gap_away_days,
            "elo_gap_days_diff": (np.asarray(gap_home_days, dtype=float) - np.asarray(gap_away_days, dtype=float)),
        },
        index=games.index,
    )
    return EloResult(game_features=features, final_ratings=ratings)


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
