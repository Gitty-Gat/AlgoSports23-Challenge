from __future__ import annotations

import textwrap
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _make_figure(figsize=(10, 6)):
    fig = plt.figure(figsize=figsize, dpi=120)
    return fig


def _df_preview(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.head(n).copy()


def build_report_bundle(
    *,
    seed: int,
    data_summary: Mapping[str, object],
    home_adv_result: Mapping[str, object],
    cv_outputs: Mapping[str, object],
    chosen_config: Mapping[str, object],
    final_outputs: Mapping[str, object],
) -> Dict[str, object]:
    config_summary: pd.DataFrame = cv_outputs["config_summary"]
    fold_metrics: pd.DataFrame = cv_outputs["fold_metrics"]
    oof_predictions: pd.DataFrame = cv_outputs["oof_predictions"]

    chosen_name = str(chosen_config["chosen_config_name"])
    chosen_oof = oof_predictions[oof_predictions["config_name"] == chosen_name].copy()
    chosen_oof = chosen_oof.sort_values(["fold", "row_index"], kind="mergesort").reset_index(drop=True)
    chosen_oof["residual"] = chosen_oof["y_true"] - chosen_oof["y_pred"]

    figures: List[dict] = []

    # 1) CV fold RMSE/MAE by configuration (top subset for readability)
    top_cfgs = config_summary.nsmallest(min(10, len(config_summary)), "rmse_mean")["config_name"].tolist()
    fm_top = fold_metrics[fold_metrics["config_name"].isin(top_cfgs)].copy()
    fig = _make_figure((12, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    rmse_pivot = fm_top.pivot(index="fold", columns="config_name", values="rmse")
    rmse_pivot.plot(ax=ax1, marker="o")
    ax1.set_title("CV Fold RMSE (Top Configs)")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("RMSE")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7, loc="best")
    ax2 = fig.add_subplot(1, 2, 2)
    mae_pivot = fm_top.pivot(index="fold", columns="config_name", values="mae")
    mae_pivot.plot(ax=ax2, marker="o")
    ax2.set_title("CV Fold MAE (Top Configs)")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("MAE")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7, loc="best")
    fig.tight_layout()
    figures.append({"title": "CV Fold Metrics", "figure": fig})

    # 2) Residual histogram + QQ-style plot
    fig = _make_figure((12, 5.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(chosen_oof["residual"], bins=30, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax1.axvline(0, color="black", lw=1)
    ax1.set_title(f"Residual Histogram ({chosen_name})")
    ax1.set_xlabel("Residual = Actual - Predicted")
    ax1.set_ylabel("Count")
    ax1.grid(alpha=0.2)
    ax2 = fig.add_subplot(1, 2, 2)
    (osm, osr), (slope, intercept, r) = stats.probplot(chosen_oof["residual"].to_numpy(), dist="norm")
    ax2.scatter(osm, osr, s=14, alpha=0.7, color="#F58518")
    xline = np.array([np.min(osm), np.max(osm)])
    ax2.plot(xline, intercept + slope * xline, color="black", lw=1)
    ax2.set_title(f"QQ-Style Residual Plot (r={r:.3f})")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Ordered Residuals")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    figures.append({"title": "Residual Distribution", "figure": fig})

    # 3) Residual vs fitted
    fig = _make_figure((10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(chosen_oof["y_pred"], chosen_oof["residual"], s=18, alpha=0.65, color="#54A24B")
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Residual vs Fitted (Validation OOF)")
    ax.set_xlabel("Fitted Margin")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figures.append({"title": "Residual vs Fitted", "figure": fig})

    # 4) Predicted vs actual scatter
    fig = _make_figure((10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(chosen_oof["y_pred"], chosen_oof["y_true"], s=18, alpha=0.65, color="#E45756")
    lims = [
        float(min(chosen_oof["y_pred"].min(), chosen_oof["y_true"].min())),
        float(max(chosen_oof["y_pred"].max(), chosen_oof["y_true"].max())),
    ]
    ax.plot(lims, lims, color="black", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Predicted vs Actual (Validation OOF)")
    ax.set_xlabel("Predicted Margin")
    ax.set_ylabel("Actual Margin")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figures.append({"title": "Predicted vs Actual", "figure": fig})

    # 5) Derby prediction distribution
    pred_df: pd.DataFrame = final_outputs["predictions"]
    fig = _make_figure((10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(pred_df["Team1_WinMargin"].astype(float), bins=20, color="#72B7B2", edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Final Derby Prediction Distribution")
    ax.set_xlabel("Team1_WinMargin")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    figures.append({"title": "Derby Prediction Distribution", "figure": fig})

    # 6) HOME_ADV sensitivity plot (config summary)
    plot_df = config_summary.nsmallest(min(12, len(config_summary)), "selection_score").copy()
    fig = _make_figure((12, 6))
    ax = fig.add_subplot(1, 1, 1)
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["sens_home_adv_pm10_mean_abs"].fillna(0.0), color="#B279A2")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["config_name"], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Prediction Sensitivity to HOME_ADV ±10 (Mean |Δpred|)")
    ax.set_xlabel("Mean Absolute Prediction Difference")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    figures.append({"title": "HOME_ADV Sensitivity", "figure": fig})

    # 7) Home advantage tuning curve
    ha_detail = home_adv_result["detail"]
    ha_summary = (
        ha_detail[ha_detail["fold"] >= 0]
        .groupby("home_adv", as_index=False)
        .agg(brier_mean=("brier", "mean"), brier_std=("brier", "std"))
        .sort_values("home_adv")
    )
    fig = _make_figure((10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ha_summary["home_adv"], ha_summary["brier_mean"], marker="o", color="#4C78A8")
    ax.fill_between(
        ha_summary["home_adv"].to_numpy(),
        (ha_summary["brier_mean"] - ha_summary["brier_std"].fillna(0)).to_numpy(),
        (ha_summary["brier_mean"] + ha_summary["brier_std"].fillna(0)).to_numpy(),
        alpha=0.2,
        color="#4C78A8",
    )
    ax.axvline(home_adv_result["best_home_adv"], color="black", lw=1, linestyle="--")
    ax.set_title("Elo Home Advantage Tuning (Brier Score)")
    ax.set_xlabel("HOME_ADV (Elo points)")
    ax.set_ylabel("Brier Score")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    figures.append({"title": "Home Advantage Tuning", "figure": fig})

    metrics_row = config_summary.loc[config_summary["config_name"] == chosen_name].iloc[0].to_dict()
    pred_preview = _df_preview(final_outputs["predictions"], 10)
    rank_preview = final_outputs["rankings"].sort_values("Rank", kind="mergesort").head(20).copy()

    section_texts = {
        "Executive Summary": "\n".join(
            [
                "Built a deterministic Python-only sports analytics pipeline for regular-season training (home venue) and derby forecasting (neutral venue).",
                f"Selected model: {chosen_name}. HOME_ADV tuned via time-aware Elo Brier-score CV = {home_adv_result['best_home_adv']} Elo points.",
                f"Validation (OOF CV): RMSE={metrics_row['rmse_mean']:.3f}, MAE={metrics_row['mae_mean']:.3f}, Bias={metrics_row['bias_mean']:.3f}, SignAcc={metrics_row['sign_acc_mean']:.3f}.",
                "Derby predictions use final season states with neutral-site home advantage set to 0, then winsorized to Train 1st/99th percentile and rounded to integer.",
            ]
        ),
        "Data & Constraints": "\n".join(
            [
                f"Train rows: {data_summary['n_train_games']} (home venue, dated {data_summary['train_date_min']} to {data_summary['train_date_max']}).",
                f"Derby rows: {data_summary['n_pred_games']} (neutral venue, date {data_summary['pred_date_min']}).",
                f"Team universe from Rankings.xlsx: {data_summary['n_teams_rankings']} teams.",
                "No derby ground truth exists; validation performed only on Train.csv with expanding-window / walk-forward splits.",
                "No-leakage rule enforced for sequential features: each game uses only strictly prior game outcomes.",
                "Massey, offense/defense, and conference ridge models are fit on training folds only during CV.",
            ]
        ),
        "Feature Engineering": "\n".join(
            [
                "Pregame features include Elo (MOV-scaled), Massey margin ratings, ridge-shrunk offense/defense strengths, conference-strength shrinkage, and rolling efficiency statistics.",
                "Rolling metrics are computed per team before each game: mean margin, EMA margin, opponent-adjusted margin, schedule strength proxy (avg opp Elo), Laplace-smoothed win rate, shrunk volatility, games played, and rest days.",
                "Matchup-level diffs are used heavily to stabilize margins and transfer cleanly to neutral-site derby predictions.",
            ]
        ),
        "Model Selection Logic": "\n".join(
            [
                "Primary metric is RMSE for margin prediction.",
                "Selection score adds penalties for fold-to-fold instability (RMSE/MAE std), bias, and extreme prediction magnitude to reduce brittle overfitting.",
                "A simple mean ensemble of the two best individual configs is also evaluated as a required baseline.",
            ]
        ),
        "Post-processing & Ranking": "\n".join(
            [
                f"Prediction clipping bounds derived from Train HomeWinMargin percentiles: {final_outputs['clip_lo']:.2f} to {final_outputs['clip_hi']:.2f}.",
                "Rankings are computed from standardized final Elo, final Massey rating, and final net rating (offense-defense) using weights estimated from OOF component-diff regression.",
                f"Ranking weights (elo, massey, net): {final_outputs['ranking_weights']['elo']:.3f}, {final_outputs['ranking_weights']['massey']:.3f}, {final_outputs['ranking_weights']['net']:.3f}.",
            ]
        ),
        "Reproducibility": "\n".join(
            [
                f"Random seed: {seed}",
                "Run command: python run_pipeline.py",
                "All outputs are written at repo root: predictions.csv, rankings.xlsx, final_report.pdf",
            ]
        ),
    }

    markdown = render_markdown_summary(
        section_texts=section_texts,
        config_summary=config_summary,
        pred_preview=pred_preview,
        rank_preview=rank_preview,
    )

    return {
        "section_texts": section_texts,
        "tables": {
            "model_sweep": config_summary.copy(),
            "fold_metrics": fold_metrics.copy(),
            "home_adv_detail": ha_detail.copy(),
            "pred_preview": pred_preview,
            "rank_preview": rank_preview,
        },
        "figures": figures,
        "markdown_text": markdown,
        "diagnostics": {
            "chosen_oof": chosen_oof,
        },
    }


def render_markdown_summary(
    *,
    section_texts: Mapping[str, str],
    config_summary: pd.DataFrame,
    pred_preview: pd.DataFrame,
    rank_preview: pd.DataFrame,
) -> str:
    out: List[str] = []
    def _table_text(df: pd.DataFrame) -> str:
        return df.to_string(index=False)
    for title, text in section_texts.items():
        out.append(f"# {title}")
        out.append("")
        out.append(text)
        out.append("")
    out.append("# Model Sweep (Top 20)")
    out.append("")
    out.append(_table_text(config_summary.head(20)))
    out.append("")
    out.append("# Predictions Preview (First 10)")
    out.append("")
    out.append(_table_text(pred_preview))
    out.append("")
    out.append("# Rankings Preview (Top 20)")
    out.append("")
    out.append(_table_text(rank_preview))
    out.append("")
    return "\n".join(out)
