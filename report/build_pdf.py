from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def _add_text_page(pdf: PdfPages, title: str, text: str, page_size=(8.5, 11), fontsize=10):
    fig = plt.figure(figsize=page_size)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.05, 0.96, title, fontsize=16, fontweight="bold", va="top", ha="left")
    y = 0.92
    wrap_width = 110
    for para in str(text).split("\n"):
        wrapped = textwrap.wrap(para, width=wrap_width) or [""]
        for line in wrapped:
            ax.text(0.05, y, line, fontsize=fontsize, va="top", ha="left")
            y -= 0.028
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=page_size)
                fig.patch.set_facecolor("white")
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                ax.text(0.05, 0.96, f"{title} (cont.)", fontsize=16, fontweight="bold", va="top", ha="left")
                y = 0.92
        y -= 0.012
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _format_cell(v):
    if pd.isna(v):
        return ""
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        return f"{float(v):.4f}"
    return str(v)


def _add_table_pages(pdf: PdfPages, title: str, df: pd.DataFrame, rows_per_page: int = 22):
    df = df.copy()
    if df.empty:
        _add_text_page(pdf, title, "No rows.")
        return
    n_pages = math.ceil(len(df) / rows_per_page)
    for p in range(n_pages):
        chunk = df.iloc[p * rows_per_page : (p + 1) * rows_per_page].copy()
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.90])
        ax.axis("off")
        ax.text(0.0, 1.03, f"{title} ({p+1}/{n_pages})", fontsize=14, fontweight="bold", transform=ax.transAxes)
        cell_text = [[_format_cell(v) for v in row] for row in chunk.to_numpy()]
        table = ax.table(
            cellText=cell_text,
            colLabels=[str(c) for c in chunk.columns],
            loc="upper left",
            cellLoc="center",
            colLoc="center",
            bbox=[0.0, 0.0, 1.0, 0.95],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_pdf(report_bundle: Mapping[str, object], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    with PdfPages(output_path) as pdf:
        pdf.infodict()["Title"] = "AlgoSports23 Final Report"
        pdf.infodict()["Author"] = "Codex (GPT-5)"

        section_texts = report_bundle.get("section_texts", {})
        for title, text in section_texts.items():
            _add_text_page(pdf, title, str(text))

        tables = report_bundle.get("tables", {})
        if "model_sweep" in tables:
            # Keep detailed sweep table but cap rows per page for readability.
            _add_table_pages(pdf, "Model Sweep Results", pd.DataFrame(tables["model_sweep"]), rows_per_page=18)
        if "fold_metrics" in tables:
            _add_table_pages(pdf, "Fold Metrics (All Configs)", pd.DataFrame(tables["fold_metrics"]), rows_per_page=20)
        if "home_adv_detail" in tables:
            _add_table_pages(pdf, "Elo HOME_ADV Tuning Detail", pd.DataFrame(tables["home_adv_detail"]), rows_per_page=24)

        for fig_item in report_bundle.get("figures", []):
            fig = fig_item.get("figure")
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # Appendix previews required by prompt
        if "pred_preview" in tables:
            _add_table_pages(pdf, "Appendix: predictions.csv Preview (First 10)", pd.DataFrame(tables["pred_preview"]), rows_per_page=12)
        if "rank_preview" in tables:
            _add_table_pages(pdf, "Appendix: Top 20 Rankings", pd.DataFrame(tables["rank_preview"]), rows_per_page=22)

        # Include markdown summary as plain text appendix for traceability.
        markdown_text = report_bundle.get("markdown_text")
        if markdown_text:
            _add_text_page(pdf, "Appendix: Markdown Summary", str(markdown_text), fontsize=8)

    return output_path

