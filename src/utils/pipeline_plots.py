from __future__ import annotations

from pathlib import Path

import matplotlib

# Offline pipeline plots are file artifacts. A GUI backend such as TkAgg is unsafe when
# Stage 10 starts parallel workers, because Tk objects may be finalized off the main thread.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import PASTEL, STAGES, TARGET


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def chart_pipeline_12(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(18, 7.5))
    axis.set_xlim(0, 6)
    axis.set_ylim(0, 2.5)
    axis.axis("off")
    colors = [
        PASTEL["purple"],
        PASTEL["blue"],
        PASTEL["pink"],
        PASTEL["orange"],
        PASTEL["yellow"],
        PASTEL["green"],
        PASTEL["blue"],
        PASTEL["purple"],
        PASTEL["pink"],
        PASTEL["yellow"],
        PASTEL["green"],
        PASTEL["orange"],
    ]
    for index, title in enumerate(STAGES):
        row = 1 if index < 6 else 0
        column = index if index < 6 else index - 6
        x, y = column + 0.08, row + 0.25
        rectangle = plt.Rectangle(
            (x, y),
            0.84,
            0.72,
            facecolor=colors[index],
            edgecolor="#FFFFFF",
            linewidth=1.5,
        )
        axis.add_patch(rectangle)
        axis.text(
            x + 0.10,
            y + 0.56,
            f"{index + 1:02d}",
            fontsize=13,
            fontweight="bold",
            color=PASTEL["ink"],
        )
        wrapped = "\n".join(__import__("textwrap").wrap(title, width=22))
        axis.text(
            x + 0.10,
            y + 0.36,
            wrapped,
            fontsize=8.5,
            va="top",
            color=PASTEL["ink"],
            fontweight="semibold",
        )
        if index < 5 or 6 <= index < 11:
            axis.annotate(
                "",
                xy=(x + 0.99, y + 0.36),
                xytext=(x + 0.86, y + 0.36),
                arrowprops=dict(arrowstyle="->", color="#7C8DA6", lw=1.7),
            )
        elif index == 5:
            axis.annotate(
                "",
                xy=(5.5, 0.96),
                xytext=(5.5, 1.25),
                arrowprops=dict(arrowstyle="->", color="#7C8DA6", lw=1.7),
            )
    axis.text(
        0.08,
        2.25,
        "AI JOB MARKET SALARY PREDICTION",
        fontsize=23,
        fontweight="bold",
        color=PASTEL["ink"],
    )
    axis.text(
        0.08,
        2.08,
        "12-stage data-quality-first, leakage-safe and deployable workflow",
        fontsize=12.5,
        color=PASTEL["muted"],
    )
    axis.text(
        0.08,
        0.05,
        "Target: annual_salary_usd | Locked temporal test: 2026-03 | "
        "Skills: TRAIN-only token vocabulary + multi-hot encoding",
        fontsize=10.5,
        color=PASTEL["muted"],
    )
    _save_figure(figure, path)


def plot_basic_outputs(clean: pd.DataFrame, issues: pd.DataFrame, out: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.hist(clean[TARGET], bins=24, edgecolor="white", color=PASTEL["blue"])
    axis.set_title(
        "Annual Salary Distribution after Basic Cleaning",
        color=PASTEL["ink"],
        fontweight="bold",
    )
    axis.set_xlabel("annual_salary_usd")
    axis.set_ylabel("Records")
    axis.grid(axis="y", alpha=0.18)
    _save_figure(figure, out / "salary_distribution.png")

    if not issues.empty:
        figure, axis = plt.subplots(figsize=(9, 4.8))
        ordered = issues.sort_values("affected_pct")
        axis.barh(ordered["issue"], ordered["affected_pct"], color=PASTEL["pink"])
        axis.set_xlabel("Affected records (%)")
        axis.set_title(
            "Material Data Logic / Integrity Findings",
            color=PASTEL["ink"],
            fontweight="bold",
        )
        axis.grid(axis="x", alpha=0.18)
        _save_figure(figure, out / "logic_issue_rates.png")

    monthly = (
        clean.groupby(["posting_year", "posting_month"])
        .size()
        .reset_index(name="records")
    )
    monthly["period"] = (
        monthly["posting_year"].astype(str)
        + "-"
        + monthly["posting_month"].astype(str).str.zfill(2)
    )
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(monthly["period"], monthly["records"], marker="o", color="#7895CB")
    axis.axvspan(
        max(0, len(monthly) - 1) - 0.3,
        len(monthly) - 0.7,
        alpha=0.18,
        color=PASTEL["pink"],
    )
    axis.set_title(
        "Posting Volume by Month (final month is locked test)",
        color=PASTEL["ink"],
        fontweight="bold",
    )
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("Records")
    axis.grid(axis="y", alpha=0.18)
    _save_figure(figure, out / "monthly_posting_distribution.png")


def plot_correlation(correlation: pd.DataFrame, out: Path) -> None:
    top = correlation.head(30).sort_values("pearson_r")
    figure, axis = plt.subplots(figsize=(10, 9))
    colors = ["#F4A6A6" if value < 0 else "#90CAF9" for value in top["pearson_r"]]
    axis.barh(top["encoded_feature"], top["pearson_r"], color=colors)
    axis.axvline(0, color="#64748B", lw=1)
    axis.set_xlabel("Pearson correlation with annual_salary_usd (TRAIN only)")
    axis.set_title(
        "Top 30 Encoded Feature Correlations", color=PASTEL["ink"], fontweight="bold"
    )
    axis.grid(axis="x", alpha=0.15)
    _save_figure(figure, out / "top30_train_target_correlation.png")


def plot_model_comparison(comparison: pd.DataFrame, out: Path) -> None:
    ordered = comparison.sort_values("CV_MAE_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5.3))
    axis.barh(
        ordered["model"],
        ordered["CV_MAE_mean"],
        xerr=ordered["CV_MAE_std"],
        color=PASTEL["purple"],
        alpha=0.95,
    )
    axis.set_xlabel("Temporal CV MAE (USD) - lower is better")
    axis.set_title(
        "Five-Model Comparison on Development Period",
        color=PASTEL["ink"],
        fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.18)
    _save_figure(figure, out / "model_comparison_cv_mae.png")

    ordered_r2 = comparison.sort_values("CV_R2_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5.3))
    axis.barh(ordered_r2["model"], ordered_r2["CV_R2_mean"], color=PASTEL["green"])
    axis.axvline(0, color="#64748B", lw=1)
    axis.set_xlabel("Temporal CV R2 - higher is better")
    axis.set_title(
        "Model Generalization: Mean Temporal CV R2",
        color=PASTEL["ink"],
        fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.18)
    _save_figure(figure, out / "model_comparison_cv_r2.png")


def plot_best_outputs(
    y_test: pd.Series,
    prediction: np.ndarray,
    encoded_importance: pd.DataFrame,
    raw_importance: pd.DataFrame,
    out: Path,
) -> None:
    top = encoded_importance.head(25).sort_values("importance")
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(top["encoded_feature"], top["importance"], color=PASTEL["blue"])
    axis.set_xlabel("Model feature importance")
    axis.set_title(
        "Top 25 Encoded Feature Importances", color=PASTEL["ink"], fontweight="bold"
    )
    axis.grid(axis="x", alpha=0.18)
    _save_figure(figure, out / "top25_encoded_feature_importance.png")

    top_raw = raw_importance.head(13).sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(9, 5.8))
    axis.barh(
        top_raw["raw_feature"],
        top_raw["importance_mean"],
        xerr=top_raw["importance_std"],
        color=PASTEL["yellow"],
    )
    axis.set_xlabel("Increase in MAE when permuted (USD)")
    axis.set_title(
        "Raw Feature-Family Permutation Importance",
        color=PASTEL["ink"],
        fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.18)
    _save_figure(figure, out / "raw_feature_permutation_importance.png")

    figure, axis = plt.subplots(figsize=(6.7, 6.2))
    axis.scatter(y_test, prediction, alpha=0.62, color="#8CB9BD", edgecolors="none")
    lower = float(min(y_test.min(), prediction.min()))
    upper = float(max(y_test.max(), prediction.max()))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#7C8DA6")
    axis.set_xlabel("Actual annual salary (USD)")
    axis.set_ylabel("Predicted annual salary (USD)")
    axis.set_title(
        "Locked Test: Actual vs Predicted", color=PASTEL["ink"], fontweight="bold"
    )
    axis.grid(alpha=0.16)
    _save_figure(figure, out / "actual_vs_predicted_locked_test.png")

    residual = np.asarray(y_test) - prediction
    figure, axis = plt.subplots(figsize=(8.3, 4.9))
    axis.hist(residual, bins=24, edgecolor="white", color=PASTEL["pink"])
    axis.axvline(0, color="#64748B", lw=1.2)
    axis.set_xlabel("Residual = Actual - Predicted (USD)")
    axis.set_ylabel("Records")
    axis.set_title(
        "Locked Test Residual Distribution", color=PASTEL["ink"], fontweight="bold"
    )
    axis.grid(axis="y", alpha=0.16)
    _save_figure(figure, out / "locked_test_residuals.png")


__all__ = [
    "chart_pipeline_12",
    "plot_basic_outputs",
    "plot_best_outputs",
    "plot_correlation",
    "plot_model_comparison",
]
