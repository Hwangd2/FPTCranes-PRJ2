from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import PASTEL, STAGES, TARGET


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pipeline_12(path: Path) -> None:
    # Snake layout keeps the numbered sequence visually continuous:
    # 01→02→03→04→05→06, then down to 07 and back left through 12.
    fig, ax = plt.subplots(figsize=(18, 7.5))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    colors = [
        PASTEL["purple"], PASTEL["blue"], PASTEL["pink"], PASTEL["orange"], PASTEL["yellow"], PASTEL["green"],
        PASTEL["blue"], PASTEL["purple"], PASTEL["pink"], PASTEL["yellow"], PASTEL["green"], PASTEL["orange"],
    ]
    positions = []
    for i in range(12):
        if i < 6:
            col, row = i, 1
        else:
            col, row = 11 - i, 0  # 07 under 06, 12 under 01
        positions.append((col + 0.08, row + 0.25))

    for i, title in enumerate(STAGES):
        x, y = positions[i]
        rect = plt.Rectangle((x, y), 0.84, 0.72, facecolor=colors[i], edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.10, y + 0.56, f"{i + 1:02d}", fontsize=13, fontweight="bold", color=PASTEL["ink"])
        ax.text(
            x + 0.10, y + 0.36, "\n".join(wrap(title, width=22)), fontsize=8.5,
            va="top", color=PASTEL["ink"], fontweight="semibold",
        )

    # Draw arrows between consecutive stages using box edges.
    for i in range(11):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        if i < 5:
            start_xy, end_xy = (x1 + 0.86, y1 + 0.36), (x2 - 0.07, y2 + 0.36)
        elif i == 5:
            start_xy, end_xy = (x1 + 0.42, y1 - 0.02), (x2 + 0.42, y2 + 0.78)
        else:
            start_xy, end_xy = (x1 - 0.02, y1 + 0.36), (x2 + 0.91, y2 + 0.36)
        ax.annotate("", xy=end_xy, xytext=start_xy, arrowprops=dict(arrowstyle="->", color="#7C8DA6", lw=1.7))

    ax.text(0.08, 2.25, "AI JOB MARKET SALARY PREDICTION", fontsize=23, fontweight="bold", color=PASTEL["ink"])
    ax.text(0.08, 2.08, "12-stage data-quality-first, leakage-safe and deployable workflow", fontsize=12.5, color=PASTEL["muted"])
    ax.text(0.08, 0.05, "Target: annual_salary_usd | Locked temporal test: 2026-03 | Skills: TRAIN-only vocabulary + multi-hot encoding", fontsize=10.5, color=PASTEL["muted"])
    _save(fig, path)

def plot_target_distribution(frame: pd.DataFrame, path: Path) -> None:
    target = pd.to_numeric(frame[TARGET], errors="coerce").dropna()
    q1, q3 = target.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.hist(target, bins=28, color=PASTEL["blue_dark"], edgecolor="white")
    ax.axvline(target.mean(), linestyle="--", linewidth=1.8, color="#B35C76", label=f"Mean ${target.mean():,.0f}")
    ax.axvline(target.median(), linestyle=":", linewidth=1.8, color="#4F6D8A", label=f"Median ${target.median():,.0f}")
    ax.axvline(upper, linestyle="--", linewidth=1.2, color="#A06A3B", label=f"IQR upper ${upper:,.0f}")
    ax.set_title("Observed Target Distribution — annual_salary_usd", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Annual salary (USD)")
    ax.set_ylabel("Job postings")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.16)
    _save(fig, path)


def plot_cardinality(profile: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    data = profile.sort_values("unique_count", ascending=False).head(top_n).sort_values("unique_count")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(data["column"], data["unique_count"], color=PASTEL["green"])
    ax.set_title(f"Top {top_n} Feature Cardinalities", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Unique values")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_missingness(quality: pd.DataFrame, path: Path) -> None:
    data = quality.sort_values("missing_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(data["column"], data["missing_pct"], color=PASTEL["purple"])
    ax.set_title("Missingness by Feature", color=PASTEL["ink"], fontweight="bold")
    ax.set_ylabel("Missing rows (%)")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(axis="y", alpha=0.15)
    _save(fig, path)


def plot_logic_issue_rates(issues: pd.DataFrame, path: Path) -> None:
    data = issues.sort_values("affected_pct")
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.barh(data["issue"], data["affected_pct"], color=PASTEL["pink"])
    for i, row in enumerate(data.itertuples()):
        ax.text(row.affected_pct + 0.8, i, f"{row.affected_pct:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(100, float(data["affected_pct"].max()) + 8))
    ax.set_xlabel("Rows affected (%)")
    ax.set_title("Data Logic & Integrity Issues — Observed Rates", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_salary_by_job_category(summary: pd.DataFrame, path: Path) -> None:
    data = summary.sort_values("mean_salary_usd")
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.barh(data["job_category"], data["mean_salary_usd"] / 1000, color=PASTEL["blue_dark"])
    ax.set_title("Mean annual_salary_usd by Job Domain — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Mean salary (USD thousands)")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_years_by_experience_level(frame: pd.DataFrame, path: Path) -> None:
    order = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]
    data = [frame.loc[frame["experience_level"].eq(level), "years_of_experience"].dropna().to_numpy() for level in order]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(PASTEL["purple"])
    ax.set_title("Years of Experience Distribution by Experience Level", color=PASTEL["ink"], fontweight="bold")
    ax.set_ylabel("Years of experience")
    ax.tick_params(axis="x", rotation=8)
    ax.grid(axis="y", alpha=0.15)
    _save(fig, path)


def plot_salary_by_experience_level(frame: pd.DataFrame, path: Path) -> None:
    order = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]
    data = [frame.loc[frame["experience_level"].eq(level), TARGET].dropna().to_numpy() for level in order]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(PASTEL["green"])
    ax.set_title("Annual Salary Distribution by Experience Level — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_ylabel("Annual salary (USD)")
    ax.tick_params(axis="x", rotation=8)
    ax.grid(axis="y", alpha=0.15)
    _save(fig, path)


def plot_salary_by_years(summary: pd.DataFrame, path: Path) -> None:
    data = summary.sort_values("years_of_experience")
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.plot(data["years_of_experience"], data["median_salary_usd"], marker="o", linewidth=2, color="#6C9BCF")
    ax.fill_between(data["years_of_experience"], data["mean_salary_usd"], data["median_salary_usd"], alpha=0.12, color="#6C9BCF")
    ax.set_title("Salary Pattern by Years of Experience — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Years of experience")
    ax.set_ylabel("Salary (USD)")
    ax.grid(alpha=0.15)
    _save(fig, path)


def plot_salary_range_integrity(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.scatter(frame["annual_salary_usd"], frame["salary_min_usd"], s=12, alpha=0.35, label="salary_min_usd", color="#D1876A")
    ax.scatter(frame["annual_salary_usd"], frame["salary_max_usd"], s=12, alpha=0.35, label="salary_max_usd", color="#6C9BCF")
    lims = [min(frame["annual_salary_usd"].min(), frame["salary_min_usd"].min()), max(frame["annual_salary_usd"].max(), frame["salary_max_usd"].max())]
    ax.plot(lims, lims, linestyle="--", linewidth=1, color="#7A7A7A", label="equal to annual salary")
    ax.set_xlabel("annual_salary_usd")
    ax.set_ylabel("stated salary bound")
    ax.set_title("Salary Range Metadata vs Target — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12)
    _save(fig, path)


def plot_feature_policy(policy: pd.DataFrame, path: Path) -> None:
    counts = policy.groupby("policy").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90, colors=[PASTEL["green"], PASTEL["pink"], PASTEL["yellow"], PASTEL["blue"]][: len(counts)])
    ax.add_artist(plt.Circle((0, 0), 0.52, color="white"))
    ax.set_title("Feature Governance Mix", color=PASTEL["ink"], fontweight="bold")
    _save(fig, path)


def plot_correlations(corr: pd.DataFrame, path: Path, top_n: int = 30) -> None:
    data = corr.head(top_n).sort_values("pearson")
    colors = ["#D8896F" if x < 0 else "#6C9BCF" for x in data["pearson"]]
    fig, ax = plt.subplots(figsize=(10, 8.0))
    ax.barh(data["feature"], data["pearson"], color=colors)
    ax.axvline(0, color="#64748B", linewidth=1)
    ax.set_title(f"Top {top_n} Encoded Features by |Pearson| — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Pearson correlation with annual_salary_usd")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_pearson_spearman(corr: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    data = corr.head(top_n).iloc[::-1]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    ax.barh(y - 0.18, data["pearson"], height=0.35, label="Pearson", color=PASTEL["blue_dark"])
    ax.barh(y + 0.18, data["spearman"], height=0.35, label="Spearman", color="#B18BC7")
    ax.set_yticks(y, data["feature"])
    ax.axvline(0, color="#64748B", linewidth=1)
    ax.legend(frameon=False)
    ax.set_title("Pearson vs Spearman — Top Encoded Features", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_vif(vif: pd.DataFrame, path: Path) -> None:
    if vif.empty:
        return
    data = vif.sort_values("vif")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(data["feature"], data["vif"], color=PASTEL["yellow"])
    ax.axvline(5, linestyle="--", color="#A06A3B", linewidth=1, label="review = 5")
    ax.axvline(10, linestyle="--", color="#A33A3A", linewidth=1, label="high = 10")
    ax.set_title("VIF — Numeric/Engineered Features on DEV", color=PASTEL["ink"], fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_skills(skill_summary: pd.DataFrame, path: Path, top_n: int = 30) -> None:
    data = skill_summary.sort_values("record_count", ascending=False).head(top_n).sort_values("record_count")
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.barh(data["skill_token"], data["record_count"], color=PASTEL["blue"])
    ax.set_title(f"Top {top_n} Skills by Record Count — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Records")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_temporal_split(monthly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = ["#F2A6A6" if p == "LOCKED_TEST" else "#AFCFE8" for p in monthly["partition"]]
    ax.bar(monthly["period"], monthly["rows"], color=colors)
    ax.set_title("Monthly Posting Volume & Locked Temporal Test", color=PASTEL["ink"], fontweight="bold")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.15)
    _save(fig, path)


def plot_model_comparison(comp: pd.DataFrame, out: Path) -> None:
    data = comp.sort_values("CV_MAE_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(data["model"], data["CV_MAE_mean"], xerr=data["CV_MAE_std"], color=PASTEL["blue_dark"], alpha=0.88)
    ax.set_xlabel("Mean temporal CV MAE (USD)")
    ax.set_title("Dummy Floor + Candidate Models — Temporal CV MAE", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, out / "model_comparison_cv_mae.png")

    data_r2 = comp.sort_values("CV_R2_mean")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(data_r2["model"], data_r2["CV_R2_mean"], color=PASTEL["green"])
    ax.axvline(0, color="#64748B", linewidth=1)
    ax.set_xlabel("Mean temporal CV R²")
    ax.set_title("Temporal CV R² by Model Family", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, out / "model_comparison_cv_r2.png")


def plot_fold_stability(folds: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for model, group in folds.groupby("model"):
        ax.plot(group["validation_month"].astype(str), group["MAE"], marker="o", linewidth=1.6, label=model)
    ax.set_title("Temporal CV Fold Stability — MAE", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Validation month")
    ax.set_ylabel("MAE (USD)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.15)
    _save(fig, path)


def plot_ablation(ablation: pd.DataFrame, path: Path) -> None:
    data = ablation.sort_values("CV_MAE_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.barh(data["experiment"], data["CV_MAE_mean"], color=PASTEL["purple"])
    ax.set_xlabel("Mean temporal CV MAE (USD)")
    ax.set_title("Feature-Family Ablation — Random Forest", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_importance_drift(drift: pd.DataFrame, path: Path, top_n: int = 10) -> None:
    if drift.empty:
        return
    pivot = drift.pivot_table(index="raw_feature", columns="fold", values="importance", aggfunc="mean").fillna(0)
    top = pivot.mean(axis=1).sort_values(ascending=False).head(top_n).index
    matrix = pivot.loc[top]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xticks(range(len(matrix.columns)), [f"Fold {c}" for c in matrix.columns])
    ax.set_title("Raw Feature Importance Stability Across Temporal Folds", color=PASTEL["ink"], fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    _save(fig, path)


def plot_best_outputs(y_true: pd.Series, pred: np.ndarray, encoded_imp: pd.DataFrame, raw_imp: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.scatter(y_true, pred, s=18, alpha=0.45, color="#6C9BCF")
    lo = min(float(np.min(y_true)), float(np.min(pred)))
    hi = max(float(np.max(y_true)), float(np.max(pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="#64748B")
    ax.set_xlabel("Actual salary (USD)")
    ax.set_ylabel("Predicted salary (USD)")
    ax.set_title("Locked Test — Actual vs Predicted", color=PASTEL["ink"], fontweight="bold")
    ax.grid(alpha=0.12)
    _save(fig, out / "actual_vs_predicted_locked_test.png")

    residual = np.asarray(y_true) - pred
    fig, ax = plt.subplots(figsize=(8.3, 4.9))
    ax.hist(residual, bins=24, edgecolor="white", color=PASTEL["pink"])
    ax.axvline(0, color="#64748B", lw=1.2)
    ax.set_xlabel("Residual = Actual - Predicted (USD)")
    ax.set_ylabel("Records")
    ax.set_title("Locked Test Residual Distribution", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="y", alpha=0.16)
    _save(fig, out / "locked_test_residuals.png")

    raw = raw_imp.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.barh(raw["raw_feature"], raw["importance_mean"], xerr=raw["importance_std"], color=PASTEL["green"])
    ax.set_title("Raw Feature Permutation Importance — Locked Test", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Increase in MAE when permuted")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, out / "raw_feature_permutation_importance.png")

    enc = encoded_imp.head(25).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    ax.barh(enc["encoded_feature"], enc["importance"], color=PASTEL["yellow"])
    ax.set_title("Top 25 Encoded Feature Importances", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Model importance")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, out / "top25_encoded_feature_importance.png")
