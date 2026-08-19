from __future__ import annotations

import pandas as pd


def format_money(value: float) -> str:
    return f"${value:,.0f}"


def model_comparison_comment(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "Model comparison outputs are not available. Run pipeline.py first."

    ranked = comparison.sort_values("CV_MAE_mean", kind="stable").reset_index(drop=True)
    best = ranked.iloc[0]
    if len(ranked) == 1:
        return (
            f"{best['model']} is the only evaluated candidate, with mean temporal CV MAE "
            f"of {format_money(float(best['CV_MAE_mean']))}. Add another candidate before "
            "making a comparative selection claim."
        )

    second = ranked.iloc[1]
    second_mae = float(second["CV_MAE_mean"])
    best_mae = float(best["CV_MAE_mean"])
    gap = 0.0 if second_mae == 0 else (second_mae - best_mae) / second_mae * 100
    return (
        f"{best['model']} ranks first by mean temporal CV MAE ({format_money(best_mae)}), "
        f"about {gap:.1f}% lower than the next model ({second['model']}). Selection therefore "
        "uses development-period temporal validation rather than the locked test."
    )
