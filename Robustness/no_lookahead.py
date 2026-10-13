#!/usr/bin/env python3
"""
No-lookahead robustness checks for the revised extension linear regressions.

This script leaves the main extension outputs unchanged. It rebuilds the
momentum-adjusted open-interest component with an expanding window, so each
date's residual uses only observations strictly before that date to estimate
the return adjustment.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"
DATA_PATH = OUTPUT_DIR / "revised_extension_dataset.csv"
MIN_HISTORY = 36

sys.path.insert(0, str(SCRIPT_DIR.parent))
import cfpi_variants as rev  # noqa: E402


HORIZONS = rev.CONFIG["HORIZONS"]
CONTROLS = rev.CONFIG["CONTROL_COLUMNS"]


def expanding_residualize(
    df: pd.DataFrame,
    y_col: str,
    controls: list[str],
    out_col: str,
    min_history: int = MIN_HISTORY,
) -> pd.DataFrame:
    """Estimate y residuals using only rows before the current observation."""
    work = df[[y_col] + controls].replace([np.inf, -np.inf], np.nan)
    residual = pd.Series(np.nan, index=df.index, name=out_col)
    minimum_obs = max(min_history, len(controls) + 2)

    for i in range(len(df)):
        current = work.iloc[i]
        history = work.iloc[:i].dropna()
        if len(history) < minimum_obs or current.isna().any():
            continue

        x_history = np.column_stack(
            [np.ones(len(history)), history[controls].to_numpy(dtype=float)]
        )
        y_history = history[y_col].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x_history, y_history, rcond=None)[0]
        x_current = np.array([1.0] + [float(current[col]) for col in controls])
        residual.iloc[i] = float(current[y_col] - x_current @ beta)

    df[out_col] = residual
    return df


def run_specifications(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        {
            "name": "individual_basis",
            "label": "Individual futures basis",
            "x_cols": ["futures_basis"],
            "main_predictor": "futures_basis",
            "columns": ("futures_basis",),
            "index_method": "no_lookahead_individual_component",
        },
        {
            "name": "individual_expanding_resid_oi",
            "label": "Individual expanding residual open-interest growth",
            "x_cols": ["cftc_open_interest_growth_resid_expanding"],
            "main_predictor": "cftc_open_interest_growth_resid_expanding",
            "columns": ("cftc_open_interest_growth_resid_expanding",),
            "index_method": "no_lookahead_individual_component",
        },
        {
            "name": "individual_gross_mm",
            "label": "Individual gross managed-money participation",
            "x_cols": ["cftc_gross_position_share"],
            "main_predictor": "cftc_gross_position_share",
            "columns": ("cftc_gross_position_share",),
            "index_method": "no_lookahead_individual_component",
        },
        {
            "name": "linear_components_expanding_resid",
            "label": "No-lookahead linear components",
            "x_cols": [
                "futures_basis",
                "cftc_open_interest_growth_resid_expanding",
                "cftc_gross_position_share",
            ],
            "main_predictor": "cftc_open_interest_growth_resid_expanding",
            "columns": (
                "futures_basis",
                "cftc_open_interest_growth_resid_expanding",
                "cftc_gross_position_share",
            ),
            "index_method": "no_lookahead_linear_components",
        },
        {
            "name": "publication_lag_linear_components_expanding_resid",
            "label": "One-week-lagged no-lookahead linear components",
            "x_cols": [
                "futures_basis_lag1",
                "cftc_open_interest_growth_resid_expanding_lag1",
                "cftc_gross_position_share_lag1",
            ],
            "main_predictor": "cftc_open_interest_growth_resid_expanding_lag1",
            "columns": (
                "futures_basis_lag1",
                "cftc_open_interest_growth_resid_expanding_lag1",
                "cftc_gross_position_share_lag1",
            ),
            "index_method": "publication_lag_no_lookahead_linear_components",
        },
    ]

    rows = []
    for spec in specs:
        variant = rev.IndexVariant(
            name=spec["name"],
            label=spec["label"],
            columns=spec["columns"],
            description="No-lookahead linear robustness check.",
        )
        for horizon in HORIZONS:
            result = rev.run_hac_ols(
                df=df,
                y_col=f"R_fwd_{horizon}",
                x_cols=CONTROLS + spec["x_cols"],
                horizon=horizon,
                variant=variant,
                index_method=spec["index_method"],
                specification=spec["name"],
                main_predictor=spec["main_predictor"],
                sample_scope=f"expanding_residual_min{MIN_HISTORY}",
            )
            rows.append(result.table)

    return pd.concat(rows, ignore_index=True)


def coef_with_stars(row: pd.Series) -> str:
    return f"{row['coef']:.3f}{rev.significance_stars(row['p_value_hac'])}"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    df = expanding_residualize(
        df=df,
        y_col="cftc_open_interest_growth",
        controls=["R_t"],
        out_col="cftc_open_interest_growth_resid_expanding",
    )
    df["futures_basis_lag1"] = df["futures_basis"].shift(1)
    df["cftc_open_interest_growth_resid_expanding_lag1"] = df[
        "cftc_open_interest_growth_resid_expanding"
    ].shift(1)
    df["cftc_gross_position_share_lag1"] = df["cftc_gross_position_share"].shift(1)

    results = run_specifications(df)
    results_path = OUTPUT_DIR / "no_lookahead_linear_regressions.csv"
    results.to_csv(results_path, index=False)

    focus = results[
        (results["status"] == "ok")
        & (results["parameter"] == results["main_predictor"])
    ].copy()
    focus["coef_stars"] = focus.apply(coef_with_stars, axis=1)
    focus = focus[
        [
            "specification",
            "index_method",
            "horizon",
            "parameter",
            "coef",
            "t_stat_hac",
            "p_value_hac",
            "coef_stars",
            "n_obs",
            "sample_start",
            "sample_end",
        ]
    ]
    summary_path = OUTPUT_DIR / "no_lookahead_linear_summary.csv"
    focus.to_csv(summary_path, index=False)

    print(f"Saved full results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
