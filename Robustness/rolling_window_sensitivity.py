#!/usr/bin/env python3
"""
Rolling PCA window sensitivity for the revised extension.

This companion script tests whether the weak rolling-PCA results are specific to
the baseline 36-week calibration window. It does not modify the main revised
extension pipeline. Outputs are written separately under revised_extension/outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cfpi_variants as rev


WINDOWS = [13, 18, 26, 36, 52, 78]
EWMA_HALFLIVES = [13, 26, 36, 52]
SHRINKAGE_ALPHA = 0.10
MIN_REPORTABLE_N = 30
VARIANT_NAMES = [
    rev.CONFIG["MAIN_VARIANT"],
    rev.CONFIG["BTC_ROBUSTNESS_VARIANT"],
]
CORRELATION_RETURN_LAGS = [1, 2, 4]


def get_variant(name: str) -> rev.IndexVariant:
    variants = {variant.name: variant for variant in rev.INDEX_VARIANTS}
    if name not in variants:
        raise ValueError(f"Unknown variant: {name}")
    return variants[name]


def run_rolling_pca_regressions(
    df: pd.DataFrame, variant: rev.IndexVariant, window: int
) -> pd.DataFrame:
    original_window = rev.CONFIG["ROLLING_WINDOW"]
    rev.CONFIG["ROLLING_WINDOW"] = window
    try:
        work = rev.build_rolling_pca_features_for_variant(df.copy(), variant)
    finally:
        rev.CONFIG["ROLLING_WINDOW"] = original_window

    score_col = f"{variant.name}_rolling_score"
    results = []
    for horizon in rev.CONFIG["HORIZONS"]:
        result = rev.run_hac_ols(
            df=work,
            y_col=f"R_fwd_{horizon}",
            x_cols=rev.CONFIG["CONTROL_COLUMNS"] + [score_col],
            horizon=horizon,
            variant=variant,
            index_method="rolling_pca",
            specification=f"{variant.name}_rolling_pca_{window}w",
            main_predictor=score_col,
            sample_scope=f"rolling_window_{window}w",
        ).table
        result["rolling_window"] = window
        results.append(result)

    return pd.concat(results, ignore_index=True)


def build_rolling_standardized_fixed_features(
    df: pd.DataFrame, variant: rev.IndexVariant, window: int
) -> pd.DataFrame:
    """
    Build a rolling-standardized index using fixed static PCA loadings.

    The rolling z-scores use only lagged observations from the calibration window,
    while the loading vector is the static PCA loading vector for the same
    component set. This separates unstable rolling PCA rotations from the effect
    of rolling/no-lookahead standardization.
    """
    original_window = rev.CONFIG["ROLLING_WINDOW"]
    rev.CONFIG["ROLLING_WINDOW"] = window
    try:
        work = rev.build_rolling_pca_features_for_variant(df.copy(), variant)
    finally:
        rev.CONFIG["ROLLING_WINDOW"] = original_window

    _, static_weights = rev.build_static_pca_features_for_variant(df.copy(), variant)
    weight_lookup = static_weights.set_index("component")["static_pc1_weight"]
    weights = [float(weight_lookup.loc[col]) for col in variant.columns]
    z_cols = [f"{variant.name}_z_rolling_{rev.clean_label(col)}" for col in variant.columns]
    score_col = f"{variant.name}_rolling_standardized_fixed_score"
    work[score_col] = work[z_cols].mul(weights, axis=1).sum(axis=1, min_count=len(z_cols))
    work[f"{variant.name}_rolling_standardized_fixed_bounded"] = rev.bounded_score(
        work[score_col]
    )
    return work


def run_rolling_standardized_fixed_regressions(
    df: pd.DataFrame, variant: rev.IndexVariant, window: int
) -> pd.DataFrame:
    work = build_rolling_standardized_fixed_features(df, variant, window)
    score_col = f"{variant.name}_rolling_standardized_fixed_score"
    results = []
    for horizon in rev.CONFIG["HORIZONS"]:
        result = rev.run_hac_ols(
            df=work,
            y_col=f"R_fwd_{horizon}",
            x_cols=rev.CONFIG["CONTROL_COLUMNS"] + [score_col],
            horizon=horizon,
            variant=variant,
            index_method="rolling_standardized_fixed_pca",
            specification=f"{variant.name}_rolling_standardized_fixed_pca_{window}w",
            main_predictor=score_col,
            sample_scope=f"rolling_window_{window}w",
        ).table
        result["rolling_window"] = window
        results.append(result)

    return pd.concat(results, ignore_index=True)


def build_ewma_shrinkage_pca_features(
    df: pd.DataFrame,
    variant: rev.IndexVariant,
    half_life: int,
    shrinkage_alpha: float = SHRINKAGE_ALPHA,
) -> pd.DataFrame:
    """
    Build exponentially weighted shrinkage PCA with static-sign anchoring.

    This replaces a hard rolling window with a forgetting factor. Standardization
    uses only past exponentially weighted moments. The covariance matrix is
    recursively updated using lagged standardized observations, shrunk toward the
    identity matrix, and its first eigenvector is sign-aligned to the static PCA
    loading vector so the index direction remains economically comparable.
    """
    out = df.copy()
    prefix = variant.name
    decay = float(np.exp(np.log(0.5) / half_life))
    update_weight = 1.0 - decay
    eps = float(rev.CONFIG["EPSILON"])
    min_history = max(10, 3 * len(variant.columns) + 4)

    _, static_weights = rev.build_static_pca_features_for_variant(df.copy(), variant)
    anchor = static_weights.set_index("component").loc[list(variant.columns), "static_pc1_weight"]
    anchor_vector = anchor.to_numpy(dtype=float)

    z_cols = []
    for col in variant.columns:
        label = rev.clean_label(col)
        x = out[col].astype(float)
        mean_past = x.ewm(alpha=update_weight, adjust=False, min_periods=min_history).mean().shift(1)
        var_past = x.ewm(alpha=update_weight, adjust=False, min_periods=min_history).var(
            bias=False
        ).shift(1)
        std_past = np.sqrt(var_past.clip(lower=0.0))
        z_col = f"{prefix}_ewma_z_{label}_{half_life}h"
        out[z_col] = np.arcsinh((x - mean_past) / (std_past + eps))
        z_cols.append(z_col)

    score_col = f"{prefix}_ewma_shrinkage_score_{half_life}h"
    bounded_col = f"{prefix}_ewma_shrinkage_bounded_{half_life}h"
    out[score_col] = np.nan
    out[bounded_col] = np.nan

    weight_cols = []
    for col in variant.columns:
        weight_col = f"{prefix}_ewma_shrinkage_w_{rev.clean_label(col)}_{half_life}h"
        out[weight_col] = np.nan
        weight_cols.append(weight_col)

    z_values = out[z_cols].to_numpy(dtype=float)
    n_components = len(variant.columns)
    covariance = np.eye(n_components)
    history_count = 0

    for i in range(len(out)):
        current_z = z_values[i, :]
        if np.isfinite(current_z).all() and history_count >= min_history:
            shrunk_covariance = (
                (1.0 - shrinkage_alpha) * covariance
                + shrinkage_alpha * np.eye(n_components)
            )
            weights, _ = rev.first_pc_from_cov(shrunk_covariance)
            if float(weights @ anchor_vector) < 0:
                weights = -weights
            score = float(current_z @ weights)
            out.loc[out.index[i], score_col] = score
            out.loc[out.index[i], bounded_col] = rev.bounded_score(score)
            out.loc[out.index[i], weight_cols] = weights

        if np.isfinite(current_z).all():
            covariance = decay * covariance + update_weight * np.outer(current_z, current_z)
            history_count += 1

    return out


def run_ewma_shrinkage_pca_regressions(
    df: pd.DataFrame, variant: rev.IndexVariant, half_life: int
) -> pd.DataFrame:
    work = build_ewma_shrinkage_pca_features(df, variant, half_life)
    score_col = f"{variant.name}_ewma_shrinkage_score_{half_life}h"
    results = []
    for horizon in rev.CONFIG["HORIZONS"]:
        result = rev.run_hac_ols(
            df=work,
            y_col=f"R_fwd_{horizon}",
            x_cols=rev.CONFIG["CONTROL_COLUMNS"] + [score_col],
            horizon=horizon,
            variant=variant,
            index_method="ewma_shrinkage_pca",
            specification=f"{variant.name}_ewma_shrinkage_pca_{half_life}h",
            main_predictor=score_col,
            sample_scope=f"ewma_halflife_{half_life}w",
        ).table
        result["rolling_window"] = half_life
        results.append(result)

    return pd.concat(results, ignore_index=True)


def summarize_window_results(results: pd.DataFrame) -> pd.DataFrame:
    score_rows = results[
        (results["parameter"] == results["main_predictor"])
        & (results["status"] == "ok")
    ].copy()
    if score_rows.empty:
        return pd.DataFrame()

    summary = (
        score_rows.groupby(
            [
                "variant",
                "variant_label",
                "rolling_window",
                "index_method",
                "sample_scope",
            ],
            dropna=False,
        )
        .agg(
            min_p_value=("p_value_hac", "min"),
            median_p_value=("p_value_hac", "median"),
            mean_abs_t=("t_stat_hac", lambda s: float(s.abs().mean())),
            significant_10pct=("p_value_hac", lambda s: int((s < 0.10).sum())),
            significant_5pct=("p_value_hac", lambda s: int((s < 0.05).sum())),
            avg_r_squared=("r_squared", "mean"),
            avg_n_obs=("n_obs", "mean"),
            sample_start=("sample_start", "min"),
            sample_end=("sample_end", "max"),
        )
        .reset_index()
        .sort_values(["variant", "rolling_window"])
    )
    summary["reportable_window"] = summary["avg_n_obs"] >= MIN_REPORTABLE_N
    return summary


def rows_for_block(
    results: pd.DataFrame, variant: str, window: int, index_method: str
) -> pd.DataFrame:
    return results[
        (results["variant"] == variant)
        & (results["rolling_window"] == window)
        & (results["index_method"] == index_method)
        & (results["parameter"] == results["main_predictor"])
    ].copy()


def build_latex_table(results: pd.DataFrame) -> str:
    horizon_cols = [rev.EXTENSION_HORIZON_LABELS[h] for h in rev.CONFIG["HORIZONS"]]
    ncols = 1 + len(horizon_cols)
    panels = [
        (
            rev.CONFIG["MAIN_VARIANT"],
            r"\textit{Panel A: Momentum-adjusted futures pressure index}",
            r"Adj. CFPI$_t$",
        ),
        (
            rev.CONFIG["BTC_ROBUSTNESS_VARIANT"],
            r"\textit{Panel B: BTC-only momentum-adjusted futures pressure index}",
            r"BTC Adj. CFPI$_t$",
        ),
    ]
    method_specs = [
        ("rolling_pca", "Rolling PCA", "{variant}_rolling_score"),
        (
            "rolling_standardized_fixed_pca",
            "Rolling-standardized fixed PCA",
            "{variant}_rolling_standardized_fixed_score",
        ),
        (
            "ewma_shrinkage_pca",
            "EW shrinkage anchored PCA",
            "{variant}_ewma_shrinkage_score_{window}h",
        ),
    ]

    body_lines = ["Variable & " + " & ".join(horizon_cols) + r" \\", r"\midrule"]
    for panel_idx, (variant, panel_title, row_label) in enumerate(panels):
        if panel_idx > 0:
            body_lines.append(r"\midrule")
        body_lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{{panel_title}}} \\")

        for window_idx, window in enumerate(WINDOWS):
            if window_idx > 0:
                body_lines.append(r"\addlinespace")

            active_method_specs = [
                spec
                for spec in method_specs
                if spec[0] != "ewma_shrinkage_pca" or window in EWMA_HALFLIVES
            ]
            for method_idx, (index_method, method_label, predictor_template) in enumerate(
                active_method_specs
            ):
                if method_idx > 0:
                    body_lines.append(r"\addlinespace[0.25em]")

                block = rows_for_block(results, variant, window, index_method)
                predictor = predictor_template.format(variant=variant, window=window)
                ok_horizons = int((block["status"] == "ok").sum()) if "status" in block else 0
                max_n = block["n_obs"].max() if "n_obs" in block else 0
                row_prefix = (
                    rf"{window}-week half-life {method_label}"
                    if index_method == "ewma_shrinkage_pca"
                    else rf"{window}-week {method_label}"
                )

                if ok_horizons < len(rev.CONFIG["HORIZONS"]) or max_n < MIN_REPORTABLE_N:
                    body_lines.append(
                        rf"{row_prefix} & "
                        rf"\multicolumn{{{len(horizon_cols)}}}{{c}}{{Insufficient effective sample}} \\"
                    )
                    continue

                coef_cells = [row_prefix]
                tstat_cells = [""]
                r2_cells = [r"R$^2$"]
                n_cells = ["N"]
                for horizon in rev.CONFIG["HORIZONS"]:
                    coef = rev.get_reg_value(block, horizon, predictor, "coef")
                    pval = rev.get_reg_value(block, horizon, predictor, "p_value_hac")
                    tval = rev.get_reg_value(block, horizon, predictor, "t_stat_hac")
                    coef_cells.append(rev.format_coef_latex(coef, pval))
                    tstat_cells.append(rev.format_tstat_latex(tval))
                    r2_cells.append(
                        rev.format_num_latex(
                            rev.get_horizon_value(block, horizon, "r_squared"), 3
                        )
                    )
                    n_obs = rev.get_horizon_value(block, horizon, "n_obs")
                    n_cells.append(str(int(n_obs)) if pd.notna(n_obs) else "")

                body_lines.append(" & ".join(coef_cells) + r" \\")
                body_lines.append(" & ".join(tstat_cells) + r" \\")
                body_lines.append(" & ".join(r2_cells) + r" \\")
                body_lines.append(" & ".join(n_cells) + r" \\")

    description = (
        r"This table reports rolling-window sensitivity checks for alternative "
        r"calibration windows and exponentially weighted alternatives. Rolling "
        r"PCA re-estimates both standardization and PCA loadings from lagged "
        r"observations. The rolling-standardized fixed PCA index uses lagged "
        r"rolling standardization but keeps the static PCA loading vector fixed. "
        r"The EW shrinkage anchored PCA replaces the hard rolling window with an "
        r"exponentially weighted covariance matrix; EW rows are labeled by "
        r"half-life rather than by a truncated window. The covariance matrix is "
        r"shrunk toward the identity matrix and the first component is sign-aligned "
        r"to the static PCA loading vector. All regressions include $R_t$ and "
        r"$Google_t$ as controls. HAC/Newey--West t-statistics are in parentheses. "
        r"Windows with too few usable observations across the forecast horizons "
        r"are flagged rather than interpreted. *, **, and *** denote significance "
        r"at the 10\%, 5\%, and 1\% levels."
    )
    return rev.make_manual_latex_table(
        caption="Rolling and exponentially weighted PCA sensitivity.",
        label="tbl-revised-extension-rolling-window-sensitivity",
        body_lines=body_lines,
        description=description,
        ncols=ncols,
    )


def build_oi_momentum_correlation_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Correlate raw and momentum-adjusted OI growth with current and lagged returns.

    This diagnostic checks whether the momentum adjustment actually reduces the
    contemporaneous return component in the OI variable used inside the PCAs.
    """
    work = df.sort_values(rev.CONFIG["DATE_COLUMN"]).copy()
    for lag in CORRELATION_RETURN_LAGS:
        work[f"R_t_lag{lag}"] = work["R_t"].shift(lag)

    specs = {
        "pooled_cftc": {
            "labels": {
                "R_t": r"R$_t$",
                "R_t_lag1": r"R$_{t-1}$",
                "R_t_lag2": r"R$_{t-2}$",
                "R_t_lag4": r"R$_{t-4}$",
                "cftc_open_interest_growth": r"Raw $\Delta OI_t$",
                "cftc_open_interest_growth_resid": r"Momentum-adj. $\Delta OI_t$",
            },
            "columns": [
                "R_t",
                "R_t_lag1",
                "R_t_lag2",
                "R_t_lag4",
                "cftc_open_interest_growth",
                "cftc_open_interest_growth_resid",
            ],
        },
        "btc_only": {
            "labels": {
                "R_t": r"R$_t$",
                "R_t_lag1": r"R$_{t-1}$",
                "R_t_lag2": r"R$_{t-2}$",
                "R_t_lag4": r"R$_{t-4}$",
                "btc_open_interest_growth": r"BTC raw $\Delta OI_t$",
                "btc_open_interest_growth_resid": r"BTC momentum-adj. $\Delta OI_t$",
            },
            "columns": [
                "R_t",
                "R_t_lag1",
                "R_t_lag2",
                "R_t_lag4",
                "btc_open_interest_growth",
                "btc_open_interest_growth_resid",
            ],
        },
    }

    matrices = {}
    for panel, spec in specs.items():
        columns = [col for col in spec["columns"] if col in work.columns]
        corr = work[columns].replace([np.inf, -np.inf], np.nan).corr()
        corr = corr.rename(index=spec["labels"], columns=spec["labels"])
        matrices[panel] = corr
    return matrices


def flatten_correlation_matrices(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for panel, matrix in matrices.items():
        for row_label, values in matrix.iterrows():
            for col_label, value in values.items():
                rows.append(
                    {
                        "panel": panel,
                        "row": row_label,
                        "column": col_label,
                        "correlation": value,
                    }
                )
    return pd.DataFrame(rows)


def build_oi_momentum_correlation_latex(matrices: dict[str, pd.DataFrame]) -> str:
    panel_titles = {
        "pooled_cftc": r"\textit{Panel A: Pooled BTC/ETH CFTC open-interest growth}",
        "btc_only": r"\textit{Panel B: BTC-only open-interest growth}",
    }

    ncols = 1 + max(len(matrix.columns) for matrix in matrices.values())
    body_lines = []
    for panel_idx, (panel, matrix) in enumerate(matrices.items()):
        if panel_idx == 0:
            body_lines.append("Variable & " + " & ".join(matrix.columns) + r" \\")
            body_lines.append(r"\midrule")
        else:
            body_lines.append(r"\midrule")
        body_lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{{panel_titles[panel]}}} \\")
        for row_label, values in matrix.iterrows():
            cells = [row_label] + [rev.format_num_latex(value, 3) for value in values]
            body_lines.append(" & ".join(cells) + r" \\")

    description = (
        r"This table reports pairwise correlations between current and lagged "
        r"weekly returns, the raw open-interest growth variable, and the "
        r"momentum-adjusted open-interest growth variable used in the PCA "
        r"indices. The momentum-adjusted variable is constructed by removing "
        r"the contemporaneous $R_t$ component from raw open-interest growth, so "
        r"its correlation with $R_t$ should be mechanically close to zero. "
        r"Lagged-return correlations are reported as a robustness check on "
        r"remaining return-momentum exposure."
    )
    return rev.make_manual_latex_table(
        caption="Return-momentum correlations for raw and adjusted open-interest growth.",
        label="tbl-revised-extension-oi-momentum-correlations",
        body_lines=body_lines,
        description=description,
        ncols=ncols,
        column_format="l" + "r" * (ncols - 1),
        arraystretch="1.08",
        table_width=r"0.94\linewidth",
    )


def main() -> None:
    rev.ensure_output_dir()
    df = rev.load_revised_input_dataset()
    df = rev.build_futures_positioning_feature_pool(df)

    variants = [get_variant(name) for name in VARIANT_NAMES]
    results = []
    for variant in variants:
        for window in WINDOWS:
            results.append(run_rolling_pca_regressions(df, variant, window))
            results.append(run_rolling_standardized_fixed_regressions(df, variant, window))
        for half_life in EWMA_HALFLIVES:
            results.append(run_ewma_shrinkage_pca_regressions(df, variant, half_life))

    results_df = pd.concat(results, ignore_index=True)
    summary = summarize_window_results(results_df)
    oi_momentum_matrices = build_oi_momentum_correlation_matrices(df)
    oi_momentum_correlations = flatten_correlation_matrices(oi_momentum_matrices)

    output_dir = Path(rev.CONFIG["OUTPUT_DIR"])
    results_df.to_csv(output_dir / "rolling_pca_window_sensitivity_results.csv", index=False)
    summary.to_csv(output_dir / "rolling_pca_window_sensitivity_summary.csv", index=False)
    oi_momentum_correlations.to_csv(
        output_dir / "oi_momentum_correlation_matrix.csv", index=False
    )

    latex_dir = output_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    (latex_dir / "revised_extension_rolling_window_sensitivity.tex").write_text(
        build_latex_table(results_df)
    )
    (latex_dir / "revised_extension_oi_momentum_correlations.tex").write_text(
        build_oi_momentum_correlation_latex(oi_momentum_matrices)
    )

    print("\nRolling PCA window sensitivity summary:")
    print(summary.to_string(index=False))
    print("\nOpen-interest momentum correlation matrices:")
    for panel, matrix in oi_momentum_matrices.items():
        print(f"\n{panel}:")
        print(matrix.to_string())
    print("\nSaved rolling-window sensitivity outputs to:")
    print(output_dir)


if __name__ == "__main__":
    main()
