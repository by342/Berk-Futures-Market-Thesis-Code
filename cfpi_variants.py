#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import re

SCRIPT_DIR_FOR_CACHE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR_FOR_CACHE / ".matplotlib"))

import numpy as np
import pandas as pd


# The Config

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.parent.name == "FI-thesis-template":
    FI_TEMPLATE_DIR = SCRIPT_DIR.parent
    PROJECT_ROOT = FI_TEMPLATE_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR.parent
    FI_TEMPLATE_DIR = PROJECT_ROOT / "FI-thesis-template"

DEFAULT_INPUT = FI_TEMPLATE_DIR / "outputs" / "final_merged_regression_dataset.csv"
FALLBACK_INPUT = FI_TEMPLATE_DIR / "outputs" / "cfpi_dataset.csv"
ASSET_LEVEL_INPUT = FI_TEMPLATE_DIR / "outputs" / "crypto_positioning_components_asset_level.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"


CONFIG = {
    "INPUT_DATASET": DEFAULT_INPUT,
    "FALLBACK_INPUT_DATASET": FALLBACK_INPUT,
    "ASSET_LEVEL_COMPONENTS_DATASET": ASSET_LEVEL_INPUT,
    "OUTPUT_DIR": OUTPUT_DIR,
    "HORIZONS": [1, 2, 3, 4, 6, 8],
    "CONTROL_COLUMNS": ["R_t", "Google_t"],
    "ROLLING_WINDOW": 36,
    "DYNAMIC_PCA_HALFLIFE": 36,
    "DYNAMIC_PCA_SHRINKAGE_ALPHA": 0.10,
    "EPSILON": 1e-8,
    "KAPPA": 1.0,
    "MIN_OBS_EXTRA": 2,
    "HAC_MAXLAGS": "horizon",
    "DATE_COLUMN": "date",
    "SAVE_LATEX_TABLES": True,
    "MAIN_VARIANT": "futures_pressure_gross_resid_3v",
    "RAW_BASELINE_VARIANT": "futures_pressure_gross_3v",
    "BTC_ROBUSTNESS_VARIANT": "btc_price_quantity_resid_2v",
    "BTC_RAW_BASELINE_VARIANT": "btc_price_quantity_pressure_2v",
}


@dataclass(frozen=True)
class IndexVariant:
    name: str
    label: str
    columns: tuple[str, ...]
    description: str
    is_new_alternative: bool = True


@dataclass
class RegressionResult:
    table: pd.DataFrame
    n_obs: int
    status: str


INDEX_VARIANTS = [
    IndexVariant(
        name="futures_price_quantity_pressure_2v",
        label="Futures price-quantity pressure",
        columns=(
            "futures_basis",
            "cftc_open_interest_growth",
        ),
        description=(
            "Parsimonious futures pressure index: the futures basis measures the "
            "price side of pressure, while open-interest growth measures the "
            "quantity side of new futures risk-bearing."
        ),
    ),
    IndexVariant(
        name="futures_pressure_short_3v",
        label="Futures pressure with short imbalance",
        columns=(
            "futures_basis",
            "cftc_open_interest_growth",
            "cftc_short_position_pressure",
        ),
        description=(
            "Main three-variable candidate: combines futures basis, open-interest "
            "growth, and inverted managed-money short share. Higher values indicate "
            "costlier futures exposure, expanding depth, and less short pressure."
        ),
    ),
    IndexVariant(
        name="futures_pressure_gross_3v",
        label="Futures pressure with gross participation",
        columns=(
            "futures_basis",
            "cftc_open_interest_growth",
            "cftc_gross_position_share",
        ),
        description=(
            "Alternative three-variable candidate: combines the futures basis, "
            "open-interest growth, and gross managed-money participation. This "
            "tests whether the signal is broad participation rather than the "
            "direction of net positioning."
        ),
    ),
    IndexVariant(
        name="futures_pressure_gross_resid_3v",
        label="Momentum-adjusted futures pressure with gross participation",
        columns=(
            "futures_basis",
            "cftc_open_interest_growth_resid",
            "cftc_gross_position_share",
        ),
        description=(
            "Main momentum-adjusted three-variable index: combines futures-basis "
            "pricing pressure, the component of open-interest growth orthogonal "
            "to contemporaneous weekly returns, and gross managed-money "
            "participation. This keeps the futures-pressure interpretation while "
            "reducing the scope for a pure return-momentum explanation."
        ),
    ),
    IndexVariant(
        name="futures_pressure_net_3v",
        label="Futures pressure with net positioning",
        columns=(
            "futures_basis",
            "cftc_open_interest_growth",
            "cftc_net_position_share",
        ),
        description=(
            "Net-positioning version of the futures pressure index. It keeps the "
            "price and quantity components fixed and asks whether directional "
            "managed-money exposure adds information."
        ),
    ),
    IndexVariant(
        name="btc_price_quantity_pressure_2v",
        label="BTC-only price-quantity pressure",
        columns=(
            "btc_futures_basis",
            "btc_open_interest_growth",
        ),
        description=(
            "BTC-only benchmark for the asset-scope critique. It uses the same "
            "price-quantity logic without pooling Bitcoin and Ethereum futures."
        ),
    ),
    IndexVariant(
        name="btc_price_quantity_resid_2v",
        label="BTC-only momentum-adjusted price-quantity pressure",
        columns=(
            "btc_futures_basis",
            "btc_open_interest_growth_resid",
        ),
        description=(
            "BTC-only momentum-adjusted benchmark: combines Bitcoin futures-basis "
            "pricing pressure with Bitcoin open-interest growth after removing "
            "the component explained by contemporaneous weekly returns."
        ),
    ),
    IndexVariant(
        name="btc_price_quantity_net_3v",
        label="BTC-only pressure with net positioning",
        columns=(
            "btc_futures_basis",
            "btc_open_interest_growth",
            "btc_net_position_share",
        ),
        description=(
            "BTC-only three-variable benchmark that adds managed-money net "
            "positioning to the price-quantity futures pressure construction."
        ),
    ),
]


# Helpers


def ensure_output_dir() -> None:
    CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)


def clean_label(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")


def print_date_range(df: pd.DataFrame, label: str, date_col: str = "date") -> None:
    if df.empty:
        print(f"{label}: empty dataframe")
        return
    print(
        f"{label}: shape={df.shape}, "
        f"date range={df[date_col].min().date()} to {df[date_col].max().date()}"
    )


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns are:\n{list(df.columns)}"
        )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def residualize_on_controls(
    df: pd.DataFrame, y_col: str, control_cols: list[str], output_col: str
) -> pd.Series:
    """
    This is where I residualize wrt contemporenaous returns
    """
    work = df[[y_col] + control_cols].replace([np.inf, -np.inf], np.nan).dropna()
    residuals = pd.Series(np.nan, index=df.index, name=output_col)
    if len(work) <= len(control_cols) + 1:
        return residuals

    x = np.column_stack([np.ones(len(work)), work[control_cols].to_numpy(dtype=float)])
    y = work[y_col].to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals.loc[work.index] = y - x @ beta
    return residuals


def load_revised_input_dataset() -> pd.DataFrame:
    input_path = Path(CONFIG["INPUT_DATASET"])
    fallback_path = Path(CONFIG["FALLBACK_INPUT_DATASET"])

    if input_path.exists():
        path = input_path
    elif fallback_path.exists():
        path = fallback_path
    else:
        raise FileNotFoundError(
            "Could not find the cleaned regression dataset. Expected either:\n"
            f"  {input_path}\n"
            f"  {fallback_path}\n"
            "Run 03_cfpi_construction.ipynb first so the cleaned weekly "
            "dataset exists."
        )

    df = pd.read_csv(path)
    date_col = CONFIG["DATE_COLUMN"]
    require_columns(df, [date_col], "input dataset")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if df[date_col].isna().any():
        bad = df.loc[df[date_col].isna()].head(10)
        raise ValueError(f"Some dates could not be parsed:\n{bad}")

    required = (
        [date_col]
        + CONFIG["CONTROL_COLUMNS"]
        + [f"R_fwd_{h}" for h in CONFIG["HORIZONS"]]
    )
    require_columns(df, required, "input dataset")

    df = df.sort_values(date_col).reset_index(drop=True)
    duplicated_dates = df[df[date_col].duplicated(keep=False)]
    if not duplicated_dates.empty:
        raise ValueError(
            "The regression dataset must have one observation per weekly date. "
            f"Duplicate dates found:\n{duplicated_dates[[date_col]].head(20)}"
        )

    print(f"Loaded input dataset from: {path}")
    print_date_range(df, "input dataset", date_col=date_col)
    return df


def load_asset_level_components() -> pd.DataFrame:
    path = Path(CONFIG["ASSET_LEVEL_COMPONENTS_DATASET"])
    if not path.exists():
        raise FileNotFoundError(
            "Could not find the asset-level futures components dataset:\n"
            f"  {path}\n"
            "Run 03_cfpi_construction.ipynb first"
        )

    asset = pd.read_csv(path)
    date_col = CONFIG["DATE_COLUMN"]
    required = [
        date_col,
        "asset",
        "am_long_notional_usd",
        "am_short_notional_usd",
        "open_interest_notional_usd",
        "log_futures_premium",
    ]
    require_columns(asset, required, "asset-level components dataset")
    asset[date_col] = pd.to_datetime(asset[date_col], errors="coerce")
    asset = asset.sort_values([date_col, "asset"]).reset_index(drop=True)

    if asset[date_col].isna().any():
        bad = asset.loc[asset[date_col].isna()].head(10)
        raise ValueError(f"Some asset-level dates could not be parsed:\n{bad}")

    print(f"Loaded asset-level futures components from: {path}")
    print_date_range(asset, "asset-level futures components", date_col=date_col)
    return asset


# The futures positioning variables


def build_futures_positioning_feature_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    I construct the variables in such a way where positive values reflect bullish sentiment, and vice versa.
    Multiply by -1 when needed
    """
    date_col = CONFIG["DATE_COLUMN"]
    asset = load_asset_level_components()
    asset = asset.copy()
    asset["asset"] = asset["asset"].astype(str).str.upper()

    asset["am_net_notional_usd"] = (
        asset["am_long_notional_usd"] - asset["am_short_notional_usd"]
    )
    asset["am_gross_notional_usd"] = (
        asset["am_long_notional_usd"] + asset["am_short_notional_usd"]
    )
    asset["net_position_share"] = safe_divide(
        asset["am_net_notional_usd"], asset["open_interest_notional_usd"]
    )
    asset["gross_position_share"] = safe_divide(
        asset["am_gross_notional_usd"], asset["open_interest_notional_usd"]
    )
    asset["long_position_share"] = safe_divide(
        asset["am_long_notional_usd"], asset["open_interest_notional_usd"]
    )
    asset["short_position_pressure"] = -safe_divide(
        asset["am_short_notional_usd"], asset["open_interest_notional_usd"]
    )
    asset["log_open_interest"] = np.log(asset["open_interest_notional_usd"].replace(0, np.nan))
    asset["open_interest_growth"] = asset.groupby("asset")["log_open_interest"].diff()

    aggregate_rows = []
    for date, group in asset.groupby(date_col, sort=True):
        total_oi = group["open_interest_notional_usd"].sum(min_count=1)
        long_notional = group["am_long_notional_usd"].sum(min_count=1)
        short_notional = group["am_short_notional_usd"].sum(min_count=1)
        gross_notional = long_notional + short_notional
        net_notional = long_notional - short_notional
        aggregate_rows.append(
            {
                date_col: date,
                "cftc_net_position_share": net_notional / total_oi if total_oi else np.nan,
                "cftc_gross_position_share": gross_notional / total_oi if total_oi else np.nan,
                "cftc_long_position_share": long_notional / total_oi if total_oi else np.nan,
                "cftc_short_position_pressure": (
                    -short_notional / total_oi if total_oi else np.nan
                ),
                "cftc_log_open_interest": np.log(total_oi) if total_oi and total_oi > 0 else np.nan,
                "futures_basis": weighted_average(
                    group["log_futures_premium"],
                    group["open_interest_notional_usd"],
                ),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows).sort_values(date_col)
    aggregate["cftc_open_interest_growth"] = aggregate["cftc_log_open_interest"].diff()

    pivot_cols = [
        "net_position_share",
        "gross_position_share",
        "long_position_share",
        "short_position_pressure",
        "log_open_interest",
        "open_interest_growth",
        "log_futures_premium",
    ]
    asset_pivot = asset[[date_col, "asset"] + pivot_cols].pivot(
        index=date_col, columns="asset", values=pivot_cols
    )
    asset_pivot.columns = [
        f"{asset_name.lower()}_{feature}"
        for feature, asset_name in asset_pivot.columns.to_flat_index()
    ]
    asset_pivot = asset_pivot.reset_index()
    asset_pivot = asset_pivot.rename(
        columns={
            "btc_log_futures_premium": "btc_futures_basis",
            "eth_log_futures_premium": "eth_futures_basis",
        }
    )

    out = df.merge(aggregate, on=date_col, how="left")
    out = out.merge(asset_pivot, on=date_col, how="left")
    out = out.sort_values(date_col).reset_index(drop=True)

    if "total_open_interest_notional_usd" in out:
        fallback_log_oi = np.log(out["total_open_interest_notional_usd"].replace(0, np.nan))
        out["cftc_log_open_interest"] = out["cftc_log_open_interest"].fillna(fallback_log_oi)

    if {"btc_futures_basis", "eth_futures_basis"}.issubset(out.columns):
        out["btc_minus_eth_basis"] = out["btc_futures_basis"] - out["eth_futures_basis"]

    out["cftc_open_interest_growth_resid"] = residualize_on_controls(
        out,
        "cftc_open_interest_growth",
        ["R_t"],
        "cftc_open_interest_growth_resid",
    )
    if "btc_open_interest_growth" in out.columns:
        out["btc_open_interest_growth_resid"] = residualize_on_controls(
            out,
            "btc_open_interest_growth",
            ["R_t"],
            "btc_open_interest_growth_resid",
        )

    required_variant_cols = sorted({col for variant in INDEX_VARIANTS for col in variant.columns})
    require_columns(out, required_variant_cols, "futures positioning feature pool")
    print("\nCryptocurrency futures positioning variants:")
    for variant in INDEX_VARIANTS:
        marker = "alternative" if variant.is_new_alternative else "reference"
        print(f"- {variant.name} ({marker}): {', '.join(variant.columns)}")
    return out


# PCA construction!

def first_pc_from_cov(cov: np.ndarray) -> tuple[np.ndarray, float]:
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigval = float(eigvals[order[0]])
    eigvec = eigvecs[:, order[0]].astype(float)
    eigvec = sign_normalize_pc1(eigvec)
    return eigvec, eigval


def sign_normalize_pc1(weights: np.ndarray) -> np.ndarray:
    """
    Higher PCA scores should (hopefully) mean stronger futures positioning pressure.
    """
    if np.dot(weights, np.ones_like(weights)) < 0:
        weights = -weights
    return weights


def bounded_score(score: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 100.0 * np.tanh(CONFIG["KAPPA"] * score)


def build_rolling_pca_features_for_variant(
    df: pd.DataFrame, variant: IndexVariant
) -> pd.DataFrame:
    """
    Where the magic happens!
    """
    if len(variant.columns) not in (2, 3):
        raise ValueError(f"{variant.name} must contain exactly two or three variables.")

    out = df.copy()
    prefix = variant.name
    window = int(CONFIG["ROLLING_WINDOW"])
    eps = float(CONFIG["EPSILON"])

    z_cols = []
    weight_cols = []
    for col in variant.columns:
        label = clean_label(col)
        sigma_past = out[col].rolling(window=window, min_periods=window).std(ddof=1).shift(1)
        out[f"{prefix}_rolling_sigma_{label}"] = sigma_past
        out[f"{prefix}_psi_rolling_{label}"] = np.arcsinh(out[col] / (sigma_past + eps))

        psi_col = f"{prefix}_psi_rolling_{label}"
        mu_past = out[psi_col].rolling(window=window, min_periods=window).mean().shift(1)
        sd_past = out[psi_col].rolling(window=window, min_periods=window).std(ddof=1).shift(1)
        z_col = f"{prefix}_z_rolling_{label}"
        out[z_col] = (out[psi_col] - mu_past) / (sd_past + eps)
        z_cols.append(z_col)

        weight_col = f"{prefix}_rolling_w_{label}"
        out[weight_col] = np.nan
        weight_cols.append(weight_col)

    score_col = f"{prefix}_rolling_score"
    bounded_col = f"{prefix}_rolling_bounded"
    eigen_col = f"{prefix}_rolling_eigenvalue_1"
    out[score_col] = np.nan
    out[bounded_col] = np.nan
    out[eigen_col] = np.nan

    z_values = out[z_cols].to_numpy(dtype=float)
    for i in range(len(out)):
        if i < window:
            continue

        current_z = z_values[i, :]
        history = z_values[i - window : i, :]
        if not np.isfinite(current_z).all():
            continue
        if not np.isfinite(history).all():
            continue

        cov = np.cov(history, rowvar=False, ddof=1)
        weights, eigval = first_pc_from_cov(cov)
        score = float(current_z @ weights)

        out.loc[out.index[i], weight_cols] = weights
        out.loc[out.index[i], eigen_col] = eigval
        out.loc[out.index[i], score_col] = score
        out.loc[out.index[i], bounded_col] = bounded_score(score)

    return out


def build_static_pca_features_for_variant(          # the static one
    df: pd.DataFrame, variant: IndexVariant
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(variant.columns) not in (2, 3):
        raise ValueError(f"{variant.name} must contain exactly two or three variables.")

    out = df.copy()
    prefix = variant.name
    eps = float(CONFIG["EPSILON"])

    z_cols = []
    for col in variant.columns:
        label = clean_label(col)
        sigma = out[col].std(ddof=1)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Cannot compute static PCA: {col} has non-positive sample std.")

        psi_col = f"{prefix}_psi_static_{label}"
        z_col = f"{prefix}_z_static_{label}"
        out[psi_col] = np.arcsinh(out[col] / (sigma + eps))
        mu = out[psi_col].mean()
        sd = out[psi_col].std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Cannot compute static PCA: transformed {col} has non-positive std.")
        out[z_col] = (out[psi_col] - mu) / (sd + eps)
        z_cols.append(z_col)

    valid = out[z_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < len(variant.columns) + CONFIG["MIN_OBS_EXTRA"]:
        raise ValueError(
            "Not enough complete observations to estimate static PCA. "
            f"Variant={variant.name}, complete rows={len(valid)}, columns={z_cols}"
        )

    cov = np.cov(valid.to_numpy(dtype=float), rowvar=False, ddof=1)
    weights, eigval = first_pc_from_cov(cov)

    score_col = f"{prefix}_static_score"
    bounded_col = f"{prefix}_static_bounded"
    eigen_col = f"{prefix}_static_eigenvalue_1"
    out[score_col] = out[z_cols].to_numpy(dtype=float) @ weights
    out[bounded_col] = bounded_score(out[score_col])
    out[eigen_col] = eigval

    weight_rows = []
    for col, weight in zip(variant.columns, weights):
        label = clean_label(col)
        out[f"{prefix}_static_w_{label}"] = weight
        weight_rows.append(
            {
                "variant": variant.name,
                "variant_label": variant.label,
                "component": col,
                "static_pc1_weight": weight,
                "static_eigenvalue_1": eigval,
                "n_complete_rows": len(valid),
                "description": variant.description,
            }
        )

    return out, pd.DataFrame(weight_rows)


def build_ewma_shrinkage_pca_features_for_variant(
    df: pd.DataFrame, variant: IndexVariant, static_weights: pd.DataFrame
) -> pd.DataFrame:
    """
    I build the no-lookahead dynamic PCA features using EW covariance shrink
    """
    if len(variant.columns) not in (2, 3):
        raise ValueError(f"{variant.name} must contain exactly two or three variables.")

    out = df.copy()
    prefix = variant.name
    half_life = int(CONFIG["DYNAMIC_PCA_HALFLIFE"])
    shrinkage_alpha = float(CONFIG["DYNAMIC_PCA_SHRINKAGE_ALPHA"])
    decay = float(np.exp(np.log(0.5) / half_life))
    update_weight = 1.0 - decay
    eps = float(CONFIG["EPSILON"])
    min_history = max(10, 3 * len(variant.columns) + 4)

    anchor = static_weights.set_index("component").loc[
        list(variant.columns), "static_pc1_weight"
    ]
    anchor_vector = anchor.to_numpy(dtype=float)

    z_cols = []
    for col in variant.columns:
        label = clean_label(col)
        x = out[col].astype(float)
        mean_past = (
            x.ewm(alpha=update_weight, adjust=False, min_periods=min_history)
            .mean()
            .shift(1)
        )
        var_past = (
            x.ewm(alpha=update_weight, adjust=False, min_periods=min_history)
            .var(bias=False)
            .shift(1)
        )
        std_past = np.sqrt(var_past.clip(lower=0.0))
        z_col = f"{prefix}_dynamic_z_{label}"
        out[z_col] = np.arcsinh((x - mean_past) / (std_past + eps))
        z_cols.append(z_col)

    score_col = f"{prefix}_dynamic_score"
    bounded_col = f"{prefix}_dynamic_bounded"
    out[score_col] = np.nan
    out[bounded_col] = np.nan

    weight_cols = []
    for col in variant.columns:
        weight_col = f"{prefix}_dynamic_w_{clean_label(col)}"
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
            weights, _ = first_pc_from_cov(shrunk_covariance)
            if float(weights @ anchor_vector) < 0:
                weights = -weights
            score = float(current_z @ weights)
            out.loc[out.index[i], score_col] = score
            out.loc[out.index[i], bounded_col] = bounded_score(score)
            out.loc[out.index[i], weight_cols] = weights

        if np.isfinite(current_z).all():
            covariance = decay * covariance + update_weight * np.outer(current_z, current_z)
            history_count += 1

    return out


def build_all_index_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    static_weight_tables = []
    for variant in INDEX_VARIANTS:
        out, static_weights = build_static_pca_features_for_variant(out, variant)
        out = build_ewma_shrinkage_pca_features_for_variant(out, variant, static_weights)
        static_weight_tables.append(static_weights)
    return out, pd.concat(static_weight_tables, ignore_index=True)


# Regressions


def hac_lags_for_horizon(horizon: int) -> int:
    rule = CONFIG["HAC_MAXLAGS"]
    if rule == "horizon":
        return int(horizon)
    if isinstance(rule, dict):
        return int(rule.get(horizon, horizon))
    return int(rule)


@dataclass
class OLSHACFit:
    param_names: list[str]
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    nobs: int
    rsquared: float
    rsquared_adj: float


def normal_two_sided_pvalue(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def fit_ols_hac(y: pd.Series, x: pd.DataFrame, maxlags: int) -> OLSHACFit:

    param_names = ["const"] + list(x.columns)
    y_arr = y.to_numpy(dtype=float)
    x_arr = np.column_stack([np.ones(len(x), dtype=float), x.to_numpy(dtype=float)])
    nobs, n_params = x_arr.shape

    beta = np.linalg.lstsq(x_arr, y_arr, rcond=None)[0]
    fitted = x_arr @ beta
    resid = y_arr - fitted

    xtx_inv = np.linalg.pinv(x_arr.T @ x_arr)
    xu = x_arr * resid[:, None]
    meat = xu.T @ xu
    lag_limit = min(int(maxlags), max(nobs - 1, 0))
    for lag in range(1, lag_limit + 1):
        weight = 1.0 - lag / (lag_limit + 1.0)
        gamma = xu[lag:].T @ xu[:-lag]
        meat += weight * (gamma + gamma.T)

    cov = xtx_inv @ meat @ xtx_inv
    diag = np.diag(cov)
    se = np.sqrt(np.where(diag >= 0, diag, np.nan))
    tvalues = beta / se
    pvalues = np.array([normal_two_sided_pvalue(t) for t in tvalues])

    centered = y_arr - y_arr.mean()
    ss_total = float(centered @ centered)
    ss_resid = float(resid @ resid)
    rsquared = 1.0 - ss_resid / ss_total if ss_total > 0 else np.nan
    if nobs > n_params and np.isfinite(rsquared):
        rsquared_adj = 1.0 - (1.0 - rsquared) * (nobs - 1.0) / (nobs - n_params)
    else:
        rsquared_adj = np.nan

    return OLSHACFit(
        param_names=param_names,
        params=pd.Series(beta, index=param_names),
        bse=pd.Series(se, index=param_names),
        tvalues=pd.Series(tvalues, index=param_names),
        pvalues=pd.Series(pvalues, index=param_names),
        nobs=nobs,
        rsquared=rsquared,
        rsquared_adj=rsquared_adj,
    )


def run_hac_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    horizon: int,
    variant: IndexVariant,
    index_method: str,
    specification: str,
    main_predictor: str,
    sample_scope: str,
) -> RegressionResult:
    require_columns(df, [y_col] + x_cols, f"{specification} regression input")

    reg = df[[CONFIG["DATE_COLUMN"], y_col] + x_cols].copy()
    reg = reg.replace([np.inf, -np.inf], np.nan).dropna()
    min_obs = len(x_cols) + CONFIG["MIN_OBS_EXTRA"] + 1
    hac_lags = hac_lags_for_horizon(horizon)

    if len(reg) < min_obs:
        rows = []
        for param in ["const"] + x_cols:
            rows.append(
                {
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "is_new_alternative": variant.is_new_alternative,
                    "index_method": index_method,
                    "specification": specification,
                    "sample_scope": sample_scope,
                    "horizon": horizon,
                    "dependent_variable": y_col,
                    "main_predictor": main_predictor,
                    "parameter": param,
                    "coef": np.nan,
                    "std_err_hac": np.nan,
                    "t_stat_hac": np.nan,
                    "p_value_hac": np.nan,
                    "n_obs": len(reg),
                    "r_squared": np.nan,
                    "adj_r_squared": np.nan,
                    "hac_maxlags": hac_lags,
                    "sample_start": pd.NaT,
                    "sample_end": pd.NaT,
                    "status": "too_few_observations",
                }
            )
        return RegressionResult(pd.DataFrame(rows), len(reg), "too_few_observations")

    y = reg[y_col].astype(float)
    x = reg[x_cols].astype(float)
    model = fit_ols_hac(y, x, maxlags=hac_lags)

    rows = []
    for param in model.param_names:
        rows.append(
            {
                "variant": variant.name,
                "variant_label": variant.label,
                "is_new_alternative": variant.is_new_alternative,
                "index_method": index_method,
                "specification": specification,
                "sample_scope": sample_scope,
                "horizon": horizon,
                "dependent_variable": y_col,
                "main_predictor": main_predictor,
                "parameter": param,
                "coef": model.params[param],
                "std_err_hac": model.bse[param],
                "t_stat_hac": model.tvalues[param],
                "p_value_hac": model.pvalues[param],
                "n_obs": int(model.nobs),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "hac_maxlags": hac_lags,
                "sample_start": reg[CONFIG["DATE_COLUMN"]].min(),
                "sample_end": reg[CONFIG["DATE_COLUMN"]].max(),
                "status": "ok",
            }
        )

    return RegressionResult(pd.DataFrame(rows), int(model.nobs), "ok")


def regression_specs_for_variant(variant: IndexVariant) -> list[dict[str, object]]:
    prefix = variant.name
    return [
        {
            "index_method": "linear_components",
            "specification": f"{prefix}_linear_components",
            "x_cols": CONFIG["CONTROL_COLUMNS"] + list(variant.columns),
            "main_predictor": "+".join(variant.columns),
        },
        {
            "index_method": "dynamic_pca",
            "specification": f"{prefix}_dynamic_pca",
            "x_cols": CONFIG["CONTROL_COLUMNS"] + [f"{prefix}_dynamic_score"],
            "main_predictor": f"{prefix}_dynamic_score",
        },
        {
            "index_method": "static_pca",
            "specification": f"{prefix}_static_pca",
            "x_cols": CONFIG["CONTROL_COLUMNS"] + [f"{prefix}_static_score"],
            "main_predictor": f"{prefix}_static_score",
        },
    ]


def run_all_revised_regressions(df: pd.DataFrame, sample_scope: str = "full_available") -> pd.DataFrame:
    results = []

    for variant in INDEX_VARIANTS:
        dynamic_score = f"{variant.name}_dynamic_score"
        static_score = f"{variant.name}_static_score"

        for h in CONFIG["HORIZONS"]:
            y_col = f"R_fwd_{h}"
            work = df.copy()

            if sample_scope == "matched_dynamic_pca_sample":
                matched_cols = (
                    [y_col]
                    + CONFIG["CONTROL_COLUMNS"]
                    + list(variant.columns)
                    + [dynamic_score, static_score]
                )
                work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=matched_cols)
            elif sample_scope != "full_available":
                raise ValueError(
                    "sample_scope must be either 'full_available' or "
                    "'matched_dynamic_pca_sample'."
                )

            for spec in regression_specs_for_variant(variant):
                result = run_hac_ols(
                    df=work,
                    y_col=y_col,
                    x_cols=spec["x_cols"],
                    horizon=h,
                    variant=variant,
                    index_method=spec["index_method"],
                    specification=spec["specification"],
                    main_predictor=spec["main_predictor"],
                    sample_scope=sample_scope,
                )
                results.append(result.table)

    return pd.concat(results, ignore_index=True)


def run_component_regressions_for_variant(df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    """
    Own predictive regression with same controls
    """
    variant_lookup = {variant.name: variant for variant in INDEX_VARIANTS}
    variant = variant_lookup[variant_name]
    results = []
    for component in variant.columns:
        for h in CONFIG["HORIZONS"]:
            y_col = f"R_fwd_{h}"
            result = run_hac_ols(
                df=df,
                y_col=y_col,
                x_cols=CONFIG["CONTROL_COLUMNS"] + [component],
                horizon=h,
                variant=variant,
                index_method="individual_component",
                specification=f"individual_{component}",
                main_predictor=component,
                sample_scope="full_available",
            )
            results.append(result.table)
    return pd.concat(results, ignore_index=True)


def run_main_component_regressions(df: pd.DataFrame) -> pd.DataFrame:
    return run_component_regressions_for_variant(df, CONFIG["MAIN_VARIANT"])


def run_btc_component_regressions(df: pd.DataFrame) -> pd.DataFrame:
    return run_component_regressions_for_variant(df, CONFIG["BTC_ROBUSTNESS_VARIANT"])


def add_lagged_robustness_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(CONFIG["DATE_COLUMN"]).copy()
    out["R_t_lag1"] = out["R_t"].shift(1)
    out["cftc_open_interest_growth_resid_lag1"] = out[
        "cftc_open_interest_growth_resid"
    ].shift(1)
    if "btc_open_interest_growth_resid" in out.columns:
        out["btc_open_interest_growth_resid_lag1"] = out[
            "btc_open_interest_growth_resid"
        ].shift(1)
    return out


def run_lagged_return_oi_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lagged return and lagged open interest controls
    """
    variant_lookup = {variant.name: variant for variant in INDEX_VARIANTS}
    specs = [
        {
            "variant": variant_lookup[CONFIG["MAIN_VARIANT"]],
            "score_col": f"{CONFIG['MAIN_VARIANT']}_static_score",
            "oi_lag_col": "cftc_open_interest_growth_resid_lag1",
            "specification": "main_static_pca_lagged_return_oi_controls",
        },
        {
            "variant": variant_lookup[CONFIG["BTC_ROBUSTNESS_VARIANT"]],
            "score_col": f"{CONFIG['BTC_ROBUSTNESS_VARIANT']}_static_score",
            "oi_lag_col": "btc_open_interest_growth_resid_lag1",
            "specification": "btc_static_pca_lagged_return_oi_controls",
        },
    ]

    results = []
    for spec in specs:
        variant = spec["variant"]
        score_col = spec["score_col"]
        oi_lag_col = spec["oi_lag_col"]
        x_cols = CONFIG["CONTROL_COLUMNS"] + [score_col, "R_t_lag1", oi_lag_col]
        for horizon in CONFIG["HORIZONS"]:
            result = run_hac_ols(
                df=df,
                y_col=f"R_fwd_{horizon}",
                x_cols=x_cols,
                horizon=horizon,
                variant=variant,
                index_method="lagged_return_oi_controls",
                specification=spec["specification"],
                main_predictor=score_col,
                sample_scope="full_available",
            )
            results.append(result.table)
    return pd.concat(results, ignore_index=True)


def run_state_dependence_robustness(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["I_Google_gt_0"] = (out["Google_t"] > 0).astype(int)
    out["I_R_gt_0"] = (out["R_t"] > 0).astype(int)
    out["R_t_x_I_Google_gt_0"] = out["R_t"] * out["I_Google_gt_0"]
    out["Google_t_x_I_R_gt_0"] = out["Google_t"] * out["I_R_gt_0"]

    variant_lookup = {variant.name: variant for variant in INDEX_VARIANTS}
    variant = variant_lookup[CONFIG["MAIN_VARIANT"]]
    score_col = f"{CONFIG['MAIN_VARIANT']}_static_score"
    specs = [
        (
            "state_dependence_baseline",
            ["R_t", "Google_t", score_col],
        ),
        (
            "state_dependence_momentum_high_google",
            ["R_t", "I_Google_gt_0", "R_t_x_I_Google_gt_0", score_col],
        ),
        (
            "state_dependence_google_high_return",
            ["Google_t", "I_R_gt_0", "Google_t_x_I_R_gt_0", score_col],
        ),
    ]

    results = []
    for specification, x_cols in specs:
        for horizon in CONFIG["HORIZONS"]:
            result = run_hac_ols(
                df=out,
                y_col=f"R_fwd_{horizon}",
                x_cols=x_cols,
                horizon=horizon,
                variant=variant,
                index_method="state_dependence_robustness",
                specification=specification,
                main_predictor=score_col,
                sample_scope="full_available",
            )
            results.append(result.table)
    return pd.concat(results, ignore_index=True)


def make_compact_results(results: pd.DataFrame) -> pd.DataFrame:
    compact_rows = []
    for variant in INDEX_VARIANTS:
        variant_rows = results[results["variant"] == variant.name]
        component_rows = variant_rows[
            (variant_rows["index_method"] == "linear_components")
            & (variant_rows["parameter"].isin(CONFIG["CONTROL_COLUMNS"] + list(variant.columns)))
        ]
        pca_rows = variant_rows[
            (variant_rows["index_method"].isin(["dynamic_pca", "static_pca"]))
            & (
                variant_rows["parameter"].isin(CONFIG["CONTROL_COLUMNS"])
                | (variant_rows["parameter"] == variant_rows["main_predictor"])
            )
        ]
        compact_rows.extend([component_rows, pca_rows])

    compact = pd.concat(compact_rows, ignore_index=True)
    cols = [
        "variant",
        "variant_label",
        "is_new_alternative",
        "index_method",
        "sample_scope",
        "horizon",
        "parameter",
        "coef",
        "std_err_hac",
        "t_stat_hac",
        "p_value_hac",
        "n_obs",
        "r_squared",
        "adj_r_squared",
        "hac_maxlags",
        "sample_start",
        "sample_end",
        "status",
    ]
    return compact[cols].sort_values(
        ["sample_scope", "horizon", "variant", "index_method", "parameter"]
    )


def make_model_comparison(results: pd.DataFrame) -> pd.DataFrame:
    model_rows = (
        results[results["parameter"] == "const"]
        [
            [
                "variant",
                "variant_label",
                "is_new_alternative",
                "index_method",
                "specification",
                "sample_scope",
                "horizon",
                "n_obs",
                "r_squared",
                "adj_r_squared",
                "hac_maxlags",
                "sample_start",
                "sample_end",
                "status",
            ]
        ]
        .drop_duplicates()
        .sort_values(["sample_scope", "horizon", "variant", "index_method"])
        .reset_index(drop=True)
    )
    return model_rows


def make_variant_performance_summary(compact: pd.DataFrame) -> pd.DataFrame:
    pca = compact[
        compact["index_method"].isin(["dynamic_pca", "static_pca"])
        & compact["parameter"].str.endswith("_score")
        & (compact["status"] == "ok")
    ].copy()
    if pca.empty:
        return pd.DataFrame()

    summary = (
        pca.groupby(
            [
                "variant",
                "variant_label",
                "is_new_alternative",
                "index_method",
                "sample_scope",
            ],
            dropna=False,
        )
        .agg(
            min_p_value=("p_value_hac", "min"),
            median_p_value=("p_value_hac", "median"),
            mean_abs_t=("t_stat_hac", lambda s: float(np.nanmean(np.abs(s)))),
            mean_adj_r_squared=("adj_r_squared", "mean"),
            significant_10pct=("p_value_hac", lambda s: int((s < 0.10).sum())),
            n_horizons=("horizon", "nunique"),
            avg_n_obs=("n_obs", "mean"),
        )
        .reset_index()
    )
    summary["rank_score"] = (
        -summary["min_p_value"].fillna(1.0)
        + summary["mean_adj_r_squared"].fillna(-1.0)
        + 0.01 * summary["significant_10pct"]
    )
    return summary.sort_values(
        ["sample_scope", "is_new_alternative", "rank_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


# Diagnostics and LaTeX helpers


def make_diagnostics(df: pd.DataFrame, static_weights: pd.DataFrame) -> dict[str, pd.DataFrame]:
    variant_component_cols = sorted({col for variant in INDEX_VARIANTS for col in variant.columns})
    variant_score_cols = []
    dynamic_weight_cols = []
    for variant in INDEX_VARIANTS:
        variant_score_cols.extend(
            [
                f"{variant.name}_dynamic_score",
                f"{variant.name}_dynamic_bounded",
                f"{variant.name}_static_score",
                f"{variant.name}_static_bounded",
            ]
        )
        for col in variant.columns:
            dynamic_weight_cols.append(f"{variant.name}_dynamic_w_{clean_label(col)}")

    diagnostic_cols = (
        CONFIG["CONTROL_COLUMNS"]
        + variant_component_cols
        + variant_score_cols
        + [f"R_fwd_{h}" for h in CONFIG["HORIZONS"]]
    )
    existing_diagnostic_cols = [col for col in diagnostic_cols if col in df.columns]

    missing = df[existing_diagnostic_cols].isna().sum().rename("missing_count").to_frame()
    summary = df[existing_diagnostic_cols].describe().T
    correlations = df[
        CONFIG["CONTROL_COLUMNS"] + variant_component_cols + variant_score_cols
    ].corr()

    dynamic_weights = df[[CONFIG["DATE_COLUMN"]] + dynamic_weight_cols].copy()
    dynamic_weights = dynamic_weights.dropna(how="all", subset=dynamic_weight_cols)

    variant_definitions = pd.DataFrame(
        [
            {
                "variant": variant.name,
                "variant_label": variant.label,
                "is_new_alternative": variant.is_new_alternative,
                "component_count": len(variant.columns),
                "component_1": variant.columns[0],
                "component_2": variant.columns[1],
                "component_3": variant.columns[2] if len(variant.columns) > 2 else "",
                "description": variant.description,
            }
            for variant in INDEX_VARIANTS
        ]
    )

    return {
        "variant_definitions": variant_definitions,
        "missing_values": missing,
        "summary_statistics": summary,
        "correlations": correlations,
        "dynamic_pca_weights": dynamic_weights,
        "static_pca_weights": static_weights,
    }


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def format_coef_with_se(row: pd.Series) -> str:
    if pd.isna(row["coef"]):
        return ""
    stars = significance_stars(row["p_value_hac"])
    return f"{row['coef']:.4f}{stars} ({row['std_err_hac']:.4f})"


EXTENSION_HORIZON_LABELS = {
    1: r"R$_{t,t+1}$ (1)",
    2: r"R$_{t,t+2}$ (2)",
    3: r"R$_{t,t+3}$ (3)",
    4: r"R$_{t,t+4}$ (4)",
    6: r"R$_{t,t+6}$ (5)",
    8: r"R$_{t,t+8}$ (6)",
}


DISPLAY_LABELS = {
    "R_t": r"R$_t$",
    "Google_t": r"Google$_t$",
    "futures_pressure_gross_resid_3v_static_score": r"Adj. CFPI$_t$",
    "futures_pressure_gross_resid_3v_dynamic_score": r"Dynamic Adj. CFPI$_t$",
    "futures_pressure_gross_3v_static_score": r"CFPI$_t$",
    "futures_pressure_gross_3v_dynamic_score": r"Dynamic CFPI$_t$",
    "btc_price_quantity_resid_2v_static_score": r"BTC Adj. CFPI$_t$",
    "btc_price_quantity_resid_2v_dynamic_score": r"BTC Dynamic Adj. CFPI$_t$",
    "btc_price_quantity_pressure_2v_static_score": r"BTC CFPI$_t$",
    "btc_price_quantity_pressure_2v_dynamic_score": r"BTC Dynamic CFPI$_t$",
    "futures_price_quantity_pressure_2v_static_score": r"Price--quantity CFPI$_t$",
    "futures_basis": r"Basis$_t$",
    "cftc_open_interest_growth": r"$\Delta OI_t$",
    "cftc_open_interest_growth_resid": r"Momentum-adjusted $\Delta OI_t$",
    "cftc_gross_position_share": r"Gross AM$_t$",
    "btc_futures_basis": r"BTC Basis$_t$",
    "btc_open_interest_growth": r"BTC $\Delta OI_t$",
    "btc_open_interest_growth_resid": r"BTC momentum-adjusted $\Delta OI_t$",
    "R_t_lag1": r"R$_{t-1}$",
    "cftc_open_interest_growth_resid_lag1": r"Momentum-adjusted $\Delta OI_{t-1}$",
    "btc_open_interest_growth_resid_lag1": r"BTC momentum-adjusted $\Delta OI_{t-1}$",
    "I_Google_gt_0": r"1\{Google$_t>0$\}",
    "I_R_gt_0": r"1\{R$_t>0$\}",
    "R_t_x_I_Google_gt_0": r"R$_t \times$ 1\{Google$_t>0$\}",
    "Google_t_x_I_R_gt_0": r"Google$_t \times$ 1\{R$_t>0$\}",
}


def format_num_latex(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def format_coef_latex(coef: float, p_value: float) -> str:
    if pd.isna(coef):
        return ""
    return f"{float(coef):.3f}{significance_stars(p_value)}"


def format_tstat_latex(t_stat: float) -> str:
    if pd.isna(t_stat):
        return ""
    return f"({float(t_stat):.2f})"


def latex_note_block(description: str, width: str = r"0.96\linewidth") -> str:
    return (
        r"\vspace{0.35em}" + "\n"
        r"\par\noindent\makebox[\linewidth][c]{%" + "\n"
        rf"\begin{{minipage}}{{{width}}}" + "\n"
        r"\footnotesize" + "\n"
        r"\raggedright" + "\n"
        + description + "\n"
        r"\end{minipage}%" + "\n"
        r"}"
    )


def make_manual_latex_table(
    caption: str,
    label: str,
    body_lines: list[str],
    description: str,
    ncols: int,
    column_format: str | None = None,
    arraystretch: str = "1.12",
    table_width: str | None = None,
) -> str:
    if column_format is None:
        column_format = "l" + "c" * (ncols - 1)
    tabular_lines = [
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if table_width is not None:
        tabular_lines = [
            r"\makebox[\linewidth][c]{%",
            rf"\resizebox{{{table_width}}}{{!}}{{%",
            *tabular_lines,
            r"}%",
            r"}",
        ]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
            *tabular_lines,
            latex_note_block(description),
            r"\end{table}",
            "",
        ]
    )


def regression_subset(
    compact: pd.DataFrame,
    variant: str,
    index_method: str,
    sample_scope: str = "full_available",
) -> pd.DataFrame:
    return compact[
        (compact["variant"] == variant)
        & (compact["index_method"] == index_method)
        & (compact["sample_scope"] == sample_scope)
    ].copy()


def get_reg_value(results: pd.DataFrame, horizon: int, parameter: str, field: str) -> float:
    sub = results[
        (results["horizon"].astype(int) == int(horizon))
        & (results["parameter"] == parameter)
    ]
    if sub.empty or field not in sub:
        return np.nan
    return sub[field].iloc[0]


def get_horizon_value(results: pd.DataFrame, horizon: int, field: str) -> float:
    sub = results[results["horizon"].astype(int) == int(horizon)]
    if sub.empty or field not in sub:
        return np.nan
    return sub[field].iloc[0]


def add_predictor_block(
    body_lines: list[str],
    results: pd.DataFrame,
    predictor: str,
    row_label: str,
    include_midrule: bool = False,
    control_rows: list[tuple[str, str]] | None = None,
) -> None:
    block = results.copy()
    if "main_predictor" in block.columns:
        filtered = block[block["main_predictor"] == predictor]
        if not filtered.empty:
            block = filtered

    if include_midrule:
        body_lines.append(r"\midrule")

    if control_rows is None:
        control_rows = [
            (control_col, DISPLAY_LABELS[control_col])
            for control_col in CONFIG["CONTROL_COLUMNS"]
        ]

    coefficient_rows = [(predictor, row_label)] + [
        (control_col, control_label)
        for control_col, control_label in control_rows
        if control_col != predictor
    ]
    for parameter, label in coefficient_rows:
        coef_cells = [label]
        tstat_cells = [""]
        for h in CONFIG["HORIZONS"]:
            coef = get_reg_value(block, h, parameter, "coef")
            pval = get_reg_value(block, h, parameter, "p_value_hac")
            tval = get_reg_value(block, h, parameter, "t_stat_hac")
            coef_cells.append(format_coef_latex(coef, pval))
            tstat_cells.append(format_tstat_latex(tval))
        body_lines.append(" & ".join(coef_cells) + r" \\")
        body_lines.append(" & ".join(tstat_cells) + r" \\")

    r2_cells = [r"R$^2$"]
    n_cells = ["N"]
    for h in CONFIG["HORIZONS"]:
        r2_cells.append(format_num_latex(get_horizon_value(block, h, "r_squared"), 3))
        n_obs = get_horizon_value(block, h, "n_obs")
        n_cells.append(str(int(n_obs)) if pd.notna(n_obs) else "")
    body_lines.append(" & ".join(r2_cells) + r" \\")
    body_lines.append(" & ".join(n_cells) + r" \\")


def build_main_index_latex(compact: pd.DataFrame) -> str:
    variant = CONFIG["MAIN_VARIANT"]
    results = regression_subset(compact, variant, "static_pca")
    predictor = f"{variant}_static_score"
    horizon_cols = [EXTENSION_HORIZON_LABELS[h] for h in CONFIG["HORIZONS"]]
    body_lines = ["Variable & " + " & ".join(horizon_cols) + r" \\", r"\midrule"]
    add_predictor_block(
        body_lines,
        results,
        predictor,
        DISPLAY_LABELS[predictor],
    )
    description = (
        r"This table reports predictive regressions of cumulative future BTC/ETH "
        r"returns on $R_t$, $Google_t$, and the static cryptocurrency futures "
        r"pressure index. The open-interest component is orthogonalized with "
        r"respect to contemporaneous $R_t$ before the index is formed. "
        r"HAC/Newey--West t-statistics are in parentheses. *, **, and *** "
        r"denote significance at the 10\%, 5\%, and 1\% levels."
    )
    return make_manual_latex_table(
        caption="Cryptocurrency futures pressure index and future returns.",
        label="tbl-revised-extension-main-index",
        body_lines=body_lines,
        description=description,
        ncols=1 + len(horizon_cols),
    )


def build_component_latex(component_results: pd.DataFrame) -> str:
    results = component_results.copy()
    horizon_cols = [EXTENSION_HORIZON_LABELS[h] for h in CONFIG["HORIZONS"]]
    ncols = 1 + len(horizon_cols)
    panel_specs = [
        ("futures_basis", r"\textit{Panel A: Futures-basis pricing pressure, } Basis$_t$"),
        (
            "cftc_open_interest_growth_resid",
            r"\textit{Panel B: Momentum-adjusted open-interest growth, } Momentum-adjusted $\Delta OI_t$",
        ),
        (
            "cftc_gross_position_share",
            r"\textit{Panel C: Gross asset-manager participation, } Gross AM$_t$",
        ),
    ]

    body_lines = ["Variable & " + " & ".join(horizon_cols) + r" \\", r"\midrule"]
    for panel_idx, (predictor, panel_title) in enumerate(panel_specs):
        if panel_idx > 0:
            body_lines.append(r"\midrule")
        body_lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{{panel_title}}} \\")
        add_predictor_block(
            body_lines,
            results,
            predictor,
            DISPLAY_LABELS[predictor],
        )

    description = (
        r"Each panel reports predictive regressions of cumulative future BTC/ETH "
        r"returns on $R_t$, $Google_t$, and one component of the futures pressure "
        r"index. Momentum-adjusted $\Delta OI_t$ is the open-interest growth "
        r"component used in the PCAs after removing contemporaneous $R_t$. "
        r"HAC/Newey--West "
        r"t-statistics are in parentheses. *, **, and *** denote significance "
        r"at the 10\%, 5\%, and 1\% levels."
    )
    return make_manual_latex_table(
        caption="Individual predictive regressions for futures-pressure variables.",
        label="tbl-revised-extension-component-regressions",
        body_lines=body_lines,
        description=description,
        ncols=ncols,
    )


def build_btc_component_latex(component_results: pd.DataFrame) -> str:
    results = component_results.copy()
    horizon_cols = [EXTENSION_HORIZON_LABELS[h] for h in CONFIG["HORIZONS"]]
    ncols = 1 + len(horizon_cols)
    panel_specs = [
        ("btc_futures_basis", r"\textit{Panel A: BTC futures-basis pricing pressure, } BTC Basis$_t$"),
        (
            "btc_open_interest_growth_resid",
            r"\textit{Panel B: BTC momentum-adjusted open-interest growth, } BTC momentum-adjusted $\Delta OI_t$",
        ),
    ]

    body_lines = ["Variable & " + " & ".join(horizon_cols) + r" \\", r"\midrule"]
    for panel_idx, (predictor, panel_title) in enumerate(panel_specs):
        if panel_idx > 0:
            body_lines.append(r"\midrule")
        body_lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{{panel_title}}} \\")
        add_predictor_block(
            body_lines,
            results,
            predictor,
            DISPLAY_LABELS[predictor],
        )

    description = (
        r"Each panel reports predictive regressions of cumulative future BTC/ETH "
        r"returns on $R_t$, $Google_t$, and one BTC-only futures pressure component. "
        r"BTC momentum-adjusted $\Delta OI_t$ is the Bitcoin open-interest growth "
        r"component used in the PCAs after removing contemporaneous $R_t$. "
        r"HAC/Newey--West "
        r"t-statistics are in parentheses. *, **, and *** denote significance "
        r"at the 10\%, 5\%, and 1\% levels."
    )
    return make_manual_latex_table(
        caption="Individual predictive regressions for BTC-only futures-pressure variables.",
        label="tbl-revised-extension-btc-component-regressions",
        body_lines=body_lines,
        description=description,
        ncols=ncols,
    )


def build_method_comparison_latex(compact: pd.DataFrame) -> str:
    horizon_cols = [EXTENSION_HORIZON_LABELS[h] for h in CONFIG["HORIZONS"]]
    ncols = 1 + len(horizon_cols)
    panels = [
        (
            CONFIG["MAIN_VARIANT"],
            "static_pca",
            f"{CONFIG['MAIN_VARIANT']}_static_score",
            r"\textit{Panel A: Momentum-adjusted futures pressure index, static PCA}",
            r"Adj. CFPI$_t$",
        ),
        (
            CONFIG["MAIN_VARIANT"],
            "dynamic_pca",
            f"{CONFIG['MAIN_VARIANT']}_dynamic_score",
            r"\textit{Panel B: Momentum-adjusted futures pressure index, EW shrinkage anchored PCA}",
            r"Dynamic Adj. CFPI$_t$",
        ),
        (
            CONFIG["BTC_ROBUSTNESS_VARIANT"],
            "static_pca",
            f"{CONFIG['BTC_ROBUSTNESS_VARIANT']}_static_score",
            r"\textit{Panel C: BTC-only momentum-adjusted futures pressure index, static PCA}",
            r"BTC Adj. CFPI$_t$",
        ),
        (
            CONFIG["BTC_ROBUSTNESS_VARIANT"],
            "dynamic_pca",
            f"{CONFIG['BTC_ROBUSTNESS_VARIANT']}_dynamic_score",
            r"\textit{Panel D: BTC-only momentum-adjusted futures pressure index, EW shrinkage anchored PCA}",
            r"BTC Dynamic Adj. CFPI$_t$",
        ),
    ]
    body_lines = ["Variable & " + " & ".join(horizon_cols) + r" \\", r"\midrule"]
    for panel_idx, (variant, method, predictor, panel_title, row_label) in enumerate(panels):
        if panel_idx > 0:
            body_lines.append(r"\midrule")
        body_lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{{panel_title}}} \\")
        add_predictor_block(
            body_lines,
            regression_subset(compact, variant, method),
            predictor,
            row_label,
        )

    description = (
        r"This table compares the main momentum-adjusted static PCA futures "
        r"pressure index with its dynamic EW shrinkage anchored PCA analogue, "
        r"and with BTC-only momentum-adjusted robustness indices. The dynamic "
        r"indices use lagged exponentially "
        r"weighted standardization, a lagged covariance matrix with a 36-week "
        r"half-life, 10\% identity shrinkage, and sign anchoring to the static "
        r"PCA loading vector. The "
        r"momentum-adjusted versions replace $\Delta OI_t$ with the residual from "
        r"regressing open-interest growth on contemporaneous $R_t$ before PCA. "
        r"All regressions include $R_t$ and $Google_t$ as controls. HAC/Newey--West "
        r"t-statistics are in parentheses. *, **, and *** denote significance at "
        r"the 10\%, 5\%, and 1\% levels."
    )
    return make_manual_latex_table(
        caption="Alternative index constructions and future returns.",
        label="tbl-revised-extension-method-comparison",
        body_lines=body_lines,
        description=description,
        ncols=ncols,
    )


def build_raw_open_interest_benchmark_latex(compact: pd.DataFrame) -> str:
    horizon_cols = [EXTENSION_HORIZON_LABELS[h] for h in CONFIG["HORIZONS"]]
    ncols = 1 + len(horizon_cols)
    panels = [
        (
            CONFIG["RAW_BASELINE_VARIANT"],
            "static_pca",
            f"{CONFIG['RAW_BASELINE_VARIANT']}_static_score",
            r"\textit{Panel A: Raw open-interest futures pressure index, static PCA}",
            r"CFPI$_t$",
        ),
        (
            CONFIG["RAW_BASELINE_VARIANT"],
            "dynamic_pca",
            f"{CONFIG['RAW_BASELINE_VARIANT']}_dynamic_score",
            r"\textit{Panel B: Raw open-interest futures pressure index, EW shrinkage anchored PCA}",
            r"Dynamic CFPI$_t$",
        ),
        (
            CONFIG["BTC_RAW_BASELINE_VARIANT"],
            "static_pca",
            f"{CONFIG['BTC_RAW_BASELINE_VARIANT']}_static_score",
            r"\textit{Panel C: BTC-only raw open-interest futures pressure index, static PCA}",
            r"BTC CFPI$_t$",
        ),
        (
            CONFIG["BTC_RAW_BASELINE_VARIANT"],
            "dynamic_pca",
            f"{CONFIG['BTC_RAW_BASELINE_VARIANT']}_dynamic_score",
            r"\textit{Panel D: BTC-only raw open-interest futures pressure index, EW shrinkage anchored PCA}",
            r"BTC Dynamic CFPI$_t$",
        ),
    ]

    body_lines = ["Variable & " + " & ".join(horizon_cols) + r" \\", r"\midrule"]
    for panel_idx, (variant, method, predictor, panel_title, row_label) in enumerate(panels):
        if panel_idx > 0:
            body_lines.append(r"\midrule")
        body_lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{{panel_title}}} \\")
        add_predictor_block(
            body_lines,
            regression_subset(compact, variant, method),
            predictor,
            row_label,
        )

    description = (
        r"This table reports raw open-interest benchmark versions of the "
        r"cryptocurrency futures pressure index. Unlike the main "
        r"momentum-adjusted specifications, these indices use raw open-interest "
        r"growth in the PCA. The table is therefore interpreted as a robustness "
        r"and diagnostic benchmark rather than the preferred specification. All "
        r"regressions include $R_t$ and $Google_t$ as controls. HAC/Newey--West "
        r"t-statistics are in parentheses. *, **, and *** denote significance at "
        r"the 10\%, 5\%, and 1\% levels."
    )
    return make_manual_latex_table(
        caption="Raw open-interest benchmark index constructions.",
        label="tbl-revised-extension-raw-oi-benchmark",
        body_lines=body_lines,
        description=description,
        ncols=ncols,
    )


def build_btc_robustness_latex(compact: pd.DataFrame) -> str:
    variant = CONFIG["BTC_ROBUSTNESS_VARIANT"]
    results = regression_subset(compact, variant, "static_pca")
    predictor = f"{variant}_static_score"
    horizon_cols = [EXTENSION_HORIZON_LABELS[h] for h in CONFIG["HORIZONS"]]
    body_lines = ["Variable & " + " & ".join(horizon_cols) + r" \\", r"\midrule"]
    add_predictor_block(
        body_lines,
        results,
        predictor,
        DISPLAY_LABELS[predictor],
    )
    description = (
        r"This table repeats the static PCA index regression using only the "
        r"Bitcoin futures basis and the momentum-adjusted Bitcoin open-interest "
        r"growth component used in the PCA. The BTC-only benchmark addresses whether the "
        r"pooled BTC/ETH result is driven by the broader asset scope while also "
        r"controlling for contemporaneous return momentum in the quantity "
        r"component. HAC/Newey--West t-statistics are in parentheses. *, **, and "
        r"*** denote significance at the 10\%, 5\%, and 1\% levels."
    )
    return make_manual_latex_table(
        caption="BTC-only futures pressure index robustness test.",
        label="tbl-revised-extension-btc-robustness",
        body_lines=body_lines,
        description=description,
        ncols=1 + len(horizon_cols),
    )


def build_static_weights_latex(static_weights: pd.DataFrame) -> str:
    variant = CONFIG["MAIN_VARIANT"]
    sub = static_weights[static_weights["variant"] == variant].copy()
    role_map = {
        "futures_basis": "Futures-basis pricing pressure",
        "cftc_open_interest_growth": "Open-interest growth",
        "cftc_open_interest_growth_resid": "Momentum-adjusted open-interest growth",
        "cftc_gross_position_share": "Gross asset-manager participation",
    }
    body_lines = [
        r"Variable & Component & Static PC1 weight \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        component = row["component"]
        body_lines.append(
            " & ".join(
                [
                    DISPLAY_LABELS.get(component, component),
                    role_map.get(component, component),
                    format_num_latex(row["static_pc1_weight"], 3),
                ]
            )
            + r" \\"
        )

    description = (
        r"This table reports the static PCA loadings used to construct the main "
        r"momentum-adjusted cryptocurrency futures pressure index. The loadings "
        r"are estimated over the full available weekly sample and are reported to "
        r"clarify what the index captures."
    )
    return make_manual_latex_table(
        caption="Static PCA construction of the cryptocurrency futures pressure index.",
        label="tbl-revised-extension-pca-weights",
        body_lines=body_lines,
        description=description,
        ncols=3,
        column_format="llc",
    )


def build_summary_statistics_latex(df: pd.DataFrame) -> str:
    summary_vars = [
        "R_t",
        "Google_t",
        "futures_basis",
        "cftc_open_interest_growth_resid",
        "cftc_gross_position_share",
        f"{CONFIG['MAIN_VARIANT']}_static_score",
    ]
    label_map = {
        "R_t": r"R$_t$",
        "Google_t": r"Google$_t$",
        "futures_basis": r"Basis$_t$",
        "cftc_open_interest_growth_resid": r"Momentum-adjusted $\Delta OI_t$",
        "cftc_gross_position_share": r"Gross AM$_t$",
        f"{CONFIG['MAIN_VARIANT']}_static_score": r"Adj. CFPI$_t$",
    }
    summary = df[summary_vars].describe().T
    body_lines = [r"Variable & N & Mean & SD & Min & Median & Max \\", r"\midrule"]
    for var, row in summary.iterrows():
        body_lines.append(
            " & ".join(
                [
                    label_map[var],
                    str(int(row["count"])),
                    format_num_latex(row["mean"], 3),
                    format_num_latex(row["std"], 3),
                    format_num_latex(row["min"], 3),
                    format_num_latex(row["50%"], 3),
                    format_num_latex(row["max"], 3),
                ]
            )
            + r" \\"
        )
    description = (
        r"This table reports weekly summary statistics for the revised extension "
        r"sample. Adj. CFPI$_t$ denotes the static momentum-adjusted "
        r"cryptocurrency futures pressure index."
    )
    return make_manual_latex_table(
        caption="Summary statistics for revised extension variables.",
        label="tbl-revised-extension-summary-statistics",
        body_lines=body_lines,
        description=description,
        ncols=7,
        column_format="lrrrrrr",
    )


def build_lagged_return_oi_robustness_latex(results: pd.DataFrame) -> str:
    horizon_cols = [EXTENSION_HORIZON_LABELS[h] for h in CONFIG["HORIZONS"]]
    ncols = 1 + len(horizon_cols)
    panel_specs = [
        (
            CONFIG["MAIN_VARIANT"],
            f"{CONFIG['MAIN_VARIANT']}_static_score",
            "cftc_open_interest_growth_resid_lag1",
            r"\textit{Panel A: Pooled BTC/ETH momentum-adjusted CFPI}",
            r"Adj. CFPI$_t$",
        ),
        (
            CONFIG["BTC_ROBUSTNESS_VARIANT"],
            f"{CONFIG['BTC_ROBUSTNESS_VARIANT']}_static_score",
            "btc_open_interest_growth_resid_lag1",
            r"\textit{Panel B: BTC-only momentum-adjusted CFPI}",
            r"BTC Adj. CFPI$_t$",
        ),
    ]

    body_lines = ["Variable & " + " & ".join(horizon_cols) + r" \\", r"\midrule"]
    for panel_idx, (variant, score_col, oi_lag_col, panel_title, score_label) in enumerate(
        panel_specs
    ):
        if panel_idx > 0:
            body_lines.append(r"\midrule")
        body_lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{{panel_title}}} \\")
        block = results[
            (results["variant"] == variant)
            & (results["index_method"] == "lagged_return_oi_controls")
        ].copy()
        for predictor, row_label in [
            (score_col, score_label),
            ("R_t", DISPLAY_LABELS["R_t"]),
            ("Google_t", DISPLAY_LABELS["Google_t"]),
            ("R_t_lag1", DISPLAY_LABELS["R_t_lag1"]),
            (oi_lag_col, DISPLAY_LABELS[oi_lag_col]),
        ]:
            coef_cells = [row_label]
            tstat_cells = [""]
            for horizon in CONFIG["HORIZONS"]:
                coef = get_reg_value(block, horizon, predictor, "coef")
                pval = get_reg_value(block, horizon, predictor, "p_value_hac")
                tval = get_reg_value(block, horizon, predictor, "t_stat_hac")
                coef_cells.append(format_coef_latex(coef, pval))
                tstat_cells.append(format_tstat_latex(tval))
            body_lines.append(" & ".join(coef_cells) + r" \\")
            body_lines.append(" & ".join(tstat_cells) + r" \\")

        r2_cells = [r"R$^2$"]
        n_cells = ["N"]
        for horizon in CONFIG["HORIZONS"]:
            r2_cells.append(format_num_latex(get_horizon_value(block, horizon, "r_squared"), 3))
            n_obs = get_horizon_value(block, horizon, "n_obs")
            n_cells.append(str(int(n_obs)) if pd.notna(n_obs) else "")
        body_lines.append(" & ".join(r2_cells) + r" \\")
        body_lines.append(" & ".join(n_cells) + r" \\")

    description = (
        r"This table adds lagged return and lagged momentum-adjusted open-interest "
        r"growth controls to the static PCA index regressions. The lagged "
        r"open-interest variable is the same momentum-adjusted quantity component "
        r"used in the PCA construction, shifted by one week. All specifications "
        r"also include contemporaneous $R_t$ and $Google_t$. HAC/Newey--West "
        r"t-statistics are in parentheses. *, **, and *** denote significance at "
        r"the 10\%, 5\%, and 1\% levels."
    )
    return make_manual_latex_table(
        caption="Lagged return and open-interest robustness check.",
        label="tbl-revised-extension-lagged-return-oi-robustness",
        body_lines=body_lines,
        description=description,
        ncols=ncols,
    )


def build_state_dependence_robustness_latex(results: pd.DataFrame) -> str:
    horizon_cols = [EXTENSION_HORIZON_LABELS[h] for h in CONFIG["HORIZONS"]]
    ncols = 1 + len(horizon_cols)
    score_col = f"{CONFIG['MAIN_VARIANT']}_static_score"
    panel_specs = [
        (
            "state_dependence_baseline",
            r"\textit{Panel A: Momentum, Google attention, and momentum-adjusted CFPI}",
            ["R_t", "Google_t", score_col],
        ),
        (
            "state_dependence_momentum_high_google",
            r"\textit{Panel B: Momentum conditional on high Google attention}",
            ["R_t", "I_Google_gt_0", "R_t_x_I_Google_gt_0", score_col],
        ),
        (
            "state_dependence_google_high_return",
            r"\textit{Panel C: Google attention conditional on positive current returns}",
            ["Google_t", "I_R_gt_0", "Google_t_x_I_R_gt_0", score_col],
        ),
    ]

    body_lines = ["Variable & " + " & ".join(horizon_cols) + r" \\", r"\midrule"]
    for panel_idx, (specification, panel_title, predictors) in enumerate(panel_specs):
        if panel_idx > 0:
            body_lines.append(r"\midrule")
        body_lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{{panel_title}}} \\")
        block = results[results["specification"] == specification].copy()
        for predictor in predictors:
            coef_cells = [DISPLAY_LABELS[predictor]]
            tstat_cells = [""]
            for horizon in CONFIG["HORIZONS"]:
                coef = get_reg_value(block, horizon, predictor, "coef")
                pval = get_reg_value(block, horizon, predictor, "p_value_hac")
                tval = get_reg_value(block, horizon, predictor, "t_stat_hac")
                coef_cells.append(format_coef_latex(coef, pval))
                tstat_cells.append(format_tstat_latex(tval))
            body_lines.append(" & ".join(coef_cells) + r" \\")
            body_lines.append(" & ".join(tstat_cells) + r" \\")

        r2_cells = [r"R$^2$"]
        n_cells = ["N"]
        for horizon in CONFIG["HORIZONS"]:
            r2_cells.append(format_num_latex(get_horizon_value(block, horizon, "r_squared"), 3))
            n_obs = get_horizon_value(block, horizon, "n_obs")
            n_cells.append(str(int(n_obs)) if pd.notna(n_obs) else "")
        body_lines.append(" & ".join(r2_cells) + r" \\")
        body_lines.append(" & ".join(n_cells) + r" \\")

    description = (
        r"This table reports Table 4.1-style state-dependence robustness checks "
        r"for the static momentum-adjusted cryptocurrency futures pressure index. "
        r"Panel A includes contemporaneous $R_t$ and $Google_t$. Panel B allows "
        r"the momentum coefficient to differ when standardized Google attention "
        r"is positive. Panel C allows the Google-attention coefficient to differ "
        r"when contemporaneous returns are positive. HAC/Newey--West t-statistics "
        r"are in parentheses with maxlags equal to the return horizon. *, **, and "
        r"*** denote significance at the 10\%, 5\%, and 1\% levels."
    )
    return make_manual_latex_table(
        caption="State-dependent cryptocurrency futures pressure index robustness.",
        label="tbl-revised-extension-state-dependence-robustness",
        body_lines=body_lines,
        description=description,
        ncols=ncols,
    )


def build_oi_momentum_correlation_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df.sort_values(CONFIG["DATE_COLUMN"]).copy()
    for lag in [1, 2, 4]:
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
        matrix = work[columns].replace([np.inf, -np.inf], np.nan).corr()
        matrices[panel] = matrix.rename(index=spec["labels"], columns=spec["labels"])
    return matrices


def build_oi_momentum_correlation_latex(df: pd.DataFrame) -> str:
    matrices = build_oi_momentum_correlation_matrices(df)
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
            cells = [row_label] + [format_num_latex(value, 3) for value in values]
            body_lines.append(" & ".join(cells) + r" \\")

    description = (
        r"This table reports pairwise correlations between current and lagged "
        r"weekly returns, raw open-interest growth, and the momentum-adjusted "
        r"open-interest growth variables used in the PCA indices. The "
        r"momentum-adjusted variable removes the contemporaneous $R_t$ component "
        r"from raw open-interest growth, so its correlation with $R_t$ should be "
        r"mechanically close to zero. Lagged-return correlations are reported as "
        r"a robustness check on remaining return-momentum exposure."
    )
    return make_manual_latex_table(
        caption="Return-momentum correlations for raw and adjusted open-interest growth.",
        label="tbl-revised-extension-oi-momentum-correlations",
        body_lines=body_lines,
        description=description,
        ncols=ncols,
        column_format="l" + "r" * (ncols - 1),
        arraystretch="1.08",
        table_width=r"0.94\linewidth",
    )


def drop_previous_methodology_columns(df: pd.DataFrame) -> pd.DataFrame:
    old_prefixes = tuple(
        f"{prefix}_{component}_crypto"
        for prefix in ("psi", "z")
        for component in ("P", "B", "C")
    )
    old_columns = {
        f"{component}_crypto" for component in ("P", "B", "C")
    }
    old_columns |= {"I" + "CPI", "I" + "CPI_score_unbounded"}
    drop_cols = [
        col
        for col in df.columns
        if col in old_columns or any(col.startswith(prefix) for prefix in old_prefixes)
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def save_latex_tables(
    compact: pd.DataFrame,
    model_comparison: pd.DataFrame,
    df: pd.DataFrame,
    static_weights: pd.DataFrame,
    component_results: pd.DataFrame,
    btc_component_results: pd.DataFrame,
    lagged_robustness_results: pd.DataFrame,
    state_dependence_results: pd.DataFrame,
) -> None:
    if not CONFIG["SAVE_LATEX_TABLES"]:
        return

    latex_dir = CONFIG["OUTPUT_DIR"] / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    table_specs = {
        "revised_extension_regression_table.tex": build_main_index_latex(compact),
        "revised_extension_component_regressions.tex": build_component_latex(component_results),
        "revised_extension_btc_component_regressions.tex": build_btc_component_latex(
            btc_component_results
        ),
        "revised_extension_model_comparison.tex": build_method_comparison_latex(compact),
        "revised_extension_raw_oi_benchmark.tex": build_raw_open_interest_benchmark_latex(
            compact
        ),
        "revised_extension_btc_robustness.tex": build_btc_robustness_latex(compact),
        "revised_extension_pca_weights.tex": build_static_weights_latex(static_weights),
        "revised_extension_summary_statistics.tex": build_summary_statistics_latex(df),
        "revised_extension_oi_momentum_correlations.tex": (
            build_oi_momentum_correlation_latex(df)
        ),
        "revised_extension_lagged_return_oi_robustness.tex": (
            build_lagged_return_oi_robustness_latex(lagged_robustness_results)
        ),
        "revised_extension_state_dependence_robustness.tex": (
            build_state_dependence_robustness_latex(state_dependence_results)
        ),
    }

    for filename, latex in table_specs.items():
        (latex_dir / filename).write_text(latex)

    all_tables = "\n\n".join(table_specs.values())
    (latex_dir / "revised_extension_all_tables.tex").write_text(all_tables)

    supervisor_table_names = [
        "revised_extension_regression_table.tex",
        "revised_extension_pca_weights.tex",
        "revised_extension_model_comparison.tex",
        "revised_extension_raw_oi_benchmark.tex",
        "revised_extension_oi_momentum_correlations.tex",
        "revised_extension_lagged_return_oi_robustness.tex",
        "revised_extension_state_dependence_robustness.tex",
    ]
    supervisor_tables = "\n\n".join(table_specs[name] for name in supervisor_table_names)
    (latex_dir / "revised_extension_supervisor_packet.tex").write_text(supervisor_tables)


# The Main execution

def main() -> None:
    ensure_output_dir()

    df = load_revised_input_dataset()
    df = build_futures_positioning_feature_pool(df)
    df, static_weights = build_all_index_features(df)
    df = add_lagged_robustness_features(df)

    revised_dataset_path = CONFIG["OUTPUT_DIR"] / "revised_extension_dataset.csv"
    drop_previous_methodology_columns(df).to_csv(revised_dataset_path, index=False)

    diagnostics = make_diagnostics(df, static_weights)
    for name, table in diagnostics.items():
        table.to_csv(CONFIG["OUTPUT_DIR"] / f"diagnostics_{name}.csv", index=False)

    results_full = run_all_revised_regressions(df, sample_scope="full_available")
    results_matched = run_all_revised_regressions(df, sample_scope="matched_dynamic_pca_sample")
    results = pd.concat([results_full, results_matched], ignore_index=True)
    compact = make_compact_results(results)
    model_comparison = make_model_comparison(results)
    performance_summary = make_variant_performance_summary(compact)
    component_results = run_main_component_regressions(df)
    btc_component_results = run_btc_component_regressions(df)
    lagged_robustness_results = run_lagged_return_oi_robustness(df)
    state_dependence_results = run_state_dependence_robustness(df)

    results.to_csv(CONFIG["OUTPUT_DIR"] / "revised_regression_results_full.csv", index=False)
    compact.to_csv(CONFIG["OUTPUT_DIR"] / "revised_regression_results_compact.csv", index=False)
    model_comparison.to_csv(CONFIG["OUTPUT_DIR"] / "revised_model_comparison.csv", index=False)
    component_results.to_csv(
        CONFIG["OUTPUT_DIR"] / "revised_component_regression_results.csv", index=False
    )
    btc_component_results.to_csv(
        CONFIG["OUTPUT_DIR"] / "revised_btc_component_regression_results.csv", index=False
    )
    performance_summary.to_csv(
        CONFIG["OUTPUT_DIR"] / "revised_variant_performance_summary.csv", index=False
    )
    lagged_robustness_results.to_csv(
        CONFIG["OUTPUT_DIR"] / "revised_lagged_return_oi_robustness.csv", index=False
    )
    state_dependence_results.to_csv(
        CONFIG["OUTPUT_DIR"] / "revised_state_dependence_robustness.csv", index=False
    )
    save_latex_tables(
        compact,
        model_comparison,
        df,
        static_weights,
        component_results,
        btc_component_results,
        lagged_robustness_results,
        state_dependence_results,
    )

    print("\nSaved revised extension outputs to:")
    print(CONFIG["OUTPUT_DIR"])
    print("\nMain compact regression table:")
    print(compact.to_string(index=False))
    print("\nModel comparison:")
    print(model_comparison.to_string(index=False))
    print("\nExploratory variant performance summary:")
    print(performance_summary.to_string(index=False))

    if not performance_summary.empty:
        primary = performance_summary[
            (performance_summary["variant"] == CONFIG["MAIN_VARIANT"])
            & (performance_summary["index_method"] == "static_pca")
            & (performance_summary["sample_scope"] == "full_available")
        ]
        if not primary.empty:
            best = primary.iloc[0]
            print("\nPrimary revised specification:")
            print(
                f"{best['variant']} / {best['index_method']} "
                f"({best['sample_scope']}), min p-value={best['min_p_value']:.4f}, "
                f"median p-value={best['median_p_value']:.4f}, "
                f"significant horizons at 10%={int(best['significant_10pct'])}/"
                f"{int(best['n_horizons'])}"
            )

    print("\nPCA notes:")
    print(
        "Dynamic PCA: "
        f"{CONFIG['DYNAMIC_PCA_HALFLIFE']}-week half-life EW covariance, "
        f"{CONFIG['DYNAMIC_PCA_SHRINKAGE_ALPHA']:.2f} identity shrinkage, "
        "static sign anchoring"
    )
    for variant in INDEX_VARIANTS:
        first_dynamic = df.loc[
            df[f"{variant.name}_dynamic_score"].notna(), CONFIG["DATE_COLUMN"]
        ].min()
        print(f"- {variant.name}: first non-missing dynamic PCA score = {first_dynamic}")
    print("\nStatic PCA weights:")
    print(static_weights.to_string(index=False))

    if CONFIG["SAVE_LATEX_TABLES"]:
        print("\nSaved chapter-style LaTeX tables to:")
        print(CONFIG["OUTPUT_DIR"] / "latex")


if __name__ == "__main__":
    main()
# Done!