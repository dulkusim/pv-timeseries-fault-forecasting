
import numpy as np
import pandas as pd

from .config import SiteConfig


N_STRINGS = 7

CURRENT_MIN = 0.0
CURRENT_MAX = 150.0

POWER_MIN = 0.0
POWER_MAX = 50.0

POA_MIN_STRING = 200.0
OPEN_CIRCUIT_RATIO = 0.10
STRING_REL_DEV_THRESHOLD = 0.20


def current_columns(n_strings=N_STRINGS):
    return [
        f"ShuntCurrent_A_Avg_{i}"
        for i in range(1, n_strings + 1)
    ]


def power_columns(n_strings=N_STRINGS):
    return [
        f"ShuntPDC_kW_Avg_{i}"
        for i in range(1, n_strings + 1)
    ]


def clean_string_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physically invalid string measurements with NaN."""

    currents = current_columns()
    powers = power_columns()

    cleaned = df[currents + powers].copy()

    for col in currents:
        cleaned[col] = cleaned[col].where(
            cleaned[col].between(CURRENT_MIN, CURRENT_MAX)
        )

    for col in powers:
        cleaned[col] = cleaned[col].where(
            cleaned[col].between(POWER_MIN, POWER_MAX)
        )

    result = df.copy()
    result[currents + powers] = cleaned

    return result


def string_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate current and power statistics across strings."""

    currents = df[current_columns()]
    powers = df[power_columns()]

    i_mean = currents.mean(axis=1)
    i_std = currents.std(axis=1)
    i_min = currents.min(axis=1)
    i_max = currents.max(axis=1)

    p_mean = powers.mean(axis=1)
    p_std = powers.std(axis=1)

    features = {
        "I_string_mean": i_mean,
        "I_string_std": i_std,
        "I_string_min": i_min,
        "I_string_max": i_max,
        "I_string_cv": i_std / i_mean.replace(0, np.nan),
        "I_string_range_ratio": (
            (i_max - i_min) / i_mean.replace(0, np.nan)
        ),
        "P_string_mean": p_mean,
        "P_string_std": p_std,
        "P_string_total": powers.sum(axis=1),
        "P_string_cv": p_std / p_mean.replace(0, np.nan),
    }

    for i in range(1, N_STRINGS + 1):
        features[f"I_string_{i}_dev"] = (
            (df[f"ShuntCurrent_A_Avg_{i}"] - i_mean)
            / i_std.replace(0, np.nan)
        )

    features["I_string_cv_roll5"] = (
        features["I_string_cv"]
        .rolling(window=5, min_periods=3)
        .mean()
    )

    return pd.DataFrame(features, index=df.index)


def string_voltage_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-string P/I voltage proxies and statistics."""

    features = {}

    for i in range(1, N_STRINGS + 1):
        current = df[f"ShuntCurrent_A_Avg_{i}"]
        power = df[f"ShuntPDC_kW_Avg_{i}"]

        features[f"R_string_{i}"] = (
            power * 1000 / current.replace(0, np.nan)
        )

    r_cols = [
        f"R_string_{i}"
        for i in range(1, N_STRINGS + 1)
    ]

    r_values = pd.DataFrame(
        features, index=df.index
    )[r_cols]

    r_mean = r_values.mean(axis=1)
    r_std = r_values.std(axis=1)

    features["R_string_mean"] = r_mean
    features["R_string_std"] = r_std
    features["R_string_cv"] = (
        r_std / r_mean.replace(0, np.nan)
    )

    for i in range(1, N_STRINGS + 1):
        features[f"R_string_{i}_dev"] = (
            (features[f"R_string_{i}"] - r_mean)
            / r_std.replace(0, np.nan)
        )

    return pd.DataFrame(features, index=df.index)


def string_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate irradiance-normalized current and power."""

    poa = df["POA_fused"].replace(0, np.nan)

    features = {}

    for i in range(1, N_STRINGS + 1):
        features[f"Pnorm_string_{i}"] = (
            df[f"ShuntPDC_kW_Avg_{i}"] / poa
        )
        features[f"Inorm_string_{i}"] = (
            df[f"ShuntCurrent_A_Avg_{i}"] / poa
        )

    pnorm_cols = [
        f"Pnorm_string_{i}"
        for i in range(1, N_STRINGS + 1)
    ]

    pnorm = pd.DataFrame(
        features, index=df.index
    )[pnorm_cols]

    p_mean = pnorm.mean(axis=1)
    p_std = pnorm.std(axis=1)

    features["Pnorm_string_std"] = p_std
    features["Pnorm_string_cv"] = (
        p_std / p_mean.replace(0, np.nan)
    )

    return pd.DataFrame(features, index=df.index)


def fit_string_baselines(
    df: pd.DataFrame,
    site: SiteConfig,
) -> dict[int, float]:
    """Fit median Inorm baselines on the supplied dataset."""

    evaluable = (
        (df["Op_Mask"] == 1)
        & (df["POA_fused"] >= POA_MIN_STRING)
        & (df["InvOpStatus_Avg"] > 0)
        & (df["InvOpStatus_Avg"] <= 5)
    )

    baselines = {}

    for i in range(1, site.n_strings + 1):
        column = f"Inorm_string_{i}"

        baseline = df.loc[evaluable, column].median()

        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError(
                f"Invalid baseline for string {i}: {baseline}"
            )

        baselines[i] = float(baseline)

    return baselines


def apply_string_baselines(
    df: pd.DataFrame,
    baselines: dict[int, float],
) -> pd.DataFrame:
    """Calculate expected current and baseline-relative deviations."""

    if set(baselines) != set(range(1, N_STRINGS + 1)):
        raise ValueError(
            "Expected one baseline for each of the 7 strings"
        )

    features = {}

    for i in range(1, N_STRINGS + 1):
        baseline = baselines[i]

        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError(f"Invalid baseline for string {i}")

        expected = baseline * df["POA_fused"]

        features[f"I_string_{i}_expected"] = expected

        features[f"I_string_{i}_rel_dev"] = (
            (
                df[f"ShuntCurrent_A_Avg_{i}"] - expected
            )
            / expected.replace(0, np.nan)
        )

    deviation_cols = [
        f"I_string_{i}_rel_dev"
        for i in range(1, N_STRINGS + 1)
    ]

    deviations = pd.DataFrame(
        features, index=df.index
    )[deviation_cols]

    max_deviation = deviations.abs().max(axis=1)

    features["I_rel_dev_max"] = max_deviation
    features["I_rel_dev_std"] = deviations.std(axis=1)
    features["I_rel_dev_max_roll5"] = (
        max_deviation
        .rolling(window=5, min_periods=3)
        .mean()
    )

    return pd.DataFrame(features, index=df.index)


def label_string(
    df: pd.DataFrame,
    core_strings: tuple[int, ...] = tuple(range(1, 8)),
) -> pd.Series:
    """Generate string-level fault labels without modifying df."""

    if not core_strings:
        raise ValueError("core_strings cannot be empty")

    if len(set(core_strings)) != len(core_strings):
        raise ValueError("core_strings contains duplicates")

    if any(i not in range(1, N_STRINGS + 1) for i in core_strings):
        raise ValueError("Invalid string index")

    evaluable = (
        (df["Op_Mask"] == 1)
        & (df["POA_fused"] >= POA_MIN_STRING)
        & (df["InvOpStatus_Avg"] > 0)
        & (df["InvOpStatus_Avg"] <= 5)
    )

    open_circuit = (
        evaluable
        & (
            df["I_string_min"]
            < OPEN_CIRCUIT_RATIO * df["I_string_mean"]
        )
    )

    # Reuse the all-string metric when all strings are included.
    if set(core_strings) == set(range(1, N_STRINGS + 1)):
        imbalance_metric = df["I_rel_dev_max_roll5"]

    else:
        core_columns = [
            f"I_string_{i}_rel_dev"
            for i in core_strings
        ]

        core_max = df[core_columns].abs().max(axis=1)

        imbalance_metric = core_max.rolling(
            window=5,
            min_periods=3,
        ).mean()

    string_imbalance = (
        evaluable
        & ~open_circuit
        & (imbalance_metric > STRING_REL_DEV_THRESHOLD)
    )

    conditions = [
        ~evaluable,
        df["I_string_mean"].isna(),
        open_circuit,
        string_imbalance,
    ]

    choices = [
        "NOT_EVALUABLE",
        "MISSING_DATA",
        "OPEN_CIRCUIT_STRING",
        "STRING_IMBALANCE",
    ]

    labels = np.select(
        conditions,
        choices,
        default="NORMAL",
    )

    return pd.Series(
        labels,
        index=df.index,
        name="label_string",
    )
