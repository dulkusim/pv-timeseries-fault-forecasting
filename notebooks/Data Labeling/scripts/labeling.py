
import pandas as pd

from .config import SiteConfig
from .feature_selection import (
    drop_features,
    drop_legacy_rtds,
)
from .cleaning import apply_missingness_policy
from .operating import (
    operating_mask,
    poa_fused,
    module_temp,
    pr_evaluable,
)
from .performance import (
    expected_power,
    performance_features,
    label_power,
)
from .strings import (
    clean_string_bounds,
    string_aggregates,
    string_voltage_proxy,
    string_normalized,
    fit_string_baselines,
    apply_string_baselines,
    label_string,
)
from .inverter import (
    InverterPolicy,
    REVISED_INVERTER_POLICY,
    inverter_features,
    label_inverter,
)


def _append(df, *features):
    """Append a batch of newly calculated columns."""

    new_columns = pd.concat(features, axis=1)

    duplicates = df.columns.intersection(
        new_columns.columns
    )

    if len(duplicates):
        raise ValueError(
            f"Columns already exist: {duplicates.tolist()}"
        )

    return pd.concat(
        [df, new_columns],
        axis=1,
    )


def run_labeling(
    df: pd.DataFrame,
    site: SiteConfig,
    *,
    baselines: dict[int, float] | None = None,
    core_strings: tuple[int, ...] | None = None,
    inverter_policy: InverterPolicy = REVISED_INVERTER_POLICY,
    missing_power_label: str = "MISSING_DATA",
) -> pd.DataFrame:
    """Run the complete PV fault-labeling pipeline."""

    # 1. Feature selection and cleaning
    result = drop_features(df)
    result = drop_legacy_rtds(result, site)
    result = apply_missingness_policy(result)
    result = clean_string_bounds(result)

    # 2. Operating features
    result = _append(
        result,
        operating_mask(result, site),
        poa_fused(result, site),
        module_temp(result, site),
    )

    result = _append(
        result,
        pr_evaluable(result, site),
    )

    # 3. Expected power and PR features
    result = _append(
        result,
        expected_power(result, site),
    )

    result = _append(
        result,
        performance_features(result),
    )

    # Preserve the original notebook's rounding.
    round_columns = [
        "module_temp",
        "P_expected_DC_kW",
        "P_expected_AC_kW",
        "PR",
        "PR_roll_5min_weighted",
        "PR_roll_15min_weighted",
        "PR_roll_60min_weighted",
        "PR_roll_std_15min",
        "power_deficit_kW",
    ]

    result = result.assign(
        **{
            col: result[col].round(3)
            for col in round_columns
        }
    )

    result = _append(
        result,
        label_power(
            result,
            missing_label=missing_power_label,
        ),
    )

    # 4. String-level features
    result = _append(
        result,
        string_aggregates(result),
        string_voltage_proxy(result),
        string_normalized(result),
    )

    if baselines is None:
        baselines = fit_string_baselines(result, site)

    result = _append(
        result,
        apply_string_baselines(result, baselines),
    )

    if core_strings is None:
        if site.name.lower() == "canopy":
            core_strings = (2, 3, 4, 5, 6)
        else:
            core_strings = tuple(range(1, site.n_strings + 1))
        
    result = _append(
        result,
        label_string(
            result,
            core_strings=core_strings,
        ),
    )

    # 5. Inverter features and labels
    result = _append(
        result,
        inverter_features(
            result,
            site,
            policy=inverter_policy,
        ),
    )

    result = _append(
        result,
        label_inverter(
            result,
            policy=inverter_policy,
        ),
    )

    return result


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts and percentages for the three label types."""

    label_columns = (
        "label_power",
        "label_string",
        "label_inverter",
    )

    missing = [
        col for col in label_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing label columns: {missing}"
        )

    records = []

    for column in label_columns:
        counts = df[column].value_counts(
            dropna=False
        )

        for label, count in counts.items():
            records.append({
                "label_type": column,
                "label": label,
                "count": int(count),
                "percentage": (
                    100 * count / len(df)
                ),
            })

    return pd.DataFrame(records)
