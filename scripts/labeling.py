
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


from .reporting import append_features

STAGES = (
    "secondary", "redundant", "rtd", "missing", "bounds", "operating",
    "poa", "temperature", "pr_mask", "expected", "performance", "round",
    "power_labels", "aggregates", "voltage", "normalized", "baselines",
    "string_labels", "inverter", "inverter_labels",
)


def run_labeling(
    df: pd.DataFrame,
    site: SiteConfig,
    *,
    baselines: dict[int, float] | None = None,
    core_strings: tuple[int, ...] | None = None,
    inverter_policy: InverterPolicy = REVISED_INVERTER_POLICY,
    missing_power_label: str = "MISSING_DATA",
    stages: tuple[str, ...] = STAGES,
) -> pd.DataFrame:
    """Run all stages or explicitly selected stages for the thesis notebook.

    Selected stages must follow STAGES order and require the preceding features
    to be present. This is the single implementation used by interactive and
    batch/regression workflows; no annual notebook contains label rules.
    """
    if not stages or any(stage not in STAGES for stage in stages):
        raise ValueError(f"Unknown or empty stage selection: {stages}")
    if list(stages) != sorted(set(stages), key=STAGES.index):
        raise ValueError("Stages must be unique and in pipeline order")
    core_strings = site.core_strings if core_strings is None else core_strings
    result = df
    for stage in stages:
        if stage == "secondary":
            result = drop_features(result, groups=("secondary",))
        elif stage == "redundant":
            result = drop_features(result, groups=("redundant",))
        elif stage == "rtd":
            result = drop_legacy_rtds(result, site)
        elif stage == "missing":
            result = apply_missingness_policy(result)
        elif stage == "bounds":
            result = clean_string_bounds(result)
        elif stage in ("operating", "poa", "temperature", "pr_mask"):
            function = {
                "operating": operating_mask, "poa": poa_fused,
                "temperature": module_temp, "pr_mask": pr_evaluable,
            }[stage]
            result = append_features(result, function(result, site).to_frame())
        elif stage == "expected":
            result = append_features(result, expected_power(result, site))
        elif stage == "performance":
            result = append_features(result, performance_features(result))
        elif stage == "round":
            columns = [
                "module_temp", "P_expected_DC_kW", "P_expected_AC_kW", "PR",
                "PR_roll_5min_weighted", "PR_roll_15min_weighted",
                "PR_roll_60min_weighted", "PR_roll_std_15min", "power_deficit_kW",
            ]
            result = result.assign(**{col: result[col].round(3) for col in columns})
        elif stage == "power_labels":
            result = append_features(result, label_power(result, missing_label=missing_power_label).to_frame())
        elif stage in ("aggregates", "voltage", "normalized"):
            function = {"aggregates": string_aggregates, "voltage": string_voltage_proxy,
                        "normalized": string_normalized}[stage]
            result = append_features(result, function(result))
        elif stage == "baselines":
            fitted = fit_string_baselines(result, site) if baselines is None else baselines
            result = append_features(result, apply_string_baselines(result, fitted))
        elif stage == "string_labels":
            result = append_features(result, label_string(result, core_strings=core_strings).to_frame())
        elif stage == "inverter":
            result = append_features(result, inverter_features(result, site, policy=inverter_policy))
        elif stage == "inverter_labels":
            result = append_features(result, label_inverter(result, policy=inverter_policy).to_frame())
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
