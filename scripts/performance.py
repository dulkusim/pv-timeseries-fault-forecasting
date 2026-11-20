import numpy as np
import pandas as pd

from .config import (
    SiteConfig,
    PR_CRITICAL,
    PR_SEVERE,
    PR_UNDERPERFORMANCE,
    PR_SUSPICIOUS_HIGH,
)


def expected_power(
    df: pd.DataFrame,
    site: SiteConfig,
) -> pd.DataFrame:
    """Calculate temperature-corrected expected DC/AC power."""

    expected_dc = (
        site.p_rated_dc_kw
        * (df["POA_fused"] / 1000.0)
        * (
            1
            + site.gamma
            * (df["module_temp"] - site.t_stc)
        )
    )

    expected_ac = expected_dc.clip(
        upper=site.p_ac_max_kw
    )

    return pd.DataFrame({
        "P_expected_DC_kW": expected_dc,
        "P_expected_AC_kW": expected_ac,
    }, index=df.index)


def performance_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate PR, rolling PR, volatility and deficit."""

    actual = df["InvPAC_kW_Avg"]
    expected = df["P_expected_AC_kW"]

    pr = actual / expected.replace(0, np.nan)

    evaluable = df["PR_evaluable"] == 1

    actual_masked = actual.where(evaluable)
    expected_masked = expected.where(evaluable)

    features = {
        "PR": pr,
    }

    for window, min_periods in (
        (5, 3),
        (15, 8),
        (60, 30),
    ):
        numerator = actual_masked.rolling(
            window=window,
            min_periods=min_periods,
        ).sum()

        denominator = expected_masked.rolling(
            window=window,
            min_periods=min_periods,
        ).sum()

        features[
            f"PR_roll_{window}min_weighted"
        ] = numerator / denominator

    features["PR_roll_std_15min"] = pr.rolling(
        window=15,
        min_periods=8,
    ).std()

    features["power_deficit_kW"] = expected - actual

    return pd.DataFrame(features, index=df.index)


def label_power(
    df: pd.DataFrame,
    missing_label: str = "MISSING_DATA",
) -> pd.Series:
    """Assign power-performance labels without mutating df."""

    pr = df["PR_roll_15min_weighted"]
    evaluable = df["PR_evaluable"] == 1

    conditions = [
        df["PR_evaluable"] == 0,

        (
            df["P_expected_AC_kW"].isna()
            | df["InvPAC_kW_Avg"].isna()
            | (evaluable & pr.isna())
        ),

        evaluable & (pr >= 0) & (pr < PR_CRITICAL),

        evaluable & (pr >= PR_CRITICAL) & (pr < PR_SEVERE),

        evaluable
        & (pr >= PR_SEVERE)
        & (pr < PR_UNDERPERFORMANCE),

        evaluable & (pr > PR_SUSPICIOUS_HIGH),

        evaluable & (df["InvPAC_kW_Avg"] < 0),
    ]

    choices = [
        "NOT_EVALUABLE",
        missing_label,
        "CRITICAL_UNDERPERFORMANCE",
        "SEVERE_UNDERPERFORMANCE",
        "UNDERPERFORMANCE",
        "SUSPICIOUSLY_HIGH",
        "NEGATIVE_AC_POWER",
    ]

    return pd.Series(
        np.select(conditions, choices, default="NORMAL"),
        index=df.index,
        name="label_power",
    )
