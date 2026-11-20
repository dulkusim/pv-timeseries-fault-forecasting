
import pandas as pd

from .config import SiteConfig, POA_MIN_OPERATING, POA_MIN_PR


def operating_mask(
    df: pd.DataFrame,
    site: SiteConfig,
) -> pd.Series:
    """Identify periods with sufficient operating irradiance."""

    if site.op_mask_agg == "max":
        irradiance = df[list(site.op_mask_columns)].max(axis=1)
    elif site.op_mask_agg == "single":
        irradiance = df[site.op_mask_columns[0]]
    else:
        raise ValueError(
            f"Unknown operating aggregation: {site.op_mask_agg}"
        )

    return (irradiance > POA_MIN_OPERATING).astype(int).rename(
        "Op_Mask"
    )


def poa_fused(
    df: pd.DataFrame,
    site: SiteConfig,
) -> pd.Series:
    """Calculate site-specific reference-cell irradiance."""

    if site.poa_agg == "mean":
        result = df[list(site.poa_columns)].mean(axis=1)
    elif site.poa_agg == "single":
        result = df[site.poa_columns[0]]
    else:
        raise ValueError(
            f"Unknown POA aggregation: {site.poa_agg}"
        )

    return result.rename("POA_fused")


def module_temp(
    df: pd.DataFrame,
    site: SiteConfig,
) -> pd.Series:
    """Median temperature from retained RTD sensors."""

    columns = [
        f"RTD_C_Avg_{i}"
        for i in site.rtd_module_temp
    ]

    return df[columns].median(axis=1).rename("module_temp")


def pr_evaluable(
    df: pd.DataFrame,
    site: SiteConfig,
) -> pd.Series:
    """Determine which observations support PR evaluation."""

    result = (
        (df["Op_Mask"] == 1)
        & (df["POA_fused"] >= POA_MIN_PR)
        & (df["InvOpStatus_Avg"] > 0)
        & (df["InvOpStatus_Avg"] <= 5)
    )

    return result.astype(int).rename("PR_evaluable")
