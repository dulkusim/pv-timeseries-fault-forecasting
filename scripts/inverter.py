from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SiteConfig


FAULT_FLAG_COLUMNS = (
    "InvMainFault_Max",
    "InvDriveFault_Max",
    "InvVoltageFault_Max",
    "InvGridFault_Max",
    "InvTempFault_Max",
    "InvSystemFault_Max",
)


@dataclass(frozen=True)
class InverterPolicy:
    shutdown_column: str
    shutdown_threshold: float
    invalid_status_above_five: bool
    efficiency_min_dc_kw: float
    fault_flags_nonzero: bool


ORIGINAL_INVERTER_POLICY = InverterPolicy(
    shutdown_column="InvPAC_kW_Avg",
    shutdown_threshold=0.05,
    invalid_status_above_five=False,
    efficiency_min_dc_kw=1.0,
    fault_flags_nonzero=False,
)

REVISED_INVERTER_POLICY = InverterPolicy(
    shutdown_column="InvPDC_kW_Avg",
    shutdown_threshold=1.0,
    invalid_status_above_five=True,
    efficiency_min_dc_kw=5.0,
    fault_flags_nonzero=True,
)


def inverter_features(
    df: pd.DataFrame,
    site: SiteConfig,
    policy: InverterPolicy = REVISED_INVERTER_POLICY,
) -> pd.DataFrame:
    """Compute inverter diagnostics without mutating df."""

    pac = df["InvPAC_kW_Avg"]
    pdc = df["InvPDC_kW_Avg"]

    efficiency = pac / pdc.replace(0, np.nan)
    efficiency_clipped = efficiency.clip(lower=0, upper=1)

    discrepancy = df["P_string_total"] - pdc

    flags = df.loc[:, FAULT_FLAG_COLUMNS]

    if policy.fault_flags_nonzero:
        # NaN must not be interpreted as a fault code.
        active_flags = flags.notna() & flags.ne(0)
    else:
        active_flags = flags.gt(0)

    evaluable = (
        df["Op_Mask"].eq(1)
        & df["POA_fused"].ge(100)
    )

    status = df["InvOpStatus_Avg"]

    shutdown_power = (
        evaluable
        & df[policy.shutdown_column].lt(
            policy.shutdown_threshold
        )
    )

    invalid_status = status.isna() | status.le(0)

    if policy.invalid_status_above_five:
        invalid_status = invalid_status | status.gt(5)

    shutdown_status = evaluable & invalid_status

    return pd.DataFrame({
        "inv_efficiency": efficiency,
        "inv_efficiency_clipped": efficiency_clipped,
        "DC_power_discrepancy": discrepancy,
        "DC_power_discrepancy_pct": (
            discrepancy / pdc.replace(0, np.nan) * 100
        ),
        "inv_loading_ratio": pac / site.p_ac_max_kw,
        "inv_any_fault_reported": (
            active_flags.any(axis=1).astype(int)
        ),
        "inv_fault_count": (
            active_flags.sum(axis=1).astype(int)
        ),
        "inv_efficiency_roll5": (
            efficiency_clipped
            .rolling(window=5, min_periods=3)
            .mean()
        ),
        "inv_shutdown_power_flag": (
            shutdown_power.astype(int)
        ),
        "inv_shutdown_status_flag": (
            shutdown_status.astype(int)
        ),
    }, index=df.index)


def label_inverter(
    df: pd.DataFrame,
    policy: InverterPolicy = REVISED_INVERTER_POLICY,
) -> pd.Series:
    """Return inverter fault labels without modifying df."""

    evaluable = (
        df["Op_Mask"].eq(1)
        & df["POA_fused"].ge(100)
    )

    status = df["InvOpStatus_Avg"]

    normal_operation = (
        status.gt(0) & status.le(5)
    )

    shutdown = (
        df["inv_shutdown_power_flag"].eq(1)
        | df["inv_shutdown_status_flag"].eq(1)
    )

    efficiency_fault = (
        evaluable
        & normal_operation
        & ~shutdown
        & df["inv_efficiency_roll5"].lt(0.85)
        & df["InvPDC_kW_Avg"].gt(
            policy.efficiency_min_dc_kw
        )
        & df["inv_loading_ratio"].gt(0.20)
    )

    conditions = [
        df["Op_Mask"].eq(0),
        (
            df["InvPAC_kW_Avg"].isna()
            | df["InvPDC_kW_Avg"].isna()
        ),
        shutdown,
        efficiency_fault,
    ]

    choices = [
        "NOT_EVALUABLE",
        "MISSING_DATA",
        "INVERTER_SHUTDOWN",
        "INVERTER_EFFICIENCY_FAULT",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="NORMAL",
        ),
        index=df.index,
        name="label_inverter",
    )
