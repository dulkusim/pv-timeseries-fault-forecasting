
import pandas as pd

from .config import SiteConfig


SECONDARY_FEATURES = (
    "RECORD",
    "CR1000Temp_C_Avg",
    "DoorOpen_Min",
    "Battery_V_Min",
    "Battery_A_Avg",
    "Load_A_Avg",
    "ChgState_Min",
    "ChgSource_Min",
    "CkBatt_Max",
    "Qloss_Ah_Max",
    "WindDirAve_deg",
    "WindDirStdDev_deg",
    "WindSpeed_ms_Max",
    "WindRef_V_Min",
    "WindHeatStateID_Avg",
    "WindValid_Avg",
    *(f"ShuntEtot_kWh_Max_{i}" for i in range(1, 8)),
)


REDUNDANT_FEATURES = (
    "PwrMtrFreqMin_Min",
    "PwrMtrFreqMax_Max",
    "PwrMtrPhaseRev_Avg",
    "PwrMtrP_kVAR_Avg",
    "PwrMtrP_kVA_Avg",
    "PwrMtrEdel_kWh_Max",
    "PwrMtrErec_kWh_Max",
    "PwrMtrEdel_kVARh_Max",
    "PwrMtrErec_kVARh_Max",
    "PwrMtrVaTHDMax_Max",
    "PwrMtrVbTHDMax_Max",
    "PwrMtrVcTHDMax_Max",
    "SEWSAmbientTemp_C_Avg",
    "SEWSModuleTemp_C_Avg",
    "SEWSPOAIrrad_Wm2_Avg",
)


FEATURE_GROUPS = {
    "secondary": SECONDARY_FEATURES,
    "redundant": REDUNDANT_FEATURES,
}


def drop_features(
    df: pd.DataFrame,
    groups: tuple[str, ...] = ("secondary", "redundant"),
) -> pd.DataFrame:
    """Drop selected feature groups without modifying df."""

    unknown = set(groups) - set(FEATURE_GROUPS)

    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")

    columns = list(dict.fromkeys(
        column
        for group in groups
        for column in FEATURE_GROUPS[group]
    ))

    return df.drop(columns=columns)

#Example: df = drop_features(df, groups=("secondary",)) removes only secondary features
# df = drop_features(df) removes both groups


RTD_COLUMNS = tuple(
    f"RTD_C_Avg_{i}" for i in range(1, 10)
)


def rtd_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report mean temperatures for all nine RTDs."""

    means = df.loc[:, RTD_COLUMNS].mean()

    return pd.DataFrame({
        "mean_c": means,
    })


def detect_faulty_rtds(
    df: pd.DataFrame,
    tol: float = 10.0,
) -> list[str]:
    """
    Detect RTDs with unusually different annual means.

    tol: Maximum allowed absolute difference (degrees C)
         from the median RTD mean.

    Diagnostic only: does not remove columns.
    """

    if tol < 0:
        raise ValueError("tol must be non-negative")

    report = rtd_report(df)
    means = report["mean_c"]

    reference = means.median()

    deviations = (means - reference).abs()

    faulty = deviations[deviations > tol]

    return faulty.index.tolist()


def aggregate_faulty_rtds(
    yearly_faults: dict[int, list[str]],
) -> list[str]:
    """Union of faulty RTDs detected across years."""

    return sorted(
        set().union(*yearly_faults.values())
    )


def drop_legacy_rtds(
    df: pd.DataFrame,
    site: SiteConfig,
) -> pd.DataFrame:
    """Reproduce the original site-specific RTD removal."""

    columns = [
        f"RTD_C_Avg_{i}"
        for i in site.rtd_drop
    ]

    return df.drop(columns=columns)


CORRELATION_COLUMNS = (
    "InvVDVoltage_V_Avg",
    "InvVa_Avg",
    "InvVb_Avg",
    "InvVc_Avg",
    "InvIa_Avg",
    "InvIb_Avg",
    "InvIc_Avg",
    "InvVDCin_Avg",
    "InvIDCin_Avg",
    "InvFreq_Avg",
    "PwrMtrIa_Avg",
    "PwrMtrIb_Avg",
    "PwrMtrIc_Avg",
    "PwrMtrFreq_Avg",
    "PwrMtrVa_Avg",
    "PwrMtrVb_Avg",
    "PwrMtrVc_Avg",
    "PwrMtrP_kW_Avg",
)


def correlation_report(
    df: pd.DataFrame,
    columns: tuple[str, ...] = CORRELATION_COLUMNS,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Return strongly correlated feature pairs."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    corr = df.loc[:, columns].corr()

    pairs = []

    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            value = corr.loc[col_a, col_b]

            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({
                    "feature_1": col_a,
                    "feature_2": col_b,
                    "correlation": value,
                })

    return pd.DataFrame(
        pairs,
        columns=["feature_1", "feature_2", "correlation"],
    ).sort_values(
        "correlation",
        key=lambda s: s.abs(),
        ascending=False,
    ).reset_index(drop=True)
