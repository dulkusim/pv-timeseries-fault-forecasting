
import pandas as pd


MISSING_THRESHOLD = 0.1

# Exact behavior of the original notebooks.
LEGACY_MISSING_DROP = ("WindSpeedAve_ms",)

PROTECTED_FEATURES = (
    "InvPAC_kW_Avg", "InvPDC_kW_Avg", "InvOpStatus_Avg", 
    "InvIa_Avg", "InvVa_Avg", "InvIb_Avg", "InvOpStatus_Avg",
    "InvVb_Avg", "InvVc_Avg", "InvPAC_kW_Avg"        
    "InvFreq_Avg", "InvIDCin_Avg", "InvVDCin_Avg",
    "InvIc_Avg", "InvPDC_kW_Avg", "InvVPVin_Avg",)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing-value counts and proportions."""

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset")

    counts = df.isna().sum()

    report = pd.DataFrame({
        "missing_count": counts,
        "missing_fraction": counts / len(df),
    })

    return report.loc[
        report["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False,
    )


def drop_high_missing(
    df: pd.DataFrame,
    threshold: float = MISSING_THRESHOLD,
    protected: tuple[str, ...] = PROTECTED_FEATURES,
) -> pd.DataFrame:
    """Drop high-missingness features unless protected."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    missing_fraction = df.isna().mean()

    columns_to_drop = [
        column
        for column in df.columns
        if missing_fraction[column] > threshold
        and column not in protected
    ]

    return df.drop(columns=columns_to_drop)



def apply_missingness_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the established cross-site feature policy."""

    return df.drop(columns=list(LEGACY_MISSING_DROP))