
import pandas as pd

from scripts.cleaning import (
    missing_report,
    drop_high_missing,
    drop_legacy_missing,
)


def test_missing_report():
    df = pd.DataFrame({
        "a": [1, None, 3, 4],
        "b": [1, 2, 3, 4],
    })

    report = missing_report(df)

    assert report.loc["a", "missing_count"] == 1
    assert report.loc["a", "missing_fraction"] == 0.25
    assert "b" not in report.index


def test_drop_high_missing():
    df = pd.DataFrame({
        "a": [None] + list(range(19)),
        "b": list(range(20)),
    })

    # Exactly 5% missing: retain.
    result = drop_high_missing(df, threshold=0.05)

    assert "a" in result.columns

    # Above 4% missing: remove.
    result = drop_high_missing(df, threshold=0.04)

    assert "a" not in result.columns


def test_legacy_missing():
    df = pd.DataFrame({
        "WindSpeedAve_ms": [1.0],
        "InvPAC_kW_Avg": [100.0],
    })

    result = drop_legacy_missing(df)

    assert result.columns.tolist() == [
        "InvPAC_kW_Avg"
    ]
