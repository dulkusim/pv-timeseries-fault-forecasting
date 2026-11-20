
import pandas as pd

from scripts.config import CANOPY, GROUND
from scripts.feature_selection import (
    drop_features,
    rtd_report,
    detect_faulty_rtds,
    drop_legacy_rtds,
    correlation_report,
)


def test_drop_features():
    df = pd.DataFrame({
        "RECORD": [1, 2],
        "PwrMtrFreqMin_Min": [49, 50],
        "InvPAC_kW_Avg": [10, 20],
    })

    # Test individual groups with their complete schemas.
    # A minimal dataframe should raise for missing columns.
    try:
        drop_features(df)
    except KeyError:
        pass
    else:
        raise AssertionError("Missing columns were not detected")


def test_rtd_detection():
    data = {
        f"RTD_C_Avg_{i}": [20.0, 21.0]
        for i in range(1, 10)
    }

    data["RTD_C_Avg_7"] = [-27.0, -28.0]

    df = pd.DataFrame(data)

    assert len(rtd_report(df)) == 9
    assert detect_faulty_rtds(df, tol=10) == [
        "RTD_C_Avg_7"
    ]


def test_legacy_rtds():
    df = pd.DataFrame({
        f"RTD_C_Avg_{i}": [20.0]
        for i in range(1, 10)
    })

    canopy = drop_legacy_rtds(df, CANOPY)
    ground = drop_legacy_rtds(df, GROUND)

    assert len(canopy.columns) == 5
    assert len(ground.columns) == 6


def test_correlation_report():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 4, 6, 8],
        "c": [4, 3, 2, 1],
    })

    result = correlation_report(
        df,
        columns=("a", "b", "c"),
        threshold=0.9,
    )

    assert len(result) == 3
