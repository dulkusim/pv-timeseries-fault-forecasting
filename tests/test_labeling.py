import numpy as np
import pandas as pd
import pytest

from scripts.config import CANOPY, GROUND
from scripts.feature_selection import REDUNDANT_FEATURES, SECONDARY_FEATURES
from scripts.inverter import (
    FAULT_FLAG_COLUMNS,
    ORIGINAL_INVERTER_POLICY,
    REVISED_INVERTER_POLICY,
    inverter_features,
    label_inverter,
)
from scripts.labeling import run_labeling
from scripts.operating import operating_mask
from scripts.performance import expected_power, label_power, performance_features
from scripts.strings import fit_string_baselines


@pytest.mark.parametrize("site", [CANOPY, GROUND])
def test_operating_mask_uses_site_sensors_and_strict_threshold(site):
    index = pd.date_range("2015-01-01", periods=4, freq="min")
    frame = pd.DataFrame(
        {
            "Pyra1_Wm2_Avg": [99, 100, 99, np.nan],
            "Pyra3_Wm2_Avg": [99, 100, 101, np.nan],
            "Pyra2_Wm2_Avg": [101, 100, 99, np.nan],
        },
        index=index,
    )
    values = [0, 0, 1, 0] if site == CANOPY else [1, 0, 0, 0]
    pd.testing.assert_series_equal(
        operating_mask(frame, site), pd.Series(values, index=index, name="Op_Mask")
    )


def test_expected_power_scales_corrects_temperature_and_clips_ac():
    frame = pd.DataFrame(
        {
            "POA_fused": [0, 500, 1000, 1000],
            "module_temp": [25, 25, 35, 25],
        }
    )
    result = expected_power(frame, GROUND)
    assert result["P_expected_DC_kW"].tolist() == pytest.approx(
        [0, 135.5, 257.8565, 271]
    )
    assert result["P_expected_AC_kW"].tolist() == pytest.approx(
        [0, 135.5, 257.8565, 260]
    )


def test_rolling_pr_weights_by_expected_power_and_skips_unevaluable_rows():
    frame = pd.DataFrame(
        {
            "InvPAC_kW_Avg": [10, 30, 999, 10],
            "P_expected_AC_kW": [10, 60, 0, 20],
            "PR_evaluable": [1, 1, 0, 1],
        }
    )
    result = performance_features(frame)
    assert pd.isna(result.loc[2, "PR"])
    assert result["PR_roll_5min_weighted"].iloc[:3].isna().all()
    assert result.loc[3, "PR_roll_5min_weighted"] == pytest.approx(50 / 90)


@pytest.mark.parametrize(
    ("pr", "evaluable", "actual", "label"),
    [
        (np.nan, 0, np.nan, "NOT_EVALUABLE"),
        (np.nan, 1, 10, "MISSING_DATA"),
        (0.49, 1, 10, "CRITICAL_UNDERPERFORMANCE"),
        (0.50, 1, 10, "SEVERE_UNDERPERFORMANCE"),
        (0.70, 1, 10, "UNDERPERFORMANCE"),
        (0.85, 1, 10, "NORMAL"),
        (1.05, 1, 10, "NORMAL"),
        (1.06, 1, 10, "SUSPICIOUSLY_HIGH"),
        (-0.1, 1, -10, "NEGATIVE_AC_POWER"),
    ],
)
def test_power_labels_at_pr_thresholds(pr, evaluable, actual, label):
    frame = pd.DataFrame(
        {
            "PR_roll_15min_weighted": [pr],
            "PR_evaluable": [evaluable],
            "InvPAC_kW_Avg": [actual],
            "P_expected_AC_kW": [100],
        }
    )
    assert label_power(frame).tolist() == [label]


def test_string_baselines_use_per_string_medians_of_evaluable_rows():
    frame = pd.DataFrame(
        {
            "Op_Mask": [1, 1, 1, 0, 1, 1],
            "POA_fused": [800, 800, 800, 800, 0, 800],
            "InvOpStatus_Avg": [1, 5, 1, 1, 1, 6],
        }
    )
    for i in range(1, 8):
        frame[f"Inorm_string_{i}"] = [i, 3 * i, 9 * i, 100, 100, 100]

    assert fit_string_baselines(frame, CANOPY) == {i: 3.0 * i for i in range(1, 8)}

    frame["Inorm_string_4"] = 0
    with pytest.raises(ValueError, match="Invalid baseline for string 4"):
        fit_string_baselines(frame, CANOPY)


@pytest.mark.parametrize(
    ("policy", "power", "status", "reported"),
    [
        (ORIGINAL_INVERTER_POLICY, [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]),
        (REVISED_INVERTER_POLICY, [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]),
    ],
)
def test_inverter_shutdown_and_reported_fault_policies(policy, power, status, reported):
    frame = pd.DataFrame(0.0, index=range(4), columns=list(FAULT_FLAG_COLUMNS))
    frame = frame.assign(
        Op_Mask=[1, 1, 1, 0],
        POA_fused=[800, 800, 800, 0],
        InvPAC_kW_Avg=[0.04, 0.05, 80, 0],
        InvPDC_kW_Avg=[1, 0.9, 90, 0],
        InvOpStatus_Avg=[1, 1, 6, 0],
        P_string_total=[1, 1, 90, 0],
        InvMainFault_Max=[-1, 1, np.nan, 0],
    )
    features = inverter_features(frame, CANOPY, policy=policy)
    assert features["inv_shutdown_power_flag"].tolist() == power
    assert features["inv_shutdown_status_flag"].tolist() == status
    assert features["inv_any_fault_reported"].tolist() == reported
    expected = [
        "INVERTER_SHUTDOWN" if p or s else "NORMAL"
        for p, s in zip(power[:3], status[:3], strict=True)
    ]
    assert label_inverter(frame.join(features), policy=policy).tolist() == [
        *expected,
        "NOT_EVALUABLE",
    ]


def test_run_labeling_preserves_a_full_day_and_emits_all_labels():
    index = pd.date_range("2015-06-01", periods=1440, freq="min", name="TIMESTAMP")
    daylight = (index.hour >= 6) & (index.hour < 18)
    columns = sorted(set(SECONDARY_FEATURES + REDUNDANT_FEATURES + FAULT_FLAG_COLUMNS))
    frame = pd.DataFrame(0.0, index=index, columns=columns)
    for column in CANOPY.op_mask_columns + CANOPY.poa_columns:
        frame[column] = np.where(daylight, 800.0, 0.0)
    for i in range(1, 10):
        frame[f"RTD_C_Avg_{i}"] = 25.0
    for i in range(1, 8):
        frame[f"ShuntCurrent_A_Avg_{i}"] = np.where(daylight, 50.0, 0.0)
        frame[f"ShuntPDC_kW_Avg_{i}"] = np.where(daylight, 25.0, 0.0)
    frame = frame.assign(
        WindSpeedAve_ms=0.0,
        InvOpStatus_Avg=daylight.astype(int),
        InvPAC_kW_Avg=np.where(daylight, 180.0, 0.0),
        InvPDC_kW_Avg=np.where(daylight, 195.0, 0.0),
    )

    result = run_labeling(frame, CANOPY)

    assert len(result) == len(frame) == 1440
    pd.testing.assert_index_equal(result.index, index)
    labels = result[["label_power", "label_string", "label_inverter"]]
    assert labels.notna().all().all()
    assert labels.loc["2015-06-01 00:00"].tolist() == ["NOT_EVALUABLE"] * 3
    assert labels.loc["2015-06-01 12:00"].tolist() == ["NORMAL"] * 3
