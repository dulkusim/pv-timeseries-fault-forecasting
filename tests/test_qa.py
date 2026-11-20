import pandas as pd
import pytest

from scripts.qa import (
    check_sentinels,
    check_year_coverage,
    cross_check_with_inverter_flags,
)


@pytest.mark.parametrize("year, expected", [(2015, 525600), (2016, 527040)])
def test_complete_fixed_offset_year(year, expected):
    index = pd.date_range(f"{year}-01-01", periods=expected, freq="min", tz="Etc/GMT+5")
    result = check_year_coverage(pd.DataFrame({"power": 1}, index=index), year)
    assert result["complete"]
    assert result["missing_minute_count"] == 0
    assert result["expected_rows"] == expected


def test_gap_and_duplicate_cannot_hide_each_other():
    grid = pd.date_range("2015-01-01", periods=525600, freq="min", tz="Etc/GMT+5")
    index = grid.delete(10).append(grid[:1]).sort_values()
    result = check_year_coverage(pd.DataFrame({"power": 1}, index=index), 2015)
    assert result["observed_rows"] == result["expected_rows"]
    assert result["missing_minutes"].equals(grid[10:11])
    assert result["duplicate_timestamps"] == 1
    assert not result["complete"]


def test_wrong_year_rejected():
    frame = pd.DataFrame({"power": [1]}, index=pd.to_datetime(["2016-01-01"]))
    with pytest.raises(ValueError, match="Expected year 2017"):
        check_year_coverage(frame, 2017)


def test_naive_partial_year_reports_real_gap_and_off_grid():
    index = pd.to_datetime(["2015-01-01 00:00:00", "2015-01-01 00:01:30"])
    result = check_year_coverage(pd.DataFrame({"power": [1, 2]}, index=index), 2015)
    assert result["missing_minute_count"] == 525599
    assert result["off_grid_count"] == 1


@pytest.mark.parametrize("value", [-999, -9999, " -999.0 ", "-99999"])
def test_numeric_and_string_sentinels_fail(value):
    with pytest.raises(ValueError, match="Sentinel fill values"):
        check_sentinels(pd.DataFrame({"sensor": [value]}))


def test_real_negatives_and_missing_values_are_not_sentinels():
    check_sentinels(pd.DataFrame({"temperature": [-20.5, None], "status": [-1, 0]}))


def test_cross_check_distinguishes_power_only_from_status_only():
    frame = pd.DataFrame(
        {
            "inv_shutdown_power_flag": [1, 0, 1, 0],
            "inv_shutdown_status_flag": [0, 1, 1, 0],
            "inv_any_fault_reported": [1, 0, 1, 1],
            "label_inverter": [
                "INVERTER_SHUTDOWN",
                "INVERTER_SHUTDOWN",
                "INVERTER_SHUTDOWN",
                "NORMAL",
            ],
            "Op_Mask": [1, 1, 1, 1],
            "POA_fused": [800] * 4,
        }
    )
    result = cross_check_with_inverter_flags(frame)["observations"]
    assert result["shutdown_power_only"] == 1
    assert result["shutdown_status_only"] == 1
    assert result["shutdown_both_flags"] == 1
    assert result["shutdown_either_flag"] == 3
    assert result["shutdown_also_reported"] == 2
    assert result["reported_only_evaluable"] == 1
