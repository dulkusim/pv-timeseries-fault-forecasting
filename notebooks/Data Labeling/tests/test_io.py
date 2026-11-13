
import pandas as pd
import pytest

from scripts.config import CANOPY, GROUND
from scripts.io import (
    get_data_root,
    count_daily_files,
    merge_daily_csvs,
    load_site_year,
    save_stage,
)


def test_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path



def test_daily_merge(tmp_path):
    source = tmp_path / "onemin-Canopy-2015"
    january = source / "01"
    february = source / "02"

    january.mkdir(parents=True)
    february.mkdir(parents=True)

    pd.DataFrame({"value": [1, 2]}).to_csv(
        january / "day01.csv", index=False
    )
    pd.DataFrame({"value": [3, 4]}).to_csv(
        january / "day02.csv", index=False
    )
    pd.DataFrame({"value": [5, 6]}).to_csv(
        february / "day01.csv", index=False
    )

    counts = count_daily_files(source)

    assert counts == {
        "01": 2,
        "02": 1,
    }

    output = tmp_path / "merged.csv"

    merge_daily_csvs(source, output)

    result = pd.read_csv(output)

    assert result["value"].tolist() == [
        1, 2, 3, 4, 5, 6
    ]


def test_ground_2017_rejects_2016_data(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    folder = tmp_path / "NIST_processed"
    folder.mkdir()

    # Deliberately place 2016 data in a 2017 file.
    wrong_data = pd.DataFrame({
        "TIMESTAMP": ["2016-01-01 00:00:00"],
        "InvPAC_kW_Avg": [100.0],
    })

    wrong_data.to_csv(
        folder / "onemin-Ground-2017.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="Expected year 2017"):
        load_site_year(GROUND, 2017)


def test_canopy_2015_load_and_save(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    folder = tmp_path / "NIST_processed"
    folder.mkdir()

    data = pd.DataFrame({
        "TIMESTAMP": [
            "2015-01-01 00:01:00",
            "2015-01-01 00:00:00",
        ],
        "InvPAC_kW_Avg": [20.0, 10.0],
    })

    data.to_csv(
        folder / "onemin-Canopy-2015.csv",
        index=False,
    )

    df = load_site_year(CANOPY, 2015)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert (df.index.year == 2015).all()

    output = save_stage(df, CANOPY, 2015, "cleaned")
    assert output.exists()

    with pytest.raises(FileExistsError):
        save_stage(df, CANOPY, 2015, "cleaned")
