
import os
from pathlib import Path

import pandas as pd

from .config import SiteConfig


def get_data_root() -> Path:
    """Return the root directory containing NIST data."""

    env_root = os.getenv("DATA_ROOT")

    if env_root:
        root = Path(env_root).expanduser()
    elif Path("/content/drive/MyDrive").is_dir():
        root = Path("/content/drive/MyDrive")
    else:
        raise RuntimeError(
            "DATA_ROOT is not set and Google Drive is "
            "not mounted at /content/drive/MyDrive."
        )

    if not root.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist: {root}"
        )

    return root


def count_daily_files(base: str | Path) -> int:
    """Count CSV files directly inside a directory."""

    base = Path(base)

    if not base.is_dir():
        raise NotADirectoryError(base)

    return sum(1 for path in base.glob("*.csv")
               if path.is_file())


def merge_daily_csvs(
    base: str | Path,
    out: str | Path,
) -> Path:
    """Merge daily CSVs with one pandas concatenation."""

    base = Path(base)
    out = Path(out)

    if not base.is_dir():
        raise NotADirectoryError(base)

    files = sorted(
        path for path in base.glob("*.csv")
        if path.is_file() and path.resolve() != out.resolve()
    )

    if not files:
        raise ValueError(f"No CSV files found in {base}")

    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    return out


def validate_site_year(
    df: pd.DataFrame,
    year: int,
) -> None:
    """Reject datasets containing incorrect timestamps."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")

    if df.index.hasnans:
        raise ValueError("Dataset contains missing timestamps")

    wrong_year = df.index.year != year

    if wrong_year.any():
        found = sorted(set(df.index.year.tolist()))
        raise ValueError(
            f"Expected year {year}, found years {found}"
        )


def load_site_year(
    site: SiteConfig,
    year: int,
) -> pd.DataFrame:
    """Load a merged site-year CSV and validate timestamps."""

    root = get_data_root()

    path = (
        root
        / "NIST_processed"
        / site.raw_file_pattern.format(year=year)
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {site.name} {year} dataset: {path}"
        )

    df = pd.read_csv(path)

    if "TIMESTAMP" not in df.columns:
        raise ValueError(
            f"TIMESTAMP column missing from {path}"
        )

    df["TIMESTAMP"] = pd.to_datetime(
        df["TIMESTAMP"],
        errors="raise",
    )

    df = df.set_index("TIMESTAMP")
    df = df.sort_index()

    validate_site_year(df, year)

    return df


def save_stage(
    df: pd.DataFrame,
    site: SiteConfig,
    year: int,
    stage: str,
) -> Path:
    """Save one processing stage with consistent naming."""

    allowed_stages = {
        "features",
        "cleaned",
        "operating",
        "performance",
        "strings",
        "inverter",
        "labeled",
    }

    if stage not in allowed_stages:
        raise ValueError(f"Unknown stage: {stage}")

    validate_site_year(df, year)

    root = get_data_root()
    output_dir = root / "NIST_labeling_stages"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / f"{site.name}_{year}_{stage}.csv"
    )

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    df.to_csv(output_path, index=True, index_label="TIMESTAMP")

    return output_path