"""Presentation helpers for the stage-by-stage labeling notebook."""

import pandas as pd

from .config import SiteConfig
from .io import save_stage


def append_features(
    frame: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach features after checking timestamps and column names."""
    if not frame.index.equals(features.index):
        raise ValueError("Feature timestamps do not match the working dataset")
    if features.columns.duplicated().any():
        raise ValueError("Duplicate names in the new feature block")

    overlap = frame.columns.intersection(features.columns)
    if len(overlap):
        raise ValueError(f"Features already exist: {overlap.tolist()}")

    return pd.concat([frame, features], axis=1)


def report_stage(
    name: str,
    frame: pd.DataFrame,
    checkpoint: str | None = None,
    *,
    records: list[dict],
    site: SiteConfig,
    year: int,
    save_checkpoints: bool = False,
) -> None:
    """Record dimensions and optionally save a stage using caller settings."""
    records.append({"stage": name, "rows": len(frame), "columns": frame.shape[1]})
    print(f"{name}: {len(frame):,} rows x {frame.shape[1]} columns")
    if save_checkpoints and checkpoint is not None:
        print(f"Checkpoint: {save_stage(frame, site, year, checkpoint)}")


def label_distribution(labels: pd.Series) -> pd.DataFrame:
    """Report label counts and percentages, including missing labels."""
    counts = labels.value_counts(dropna=False)
    return pd.DataFrame({
        "count": counts,
        "percent_of_all_rows": 100 * counts / len(labels),
    })
