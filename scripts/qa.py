"""Reusable input validation and physics-versus-reported-fault diagnostics."""

import pandas as pd

from .io import validate_site_year


def check_year_coverage(df: pd.DataFrame, year: int) -> dict:
    """Reject wrong-year timestamps; report gaps without rejecting partial years.

    Build the reference grid in the index timezone, preserving recorded local
    time. A fixed-offset NIST year has 525,600 or 527,040 expected minutes.
    ``complete`` also requires unique, ordered, on-grid timestamps.
    """
    validate_site_year(df, year)
    grid = pd.date_range(
        pd.Timestamp(year=year, month=1, day=1, tz=df.index.tz),
        pd.Timestamp(year=year + 1, month=1, day=1, tz=df.index.tz),
        freq="min",
        inclusive="left",
    )
    unique = df.index.unique()
    missing = grid.difference(unique)
    off_grid = unique.difference(grid)
    duplicates = int(df.index.duplicated().sum())
    steps = df.index.to_series().diff().dropna()
    return {
        "year": year,
        "expected_rows": len(grid),
        "observed_rows": len(df),
        "unique_timestamps": len(unique),
        "duplicate_timestamps": duplicates,
        "missing_minute_count": len(missing),
        "missing_minutes": missing,
        "off_grid_count": len(off_grid),
        "off_grid_timestamps": off_grid,
        "nonminute_steps": int(steps.ne(pd.Timedelta(minutes=1)).sum()),
        "complete": (
            len(df) == len(grid)
            and not duplicates
            and not len(missing)
            and not len(off_grid)
            and df.index.is_monotonic_increasing
        ),
    }


def check_sentinels(
    df: pd.DataFrame,
    values: tuple[float, ...] = (-999, -9999, -99999, -999999),
) -> None:
    """Fail on explicit numeric/string fill codes, without rejecting all negatives.

    Negative temperatures and status codes can be valid. The sentinel list is
    explicit and configurable; missing values (NaN) remain available to labels.
    """
    found = {}
    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        count = int(numeric.isin(values).sum())
        if count:
            found[column] = count
    if found:
        raise ValueError(f"Sentinel fill values detected (column: count): {found}")


def cross_check_with_inverter_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return correctly grouped shutdown and physics/reported count comparisons."""
    power = df["inv_shutdown_power_flag"].eq(1)
    status = df["inv_shutdown_status_flag"].eq(1)
    reported = df["inv_any_fault_reported"].eq(1)
    shutdown = df["label_inverter"].eq("INVERTER_SHUTDOWN")
    efficiency = df["label_inverter"].eq("INVERTER_EFFICIENCY_FAULT")
    evaluable = df["Op_Mask"].eq(1) & df["POA_fused"].ge(100)
    return pd.Series(
        {
            "shutdown_power_only": int((power & ~status).sum()),
            "shutdown_status_only": int((status & ~power).sum()),
            "shutdown_both_flags": int((power & status).sum()),
            "shutdown_either_flag": int((power | status).sum()),
            "shutdown_label": int(shutdown.sum()),
            "shutdown_also_reported": int((shutdown & reported).sum()),
            "shutdown_without_report": int((shutdown & ~reported).sum()),
            "efficiency_fault_label": int(efficiency.sum()),
            "efficiency_also_reported": int((efficiency & reported).sum()),
            "efficiency_without_report": int((efficiency & ~reported).sum()),
            "reported_only_evaluable": int(
                (reported & ~shutdown & ~efficiency & evaluable).sum()
            ),
        },
        name="observations",
    ).to_frame()
