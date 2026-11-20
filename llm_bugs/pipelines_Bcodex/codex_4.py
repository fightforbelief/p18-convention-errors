"""Build a quarterly commercial-bank panel for a four-quarter holdout."""

import numpy as np
import pandas as pd


HOLDOUT_QUARTERS = 4
MIN_TRAIN_QUARTERS = 4
MIN_TRAIN_COVERAGE = 0.80
MAX_TRAIN_GAP = 4


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Return each bank's latest filed commercial spell on a quarterly grid.

    Charter is evaluated at quarter end using the latest available snapshot.
    Training gaps are forward-filled from prior observations. The final four
    quarters must all have reported values and are retained for the benchmark.
    """
    reports = calls[["rssd_id", "report_date", "value"]].copy()
    reports["rssd_id"] = reports["rssd_id"].astype("int64")
    reports["date"] = pd.to_datetime(reports["report_date"], format="%Y%m%d")
    reports["quarter"] = reports["date"].dt.to_period("Q").astype("int64")
    reports["y"] = reports["value"].astype(float).replace([np.inf, -np.inf], np.nan)
    reports = reports.sort_values("date", kind="stable").drop_duplicates(
        ["rssd_id", "quarter"], keep="last"
    )

    bounds = reports.groupby("rssd_id", sort=True)["quarter"].agg(["min", "max"])
    lengths = (bounds["max"] - bounds["min"] + 1).to_numpy()
    offsets = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)
    quarters = np.repeat(bounds["min"].to_numpy(), lengths) + offsets
    dates = pd.PeriodIndex.from_ordinals(quarters, freq="Q").to_timestamp(how="start")
    grid = pd.DataFrame({
        "rssd_id": np.repeat(bounds.index.to_numpy(dtype="int64"), lengths),
        "quarter": quarters,
        "date": dates + pd.offsets.QuarterEnd(startingMonth=12),
    })
    grid = grid.merge(
        reports[["rssd_id", "quarter", "y"]].assign(filed=True),
        on=["rssd_id", "quarter"],
        how="left",
        sort=False,
    )

    snapshots = attrs[["rssd_id", "as_of", "charter"]].copy()
    snapshots["as_of"] = pd.to_datetime(
        snapshots["as_of"], format="%Y%m%d"
    ).astype(grid["date"].dtype)
    snapshots["rssd_id"] = snapshots["rssd_id"].astype("int64")
    ordered = grid[["rssd_id", "date"]].sort_values("date", kind="stable")
    matched = pd.merge_asof(
        ordered,
        snapshots.sort_values("as_of", kind="stable"),
        left_on="date",
        right_on="as_of",
        by="rssd_id",
    )
    commercial = np.empty(len(grid), dtype=bool)
    commercial[ordered.index] = matched["charter"].eq("commercial").fillna(False).to_numpy()

    values = grid["y"].to_numpy(copy=True)
    filed = grid["filed"].eq(True).to_numpy()
    keep = np.zeros(len(grid), dtype=bool)
    for start, stop in zip(lengths.cumsum() - lengths, lengths.cumsum()):
        commercial_reports = np.flatnonzero(commercial[start:stop] & filed[start:stop])
        if not commercial_reports.size:
            continue
        end = start + commercial_reports[-1] + 1
        noncommercial = np.flatnonzero(~commercial[start:end])
        if noncommercial.size:
            start += noncommercial[-1] + 1
        observed = np.flatnonzero(np.isfinite(values[start:end]))
        if not observed.size:
            continue
        start += observed[0]
        series = values[start:end]
        train_size = len(series) - HOLDOUT_QUARTERS
        if train_size < MIN_TRAIN_QUARTERS or not np.isfinite(series[train_size:]).all():
            continue
        train_observed = np.isfinite(series[:train_size])
        if train_observed.mean() < MIN_TRAIN_COVERAGE:
            continue
        positions = np.arange(train_size)
        previous = np.maximum.accumulate(np.where(train_observed, positions, 0))
        if (positions - previous).max() > MAX_TRAIN_GAP:
            continue
        series[:train_size] = series[previous]
        keep[start:end] = True

    grid["y"] = values
    return (
        grid.loc[keep, ["rssd_id", "date", "y"]]
        .rename(columns={"rssd_id": "id"})
        .reset_index(drop=True)
    )
