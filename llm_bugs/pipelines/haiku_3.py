"""Build a quarterly bank panel from Call Report ratios and charter snapshots.

Uses only pandas and numpy.
"""

import numpy as np
import pandas as pd


def _to_quarter_end(yyyymmdd: pd.Series) -> pd.Series:
    """Map YYYYMMDD integers to the calendar quarter-end Timestamp."""
    ts = pd.to_datetime(yyyymmdd.astype(str), format="%Y%m%d")
    return ts.dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- Prepare calls -----------------------------------------------------
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["date"] = _to_quarter_end(c["report_date"])
    c["value"] = c["value"].astype(float)
    # Treat infinities as not reported.
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    # If a bank filed more than one report for the same quarter, keep the
    # last non-missing value (sorted by original report_date so the latest
    # filing wins among non-missing values).
    c = c.sort_values(["rssd_id", "date", "report_date"])
    c = c.groupby(["rssd_id", "date"], sort=False, as_index=False)["value"].last()

    # ---- Rule 1: regular quarterly grid over each bank's active span --------
    span = c.groupby("rssd_id")["date"].agg(["min", "max"]).reset_index()
    # Number of quarters from first to last, inclusive.
    q_first = span["min"].dt.year * 4 + (span["min"].dt.month - 1) // 3
    q_last = span["max"].dt.year * 4 + (span["max"].dt.month - 1) // 3
    n_q = (q_last - q_first + 1).astype(int)

    counts = n_q.to_numpy()
    ids = np.repeat(span["rssd_id"].to_numpy(), counts)
    starts = np.repeat(q_first.to_numpy(), counts)
    offset = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    qidx = starts + offset
    grid = pd.DataFrame(
        {
            "id": ids,
            "date": pd.to_datetime(
                {
                    "year": qidx // 4,
                    "month": (qidx % 4) * 3 + 3,
                    "day": 1,
                }
            )
            + pd.offsets.MonthEnd(0),
        }
    )
    grid["date"] = grid["date"].dt.normalize()

    # Attach observed values to the grid; missing reports become NaN.
    grid = grid.merge(
        c.rename(columns={"rssd_id": "id"})[["id", "date", "value"]],
        on=["id", "date"],
        how="left",
    )

    # ---- Rule 2: keep quarters where charter at that quarter is commercial --
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["as_of_date"] = pd.to_datetime(a["as_of"].astype(str), format="%Y%m%d")
    a = a.sort_values(["rssd_id", "as_of_date"])
    a = a.groupby(["rssd_id", "as_of_date"], sort=False, as_index=False)["charter"].last()

    grid = grid.sort_values(["date", "id"]).reset_index(drop=True)
    grid = pd.merge_asof(
        grid,
        a.rename(columns={"rssd_id": "id", "as_of_date": "date"}).sort_values(["date", "id"]),
        on="date",
        by="id",
        direction="backward",
        allow_exact_matches=True,
    )
    grid = grid.sort_values(["id", "date"]).reset_index(drop=True)
    # Quarters with no snapshot on or before them have no charter -> not kept.
    grid = grid[grid["charter"] == "commercial"].copy()

    # ---- Rule 3: drop banks with more than 20% missing among kept quarters --
    grid["is_missing"] = grid["value"].isna()
    miss = grid.groupby("id")["is_missing"].agg(["mean", "size"])
    keep_ids = miss.index[miss["mean"] <= 0.2]
    grid = grid[grid["id"].isin(keep_ids)]

    # ---- Rule 4: drop missing bank-quarters, no imputation ------------------
    out = grid.loc[~grid["is_missing"], ["id", "date", "value"]].rename(
        columns={"value": "y"}
    )
    out["id"] = out["id"].astype(int)
    out["date"] = pd.to_datetime(out["date"])
    out["y"] = out["y"].astype(float)
    out = out.sort_values(["id", "date"]).reset_index(drop=True)
    return out[["id", "date", "y"]]
