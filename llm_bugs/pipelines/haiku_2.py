"""Build a quarterly bank panel from Call Report ratios and charter snapshots.

Only pandas and numpy are used.
"""

import numpy as np
import pandas as pd


def _to_quarter_end(yyyymmdd: pd.Series) -> pd.Series:
    """Convert YYYYMMDD integers to pandas Timestamps."""
    return pd.to_datetime(yyyymmdd.astype(str), format="%Y%m%d")


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "y": pd.Series(dtype="float64"),
        }
    )

    # ---- Normalise calls -------------------------------------------------
    c = pd.DataFrame(
        {
            "id": calls["rssd_id"].to_numpy(dtype="int64"),
            "date": _to_quarter_end(calls["report_date"]),
            "y": calls["value"].to_numpy(dtype="float64"),
        }
    )
    if c.empty:
        return empty

    # Quarter-end dates are normalised to calendar quarter ends.
    c["date"] = c["date"] + pd.offsets.QuarterEnd(0)

    # One row per (id, date). If duplicates exist, keep the last non-NaN value.
    c = (
        c.sort_values(["id", "date"], kind="mergesort")
        .groupby(["id", "date"], as_index=False, sort=True)["y"]
        .last()
    )

    # ---- Regular quarterly grid per bank over its active span ------------
    span = c.groupby("id")["date"].agg(["min", "max"]).reset_index()
    span["n"] = (
        (span["max"].dt.year - span["min"].dt.year) * 4
        + (span["max"].dt.month - span["min"].dt.month) // 3
        + 1
    )
    rep = np.repeat(span["id"].to_numpy(), span["n"].to_numpy())
    # Offsets 0..n-1 within each bank's span.
    starts = np.repeat(np.cumsum(span["n"].to_numpy()) - span["n"].to_numpy(), span["n"].to_numpy())
    k = np.arange(len(rep)) - starts
    base_min = np.repeat(span["min"].to_numpy(), span["n"].to_numpy())
    base_year = pd.DatetimeIndex(base_min).year.to_numpy()
    base_month = pd.DatetimeIndex(base_min).month.to_numpy()
    total_months = base_year * 12 + (base_month - 1) + 3 * k
    g_year = total_months // 12
    g_month = total_months % 12 + 1
    grid_date = pd.to_datetime(
        {"year": g_year, "month": g_month, "day": 1}
    ) + pd.offsets.MonthEnd(0)
    grid = pd.DataFrame({"id": rep.astype("int64"), "date": grid_date})

    # Attach reported values (NaN where missing).
    grid = grid.merge(c, on=["id", "date"], how="left")

    # ---- Charter as of each quarter (rule 2) ------------------------------
    a = pd.DataFrame(
        {
            "id": attrs["rssd_id"].to_numpy(dtype="int64"),
            "as_of": _to_quarter_end(attrs["as_of"]),
            "charter": attrs["charter"].astype(str).str.strip().str.lower().to_numpy(),
        }
    ).sort_values(["id", "as_of"], kind="mergesort")

    grid = grid.sort_values(["date", "id"], kind="mergesort")
    a = a.sort_values(["as_of", "id"], kind="mergesort")
    grid = pd.merge_asof(
        grid,
        a,
        left_on="date",
        right_on="as_of",
        by="id",
        direction="backward",
    )
    grid = grid.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)

    # Keep only quarters where the bank's charter is commercial.
    grid = grid[grid["charter"] == "commercial"].copy()

    # ---- Drop banks with >20% missing on remaining grid quarters ---------
    grid["miss"] = grid["y"].isna()
    miss_share = grid.groupby("id")["miss"].mean()
    keep_ids = miss_share.index[miss_share <= 0.20]
    grid = grid[grid["id"].isin(keep_ids)]

    # ---- Drop missing bank-quarters; no imputation -----------------------
    out = grid.loc[~grid["miss"], ["id", "date", "y"]].copy()
    out["id"] = out["id"].astype("int64")
    out["date"] = pd.to_datetime(out["date"])
    out["y"] = out["y"].astype("float64")
    out = out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    return out
