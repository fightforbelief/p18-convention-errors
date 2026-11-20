"""Build a quarterly bank panel from Call Report ratios and charter snapshots.

Only pandas and numpy are used.
"""

import numpy as np
import pandas as pd

MAX_MISSING_SHARE = 0.20


def _quarter_index(yyyymmdd: np.ndarray) -> np.ndarray:
    """Map YYYYMMDD ints to an integer quarter index (year * 4 + quarter - 1)."""
    yyyymmdd = np.asarray(yyyymmdd, dtype=np.int64)
    year = yyyymmdd // 10000
    month = (yyyymmdd // 100) % 100
    return year * 4 + (month - 1) // 3


def _quarter_end(qidx: np.ndarray) -> pd.Series:
    """Map integer quarter indices to calendar quarter-end Timestamps."""
    qidx = np.asarray(qidx, dtype=np.int64)
    year = qidx // 4
    month = (qidx % 4) * 3 + 3
    first = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": 1})
    )
    return pd.Series(first + pd.offsets.MonthEnd(0))


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "y": pd.Series(dtype="float64"),
        }
    )

    # ---- Reported values: one value per bank-quarter -----------------------
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["qi"] = _quarter_index(c["report_date"].to_numpy())
    # If a bank-quarter somehow has several filings, keep the last non-NaN value.
    vals = (
        c.groupby(["rssd_id", "qi"], sort=False)["value"]
        .last()
        .reset_index()
    )
    if vals.empty:
        return empty

    # ---- Active span and regular quarterly grid ----------------------------
    span = vals.groupby("rssd_id", sort=False)["qi"].agg(["min", "max"])
    n_q = (span["max"] - span["min"] + 1).to_numpy(dtype=np.int64)
    ids = np.repeat(span.index.to_numpy(dtype=np.int64), n_q)
    # offsets 0..n_q-1 within each bank, built without a Python loop
    starts = np.repeat(np.cumsum(n_q) - n_q, n_q)
    offs = np.arange(n_q.sum(), dtype=np.int64) - starts
    qi = np.repeat(span["min"].to_numpy(dtype=np.int64), n_q) + offs
    grid = pd.DataFrame({"rssd_id": ids, "qi": qi})
    grid["date"] = _quarter_end(grid["qi"].to_numpy())
    grid = grid.merge(vals, on=["rssd_id", "qi"], how="left")

    # ---- Charter at each quarter (latest snapshot on or before quarter end) -
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["as_of"] = pd.to_datetime(a["as_of"].astype(str), format="%Y%m%d")
    a = a.sort_values(["as_of", "rssd_id"]).drop_duplicates(
        ["rssd_id", "as_of"], keep="last"
    )
    grid = grid.sort_values(["date", "rssd_id"]).reset_index(drop=True)
    a = a.sort_values(["as_of", "rssd_id"]).reset_index(drop=True)
    grid = pd.merge_asof(
        grid,
        a.rename(columns={"as_of": "date"}),
        on="date",
        by="rssd_id",
        direction="backward",
        allow_exact_matches=True,
    )
    # Quarters with no snapshot on or before them have unknown charter -> dropped.
    grid = grid[grid["charter"] == "commercial"]
    if grid.empty:
        return empty

    # ---- Drop banks with too many missing quarters (after charter filter) --
    grid["missing"] = grid["value"].isna()
    stats = grid.groupby("rssd_id", sort=False)["missing"].agg(["sum", "size"])
    share = stats["sum"] / stats["size"]
    keep_ids = share.index[share <= MAX_MISSING_SHARE]
    grid = grid[grid["rssd_id"].isin(keep_ids)]

    # ---- Drop missing bank-quarters; no imputation ---------------------------
    out = grid.loc[~grid["missing"], ["rssd_id", "date", "value"]].rename(
        columns={"rssd_id": "id", "value": "y"}
    )
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    out = out.sort_values(["id", "date"]).reset_index(drop=True)
    out["date"] = out["date"].astype("datetime64[ns]")
    return out[["id", "date", "y"]]
