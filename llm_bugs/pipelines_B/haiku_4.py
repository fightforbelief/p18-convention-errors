"""Build a clean quarterly panel of commercial banks from Call Report ratios.

Steps
-----
1. Attach point-in-time charter: each filed report takes the bank's most recent
   attribute snapshot on or before its report date (no look-ahead). Reports that
   predate a bank's first snapshot take that first snapshot's charter.
2. Keep only commercial-bank reports.
3. Map each report to its calendar quarter; if a bank filed more than once for a
   quarter, average the non-missing values.
4. Put each bank on a regular quarterly grid from its first to its last observed
   quarter.
5. Drop banks with too little data: fewer than MIN_OBS observed quarters or a
   coverage ratio (observed / grid length) below MIN_COVERAGE.
6. Fill interior gaps of at most MAX_INTERP_GAP consecutive missing quarters by
   linear interpolation between the neighbouring observations. Longer gaps stay
   missing and their quarters are dropped, so `y` has no missing values.
"""

import numpy as np
import pandas as pd

MIN_OBS = 12            # minimum observed quarters for a bank to be kept
MIN_COVERAGE = 0.8      # minimum share of grid quarters observed
MAX_INTERP_GAP = 2      # longest run of missing quarters to interpolate

_OUT_COLUMNS = ["id", "date", "y"]


def _quarter_index(dates: pd.Series) -> pd.Series:
    """Integer quarter index: year * 4 + (quarter - 1)."""
    return dates.dt.year * 4 + dates.dt.quarter - 1


def _quarter_end(qidx: pd.Series) -> pd.Series:
    """Timestamp of the calendar quarter end for each integer quarter index."""
    year = qidx // 4
    month = (qidx % 4) * 3 + 3
    first_of_month = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": 1})
    )
    return first_of_month + pd.offsets.MonthEnd(0)


def _point_in_time_charter(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Return calls with a `charter` column taken as of each report date."""
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["charter"] = a["charter"].astype(str).str.strip().str.lower()
    a["as_of"] = a["as_of"].astype("int64")
    a = a.drop_duplicates(["rssd_id", "as_of"], keep="last").sort_values("as_of")

    c = calls.sort_values("report_date", kind="mergesort")
    m = pd.merge_asof(
        c,
        a,
        left_on="report_date",
        right_on="as_of",
        by="rssd_id",
        direction="backward",
    )
    m = m.sort_values(["rssd_id", "report_date"], kind="mergesort")
    # Reports before a bank's first snapshot have no backward match; use the
    # earliest known charter for them.
    m["charter"] = m.groupby("rssd_id")["charter"].bfill()
    return m


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({
        "id": pd.Series(dtype="int64"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "y": pd.Series(dtype="float64"),
    })
    if calls.empty or attrs.empty:
        return empty

    c = pd.DataFrame({
        "rssd_id": calls["rssd_id"].astype("int64"),
        "report_date": calls["report_date"].astype("int64"),
        "value": calls["value"].astype("float64"),
    })

    # 1-2. Point-in-time charter, keep commercial banks only.
    m = _point_in_time_charter(c, attrs)
    m = m[m["charter"] == "commercial"]
    if m.empty:
        return empty

    # 3. Map reports to calendar quarters; average duplicate filings.
    dates = pd.to_datetime(m["report_date"].astype(str), format="%Y%m%d")
    q = pd.DataFrame({
        "rssd_id": m["rssd_id"].to_numpy(),
        "q": _quarter_index(dates).to_numpy(),
        "value": m["value"].to_numpy(),
    })
    q = q.groupby(["rssd_id", "q"], as_index=False)["value"].mean()

    # 4. Regular quarterly grid from first to last observed quarter per bank.
    span = q.groupby("rssd_id")["q"].agg(["min", "max"])
    n_q = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n_q)
    starts = np.repeat(span["min"].to_numpy(), n_q)
    offsets = np.arange(n_q.sum()) - np.repeat(np.cumsum(n_q) - n_q, n_q)
    grid = pd.DataFrame({"rssd_id": ids, "q": starts + offsets})
    grid = grid.merge(q, on=["rssd_id", "q"], how="left")
    grid = grid.sort_values(["rssd_id", "q"], kind="mergesort").reset_index(drop=True)

    # 5. Coverage screen.
    grid["obs"] = grid["value"].notna()
    stats = grid.groupby("rssd_id")["obs"].agg(n_obs="sum", n_grid="size")
    ok = stats.index[
        (stats["n_obs"] >= MIN_OBS)
        & (stats["n_obs"] / stats["n_grid"] >= MIN_COVERAGE)
    ]
    grid = grid[grid["rssd_id"].isin(ok)].reset_index(drop=True)
    if grid.empty:
        return empty

    # 6. Linear interpolation for short interior gaps.
    key = grid["rssd_id"]
    obs = grid["obs"]
    q_obs = grid["q"].where(obs)
    y_obs = grid["value"]
    prev_q = q_obs.groupby(key).ffill()
    next_q = q_obs.groupby(key).bfill()
    prev_y = y_obs.groupby(key).ffill()
    next_y = y_obs.groupby(key).bfill()

    gap = next_q - prev_q - 1
    fillable = (~obs) & (gap <= MAX_INTERP_GAP) & prev_y.notna() & next_y.notna()
    weight = (grid["q"] - prev_q) / (next_q - prev_q)
    filled = prev_y + (next_y - prev_y) * weight

    y = y_obs.copy()
    y[fillable] = filled[fillable]
    keep_rows = y.notna()

    out = pd.DataFrame({
        "id": grid.loc[keep_rows, "rssd_id"].astype("int64").to_numpy(),
        "date": _quarter_end(grid.loc[keep_rows, "q"]).to_numpy(),
        "y": y[keep_rows].astype("float64").to_numpy(),
    })
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    out = out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    return out[_OUT_COLUMNS]
