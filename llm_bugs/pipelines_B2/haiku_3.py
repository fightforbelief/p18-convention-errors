"""Quarterly panel of year-over-year changes in Call Report ratios.

Only pandas and numpy are used.

Method
------
1. Clean ``calls``: drop missing / non-finite values, map each ``report_date``
   to its calendar quarter end, and keep one value per bank-quarter (the latest
   filing that carries a value).
2. Lay the values out on a dense quarterly grid per bank (rows = banks,
   columns = quarters). The year-over-year change for quarter t is the value
   at t minus the value four grid columns earlier, so the comparison is always
   with the same quarter one year earlier. Gaps are not interpolated: if either
   end of a comparison is missing, no y is produced for that quarter.
3. Attach the charter type as of each quarter (latest attribute snapshot dated
   on or before the quarter end; if a bank has no earlier snapshot, its
   earliest snapshot is used). Keep only commercial-bank quarters.
4. Drop banks with fewer than ``MIN_YOY_OBS`` valid y observations (thin data).
"""

import numpy as np
import pandas as pd

# A bank must have at least this many valid year-over-year changes (commercial
# quarters only) to be included. Twelve observations gives three years of
# history beyond the last four quarters held out for the benchmark.
MIN_YOY_OBS = 12

LAG_QUARTERS = 4

_OUT_COLUMNS = ["id", "date", "y"]


def _yyyymmdd_to_quarter_end(s: pd.Series) -> pd.Series:
    """Map YYYYMMDD integers to the calendar quarter-end timestamp (NaT if invalid)."""
    dt = pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    # Rolls any date forward to its quarter end; quarter-end dates are unchanged.
    return (dt + pd.offsets.QuarterEnd(0)).dt.normalize().astype("datetime64[ns]")


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- 1. clean calls -------------------------------------------------
    c = pd.DataFrame(
        {
            "id": calls["rssd_id"].astype("int64").to_numpy(),
            "report_date": calls["report_date"].astype("int64").to_numpy(),
            "value": pd.to_numeric(calls["value"], errors="coerce").to_numpy(dtype="float64"),
        }
    )
    c = c[np.isfinite(c["value"])]
    c = c.assign(date=_yyyymmdd_to_quarter_end(c["report_date"]))
    c = c.dropna(subset=["date"])
    # One value per bank-quarter: latest filing that has a value.
    c = (
        c.sort_values(["id", "date", "report_date"])
        .drop_duplicates(["id", "date"], keep="last")
        .sort_values("date", kind="mergesort")
    )

    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    # ---- 2. dense quarterly grid and lag-4 differences ------------------
    lo, hi = c["date"].min(), c["date"].max()
    grid = pd.date_range(lo, hi, freq=pd.offsets.QuarterEnd(startingMonth=12)).astype("datetime64[ns]")

    wide = c.pivot(index="id", columns="date", values="value").reindex(columns=grid)
    vals = wide.to_numpy(dtype="float64")
    ids = wide.index.to_numpy(dtype="int64")

    ymat = np.full(vals.shape, np.nan)
    if vals.shape[1] > LAG_QUARTERS:
        ymat[:, LAG_QUARTERS:] = vals[:, LAG_QUARTERS:] - vals[:, :-LAG_QUARTERS]

    rows, cols = np.nonzero(np.isfinite(ymat))
    out = pd.DataFrame(
        {
            "id": ids[rows],
            "date": grid.to_numpy()[cols],
            "y": ymat[rows, cols],
        }
    )
    if out.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    # ---- 3. charter as of each quarter; keep commercial banks -----------
    at = pd.DataFrame(
        {
            "id": attrs["rssd_id"].astype("int64").to_numpy(),
            "as_of": attrs["as_of"].astype("int64").to_numpy(),
            "charter": attrs["charter"].astype("string").str.strip().str.lower().to_numpy(),
        }
    )
    at = at.assign(as_of_dt=_yyyymmdd_to_quarter_end(at["as_of"])).dropna(subset=["as_of_dt"])
    at = at.sort_values(["as_of_dt", "id"], kind="mergesort")
    at["as_of_dt"] = at["as_of_dt"].astype("datetime64[ns]")

    out = out.sort_values("date", kind="mergesort")
    out = pd.merge_asof(
        out,
        at[["id", "as_of_dt", "charter"]],
        left_on="date",
        right_on="as_of_dt",
        by="id",
        direction="backward",
    )
    # Quarters before a bank's first snapshot fall back to that first snapshot.
    first = at.drop_duplicates("id", keep="first").set_index("id")["charter"]
    missing = out["charter"].isna()
    if missing.any():
        out.loc[missing, "charter"] = out.loc[missing, "id"].map(first).to_numpy()

    out = out[out["charter"] == "commercial"]

    # ---- 4. drop thin banks --------------------------------------------
    counts = out.groupby("id")["y"].transform("size")
    out = out[counts >= MIN_YOY_OBS]

    out = out[_OUT_COLUMNS].sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    out["date"] = pd.to_datetime(out["date"])
    return out
