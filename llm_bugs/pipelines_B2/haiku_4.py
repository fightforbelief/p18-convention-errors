"""Quarterly panel of year-over-year changes in a Call Report ratio.

build_panel(calls, attrs) returns one row per commercial bank-quarter with
    id   : rssd_id (int)
    date : calendar quarter end (Timestamp)
    y    : value(q) - value(q - 4), the change against the same quarter a year earlier

Design choices
- Reports are mapped to calendar quarter ends. Several reports for the same
  bank-quarter are averaged over their non-missing values.
- The lag is taken on a full quarterly grid per bank, so a missing quarter breaks
  the comparison (y is NaN) instead of silently pairing with a different quarter.
  No interpolation or imputation is done.
- Charter is taken from the latest attrs snapshot on or before each quarter end.
  Quarters before a bank's first snapshot use that first snapshot. A bank-quarter
  is kept only when it is commercial at that date.
- Banks with fewer than MIN_Y_OBS valid y observations are dropped as too thin.
"""

import numpy as np
import pandas as pd

LAG = 4          # quarters: same quarter one year earlier
MIN_Y_OBS = 8    # minimum valid y observations per bank (two years of changes)


def _quarter_end(yyyymmdd: pd.Series) -> pd.Series:
    """Map YYYYMMDD integers to the calendar quarter end Timestamp."""
    d = pd.to_datetime(yyyymmdd.astype("int64").astype(str), format="%Y%m%d")
    return d + pd.offsets.QuarterEnd(0)


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- clean reports ------------------------------------------------------
    c = pd.DataFrame({
        "id": pd.to_numeric(calls["rssd_id"], errors="coerce"),
        "rdate": pd.to_numeric(calls["report_date"], errors="coerce"),
        "value": pd.to_numeric(calls["value"], errors="coerce"),
    }).dropna(subset=["id", "rdate"])
    c["id"] = c["id"].astype("int64")
    c["date"] = _quarter_end(c["rdate"])
    c["q"] = c["date"].dt.year * 4 + c["date"].dt.quarter - 1  # integer quarter index

    # ---- bank x quarter grid of ratios --------------------------------------
    bq = c.groupby(["id", "q"])["value"].mean()          # NaN if all reports missing
    wide = bq.unstack("q")
    qcols = np.arange(int(c["q"].min()), int(c["q"].max()) + 1)
    wide = wide.reindex(columns=qcols)
    ids = wide.index.to_numpy()
    arr = wide.to_numpy(dtype=float)

    # ---- year-over-year change on the calendar grid -------------------------
    y = np.full(arr.shape, np.nan)
    y[:, LAG:] = arr[:, LAG:] - arr[:, :-LAG]
    rows, cols = np.nonzero(~np.isnan(y))
    long = pd.DataFrame({
        "id": ids[rows],
        "q": qcols[cols],
        "y": y[rows, cols],
    })
    year = long["q"] // 4
    quarter = long["q"] % 4 + 1
    long["date"] = pd.to_datetime({"year": year, "month": quarter * 3, "day": 1}) \
        + pd.offsets.MonthEnd(0)
    long = long.drop(columns="q").sort_values(["date", "id"], kind="stable")

    # ---- charter as of each quarter -----------------------------------------
    a = pd.DataFrame({
        "id": pd.to_numeric(attrs["rssd_id"], errors="coerce"),
        "as_of": pd.to_numeric(attrs["as_of"], errors="coerce"),
        "charter": attrs["charter"].astype(str).str.strip().str.lower(),
    }).dropna(subset=["id", "as_of"])
    a["id"] = a["id"].astype("int64")
    a["as_of"] = pd.to_datetime(a["as_of"].astype("int64").astype(str), format="%Y%m%d")
    a = a.drop_duplicates(["id", "as_of"], keep="last").sort_values("as_of")

    merged = pd.merge_asof(
        long, a, left_on="date", right_on="as_of", by="id", direction="backward"
    )
    # Quarters before a bank's first snapshot take its earliest snapshot.
    merged = merged.sort_values("date", kind="stable")
    merged["charter"] = merged.groupby("id")["charter"].bfill()

    panel = merged[merged["charter"] == "commercial"]

    # ---- drop thin banks ----------------------------------------------------
    counts = panel.groupby("id")["y"].size()
    keep = counts[counts >= MIN_Y_OBS].index
    panel = panel[panel["id"].isin(keep)]

    out = panel[["id", "date", "y"]].sort_values(["id", "date"]).reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["date"] = pd.to_datetime(out["date"])
    out["y"] = out["y"].astype("float64")
    return out


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    calls = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    attrs = pd.read_csv(os.path.join(here, "sample_attrs.csv"))
    panel = build_panel(calls, attrs)
    print(panel.head())
    print(panel.dtypes)
    print("rows", len(panel), "banks", panel["id"].nunique())
