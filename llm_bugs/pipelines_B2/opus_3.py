"""Quarterly panel of year-over-year changes in a Call Report ratio, commercial banks only.

Design choices
--------------
* Dates are snapped to calendar quarters; if a bank files more than once in the same
  quarter, the latest filing (last non-missing value) wins.
* YoY change is ``value(q) - value(q - 4 quarters)`` matched by *calendar quarter*, not by
  row position, so a missing filing never makes us compare against the wrong quarter.
  Gaps are not filled: if either end is missing (no filing or NaN value) there is no y.
* Charter is taken point-in-time: each quarter uses the latest attribute snapshot with
  ``as_of <= quarter end`` (quarters before a bank's first snapshot use that first
  snapshot). A YoY change is kept only if the bank is commercial at both ends of the
  comparison, so charter conversions do not leak savings-institution data in.
* Banks with fewer than ``MIN_OBS`` valid y observations are dropped: the benchmark holds
  out the last 4 quarters, so we require 4 hold-out + 8 training quarters.
"""
import numpy as np
import pandas as pd

HOLDOUT = 4
MIN_TRAIN = 8
MIN_OBS = HOLDOUT + MIN_TRAIN


def _quarter_index(yyyymmdd: pd.Series) -> np.ndarray:
    """Integer quarter index (year*4 + quarter-1) from YYYYMMDD ints, vectorised."""
    v = pd.to_numeric(yyyymmdd, errors="coerce").to_numpy(dtype="float64")
    year = np.floor(v / 10000)
    month = np.floor(v / 100) % 100
    q = year * 4 + np.floor((month - 1) / 3)
    ok = np.isfinite(q) & (month >= 1) & (month <= 12)
    return np.where(ok, q, np.nan)


def _quarter_end(qidx: np.ndarray) -> pd.DatetimeIndex:
    year = (qidx // 4).astype(int)
    month = ((qidx % 4) * 3 + 3).astype(int)
    first = pd.to_datetime({"year": year, "month": month, "day": np.ones_like(year)})
    return pd.DatetimeIndex(first + pd.offsets.MonthEnd(0))


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})

    # ---- calls: one value per bank-quarter -------------------------------------------
    c = pd.DataFrame({
        "id": pd.to_numeric(calls["rssd_id"], errors="coerce"),
        "q": _quarter_index(calls["report_date"]),
        "d": pd.to_numeric(calls["report_date"], errors="coerce"),
        "value": pd.to_numeric(calls["value"], errors="coerce"),
    })
    c = c.dropna(subset=["id", "q"])
    c = c[np.isfinite(c["value"])]  # NaN/inf values carry no information
    if c.empty:
        return empty
    c["id"] = c["id"].astype("int64")
    c["q"] = c["q"].astype("int64")
    c = (c.sort_values(["id", "q", "d"], kind="mergesort")
          .drop_duplicates(["id", "q"], keep="last")[["id", "q", "value"]])

    # ---- point-in-time charter -----------------------------------------------------
    a = pd.DataFrame({
        "id": pd.to_numeric(attrs["rssd_id"], errors="coerce"),
        "as_of": pd.to_numeric(attrs["as_of"], errors="coerce"),
        "commercial": attrs["charter"].astype(str).str.strip().str.lower().eq("commercial"),
    }).dropna(subset=["id", "as_of"])
    a["id"] = a["id"].astype("int64")
    a["qa"] = _quarter_index(a["as_of"])
    a = a.dropna(subset=["qa"])
    a["qa"] = a["qa"].astype("int64")
    a = (a.sort_values(["id", "qa", "as_of"], kind="mergesort")
          .drop_duplicates(["id", "qa"], keep="last")[["id", "qa", "commercial"]])

    # charter in force at quarter q: last snapshot dated within or before q
    c = c.sort_values(["q", "id"], kind="mergesort")
    a_sorted = a.sort_values(["qa", "id"], kind="mergesort")
    m = pd.merge_asof(c, a_sorted, left_on="q", right_on="qa", by="id", direction="backward")
    # before the first snapshot: fall back to the earliest snapshot
    first = a.drop_duplicates("id", keep="first").set_index("id")["commercial"]
    miss = m["commercial"].isna()
    m.loc[miss, "commercial"] = m.loc[miss, "id"].map(first)
    m["commercial"] = m["commercial"].fillna(False).astype(bool)  # no attrs -> excluded
    m = m[["id", "q", "value", "commercial"]]

    # ---- YoY by calendar quarter ----------------------------------------------------
    lag = m.rename(columns={"value": "value_lag", "commercial": "commercial_lag"})
    lag["q"] = lag["q"] + 4
    p = m.merge(lag, on=["id", "q"], how="inner")
    p = p[p["commercial"] & p["commercial_lag"]]
    p["y"] = p["value"] - p["value_lag"]
    p = p[np.isfinite(p["y"])]

    # ---- drop thin banks -------------------------------------------------------------
    n = p.groupby("id")["y"].transform("size")
    p = p[n >= MIN_OBS]
    if p.empty:
        return empty

    p = p.sort_values(["id", "q"], kind="mergesort")
    out = pd.DataFrame({
        "id": p["id"].to_numpy(dtype="int64"),
        "date": _quarter_end(p["q"].to_numpy()),
        "y": p["y"].to_numpy(dtype="float64"),
    })
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    import time
    calls_path = sys.argv[1] if len(sys.argv) > 1 else "sample_calls.csv"
    attrs_path = sys.argv[2] if len(sys.argv) > 2 else "sample_attrs.csv"
    t0 = time.time()
    res = build_panel(pd.read_csv(calls_path), pd.read_csv(attrs_path))
    print(res.head())
    print(res.dtypes)
    print(f"{len(res)} rows, {res['id'].nunique()} banks, {time.time() - t0:.2f}s")
