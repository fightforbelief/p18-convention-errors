"""Quarterly panel of year-over-year changes in a Call Report ratio for commercial banks.

Design choices
--------------
* Each report is mapped to its calendar quarter; if a bank files more than once in a
  quarter the latest filing with a finite value wins.
* Charter is taken point-in-time: each quarter gets the most recent attribute snapshot
  at or before the quarter end (the first snapshot is used for quarters preceding it).
  A bank-quarter is kept only when the bank is commercial both in that quarter and in
  the comparison quarter a year earlier, so changes never straddle a charter switch.
* y = value(q) - value(q - 4 quarters), matched on the calendar quarter itself (not on
  row position), so missing quarters or NaN values never produce a mis-aligned
  comparison: if either end is missing, that quarter simply has no y.
  A difference (not a percentage change) is used because the ratio can be ~0 or negative.
* Banks with fewer than MIN_OBS valid y observations are dropped: the benchmark holds
  out the last 4 quarters, so each bank needs a reasonable training history before them.
"""
import numpy as np
import pandas as pd

HOLDOUT = 4
MIN_TRAIN = 8
MIN_OBS = HOLDOUT + MIN_TRAIN  # 12 bank-quarters of y


def _to_quarter_ordinal(yyyymmdd: pd.Series) -> pd.Series:
    """YYYYMMDD ints -> quarter ordinal (year*4 + q-1); invalid dates -> NaN."""
    d = pd.to_datetime(pd.to_numeric(yyyymmdd, errors="coerce").astype("Int64").astype(str),
                       format="%Y%m%d", errors="coerce")
    return d.dt.year * 4 + (d.dt.month - 1) // 3, d


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- calls: clean, quarterise, dedupe -------------------------------------------
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = pd.to_numeric(c["rssd_id"], errors="coerce")
    c["value"] = pd.to_numeric(c["value"], errors="coerce").astype("float64")
    c["qo"], c["rd"] = _to_quarter_ordinal(c["report_date"])
    c = c[c["rssd_id"].notna() & c["qo"].notna() & np.isfinite(c["value"])]
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["qo"] = c["qo"].astype("int64")
    c = (c.sort_values(["rssd_id", "qo", "rd"], kind="mergesort")
          .drop_duplicates(["rssd_id", "qo"], keep="last"))

    # ---- attrs: point-in-time charter at each quarter end -----------------------------
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["rssd_id"] = pd.to_numeric(a["rssd_id"], errors="coerce")
    a["charter"] = a["charter"].astype("string").str.strip().str.lower()
    a["asd"] = pd.to_datetime(pd.to_numeric(a["as_of"], errors="coerce").astype("Int64").astype(str),
                              format="%Y%m%d", errors="coerce")
    a = a[a["rssd_id"].notna() & a["asd"].notna() & a["charter"].notna()]
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a = (a.sort_values(["rssd_id", "asd"], kind="mergesort")
          .drop_duplicates(["rssd_id", "asd"], keep="last"))
    a["is_comm"] = a["charter"].eq("commercial")

    # quarter-end timestamp for every bank-quarter
    yr, qi = c["qo"] // 4, c["qo"] % 4
    c["date"] = (pd.PeriodIndex.from_fields(year=yr.to_numpy(), quarter=(qi + 1).to_numpy(), freq="Q")
                 .to_timestamp(how="end").normalize())
    c = c.sort_values("date", kind="mergesort")
    a = a.sort_values("asd", kind="mergesort")
    c = pd.merge_asof(c, a[["rssd_id", "asd", "is_comm"]], left_on="date", right_on="asd",
                      by="rssd_id", direction="backward")
    # quarters before the first snapshot: fall back to the earliest snapshot
    first = a.groupby("rssd_id")["is_comm"].first()
    miss = c["is_comm"].isna()
    c.loc[miss, "is_comm"] = c.loc[miss, "rssd_id"].map(first)
    c = c[c["is_comm"].fillna(False).astype(bool)]  # banks with no attrs are excluded

    # ---- year-over-year change, matched on calendar quarter --------------------------
    cur = c[["rssd_id", "qo", "date", "value"]]
    prev = c[["rssd_id", "qo", "value"]].rename(columns={"value": "value_lag"})
    prev = prev.assign(qo=prev["qo"] + 4)
    p = cur.merge(prev, on=["rssd_id", "qo"], how="inner")
    p["y"] = p["value"] - p["value_lag"]
    p = p[np.isfinite(p["y"])]

    # ---- drop thin banks --------------------------------------------------------------
    n = p.groupby("rssd_id")["y"].transform("size")
    p = p[n >= MIN_OBS]

    out = (p.rename(columns={"rssd_id": "id"})[["id", "date", "y"]]
             .sort_values(["id", "date"], kind="mergesort")
             .reset_index(drop=True))
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out


if __name__ == "__main__":
    import sys, time
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    t = time.time()
    res = build_panel(pd.read_csv(f"{d}/sample_calls.csv"), pd.read_csv(f"{d}/sample_attrs.csv"))
    print(res.head(10)); print(res.dtypes); print(len(res), res["id"].nunique(), f"{time.time()-t:.2f}s")
