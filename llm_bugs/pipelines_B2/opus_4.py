"""Quarterly panel of year-over-year changes in a Call Report ratio (commercial banks only).

Design choices
--------------
* Dates: every report_date is mapped to its calendar quarter; the output `date` is that
  quarter's end. If a bank filed more than once in a quarter, the latest filing wins.
* Charter is time-varying (banks convert between savings and commercial). Each filing gets
  the charter from the most recent attribute snapshot on or before its report_date
  (as-of join); filings before a bank's first snapshot take that first snapshot. A bank-quarter
  is kept only when the bank is commercial in BOTH the current quarter and the comparison
  quarter, so no change straddles a charter conversion. Banks with no attributes are dropped.
* YoY change: y = value(q) - value(q - 4 quarters), matched on the exact calendar quarter
  (not a positional shift), so gaps in filing never compare against the wrong quarter. If the
  same quarter a year earlier is missing or NaN, that quarter has no y and is dropped (no
  interpolation, which would leak information into a forecasting benchmark). The difference
  (not percentage change) is used because the series is a ratio that can be near zero or
  negative.
* Thin banks: a bank is kept only if it has at least MIN_OBS valid y values, i.e. at least
  8 quarters of history beyond the 4 held-out final quarters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLDOUT = 4
MIN_TRAIN = 8
MIN_OBS = HOLDOUT + MIN_TRAIN


def _quarter_index(yyyymmdd: pd.Series) -> np.ndarray:
    """YYYYMMDD int -> integer quarter index (year*4 + quarter-1). Invalid -> -1."""
    v = pd.to_numeric(yyyymmdd, errors="coerce")
    year = (v // 10000)
    month = (v // 100) % 100
    ok = v.notna() & month.between(1, 12)
    q = (year * 4 + (month - 1) // 3).where(ok, -1)
    return q.astype("int64").to_numpy()


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})

    # ---- calls: clean, one row per bank-quarter -------------------------------------
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c[c["rssd_id"].notna() & c["report_date"].notna()]
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = pd.to_numeric(c["report_date"], errors="coerce")
    c = c[c["report_date"].notna()]
    c["report_date"] = c["report_date"].astype("int64")
    c["value"] = pd.to_numeric(c["value"], errors="coerce").astype("float64")
    c.loc[~np.isfinite(c["value"].to_numpy()), "value"] = np.nan
    c["q"] = _quarter_index(c["report_date"])
    c = c[c["q"] >= 0]
    # latest filing within a quarter wins
    c = c.sort_values(["rssd_id", "q", "report_date"], kind="mergesort")
    c = c.drop_duplicates(["rssd_id", "q"], keep="last")
    if c.empty:
        return empty

    # ---- attrs: as-of charter -------------------------------------------------------
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a = a[a["rssd_id"].notna() & a["as_of"].notna()]
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["as_of"] = pd.to_numeric(a["as_of"], errors="coerce")
    a = a[a["as_of"].notna()]
    a["as_of"] = a["as_of"].astype("int64")
    a["is_comm"] = a["charter"].astype(str).str.strip().str.lower().eq("commercial")
    a = a.sort_values(["rssd_id", "as_of"], kind="mergesort").drop_duplicates(
        ["rssd_id", "as_of"], keep="last")

    c = c.sort_values("report_date", kind="mergesort")
    a_s = a.sort_values("as_of", kind="mergesort")
    m = pd.merge_asof(c, a_s[["rssd_id", "as_of", "is_comm"]], left_on="report_date",
                      right_on="as_of", by="rssd_id", direction="backward")
    first = a.drop_duplicates("rssd_id", keep="first").set_index("rssd_id")["is_comm"]
    miss = m["is_comm"].isna()
    m.loc[miss, "is_comm"] = m.loc[miss, "rssd_id"].map(first)
    m["is_comm"] = m["is_comm"].fillna(False).astype(bool)

    # ---- YoY on exact calendar quarter ----------------------------------------------
    cur = m.loc[m["is_comm"] & m["value"].notna(), ["rssd_id", "q", "value"]]
    lag = cur.rename(columns={"value": "value_lag"}).assign(q=lambda d: d["q"] + HOLDOUT)
    p = cur.merge(lag, on=["rssd_id", "q"], how="inner")
    p["y"] = p["value"] - p["value_lag"]
    p = p[np.isfinite(p["y"].to_numpy())]

    # ---- drop thin banks ------------------------------------------------------------
    n = p.groupby("rssd_id")["y"].transform("size")
    p = p[n >= MIN_OBS]
    if p.empty:
        return empty

    # ---- output ---------------------------------------------------------------------
    qi = p["q"].to_numpy()
    year = qi // 4
    qtr = qi % 4 + 1
    date = (pd.PeriodIndex.from_fields(year=year, quarter=qtr, freq="Q")
            .to_timestamp(how="end").normalize())
    out = pd.DataFrame({"id": p["rssd_id"].astype("int64").to_numpy(),
                        "date": date.astype("datetime64[ns]"),
                        "y": p["y"].astype("float64").to_numpy()})
    return out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    import time
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    t = time.time()
    res = build_panel(pd.read_csv(f"{base}/sample_calls.csv"),
                      pd.read_csv(f"{base}/sample_attrs.csv"))
    print(res.head(10))
    print(res.dtypes)
    print(len(res), "rows,", res["id"].nunique(), "banks,", f"{time.time() - t:.2f}s")
