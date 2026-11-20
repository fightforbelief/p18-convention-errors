"""Quarterly panel of year-over-year changes in a Call Report ratio for commercial banks.

Design decisions
----------------
* Every report_date is mapped to its calendar quarter end, so off-cycle or
  mis-keyed filing dates still land in the right quarter.
* If a bank has several filings in one quarter (amendments / duplicates), the
  one with the latest report_date is kept (ties: the last row in input order),
  falling back to an earlier non-missing filing only if the latest is NaN.
* y(t) = value(t) - value(t - 4 quarters), the change against the *same
  calendar quarter one year earlier*. The lag is looked up by calendar quarter,
  not by row position, so gaps in filing never produce a comparison against the
  wrong quarter; if the year-ago quarter is missing or NaN, y is undefined and
  the row is dropped. Non-finite values are treated as missing.
* Charter is taken point-in-time from the most recent attrs snapshot at or
  before the quarter (quarters preceding a bank's first snapshot use that first
  snapshot). A bank-quarter is kept only if the bank is commercial both at t
  and at t-4, so no change straddles a charter conversion.
* Thin banks are dropped: a bank needs at least MIN_OBS valid y values so that
  after holding out its last HOLDOUT quarters at least MIN_TRAIN remain.
"""

import numpy as np
import pandas as pd

HOLDOUT = 4
MIN_TRAIN = 8
MIN_OBS = HOLDOUT + MIN_TRAIN


def _quarter_index(yyyymmdd: pd.Series) -> pd.Series:
    """YYYYMMDD int -> integer quarter count (year*4 + quarter-1); NaN if invalid."""
    d = pd.to_datetime(yyyymmdd.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return d.dt.year * 4 + (d.dt.month - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]

    # ---- calls: clean and collapse to one value per bank-quarter ----
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c.dropna(subset=["rssd_id", "report_date"])
    c["rssd_id"] = c["rssd_id"].astype(np.int64)
    c["report_date"] = pd.to_numeric(c["report_date"], errors="coerce")
    c["value"] = pd.to_numeric(c["value"], errors="coerce").astype(float)
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    c["q"] = _quarter_index(c["report_date"])
    c = c.dropna(subset=["q"])
    c["q"] = c["q"].astype(np.int64)
    c["_ord"] = np.arange(len(c))
    # latest filing per quarter; NaN filings sorted to the front so a later
    # non-missing amendment wins, but a lone NaN still represents the quarter
    c["_has"] = c["value"].notna()
    c = c.sort_values(["rssd_id", "q", "_has", "report_date", "_ord"])
    c = c.drop_duplicates(["rssd_id", "q"], keep="last")[["rssd_id", "q", "value"]]

    # ---- year-over-year change by calendar quarter ----
    lag = c[["rssd_id", "q", "value"]].rename(columns={"value": "value_lag"})
    lag["q"] = lag["q"] + 4
    p = c.merge(lag, on=["rssd_id", "q"], how="inner")
    p["y"] = p["value"] - p["value_lag"]
    p = p[np.isfinite(p["y"])][["rssd_id", "q", "y"]]
    if p.empty:
        return pd.DataFrame({"id": pd.Series(dtype=np.int64),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype=float)})[out_cols]

    # ---- point-in-time charter ----
    a = attrs[["rssd_id", "as_of", "charter"]].dropna(subset=["rssd_id", "as_of"]).copy()
    a["rssd_id"] = a["rssd_id"].astype(np.int64)
    a["comm"] = a["charter"].astype(str).str.strip().str.lower().eq("commercial")
    a["q"] = _quarter_index(pd.to_numeric(a["as_of"], errors="coerce"))
    a = a.dropna(subset=["q"])
    a["q"] = a["q"].astype(np.int64)
    a["_ord"] = np.arange(len(a))
    a = (a.sort_values(["rssd_id", "q", "as_of", "_ord"])
          .drop_duplicates(["rssd_id", "q"], keep="last")[["rssd_id", "q", "comm"]])
    first = a.drop_duplicates("rssd_id", keep="first").set_index("rssd_id")["comm"]

    def charter_at(keys: pd.DataFrame) -> np.ndarray:
        k = keys.reset_index(drop=True).copy()
        k["_row"] = np.arange(len(k))
        m = pd.merge_asof(k.sort_values("q"), a.sort_values("q"),
                          on="q", by="rssd_id", direction="backward")
        m = m.sort_values("_row")
        comm = m["comm"].astype("boolean").fillna(m["rssd_id"].map(first).astype("boolean"))
        return comm.fillna(False).to_numpy(dtype=bool)

    keys = p[["rssd_id", "q"]]
    ok_t = charter_at(keys)
    ok_lag = charter_at(keys.assign(q=keys["q"] - 4))
    p = p[ok_t & ok_lag]

    # ---- drop thin banks ----
    n = p.groupby("rssd_id")["y"].transform("size")
    p = p[n >= MIN_OBS]

    # ---- format ----
    q = p["q"].to_numpy()
    # first day of the following quarter minus one day = calendar quarter end
    nxt = q + 1
    date = pd.to_datetime(pd.DataFrame({"year": nxt // 4, "month": (nxt % 4) * 3 + 1, "day": 1})) \
        - pd.Timedelta(days=1)
    res = pd.DataFrame({"id": p["rssd_id"].astype(np.int64).to_numpy(),
                        "date": pd.DatetimeIndex(date).astype("datetime64[ns]"),
                        "y": p["y"].astype(float).to_numpy()})
    return res.sort_values(["id", "date"]).reset_index(drop=True)[out_cols]


if __name__ == "__main__":
    import os
    import time
    here = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    out = build_panel(pd.read_csv(os.path.join(here, "sample_calls.csv")),
                      pd.read_csv(os.path.join(here, "sample_attrs.csv")))
    print(out.head(10))
    print(out.dtypes)
    print(f"{len(out)} rows, {out['id'].nunique()} banks, {time.time() - t0:.2f}s")
