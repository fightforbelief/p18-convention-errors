"""Quarterly panel of year-over-year changes in a Call Report ratio.

Rules (see spec.md):
1. A bank-quarter is usable if it has a non-NaN value and the bank's charter in
   effect at that quarter (latest attrs snapshot on or before the quarter end)
   is "commercial".
2. y at quarter t = value(t) - value(t - 4 quarters), defined only when both
   bank-quarters are usable. No imputation.
3. Keep only rows where y is defined, then drop banks with fewer than 16 such rows.
"""

import pandas as pd

LAG_QUARTERS = 4
MIN_QUARTERS = 16


def _quarter_end(yyyymmdd: pd.Series) -> pd.Series:
    """Map YYYYMMDD integers to the calendar quarter-end Timestamp (datetime64[ns])."""
    d = pd.to_datetime(yyyymmdd.astype("int64").astype(str), format="%Y%m%d")
    return d.dt.to_period("Q").dt.end_time.dt.normalize().astype("datetime64[ns]")


def _quarter_index(dates: pd.Series) -> pd.Series:
    """Integer calendar-quarter index, so that t - 4 quarters is q - 4."""
    return dates.dt.year * 4 + (dates.dt.month - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # Reported values only, one per bank-quarter, dated at quarter end.
    # If a bank has several non-NaN reports for one quarter, the last one in input order is kept.
    c = calls[["rssd_id", "report_date", "value"]]
    c = c[c["value"].notna()]
    c = c.assign(date=_quarter_end(c["report_date"]))
    c = c.drop_duplicates(["rssd_id", "date"], keep="last")
    c = c[["rssd_id", "date", "value"]].sort_values("date", kind="stable")

    # Charter in effect at each quarter end: latest snapshot with as_of <= quarter end.
    a = attrs[["rssd_id", "as_of", "charter"]]
    a = a.assign(as_of=_quarter_end_exact(a["as_of"]))
    a = a.drop_duplicates(["rssd_id", "as_of"], keep="last").sort_values("as_of", kind="stable")
    c = pd.merge_asof(
        c,
        a,
        left_on="date",
        right_on="as_of",
        by="rssd_id",
        direction="backward",
    )

    # Usable bank-quarters: commercial charter in effect. Banks with no snapshot
    # yet have charter NaN and are excluded.
    u = c[c["charter"] == "commercial"][["rssd_id", "date", "value"]].copy()
    u["q"] = _quarter_index(u["date"])

    # y_t = value_t - value_{t-4}; both bank-quarters must be usable (inner join).
    prev = u.assign(q=u["q"] + LAG_QUARTERS)[["rssd_id", "q", "value"]]
    prev = prev.rename(columns={"value": "prev_value"})
    p = u.merge(prev, on=["rssd_id", "q"], how="inner")
    p = p.assign(y=p["value"] - p["prev_value"])

    out = p[["rssd_id", "date", "y"]].rename(columns={"rssd_id": "id"})

    # Drop banks with fewer than MIN_QUARTERS defined y values.
    n_per_bank = out.groupby("id")["y"].transform("size")
    out = out[n_per_bank >= MIN_QUARTERS]

    out = out.sort_values(["id", "date"], kind="stable").reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["date"] = out["date"].astype("datetime64[ns]")
    out["y"] = out["y"].astype("float64")
    return out[["id", "date", "y"]]


def _quarter_end_exact(yyyymmdd: pd.Series) -> pd.Series:
    """Snapshot dates are used as-is (not moved to quarter end) for the as-of lookup."""
    d = pd.to_datetime(yyyymmdd.astype("int64").astype(str), format="%Y%m%d")
    return d.astype("datetime64[ns]")


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    calls_df = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    attrs_df = pd.read_csv(os.path.join(here, "sample_attrs.csv"))
    panel = build_panel(calls_df, attrs_df)
    print(panel.head())
    print(panel.dtypes)
    print("rows:", len(panel), "banks:", panel["id"].nunique())
