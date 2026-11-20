"""Quarterly panel of year-over-year changes (see spec.md)."""
import numpy as np
import pandas as pd

MIN_QUARTERS = 16


def _quarter_index(yyyymmdd: pd.Series) -> np.ndarray:
    """Map YYYYMMDD ints to a running calendar-quarter index (year*4 + q-1)."""
    d = yyyymmdd.astype(np.int64).to_numpy()
    year = d // 10000
    month = (d // 100) % 100
    return year * 4 + (month - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # --- calls: one value per bank-quarter --------------------------------
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype(np.int64).to_numpy(),
        "q": _quarter_index(calls["report_date"]),
        "date_int": calls["report_date"].astype(np.int64).to_numpy(),
        "value": pd.to_numeric(calls["value"], errors="coerce").to_numpy(dtype=float),
    })
    # If a bank-quarter appears more than once, keep the last filed row.
    c = c.drop_duplicates(["id", "q"], keep="last")
    c = c[np.isfinite(c["value"].to_numpy())]

    # --- attrs: charter in force at each quarter (as-of join) -------------
    a = pd.DataFrame({
        "id": attrs["rssd_id"].astype(np.int64).to_numpy(),
        "as_of": attrs["as_of"].astype(np.int64).to_numpy(),
        "commercial": attrs["charter"].astype(str).str.strip().str.lower().eq("commercial").to_numpy(),
    })
    a = a.drop_duplicates(["id", "as_of"], keep="last").sort_values("as_of", kind="mergesort")

    # Charter "at the quarter" = latest snapshot on or before the quarter-end date.
    qend = pd.PeriodIndex.from_fields(year=c["q"].to_numpy() // 4,
                                      quarter=c["q"].to_numpy() % 4 + 1, freq="Q")
    c["qend"] = (qend.end_time.year * 10000 + qend.end_time.month * 100
                 + qend.end_time.day).astype(np.int64)
    c = c.sort_values("qend", kind="mergesort")
    c = pd.merge_asof(c, a, left_on="qend", right_on="as_of", by="id",
                      direction="backward")
    # No snapshot on or before the quarter -> charter unknown -> not usable.
    c = c[c["commercial"].fillna(False).astype(bool)]

    usable = c[["id", "q", "value"]]

    # --- YoY change: value(t) - value(t-4), both usable --------------------
    lag = usable.rename(columns={"value": "value_lag"})
    lag = lag.assign(q=lag["q"] + 4)
    p = usable.merge(lag, on=["id", "q"], how="inner")
    p["y"] = p["value"] - p["value_lag"]
    p = p[np.isfinite(p["y"].to_numpy())]

    # --- drop banks with fewer than 16 defined quarters --------------------
    counts = p.groupby("id")["q"].transform("size")
    p = p[counts >= MIN_QUARTERS]

    p = p.sort_values(["id", "q"], kind="mergesort")
    q = p["q"].to_numpy()
    dates = pd.PeriodIndex.from_fields(year=q // 4, quarter=q % 4 + 1, freq="Q") \
        .to_timestamp(how="end").normalize()

    out = pd.DataFrame({
        "id": p["id"].astype(np.int64).to_numpy(),
        "date": pd.DatetimeIndex(dates).astype("datetime64[ns]"),
        "y": p["y"].astype(float).to_numpy(),
    })
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    calls_path = sys.argv[1] if len(sys.argv) > 1 else "sample_calls.csv"
    attrs_path = sys.argv[2] if len(sys.argv) > 2 else "sample_attrs.csv"
    panel = build_panel(pd.read_csv(calls_path), pd.read_csv(attrs_path))
    print(panel.head())
    print(panel.shape, panel["id"].nunique())
