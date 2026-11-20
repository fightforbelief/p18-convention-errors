"""Quarterly panel of year-over-year changes in a Call Report ratio (see spec.md)."""
import numpy as np
import pandas as pd

MIN_QUARTERS = 16


def _quarter_index(yyyymmdd: pd.Series) -> np.ndarray:
    """Map an int YYYYMMDD date to a consecutive integer calendar-quarter index."""
    d = yyyymmdd.astype(np.int64).to_numpy()
    year = d // 10000
    month = (d // 100) % 100
    return year * 4 + (month - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # --- calls: one non-NaN value per bank-calendar-quarter ---
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype(np.int64).to_numpy(),
        "q": _quarter_index(calls["report_date"]),
        "value": pd.to_numeric(calls["value"], errors="coerce").astype(float).to_numpy(),
    })
    c = c[np.isfinite(c["value"].to_numpy())]
    # If a bank filed more than once for the same quarter, keep the last filing.
    c = c.drop_duplicates(["id", "q"], keep="last")

    # --- attrs: charter in force at each quarter (latest snapshot on or before quarter end) ---
    a = pd.DataFrame({
        "id": attrs["rssd_id"].astype(np.int64).to_numpy(),
        "as_of": attrs["as_of"].astype(np.int64).to_numpy(),
        "commercial": attrs["charter"].astype(str).str.strip().str.lower().eq("commercial").to_numpy(),
    })
    a = a.drop_duplicates(["id", "as_of"], keep="last").sort_values("as_of", kind="stable")

    # Quarter-end date as YYYYMMDD int for the as-of comparison.
    year = c["q"].to_numpy() // 4
    qn = c["q"].to_numpy() % 4
    qend_md = np.array([331, 630, 930, 1231])[qn]
    c["qend"] = year * 10000 + qend_md
    c = c.sort_values("qend", kind="stable")
    c = pd.merge_asof(c, a, left_on="qend", right_on="as_of", by="id", direction="backward")
    usable = c[c["commercial"].eq(True)][["id", "q", "qend", "value"]]

    # --- y_t = value_t - value_{t-4}, both usable ---
    lag = usable[["id", "q", "value"]].copy()
    lag["q"] = lag["q"] + 4
    m = usable.merge(lag, on=["id", "q"], how="inner", suffixes=("", "_lag"))
    m["y"] = m["value"] - m["value_lag"]
    m = m[np.isfinite(m["y"].to_numpy())]

    # --- drop banks with fewer than 16 defined quarters ---
    counts = m.groupby("id")["y"].transform("size")
    m = m[counts >= MIN_QUARTERS]

    out = pd.DataFrame({
        "id": m["id"].astype(np.int64).to_numpy(),
        "date": pd.to_datetime(m["qend"].astype(str), format="%Y%m%d").to_numpy(),
        "y": m["y"].astype(float).to_numpy(),
    })
    return out.sort_values(["id", "date"], kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    calls = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sample_calls.csv")
    attrs = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "sample_attrs.csv")
    panel = build_panel(calls, attrs)
    print(panel.head())
    print(panel.shape, panel["id"].nunique())
