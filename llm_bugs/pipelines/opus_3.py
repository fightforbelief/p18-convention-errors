"""Build a quarterly bank panel from Call Report data (see spec.md)."""
import numpy as np
import pandas as pd

MAX_MISSING_SHARE = 0.20


def _quarter_ordinal(yyyymmdd: pd.Series) -> np.ndarray:
    """Map YYYYMMDD ints to an integer quarter index (year*4 + quarter-1)."""
    v = yyyymmdd.astype("int64").to_numpy()
    year = v // 10000
    month = (v // 100) % 100
    return year * 4 + (month - 1) // 3


def _ordinal_to_quarter_end(q: np.ndarray) -> pd.DatetimeIndex:
    year = q // 4
    qtr = q % 4 + 1
    return pd.PeriodIndex.from_fields(year=year, quarter=qtr, freq="Q").to_timestamp(how="end").normalize()


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]

    # ---- calls: one observation per bank-quarter ----
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype("int64").to_numpy(),
        "q": _quarter_ordinal(calls["report_date"]),
        "y": pd.to_numeric(calls["value"], errors="coerce").astype("float64").to_numpy(),
    })
    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})[out_cols]
    # Guard against duplicate filings for the same quarter: keep first non-NaN value.
    c = c.groupby(["id", "q"], sort=True, as_index=False)["y"].first()

    # ---- rule 1: regular quarterly grid over each bank's active span ----
    span = c.groupby("id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    grid_id = np.repeat(span.index.to_numpy(), n)
    starts = np.repeat(span["min"].to_numpy(), n)
    offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    grid = pd.DataFrame({"id": grid_id, "q": starts + offsets})
    grid = grid.merge(c, on=["id", "q"], how="left")  # missing report -> NaN

    # ---- rule 2: charter at each quarter = latest snapshot on or before quarter end ----
    a = pd.DataFrame({
        "id": attrs["rssd_id"].astype("int64").to_numpy(),
        "as_of": pd.to_datetime(attrs["as_of"].astype("int64").astype(str), format="%Y%m%d"),
        "charter": attrs["charter"].astype(str).str.strip().str.lower().to_numpy(),
    })
    # If several snapshots share a date, the last listed one wins.
    a = a.drop_duplicates(["id", "as_of"], keep="last").sort_values("as_of", kind="mergesort")

    grid["date"] = _ordinal_to_quarter_end(grid["q"].to_numpy())
    grid = grid.sort_values("date", kind="mergesort")
    grid = pd.merge_asof(grid, a, left_on="date", right_on="as_of", by="id", direction="backward")
    grid = grid[grid["charter"] == "commercial"]  # no snapshot yet -> not commercial

    # ---- rule 3: drop banks with >20% missing among remaining grid quarters ----
    miss_share = grid["y"].isna().groupby(grid["id"]).mean()
    keep_ids = miss_share.index[miss_share <= MAX_MISSING_SHARE]
    grid = grid[grid["id"].isin(keep_ids)]

    # ---- rule 4: drop missing bank-quarters, no imputation ----
    grid = grid[grid["y"].notna()]

    out = grid[out_cols].sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    return out


if __name__ == "__main__":
    import sys
    import time
    calls_path = sys.argv[1] if len(sys.argv) > 1 else "sample_calls.csv"
    attrs_path = sys.argv[2] if len(sys.argv) > 2 else "sample_attrs.csv"
    t0 = time.time()
    panel = build_panel(pd.read_csv(calls_path), pd.read_csv(attrs_path))
    print(panel.head())
    print(panel.shape, panel["id"].nunique(), "banks", f"{time.time() - t0:.2f}s")
