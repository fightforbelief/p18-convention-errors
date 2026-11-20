"""Build a quarterly commercial-bank panel from Call Report data (see spec.md)."""
import numpy as np
import pandas as pd

MAX_MISSING_SHARE = 0.20


def _to_qindex(yyyymmdd: pd.Series) -> np.ndarray:
    """YYYYMMDD ints -> integer quarter index (year*4 + quarter-1)."""
    v = yyyymmdd.to_numpy(dtype=np.int64)
    year = v // 10000
    month = (v // 100) % 100
    return year * 4 + (month - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]

    # --- Calls: one observation per bank-quarter -------------------------
    c = pd.DataFrame({
        "id": calls["rssd_id"].to_numpy(dtype=np.int64),
        "q": _to_qindex(calls["report_date"]),
        "y": pd.to_numeric(calls["value"], errors="coerce").to_numpy(dtype=float),
    })
    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype=np.int64),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype=float)})[out_cols]
    # Duplicate filings for a bank-quarter: keep the last non-NaN value.
    c = c.groupby(["id", "q"], sort=True)["y"].last().reset_index()

    # --- Rule 1: regular quarterly grid over each bank's active span ------
    span = c.groupby("id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    grid_id = np.repeat(span.index.to_numpy(), n)
    offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    grid_q = np.repeat(span["min"].to_numpy(), n) + offsets
    grid = pd.DataFrame({"id": grid_id, "q": grid_q})
    grid = grid.merge(c, on=["id", "q"], how="left")  # missing quarters -> NaN

    # --- Rule 2: charter in force at each quarter (latest snapshot <= quarter end)
    a = pd.DataFrame({
        "id": attrs["rssd_id"].to_numpy(dtype=np.int64),
        "q": _to_qindex(attrs["as_of"]),
        "asof": attrs["as_of"].to_numpy(dtype=np.int64),
        "commercial": attrs["charter"].astype(str).str.strip().str.lower()
                      .eq("commercial").to_numpy(),
    })
    # Several snapshots in one quarter: the latest-dated one governs that quarter end.
    a = (a.sort_values(["id", "q", "asof"], kind="mergesort")
          .drop_duplicates(["id", "q"], keep="last")[["id", "q", "commercial"]])
    grid = grid.sort_values(["q", "id"], kind="mergesort")
    a = a.sort_values(["q", "id"], kind="mergesort")
    grid = pd.merge_asof(grid, a, on="q", by="id", direction="backward")
    # No snapshot on/before the quarter -> charter unknown -> not commercial.
    grid = grid[grid["commercial"].fillna(False).astype(bool)]

    # --- Rule 3: drop banks with >20% missing among remaining quarters ----
    miss_share = grid["y"].isna().groupby(grid["id"]).mean()
    keep_ids = miss_share.index[miss_share <= MAX_MISSING_SHARE]
    grid = grid[grid["id"].isin(keep_ids)]

    # --- Rule 4: drop missing bank-quarters, no imputation ---------------
    grid = grid[grid["y"].notna()].sort_values(["id", "q"], kind="mergesort")

    q = grid["q"].to_numpy()
    date = (pd.PeriodIndex.from_fields(year=q // 4, quarter=q % 4 + 1, freq="Q")
            .to_timestamp(how="end").normalize())
    return pd.DataFrame({
        "id": grid["id"].to_numpy(dtype=np.int64),
        "date": date.to_numpy(),
        "y": grid["y"].to_numpy(dtype=float),
    })[out_cols].reset_index(drop=True)
