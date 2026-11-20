"""Build a quarterly bank panel from Call Report ratios and charter snapshots (see spec.md)."""
import numpy as np
import pandas as pd

MAX_MISSING_FRAC = 0.20


def _to_qidx(yyyymmdd: pd.Series) -> np.ndarray:
    """YYYYMMDD int -> integer quarter index (year*4 + quarter-1)."""
    v = yyyymmdd.astype(np.int64).to_numpy()
    year = v // 10000
    month = (v // 100) % 100
    return year * 4 + (month - 1) // 3


def _qidx_to_qend(q: np.ndarray) -> pd.DatetimeIndex:
    """Quarter index -> calendar quarter-end Timestamp."""
    return pd.PeriodIndex.from_ordinals(
        (q - 1970 * 4).astype(np.int64), freq="Q-DEC"
    ).to_timestamp(how="end").normalize()


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype=np.int64),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype=float)})
    if len(calls) == 0:
        return empty

    # --- calls: one value per bank-quarter (prefer a non-NaN value if duplicated) ---
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype(np.int64).to_numpy(),
        "q": _to_qidx(calls["report_date"]),
        "y": pd.to_numeric(calls["value"], errors="coerce").astype(float).to_numpy(),
    })
    c["_has"] = c["y"].notna()
    c = (c.sort_values(["id", "q", "_has"], kind="mergesort")
          .drop_duplicates(["id", "q"], keep="last")
          .drop(columns="_has"))

    # --- rule 1: regular quarterly grid over each bank's active span ---
    span = c.groupby("id", sort=True)["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    qs = np.repeat(span["min"].to_numpy(), n) + offs
    grid = pd.DataFrame({"id": ids, "q": qs})
    grid = grid.merge(c, on=["id", "q"], how="left")  # missing quarters -> NaN

    # --- rule 2: charter as of each quarter (latest snapshot on or before quarter end) ---
    a = pd.DataFrame({
        "id": attrs["rssd_id"].astype(np.int64).to_numpy(),
        "as_of": attrs["as_of"].astype(np.int64).to_numpy(),
        "charter": attrs["charter"].astype(str).str.strip().str.lower().to_numpy(),
    })
    a = (a.sort_values(["id", "as_of"], kind="mergesort")
          .drop_duplicates(["id", "as_of"], keep="last"))
    # quarter-end as YYYYMMDD int, so snapshots dated mid-quarter apply to that quarter
    qend = _qidx_to_qend(grid["q"].to_numpy())
    grid["qend"] = (qend.year * 10000 + qend.month * 100 + qend.day).astype(np.int64)
    grid = grid.sort_values("qend", kind="mergesort")
    a = a.sort_values("as_of", kind="mergesort")
    grid = pd.merge_asof(grid, a, left_on="qend", right_on="as_of", by="id",
                         direction="backward")
    grid = grid[grid["charter"] == "commercial"]  # no snapshot yet -> dropped

    # --- rule 3: drop banks with >20% missing among remaining grid quarters ---
    miss_frac = grid["y"].isna().groupby(grid["id"]).transform("mean")
    grid = grid[miss_frac <= MAX_MISSING_FRAC]

    # --- rule 4: drop missing bank-quarters, no imputation ---
    grid = grid[grid["y"].notna()].sort_values(["id", "q"], kind="mergesort")
    if len(grid) == 0:
        return empty

    out = pd.DataFrame({
        "id": grid["id"].astype(np.int64).to_numpy(),
        "date": _qidx_to_qend(grid["q"].to_numpy()).astype("datetime64[ns]"),
        "y": grid["y"].astype(float).to_numpy(),
    })
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    calls = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sample_calls.csv")
    attrs = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "sample_attrs.csv")
    panel = build_panel(calls, attrs)
    print(panel.head())
    print(panel.shape, panel["id"].nunique(), "banks")
