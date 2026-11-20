"""Build a quarterly bank panel from Call Report ratios (see spec.md)."""
import numpy as np
import pandas as pd

MAX_MISSING_SHARE = 0.20


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]

    # --- reports: one value per bank-quarter -------------------------------
    rd = calls["report_date"].astype("int64").to_numpy()
    year, month = rd // 10000, (rd // 100) % 100
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype("int64").to_numpy(),
        # pandas quarterly Period ordinal: (year - 1970) * 4 + quarter - 1
        "qn": (year - 1970) * 4 + (month - 1) // 3,
        "y": pd.to_numeric(calls["value"], errors="coerce").astype(float).to_numpy(),
    })
    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype=float)})
    # Duplicate filings for a quarter: keep the last non-missing value.
    c = c.sort_values(["id", "qn"], kind="mergesort")
    vals = (c.dropna(subset=["y"]).groupby(["id", "qn"], sort=False)["y"].last())

    # --- regular quarterly grid over each bank's active span (rule 1) ------
    span = c.groupby("id")["qn"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    qn = np.repeat(span["min"].to_numpy(), n) + offs
    grid = pd.DataFrame({"id": ids, "qn": qn})
    grid["date"] = pd.PeriodIndex.from_ordinals(qn, freq="Q").end_time.normalize()
    grid = grid.join(vals, on=["id", "qn"])

    # --- charter at each quarter-end (rule 2) ------------------------------
    a = pd.DataFrame({
        "id": attrs["rssd_id"].astype("int64").to_numpy(),
        "as_of": pd.to_datetime(attrs["as_of"].astype("int64").astype(str),
                                format="%Y%m%d").to_numpy(),
        "charter": attrs["charter"].astype(str).str.strip().str.lower().to_numpy(),
    })
    # Same-day duplicate snapshots: keep the last one listed.
    a = a.drop_duplicates(["id", "as_of"], keep="last").sort_values("as_of", kind="mergesort")
    grid = grid.sort_values("date", kind="mergesort")
    grid = pd.merge_asof(grid, a, left_on="date", right_on="as_of", by="id",
                         direction="backward", allow_exact_matches=True)
    grid = grid[grid["charter"] == "commercial"]

    # --- drop banks with >20% missing among remaining quarters (rule 3) ----
    miss = grid["y"].isna()
    share = miss.groupby(grid["id"]).mean()
    keep = share.index[share <= MAX_MISSING_SHARE]
    grid = grid[grid["id"].isin(keep) & ~miss]

    # --- drop missing quarters, no imputation (rule 4) ---------------------
    out = (grid[out_cols].sort_values(["id", "date"], kind="mergesort")
           .reset_index(drop=True))
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype(float)
    return out


if __name__ == "__main__":
    import sys, time
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    t = time.time()
    panel = build_panel(pd.read_csv(f"{folder}/sample_calls.csv"),
                        pd.read_csv(f"{folder}/sample_attrs.csv"))
    print(panel.head(), panel.shape, panel["id"].nunique(), f"{time.time()-t:.2f}s")
