"""Quarterly bank panel builder (see spec.md)."""
import numpy as np
import pandas as pd


def _qidx(yyyymmdd):
    """YYYYMMDD int -> integer quarter index (year*4 + quarter-1)."""
    d = np.asarray(yyyymmdd, dtype=np.int64)
    year = d // 10000
    month = (d // 100) % 100
    return year * 4 + (month - 1) // 3


def _qend(qidx):
    """Quarter index -> Timestamp at calendar quarter end."""
    q = np.asarray(qidx, dtype=np.int64)
    year = q // 4
    month = (q % 4) * 3 + 3
    first = pd.to_datetime(dict(year=year, month=month, day=1))
    return (first + pd.offsets.MonthEnd(0)).dt.normalize()


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]

    # --- calls: map to quarter index, collapse any duplicate filings ---
    c = pd.DataFrame({
        "id": calls["rssd_id"].to_numpy(dtype=np.int64),
        "q": _qidx(calls["report_date"]),
        "y": pd.to_numeric(calls["value"], errors="coerce").to_numpy(dtype=float),
    })
    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype=np.int64),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype=float)})[out_cols]
    # Active span uses all report dates (including NaN-valued reports).
    span = c.groupby("id")["q"].agg(["min", "max"])
    # Duplicates within a bank-quarter: keep the last non-missing value.
    vals = (c.dropna(subset=["y"])
             .drop_duplicates(["id", "q"], keep="last"))

    # --- regular quarterly grid over each bank's active span ---
    lengths = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), lengths)
    starts = np.repeat(span["min"].to_numpy(), lengths)
    offs = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    grid = pd.DataFrame({"id": ids, "q": starts + offs})
    grid = grid.merge(vals, on=["id", "q"], how="left")

    # --- charter as of each quarter: latest snapshot on or before quarter end ---
    a = pd.DataFrame({
        "id": attrs["rssd_id"].to_numpy(dtype=np.int64),
        "q": _qidx(attrs["as_of"]),
        "as_of": attrs["as_of"].to_numpy(dtype=np.int64),
        "commercial": attrs["charter"].astype(str).str.strip().str.lower().eq("commercial").to_numpy(),
    })
    # A snapshot dated anywhere within a quarter is on/before that quarter's end.
    a = a.sort_values(["q", "id", "as_of"], kind="mergesort")
    a = a.drop_duplicates(["id", "q"], keep="last")[["id", "q", "commercial"]]
    grid = grid.sort_values(["q", "id"], kind="mergesort")
    grid = pd.merge_asof(grid, a, on="q", by="id", direction="backward")
    # No snapshot yet -> charter unknown -> not commercial.
    grid = grid[grid["commercial"].eq(True)].drop(columns="commercial")

    # --- drop banks with >20% missing among remaining grid quarters ---
    miss = grid["y"].isna().groupby(grid["id"]).mean()
    keep = miss.index[miss <= 0.20]
    grid = grid[grid["id"].isin(keep) & grid["y"].notna()]

    grid = grid.sort_values(["id", "q"], kind="mergesort").reset_index(drop=True)
    out = pd.DataFrame({
        "id": grid["id"].astype(np.int64),
        "date": _qend(grid["q"]) if len(grid) else pd.Series(dtype="datetime64[ns]"),
        "y": grid["y"].astype(float),
    })
    return out[out_cols].reset_index(drop=True)
