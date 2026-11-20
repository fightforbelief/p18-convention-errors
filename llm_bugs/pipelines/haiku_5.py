"""Build a quarterly bank panel from Call Report ratios and charter snapshots.

Only pandas and numpy are used.
"""

import os

import numpy as np
import pandas as pd

MAX_MISSING_SHARE = 0.20
COMMERCIAL = "commercial"
OUT_COLUMNS = ["id", "date", "y"]


def _yyyymmdd_to_datetime(s):
    """Parse YYYYMMDD integers (or strings/floats of them) to datetime64[ns]; bad values -> NaT."""
    ints = pd.to_numeric(pd.Series(s, copy=False), errors="coerce")
    text = ints.astype("Int64").astype("string")
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce").astype("datetime64[ns]")


def _quarter_index(dt):
    """Integer quarter index (year*4 + quarter-1); monotone in time, so differences are quarter counts."""
    return (dt.dt.year * 4 + dt.dt.quarter - 1).astype("int64")


def _quarter_end(qidx):
    """Quarter-end Timestamps for integer quarter indices."""
    year = qidx // 4
    month = (qidx % 4 + 1) * 3
    day = month.map({3: 31, 6: 30, 9: 30, 12: 31})
    text = (year * 10000 + month * 100 + day).astype("string")
    return pd.to_datetime(text, format="%Y%m%d").astype("datetime64[ns]")


def _empty_panel():
    return pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "y": pd.Series(dtype="float64"),
        }
    )


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- Clean the call reports -------------------------------------------------
    c = pd.DataFrame(
        {
            "id": pd.to_numeric(calls["rssd_id"], errors="coerce"),
            "dt": _yyyymmdd_to_datetime(calls["report_date"]),
            "value": pd.to_numeric(calls["value"], errors="coerce").astype("float64"),
        }
    )
    c = c.dropna(subset=["id", "dt"])
    if c.empty:
        return _empty_panel()
    c["id"] = c["id"].astype("int64")
    c["qidx"] = _quarter_index(c["dt"])

    # ---- Rule 1: regular quarterly grid over each bank's active span -------------
    span = c.groupby("id")["qidx"].agg(["min", "max"])
    lengths = (span["max"] - span["min"] + 1).to_numpy(np.int64)
    total = int(lengths.sum())
    offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
    grid = pd.DataFrame(
        {
            "id": np.repeat(span.index.to_numpy(np.int64), lengths),
            "qidx": np.repeat(span["min"].to_numpy(np.int64), lengths)
            + (np.arange(total, dtype=np.int64) - offsets),
        }
    )
    grid["date"] = _quarter_end(grid["qidx"])

    # ---- Rule 2: keep quarters where the latest snapshot on/before the quarter is commercial
    a = pd.DataFrame(
        {
            "id": pd.to_numeric(attrs["rssd_id"], errors="coerce"),
            "as_of": _yyyymmdd_to_datetime(attrs["as_of"]),
            "charter": attrs["charter"].astype("string").str.strip().str.lower(),
        }
    )
    a = a.dropna(subset=["id", "as_of"])
    a["id"] = a["id"].astype("int64")
    a = a.sort_values(["id", "as_of"], kind="mergesort").drop_duplicates(
        ["id", "as_of"], keep="last"
    )
    grid = grid.sort_values("date", kind="mergesort")
    a = a.sort_values("as_of", kind="mergesort")
    grid = pd.merge_asof(
        grid,
        a[["id", "as_of", "charter"]],
        left_on="date",
        right_on="as_of",
        by="id",
        direction="backward",
        allow_exact_matches=True,
    )
    grid = grid[grid["charter"] == COMMERCIAL]

    # ---- Attach reported values (one per bank-quarter; first non-NaN if duplicated)
    v = c.groupby(["id", "qidx"], sort=False)["value"].first().reset_index()
    grid = grid.merge(v, on=["id", "qidx"], how="left")

    # ---- Rule 3: drop banks with more than 20% missing among remaining grid quarters
    stats = grid.groupby("id")["value"].agg(
        n_missing=lambda s: int(s.isna().sum()), n_total="size"
    )
    # missing / total > 0.2  <=>  5 * missing > total (integer-exact)
    good_ids = stats.index[stats["n_missing"] * 5 <= stats["n_total"]]
    grid = grid[grid["id"].isin(good_ids)]

    # ---- Rule 4: drop missing bank-quarters -------------------------------------
    out = grid.loc[grid["value"].notna(), ["id", "date", "value"]].rename(
        columns={"value": "y"}
    )
    out = out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["date"] = out["date"].astype("datetime64[ns]")
    out["y"] = out["y"].astype("float64")
    return out[OUT_COLUMNS]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    calls_df = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    attrs_df = pd.read_csv(os.path.join(here, "sample_attrs.csv"))
    panel = build_panel(calls_df, attrs_df)
    print(panel.dtypes.to_dict())
    print(panel.head())
    print("rows:", len(panel), "banks:", panel["id"].nunique())
