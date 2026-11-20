import numpy as np
import pandas as pd


def _to_qord(d):
    """YYYYMMDD int array -> quarter ordinal (year*4 + quarter-1)."""
    d = np.asarray(d, dtype=np.int64)
    y = d // 10000
    m = (d // 100) % 100
    return y * 4 + (m - 1) // 3


def _qord_to_end(q):
    """Quarter ordinal -> Timestamp at the calendar quarter end."""
    q = np.asarray(q, dtype=np.int64)
    y = q // 4
    qq = q % 4
    start = pd.to_datetime(pd.DataFrame({"year": y, "month": qq * 3 + 1, "day": 1}))
    return start + pd.offsets.QuarterEnd(0)


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": np.array([], dtype="int64"),
                          "date": pd.to_datetime([]),
                          "y": np.array([], dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c[c["rssd_id"].notna() & c["report_date"].notna()]
    if c.empty:
        return empty
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["q"] = _to_qord(c["report_date"].astype("int64"))

    # Rule 1: active span from first to last report (NaN-valued reports count).
    span = c.groupby("rssd_id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    q = np.repeat(span["min"].to_numpy(), n) + offs
    grid = pd.DataFrame({"id": ids, "q": q})

    # Values: one per bank-quarter (if duplicated, keep last non-NaN).
    v = c.dropna(subset=["value"]).drop_duplicates(["rssd_id", "q"], keep="last")
    v = v.rename(columns={"rssd_id": "id"})[["id", "q", "value"]]
    grid = grid.merge(v, on=["id", "q"], how="left")
    grid["date"] = _qord_to_end(grid["q"])

    # Rule 2: charter as of the quarter end (latest snapshot on or before it).
    a = attrs[["rssd_id", "as_of", "charter"]].dropna(subset=["rssd_id", "as_of"]).copy()
    a["id"] = a["rssd_id"].astype("int64")
    a["as_of_dt"] = pd.to_datetime(a["as_of"].astype("int64").astype(str), format="%Y%m%d")
    a = a.sort_values("as_of_dt", kind="stable").drop_duplicates(["id", "as_of_dt"], keep="last")
    grid = grid.sort_values("date", kind="stable")
    grid = pd.merge_asof(grid, a[["id", "as_of_dt", "charter"]],
                         left_on="date", right_on="as_of_dt", by="id",
                         direction="backward")
    grid = grid[grid["charter"].astype(str).str.strip().str.lower() == "commercial"]
    if grid.empty:
        return empty

    # Rule 3: drop banks with >20% missing among remaining quarters.
    frac = grid["value"].isna().groupby(grid["id"]).mean()
    keep = frac.index[frac <= 0.2]
    grid = grid[grid["id"].isin(keep)]

    # Rule 4: drop missing bank-quarters.
    out = (grid.dropna(subset=["value"])[["id", "date", "value"]]
           .rename(columns={"value": "y"})
           .sort_values(["id", "date"]).reset_index(drop=True))
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    return out
