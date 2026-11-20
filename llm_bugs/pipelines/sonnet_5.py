import numpy as np
import pandas as pd


def _qidx(d):
    d = np.asarray(d, dtype="int64")
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    if len(calls) == 0:
        return empty
    c = pd.DataFrame({"id": calls["rssd_id"].to_numpy(dtype="int64"),
                      "q": _qidx(calls["report_date"]),
                      "y": calls["value"].to_numpy(dtype="float64")})
    # active span from all reports (even NaN-valued ones)
    span = c.groupby("id")["q"].agg(["min", "max"])
    # one value per bank-quarter (last reported)
    c = c.drop_duplicates(["id", "q"], keep="last")

    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    start = np.repeat(span["min"].to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    grid = pd.DataFrame({"id": ids, "q": start + offs})
    grid = grid.merge(c, on=["id", "q"], how="left")

    # charter at quarter end (latest snapshot on or before the quarter end)
    year = grid["q"] // 4
    month = (grid["q"] % 4 + 1) * 3
    qend = (pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))
            + pd.offsets.MonthEnd(0))
    grid["date"] = qend.astype("datetime64[ns]")

    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["as_of"] = pd.to_datetime(a["as_of"].astype(str), format="%Y%m%d").astype("datetime64[ns]")
    a = a.rename(columns={"rssd_id": "id"}).sort_values("as_of", kind="stable")
    a = a.drop_duplicates(["id", "as_of"], keep="last")
    grid = grid.sort_values("date", kind="stable")
    grid = pd.merge_asof(grid, a, left_on="date", right_on="as_of", by="id", direction="backward")
    grid = grid[grid["charter"] == "commercial"]

    miss = grid["y"].isna()
    frac = miss.groupby(grid["id"]).mean()
    keep = frac[frac <= 0.2].index
    out = grid[grid["id"].isin(keep) & ~miss][["id", "date", "y"]]
    out = out.sort_values(["id", "date"]).reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    return out
