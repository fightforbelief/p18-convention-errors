import numpy as np
import pandas as pd


def _qidx(yyyymmdd):
    s = pd.Series(yyyymmdd).astype("int64")
    y = s // 10000
    m = (s // 100) % 100
    return (y * 4 + (m - 1) // 3).to_numpy()


def _qend(q):
    y = q // 4
    m = (q % 4) * 3 + 3
    return pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": 1})) + pd.offsets.MonthEnd(0)


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    if len(calls) == 0:
        return empty
    c = pd.DataFrame({"id": calls["rssd_id"].to_numpy(),
                      "q": _qidx(calls["report_date"]),
                      "y": calls["value"].astype(float).to_numpy()})
    # Rule 1: active span from first to last report (including NaN-valued reports)
    span = c.groupby("id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    start = np.repeat(span["min"].to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    grid = pd.DataFrame({"id": ids, "q": start + offs})

    # observed values; duplicates: keep last non-missing
    obs = c.dropna(subset=["y"]).drop_duplicates(["id", "q"], keep="last")
    grid = grid.merge(obs, on=["id", "q"], how="left")
    grid["date"] = _qend(grid["q"].to_numpy())

    # Rule 2: charter as of quarter end (latest snapshot on/before)
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["as_of_dt"] = pd.to_datetime(a["as_of"].astype("int64").astype(str), format="%Y%m%d")
    a = a.sort_values(["as_of_dt"], kind="stable").drop_duplicates(["rssd_id", "as_of_dt"], keep="last")
    a = a.rename(columns={"rssd_id": "id"})[["id", "as_of_dt", "charter"]]
    grid = grid.sort_values("date", kind="stable")
    grid = pd.merge_asof(grid, a.sort_values("as_of_dt"), left_on="date", right_on="as_of_dt", by="id")
    grid = grid[grid["charter"] == "commercial"]

    # Rule 3: drop banks with >20% missing among remaining grid quarters
    miss = grid["y"].isna()
    frac = miss.groupby(grid["id"]).mean()
    keep = frac[frac <= 0.2].index
    grid = grid[grid["id"].isin(keep)]

    # Rule 4
    grid = grid.dropna(subset=["y"])
    out = grid[["id", "date", "y"]].sort_values(["id", "date"]).reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    return out
