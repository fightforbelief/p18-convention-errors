import numpy as np
import pandas as pd


def _to_ts(x):
    return pd.to_datetime(pd.Series(x).astype("int64").astype(str), format="%Y%m%d")


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]
    if len(calls) == 0:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    d = _to_ts(c["report_date"].values)
    c["q"] = (d.dt.year.values * 4 + (d.dt.month.values - 1) // 3).astype("int64")
    c["rssd_id"] = c["rssd_id"].astype("int64")

    # Active span (from all reports, including NaN-valued ones)
    span = c.groupby("rssd_id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    start = np.repeat(span["min"].to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    grid = pd.DataFrame({"rssd_id": ids, "q": start + offs})

    # Values: duplicates collapse to last non-missing report in that quarter
    cv = c.dropna(subset=["value"]).sort_values("report_date")
    cv = cv.drop_duplicates(["rssd_id", "q"], keep="last")[["rssd_id", "q", "value"]]
    grid = grid.merge(cv, on=["rssd_id", "q"], how="left")

    # Quarter-end date
    yr = grid["q"] // 4
    mo = (grid["q"] % 4) * 3 + 3
    grid["date"] = (pd.to_datetime({"year": yr, "month": mo, "day": 1})
                    + pd.offsets.MonthEnd(0))

    # Charter as of quarter end (latest snapshot on or before)
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["as_ts"] = _to_ts(a["as_of"].values).values
    a = a.sort_values(["as_ts"], kind="stable").drop_duplicates(["rssd_id", "as_ts"], keep="last")
    grid = grid.sort_values("date", kind="stable")
    grid = pd.merge_asof(grid, a[["rssd_id", "as_ts", "charter"]],
                         left_on="date", right_on="as_ts", by="rssd_id",
                         direction="backward")
    grid = grid[grid["charter"] == "commercial"].copy()

    # Missingness filter over remaining quarters
    grid["miss"] = grid["value"].isna()
    frac = grid.groupby("rssd_id")["miss"].mean()
    keep = frac.index[frac.to_numpy() <= 0.2]
    grid = grid[grid["rssd_id"].isin(keep) & ~grid["miss"]]

    res = pd.DataFrame({"id": grid["rssd_id"].astype("int64").to_numpy(),
                        "date": grid["date"].to_numpy(),
                        "y": grid["value"].astype("float64").to_numpy()})
    return res.sort_values(["id", "date"]).reset_index(drop=True)[out_cols]
