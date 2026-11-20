import numpy as np
import pandas as pd


def _to_ts(s):
    return pd.to_datetime(s.astype("int64").astype(str), format="%Y%m%d")


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    if len(calls) == 0:
        return empty

    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["q"] = _to_ts(c["report_date"]).dt.to_period("Q").astype("int64")  # quarter ordinal
    # active span uses every report row (even NaN-valued ones)
    span = c.groupby("rssd_id")["q"].agg(["min", "max"])
    # one value per bank-quarter: last non-NaN report
    vals = (c.dropna(subset=["value"]).sort_values(["rssd_id", "q"], kind="stable")
              .drop_duplicates(["rssd_id", "q"], keep="last")[["rssd_id", "q", "value"]])

    # regular quarterly grid per bank
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    grid = pd.DataFrame({"rssd_id": ids, "q": np.repeat(span["min"].to_numpy(), n) + offs})
    grid = grid.merge(vals, on=["rssd_id", "q"], how="left")
    grid["date"] = pd.PeriodIndex.from_ordinals(grid["q"].to_numpy(), freq="Q").to_timestamp(how="end").normalize()

    # charter at each quarter: latest snapshot on or before quarter end
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["as_of_ts"] = _to_ts(a["as_of"])
    a = a.sort_values(["as_of_ts"], kind="stable")
    grid = grid.sort_values("date", kind="stable")
    grid = pd.merge_asof(grid, a[["rssd_id", "as_of_ts", "charter"]],
                         left_on="date", right_on="as_of_ts", by="rssd_id",
                         direction="backward")
    grid = grid[grid["charter"] == "commercial"]

    # drop banks with > 20% missing among remaining quarters
    miss = grid["value"].isna()
    stats = pd.DataFrame({"rssd_id": grid["rssd_id"], "m": miss}).groupby("rssd_id")["m"].agg(["sum", "size"])
    keep = stats.index[(stats["sum"] * 5 <= stats["size"]).to_numpy()]
    grid = grid[grid["rssd_id"].isin(keep) & grid["value"].notna()]

    out = pd.DataFrame({"id": grid["rssd_id"].astype("int64").to_numpy(),
                        "date": grid["date"].to_numpy(),
                        "y": grid["value"].astype(float).to_numpy()})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
