import numpy as np
import pandas as pd

MIDWEST = ["IL", "IN", "IA", "MI", "MN", "OH", "WI"]
MAX_FILL = 2      # longest interior gap (quarters) that is linearly interpolated
MIN_LEN = 20      # minimum contiguous quarters (must comfortably exceed the 4-quarter holdout)
MIN_OBS_SHARE = 0.8  # min share of segment quarters that were actually reported (not filled)


def _qidx(d):
    d = np.asarray(d, dtype="int64")
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out_empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                              "date": pd.Series(dtype="datetime64[ns]"),
                              "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c.dropna(subset=["rssd_id", "report_date"])
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c["value"] = c["value"].astype("float64").replace([np.inf, -np.inf], np.nan)
    mm = (c["report_date"] // 100) % 100
    c = c[mm.isin([3, 6, 9, 12])]
    c["q"] = _qidx(c["report_date"].to_numpy())
    # duplicates: prefer a reported value, then the last one
    c = c.assign(_na=c["value"].isna()).sort_values(["rssd_id", "q", "_na"], kind="stable")
    c = c.drop_duplicates(["rssd_id", "q"], keep="first")[["rssd_id", "q", "value"]]

    # point-in-time headquarters state from history (directory is end-of-sample only,
    # lacks failed/merged banks and ignores moves)
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h["rssd_id"] = h["rssd_id"].astype("int64")
    sq = _qidx(h["start_date"].to_numpy())
    end = h["end_date"]
    eq = np.where(end.isna(), 10**6, _qidx(end.fillna(0).astype("int64").to_numpy()))
    # a spell ending before the quarter-end (e.g. 19971230) does not cover that quarter,
    # but one ending exactly at quarter-end does; a start after quarter-end starts next quarter
    ed = end.fillna(0).astype("int64").to_numpy()
    eq = np.where(end.isna(), 10**6,
                  np.where(((ed % 10000) == 1231) | ((ed % 10000) == 331) | ((ed % 10000) == 630) | ((ed % 10000) == 930),
                           eq, eq - 1))
    st = h["start_date"].astype("int64").to_numpy()
    sq = np.where((st % 10000) == 1231, sq, np.where(np.isin(st % 10000, [331, 630, 930]), sq, sq + 1))
    h = pd.DataFrame({"rssd_id": h["rssd_id"].to_numpy(), "sq": sq, "eq": eq,
                      "state": h["state"].astype(str).str.strip().str.upper().to_numpy()})
    h = h[h["state"].isin(MIDWEST)]
    m = c.merge(h, on="rssd_id", how="inner")
    m = m[(m["q"] >= m["sq"]) & (m["q"] <= m["eq"])]
    m = m.sort_values(["rssd_id", "q"]).drop_duplicates(["rssd_id", "q"])[["rssd_id", "q", "value"]]
    if m.empty:
        return out_empty

    # regular quarterly grid per bank from first to last Midwest quarter
    g = m.groupby("rssd_id")["q"].agg(["min", "max"])
    n = (g["max"] - g["min"] + 1).to_numpy()
    ids = np.repeat(g.index.to_numpy(), n)
    off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    qs = np.repeat(g["min"].to_numpy(), n) + off
    grid = pd.DataFrame({"rssd_id": ids, "q": qs})
    grid = grid.merge(m, on=["rssd_id", "q"], how="left")
    grid["obs"] = grid["value"].notna()

    # fill only short interior gaps
    isna = ~grid["obs"]
    new_run = (isna != isna.shift()) | (grid["rssd_id"] != grid["rssd_id"].shift())
    run = new_run.cumsum()
    runlen = isna.groupby(run).transform("size")
    interp = grid.groupby("rssd_id")["value"].transform(lambda s: s.interpolate(limit_area="inside"))
    fill = isna & (runlen <= MAX_FILL) & interp.notna()
    grid["y"] = np.where(fill, interp, grid["value"])

    # keep each bank's most recent contiguous segment (what a benchmark holds out from)
    bad = grid["y"].isna()
    seg = bad.groupby(grid["rssd_id"]).cumsum()
    grid = grid[~bad].copy()
    seg = seg[~bad]
    grid["seg"] = seg
    last = grid.groupby("rssd_id")["seg"].transform("max")
    grid = grid[grid["seg"] == last]
    stats = grid.groupby("rssd_id").agg(n=("q", "size"), o=("obs", "sum"))
    keep = stats[(stats["n"] >= MIN_LEN) & (stats["o"] / stats["n"] >= MIN_OBS_SHARE)].index
    grid = grid[grid["rssd_id"].isin(keep)]
    if grid.empty:
        return out_empty

    yr = grid["q"] // 4
    mo = (grid["q"] % 4) * 3 + 3
    date = pd.to_datetime(pd.DataFrame({"year": yr, "month": mo, "day": 1})) + pd.offsets.MonthEnd(0)
    res = pd.DataFrame({"id": grid["rssd_id"].astype("int64").to_numpy(),
                        "date": date.to_numpy(), "y": grid["y"].astype("float64").to_numpy()})
    return res.sort_values(["id", "date"]).reset_index(drop=True)
