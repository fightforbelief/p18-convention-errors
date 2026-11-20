import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}
MAX_FILL_GAP = 4      # interpolate interior gaps up to this many quarters
MIN_COVERAGE = 0.80   # observed quarters / quarters in the bank's span
MIN_QUARTERS = 24     # minimum contiguous length (holdout takes last 4)


def _qidx(d):
    d = np.asarray(d, dtype="int64")
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c["value"] = pd.to_numeric(c["value"], errors="coerce")
    c = c.replace({"value": {np.inf: np.nan, -np.inf: np.nan}})
    c = c.dropna(subset=["value"])
    c = c.drop_duplicates(["rssd_id", "report_date"], keep="last")
    if c.empty:
        return empty

    # point-in-time headquarters state: history spell covering the report date;
    # directory (end-of-sample) state only for banks absent from history.
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h["rssd_id"] = h["rssd_id"].astype("int64")
    h["start_date"] = h["start_date"].astype("int64")
    h["end_date"] = pd.to_numeric(h["end_date"], errors="coerce").fillna(99991231).astype("int64")
    c = c.sort_values("report_date")
    h = h.sort_values("start_date")
    m = pd.merge_asof(c, h.rename(columns={"state": "hstate"}), left_on="report_date",
                      right_on="start_date", by="rssd_id", direction="backward")
    ok = m["report_date"] <= m["end_date"]
    m["st"] = m["hstate"].where(ok)
    in_hist = m["rssd_id"].isin(h["rssd_id"].unique())
    dmap = directory.drop_duplicates("rssd_id", keep="last").set_index("rssd_id")["state"]
    m["st"] = np.where(in_hist, m["st"], m["rssd_id"].map(dmap))
    m["st"] = m["st"].astype(object).where(m["st"].notna(), None)
    m = m[pd.Series(m["st"]).str.upper().str.strip().isin(MIDWEST).values]
    if m.empty:
        return empty

    m = m.assign(q=_qidx(m["report_date"].values))[["rssd_id", "q", "value"]]
    m = m.sort_values(["rssd_id", "q"])

    # regular quarterly grid per bank over its observed span
    g = m.groupby("rssd_id")["q"].agg(["min", "max", "count"])
    g["span"] = g["max"] - g["min"] + 1
    g = g[(g["count"] / g["span"] >= MIN_COVERAGE) & (g["span"] >= MIN_QUARTERS)]
    if g.empty:
        return empty
    ids = np.repeat(g.index.values, g["span"].values)
    offs = np.arange(len(ids)) - np.repeat(np.cumsum(g["span"].values) - g["span"].values, g["span"].values)
    grid = pd.DataFrame({"rssd_id": ids, "q": np.repeat(g["min"].values, g["span"].values) + offs})
    grid = grid.merge(m, on=["rssd_id", "q"], how="left")

    # fill short interior gaps by linear interpolation; leave long gaps missing
    obs = grid["value"].notna()
    runid = (obs.groupby(grid["rssd_id"]).cumsum())  # constant within a NaN run
    runlen = grid.groupby(["rssd_id", runid])["value"].transform(lambda s: s.isna().sum()) \
        if False else (~obs).groupby([grid["rssd_id"], runid]).transform("sum")
    filled = grid.groupby("rssd_id")["value"].transform(lambda s: s.interpolate(limit_area="inside"))
    grid["y"] = np.where(obs, grid["value"], np.where(runlen <= MAX_FILL_GAP, filled, np.nan))

    # longest contiguous segment of non-missing quarters per bank
    valid = grid["y"].notna()
    seg = (~valid).groupby(grid["rssd_id"]).cumsum()
    grid = grid[valid].copy()
    seg = seg[valid]
    grid["seg"] = seg.values
    size = grid.groupby(["rssd_id", "seg"])["y"].transform("size")
    best = size.groupby(grid["rssd_id"]).transform("max")
    grid = grid[size == best]
    # tie-break: keep the latest segment
    lastseg = grid.groupby("rssd_id")["seg"].transform("max")
    grid = grid[grid["seg"] == lastseg]
    grid = grid[best[grid.index] >= MIN_QUARTERS]

    year = grid["q"] // 4
    qtr = grid["q"] % 4
    date = pd.to_datetime(year.astype(str) + "-" + ((qtr + 1) * 3).astype(str).str.zfill(2) + "-01") \
        + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"id": grid["rssd_id"].astype("int64").values,
                        "date": date.values, "y": grid["y"].astype(float).values})
    return out.sort_values(["id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    c = pd.read_csv("sample_calls.csv")
    d = pd.read_csv("sample_directory.csv")
    h = pd.read_csv("sample_history.csv")
    p = build_panel(c, d, h)
    print(p.head(), p.shape, p.id.nunique())
