import numpy as np
import pandas as pd

MAX_FILL_GAP = 3      # interpolate interior gaps of up to this many missing quarters
MIN_QUARTERS = 20     # min contiguous quarters (>=16 train + 4 held out)
MIN_OBS_SHARE = 0.8   # min share of the kept span that was truly reported


def _qend(qi):
    ym = (qi // 4) * 10000 + ((qi % 4) * 3 + 3) * 100 + 1
    return pd.DatetimeIndex(pd.to_datetime(ym.astype(str), format="%Y%m%d") + pd.offsets.MonthEnd(0))


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["value"] = pd.to_numeric(c["value"], errors="coerce")
    c = c[np.isfinite(c["value"])]
    d = pd.to_datetime(c["report_date"].astype("int64").astype(str), format="%Y%m%d", errors="coerce")
    c = c[d.notna()].copy()
    d = d[d.notna()]
    c["qi"] = d.dt.year.values * 4 + (d.dt.month.values - 1) // 3  # quarter index
    c["rssd_id"] = c["rssd_id"].astype("int64")
    # duplicates within a quarter: keep the latest filing
    c = c.sort_values(["rssd_id", "report_date"]).drop_duplicates(["rssd_id", "qi"], keep="last")
    if c.empty:
        return empty

    # point-in-time charter: latest snapshot at or before the quarter end;
    # for quarters before the first snapshot use the first snapshot.
    a = attrs[["rssd_id", "as_of", "charter"]].dropna().copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["ad"] = pd.to_datetime(a["as_of"].astype("int64").astype(str), format="%Y%m%d", errors="coerce")
    a = a.dropna(subset=["ad"]).sort_values(["rssd_id", "ad"]).drop_duplicates(["rssd_id", "ad"], keep="last")
    qend = _qend(c["qi"].values)
    c["date"] = qend
    c = c.sort_values("date")
    m = pd.merge_asof(c, a[["rssd_id", "ad", "charter"]].sort_values("ad"),
                      left_on="date", right_on="ad", by="rssd_id", direction="backward")
    first = a.groupby("rssd_id")["charter"].first()
    m["charter"] = m["charter"].fillna(m["rssd_id"].map(first))
    m = m[m["charter"] == "commercial"]
    if m.empty:
        return empty
    m = m.sort_values(["rssd_id", "qi"])

    ids = m["rssd_id"].values
    qi = m["qi"].values
    val = m["value"].values
    # split into segments: new segment when bank changes or gap > MAX_FILL_GAP
    newseg = np.ones(len(m), bool)
    newseg[1:] = (ids[1:] != ids[:-1]) | (qi[1:] - qi[:-1] - 1 > MAX_FILL_GAP)
    seg = np.cumsum(newseg) - 1
    sdf = pd.DataFrame({"id": ids, "qi": qi, "y": val, "seg": seg})
    g = sdf.groupby("seg")
    seg_len = g["qi"].transform(lambda s: s.iloc[-1] - s.iloc[0] + 1) if False else (
        g["qi"].transform("max") - g["qi"].transform("min") + 1)
    sdf["span"] = seg_len
    sdf["nobs"] = g["qi"].transform("size")
    # per bank: choose the most recent qualifying segment (keeps the end of the history,
    # which is what the benchmark holds out)
    ok = (sdf["span"] >= MIN_QUARTERS) & (sdf["nobs"] / sdf["span"] >= MIN_OBS_SHARE)
    sdf = sdf[ok]
    if sdf.empty:
        return empty
    last_seg = sdf.groupby("id")["seg"].transform("max")
    sdf = sdf[sdf["seg"] == last_seg]

    # expand each segment to a regular quarterly grid and interpolate gaps
    s = sdf.groupby("seg")
    start = s["qi"].min()
    span = s["span"].first()
    segid = np.repeat(start.index.values, span.values)
    offs = np.arange(len(segid)) - np.repeat(np.cumsum(span.values) - span.values, span.values)
    gq = np.repeat(start.values, span.values) + offs
    grid = pd.DataFrame({"seg": segid, "qi": gq})
    grid = grid.merge(sdf[["seg", "id", "qi", "y"]], on=["seg", "qi"], how="left")
    grid["id"] = grid.groupby("seg")["id"].transform("first")
    grid["y"] = grid.groupby("seg")["y"].transform(lambda x: x.interpolate(method="linear", limit_area="inside"))
    grid = grid.dropna(subset=["y"])
    grid["date"] = _qend(grid["qi"].values)
    res = grid[out_cols].copy()
    res["id"] = res["id"].astype("int64")
    res["y"] = res["y"].astype(float)
    return res.sort_values(["id", "date"]).reset_index(drop=True)
