import numpy as np
import pandas as pd

MAX_GAP = 2        # at most this many consecutive missing quarters are interpolated
MIN_QUARTERS = 24  # minimum regular-series length (leaves >=20 for training after 4 held out)
MIN_OBS_FRAC = 0.8  # minimum share of quarters that are real observations


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")

    # point-in-time charter: latest snapshot at or before the report date
    # (reports earlier than a bank's first snapshot use that first snapshot)
    a = attrs[["rssd_id", "as_of", "charter"]].dropna()
    a = a.drop_duplicates(["rssd_id", "as_of"], keep="last").sort_values(["as_of", "rssd_id"])
    c = c.sort_values(["report_date", "rssd_id"])
    c = pd.merge_asof(c, a.rename(columns={"as_of": "report_date"}),
                      on="report_date", by="rssd_id", direction="backward")
    first = a.sort_values("as_of").groupby("rssd_id")["charter"].first()
    c["charter"] = c["charter"].fillna(c["rssd_id"].map(first))
    c = c[c["charter"].astype(str).str.lower().eq("commercial")]

    # calendar quarter index; duplicate reports for a quarter: keep last non-missing
    ds = c["report_date"].to_numpy()
    year, month = ds // 10000, (ds // 100) % 100
    c["q"] = year * 4 + (month - 1) // 3
    c = c.dropna(subset=["value"])
    c = c[np.isfinite(c["value"])]
    c = c.sort_values(["rssd_id", "q", "report_date"]).drop_duplicates(["rssd_id", "q"], keep="last")
    if c.empty:
        return empty

    # split at gaps longer than MAX_GAP, keep only each bank's most recent segment
    ids = c["rssd_id"].to_numpy()
    q = c["q"].to_numpy()
    newbank = np.r_[True, ids[1:] != ids[:-1]]
    brk = newbank | np.r_[False, (q[1:] - q[:-1]) > MAX_GAP + 1]
    seg = np.cumsum(brk)
    last_seg = pd.Series(seg).groupby(ids).transform("max").to_numpy()
    keep = seg == last_seg
    ids, q, seg = ids[keep], q[keep], seg[keep]
    v = c["value"].to_numpy()[keep]

    # regular grid between first and last observation (no extrapolation at the ends)
    s = pd.DataFrame({"id": ids, "q": q, "v": v})
    g = s.groupby("id")["q"].agg(["min", "max", "size"])
    g["n"] = g["max"] - g["min"] + 1
    g = g[(g["n"] >= MIN_QUARTERS) & (g["size"] >= MIN_OBS_FRAC * g["n"])]
    if g.empty:
        return empty
    s = s[s["id"].isin(g.index)]
    gid = np.repeat(g.index.to_numpy(), g["n"].to_numpy())
    offs = np.arange(len(gid)) - np.repeat(np.cumsum(g["n"].to_numpy()) - g["n"].to_numpy(), g["n"].to_numpy())
    gq = np.repeat(g["min"].to_numpy(), g["n"].to_numpy()) + offs
    grid = pd.DataFrame({"id": gid, "q": gq})
    grid = grid.merge(s, on=["id", "q"], how="left")
    # linear interpolation within bank (gaps are interior and short by construction)
    grid["y"] = grid.groupby("id")["v"].transform(lambda x: x.interpolate(method="linear", limit_area="inside"))
    grid = grid.dropna(subset=["y"])

    yr, qt = grid["q"] // 4, grid["q"] % 4
    date = pd.to_datetime(yr.astype(str) + "-" + (qt * 3 + 3).astype(str).str.zfill(2) + "-01") + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"id": grid["id"].astype("int64").to_numpy(),
                        "date": date.to_numpy(), "y": grid["y"].astype(float).to_numpy()})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
