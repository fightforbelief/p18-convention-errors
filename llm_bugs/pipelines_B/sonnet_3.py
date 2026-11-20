import numpy as np
import pandas as pd

MAX_FILL = 2        # interpolate internal gaps of at most this many quarters
MIN_QUARTERS = 20   # need history beyond the 4 held-out quarters
MIN_OBS_FRAC = 0.8  # share of quarters in the final series that are real observations


def _qidx(date_int):
    d = np.asarray(date_int, dtype=np.int64)
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c = c.dropna(subset=["value"])
    c = c[np.isfinite(c["value"])]

    # Commercial banks only: charter in force at the report date (latest
    # snapshot at or before it; the earliest snapshot for earlier reports).
    a = attrs[["rssd_id", "as_of", "charter"]].dropna().copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["as_of"] = a["as_of"].astype("int64")
    a = a.sort_values(["as_of", "rssd_id"]).drop_duplicates(["rssd_id", "as_of"], keep="last")
    c = c.sort_values("report_date")
    c["_k"] = _qidx(c["report_date"])
    a_first = a.sort_values("as_of").drop_duplicates("rssd_id", keep="first")
    m = pd.merge_asof(c, a.rename(columns={"charter": "ch"}), left_on="report_date",
                      right_on="as_of", by="rssd_id", direction="backward")
    first = a_first.set_index("rssd_id")["charter"]
    m["ch"] = m["ch"].where(m["as_of"].notna(), m["rssd_id"].map(first))
    m = m[m["ch"] == "commercial"]
    if m.empty:
        return empty

    # One value per bank-quarter (latest report wins), sorted.
    m = m.sort_values(["rssd_id", "_k", "report_date"])
    m = m.drop_duplicates(["rssd_id", "_k"], keep="last")
    ids = m["rssd_id"].to_numpy()
    k = m["_k"].to_numpy()
    v = m["value"].to_numpy(float)

    # Break series at gaps longer than MAX_FILL missing quarters; keep latest segment.
    same = np.r_[False, ids[1:] == ids[:-1]]
    gap = np.where(same, k - np.r_[k[:1], k[:-1]] - 1, 0)
    newseg = (~same) | (gap > MAX_FILL)
    seg = np.cumsum(newseg)
    last_seg = pd.Series(seg).groupby(ids).transform("max").to_numpy()
    keep = seg == last_seg
    ids, k, v, gap, seg = ids[keep], k[keep], v[keep], gap[keep], seg[keep]
    gap = np.where(np.r_[True, seg[1:] != seg[:-1]], 0, gap)

    # Coverage filter on the kept segment.
    n_obs = pd.Series(k).groupby(ids).transform("size").to_numpy()
    span = pd.Series(k).groupby(ids).transform("max").to_numpy() - \
        pd.Series(k).groupby(ids).transform("min").to_numpy() + 1
    ok = (span >= MIN_QUARTERS) & (n_obs / span >= MIN_OBS_FRAC)
    ids, k, v, gap = ids[ok], k[ok], v[ok], gap[ok]
    if len(ids) == 0:
        return empty

    # Linear interpolation of the short internal gaps.
    prev_v = np.r_[v[:1], v[:-1]]
    rep = gap.astype(int) + 1  # each row emits `gap` filled rows followed by itself
    out_ids = np.repeat(ids, rep)
    row_start = np.cumsum(rep) - rep
    j = np.arange(rep.sum()) - np.repeat(row_start, rep)  # 0..gap
    g = np.repeat(gap, rep)
    cur = np.repeat(v, rep)
    prv = np.repeat(prev_v, rep)
    kk = np.repeat(k, rep) - (g - j)
    y = np.where(j == g, cur, prv + (cur - prv) * (j + 1) / (g + 1))

    date = (pd.to_datetime(pd.DataFrame({"year": kk // 4, "month": (kk % 4) * 3 + 1, "day": 1}))
            + pd.offsets.MonthEnd(3))
    res = pd.DataFrame({"id": out_ids.astype("int64"), "date": date.to_numpy(), "y": y})
    res = res.dropna(subset=["y"]).sort_values(["id", "date"]).reset_index(drop=True)
    return res
