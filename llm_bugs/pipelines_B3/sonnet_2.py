import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}
MAX_GAP = 2          # interpolate internal gaps up to this many quarters
MIN_OBS = 16         # min quarters in final series (train + 4 holdout)
MIN_COVERAGE = 0.8   # min share of real (non-interpolated) obs in the series


def _qidx(dates):
    """Integer quarter index (year*4 + quarter-1) from YYYYMMDD ints."""
    d = np.asarray(dates, dtype="int64")
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c.dropna(subset=["value"])
    c = c[np.isfinite(c["value"])]
    c["q"] = _qidx(c["report_date"])
    c = c.sort_values(["rssd_id", "q", "report_date"]).drop_duplicates(["rssd_id", "q"], keep="last")

    # Point-in-time headquarters state from history (directory only reflects the end of sample).
    h = history[["rssd_id", "start_date", "state"]].dropna(subset=["start_date"]).copy()
    h["sq"] = _qidx(h["start_date"].astype("int64"))
    h = h.sort_values(["sq", "rssd_id"])
    c = c.sort_values("q")
    m = pd.merge_asof(c, h[["rssd_id", "sq", "state"]], left_on="q", right_on="sq",
                      by="rssd_id", direction="backward")
    # Before first spell: use that bank's earliest spell; no history at all: directory.
    first = h.sort_values("sq").drop_duplicates("rssd_id")[["rssd_id", "state"]].rename(columns={"state": "st0"})
    m = m.merge(first, on="rssd_id", how="left")
    m["state"] = m["state"].fillna(m["st0"])
    dmap = directory.drop_duplicates("rssd_id").set_index("rssd_id")["state"]
    m["state"] = m["state"].fillna(m["rssd_id"].map(dmap))
    m = m[m["state"].isin(MIDWEST)]
    if m.empty:
        return empty
    m = m.sort_values(["rssd_id", "q"])

    out = []
    for bid, g in m.groupby("rssd_id", sort=True):
        q = g["q"].to_numpy()
        v = g["value"].to_numpy(dtype=float)
        # split into segments where gaps exceed MAX_GAP missing quarters; keep latest segment
        brk = np.where(np.diff(q) > MAX_GAP + 1)[0]
        start = brk[-1] + 1 if len(brk) else 0
        q, v = q[start:], v[start:]
        full = np.arange(q[0], q[-1] + 1)
        if len(full) and len(q) / len(full) < MIN_COVERAGE:
            continue
        if len(full) < MIN_OBS:
            continue
        y = np.interp(full, q, v)
        out.append(pd.DataFrame({"id": np.int64(bid), "q": full, "y": y}))
    if not out:
        return empty
    p = pd.concat(out, ignore_index=True)
    yr, qt = p["q"] // 4, p["q"] % 4
    p["date"] = pd.to_datetime(dict(year=yr, month=qt * 3 + 3, day=1)) + pd.offsets.MonthEnd(0)
    p = p[["id", "date", "y"]].sort_values(["id", "date"]).reset_index(drop=True)
    p["id"] = p["id"].astype("int64")
    return p[p["y"].notna()].reset_index(drop=True)
