import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}
MAX_GAP = 2        # fill at most this many consecutive missing quarters (linear interpolation)
MIN_OBS = 20       # min reported quarters in the kept contiguous series (>= 5y; 4 are held out)


def _qidx(d):
    """YYYYMMDD int -> integer quarter index (year*4 + quarter-1)."""
    d = np.asarray(d, dtype="int64")
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].dropna(subset=["rssd_id", "report_date"]).copy()
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["q"] = _qidx(c["report_date"].astype("int64"))
    c["value"] = c["value"].astype(float).replace([np.inf, -np.inf], np.nan)
    # one row per bank-quarter (a non-missing report wins over a missing one; last wins)
    c = c.sort_values(["rssd_id", "q", "value"], na_position="first").drop_duplicates(["rssd_id", "q"], keep="last")
    c = c.dropna(subset=["value"])
    if c.empty:
        return empty

    # Point-in-time headquarters state from the history (the directory only holds the
    # end-of-sample state and omits banks that disappeared, so it is only a fallback).
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h["rssd_id"] = h["rssd_id"].astype("int64")
    h["sd"] = h["start_date"].astype("int64")
    h["ed"] = pd.to_numeric(h["end_date"], errors="coerce").fillna(99991231).astype("int64")
    h["mw"] = h["state"].astype(str).str.upper().isin(MIDWEST)
    # quarter-end date (YYYYMMDD) of each report, compared against spell start/end dates
    qe = pd.Series(c["q"] // 4 * 10000 + (c["q"] % 4 * 3 + 3) * 100, index=c.index)
    c["qd"] = qe + np.where(c["q"] % 4 == 0, 31, np.where(c["q"] % 4 == 1, 30, np.where(c["q"] % 4 == 2, 30, 31)))
    h = h.rename(columns={"sd": "sq", "ed": "eq"})
    m = c.merge(h[["rssd_id", "sq", "eq", "mw"]], on="rssd_id", how="left")
    cover = (m["sq"] <= m["qd"]) & (m["qd"] <= m["eq"])
    m["hit"] = cover
    m = m.sort_values(["rssd_id", "q", "hit", "sq"])
    m = m.drop_duplicates(["rssd_id", "q"], keep="last")        # prefer covering spell, latest start
    has_hist = m["sq"].notna() & m["hit"]
    # fallback: directory state for banks with no history spell covering the quarter
    dstate = directory.drop_duplicates("rssd_id").set_index("rssd_id")["state"].astype(str).str.upper()
    dmw = m["rssd_id"].map(dstate).isin(MIDWEST)
    no_hist_bank = ~m["rssd_id"].isin(h["rssd_id"])
    keep = np.where(has_hist, m["mw"].fillna(False).astype(bool), no_hist_bank & dmw)
    m = m[keep.astype(bool)][["rssd_id", "q", "value"]].sort_values(["rssd_id", "q"]).reset_index(drop=True)
    if m.empty:
        return empty

    # segments: break wherever the gap exceeds MAX_GAP missing quarters
    new_bank = m["rssd_id"].ne(m["rssd_id"].shift())
    gap = m["q"].diff() - 1
    brk = new_bank | (gap > MAX_GAP)
    m["seg"] = brk.cumsum()
    # keep only each bank's most recent segment (contiguous, forecast-ready series)
    last_seg = m.groupby("rssd_id")["seg"].transform("max")
    m = m[m["seg"] == last_seg]
    # coverage filter
    cnt = m.groupby("rssd_id")["q"].transform("size")
    m = m[cnt >= MIN_OBS]
    if m.empty:
        return empty

    # regular quarterly grid between first and last report, short gaps interpolated
    g = m.groupby("rssd_id")["q"].agg(["min", "max"])
    lens = (g["max"] - g["min"] + 1).to_numpy()
    ids = np.repeat(g.index.to_numpy(), lens)
    start = np.repeat(g["min"].to_numpy(), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    grid = pd.DataFrame({"id": ids, "q": start + offs})
    grid = grid.merge(m[["rssd_id", "q", "value"]].rename(columns={"rssd_id": "id"}), on=["id", "q"], how="left")
    grid["y"] = grid.groupby("id")["value"].transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
    grid = grid.dropna(subset=["y"])

    yr = grid["q"] // 4
    qq = grid["q"] % 4
    date = pd.to_datetime(pd.DataFrame({"year": yr, "month": qq * 3 + 3, "day": 1})) + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"id": grid["id"].astype("int64").to_numpy(), "date": date.to_numpy(),
                        "y": grid["y"].astype(float).to_numpy()})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
