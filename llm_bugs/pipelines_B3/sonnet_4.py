import numpy as np
import pandas as pd

MIDWEST = ["IL", "IN", "IA", "MI", "MN", "OH", "WI"]
MAX_GAP = 2       # interior gaps up to this many quarters are linearly interpolated
MIN_QUARTERS = 20  # minimum length of the contiguous series kept per bank (>= 4 holdout + 16 train)


def _qindex(yyyymmdd):
    s = np.asarray(yyyymmdd, dtype="int64")
    return (s // 10000) * 4 + ((s // 100) % 100 - 1) // 3


def _state_at(ids, qidx_dates, history, directory):
    """Point-in-time headquarters state for (id, date) pairs, from the history spells."""
    keys = pd.DataFrame({"rssd_id": ids, "date": qidx_dates, "_o": np.arange(len(ids))})
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h["start_date"] = pd.to_numeric(h["start_date"], errors="coerce")
    h = h.dropna(subset=["start_date"])
    h["start_date"] = h["start_date"].astype("int64")
    h = h.sort_values(["start_date", "rssd_id"])
    keys = keys.sort_values(["date", "rssd_id"])
    m = pd.merge_asof(keys, h[["rssd_id", "start_date", "state"]], left_on="date",
                      right_on="start_date", by="rssd_id", direction="backward")
    # banks with no history spell covering the date: fall back to directory only if
    # the bank has no history at all
    no_hist = ~m["rssd_id"].isin(history["rssd_id"].unique())
    if no_hist.any():
        dmap = directory.drop_duplicates("rssd_id").set_index("rssd_id")["state"]
        m.loc[no_hist, "state"] = m.loc[no_hist, "rssd_id"].map(dmap)
    m = m.sort_values("_o")
    return m["state"].to_numpy()


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c.dropna(subset=["rssd_id", "report_date"])
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c["value"] = pd.to_numeric(c["value"], errors="coerce")
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    c["q"] = _qindex(c["report_date"])
    # one observation per bank-quarter (prefer a non-missing value, last filed wins)
    c = c.sort_values(["rssd_id", "q", "value"], na_position="first")
    c = c.drop_duplicates(["rssd_id", "q"], keep="last")
    valid = c.dropna(subset=["value"])
    if valid.empty:
        return empty

    # regular quarterly grid from first to last valid observation per bank
    span = valid.groupby("rssd_id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    q = np.repeat(span["min"].to_numpy(), n) + offs
    g = pd.DataFrame({"id": ids, "q": q})
    g = g.merge(valid[["rssd_id", "q", "value"]].rename(columns={"rssd_id": "id"}),
                on=["id", "q"], how="left")
    g = g.sort_values(["id", "q"], kind="stable").reset_index(drop=True)

    # linear interpolation of short interior gaps only
    has = g["value"].notna()
    q_prev = g["q"].where(has).groupby(g["id"]).ffill()
    q_next = g["q"].where(has).groupby(g["id"]).bfill()
    v_prev = g["value"].groupby(g["id"]).ffill()
    v_next = g["value"].groupby(g["id"]).bfill()
    gap = q_next - q_prev - 1
    w = (g["q"] - q_prev) / (q_next - q_prev)
    interp = v_prev + (v_next - v_prev) * w
    fill = (~has) & (gap <= MAX_GAP)
    g["y"] = np.where(has, g["value"], np.where(fill, interp, np.nan))

    # Midwest membership at each quarter, from headquarters history
    date_int = ((g["q"] // 4) * 10000 + (g["q"] % 4 * 3 + 3) * 100).to_numpy()
    date_int = date_int + np.where(g["q"] % 4 == 0, 31, np.where(g["q"] % 4 == 3, 31, 30)).astype(int)
    st = _state_at(g["id"].to_numpy(), date_int, history, directory)
    keep = pd.Series(st).isin(MIDWEST).to_numpy() & g["y"].notna().to_numpy()
    g = g[keep].reset_index(drop=True)
    if g.empty:
        return empty

    # longest contiguous run per bank
    brk = (g["id"] != g["id"].shift()) | (g["q"] != g["q"].shift() + 1)
    g["run"] = brk.cumsum()
    g["rl"] = g.groupby("run")["q"].transform("size")
    best = g.groupby("id")["rl"].transform("max")
    # tie -> most recent run
    g = g[g["rl"] == best]
    last_run = g.groupby("id")["run"].transform("max")
    g = g[(g["run"] == last_run) & (g["rl"] >= MIN_QUARTERS)]

    date = pd.to_datetime(pd.Series(((g["q"] // 4) * 100 + (g["q"] % 4) * 3 + 3).to_numpy()).astype(str)
                          + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"id": g["id"].astype("int64").to_numpy(),
                        "date": date.to_numpy(),
                        "y": g["y"].astype(float).to_numpy()})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
