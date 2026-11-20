import numpy as np
import pandas as pd

MAX_FILL = 2      # interpolate interior gaps up to this many quarters
MIN_LEN = 24      # min quarters in final series (>= 4 holdout + 20 history)
MIN_OBS_FRAC = 0.8  # min share of truly observed quarters in the kept run


def _qidx(d):
    """YYYYMMDD int array -> quarter index (year*4 + quarter-1)."""
    d = np.asarray(d, dtype="int64")
    y, m = d // 10000, (d // 100) % 100
    return y * 4 + (m - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c.dropna(subset=["rssd_id", "report_date"])
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["q"] = _qidx(c["report_date"])
    c = c.replace([np.inf, -np.inf], np.nan)

    # charter in effect at each report (latest snapshot at/before the report;
    # before the first snapshot, use the first one). Keep commercial only.
    a = attrs[["rssd_id", "as_of", "charter"]].dropna().copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["aq"] = _qidx(a["as_of"])
    a = a.sort_values(["rssd_id", "as_of"]).drop_duplicates(["rssd_id", "aq"], keep="last")
    first = a.groupby("rssd_id")["aq"].transform("min")
    a.loc[a["aq"] == first, "aq"] = -10**9  # first snapshot covers all earlier dates
    a = a.sort_values("aq")
    c = c.sort_values("q")
    c = pd.merge_asof(c, a[["rssd_id", "aq", "charter"]], left_on="q", right_on="aq",
                      by="rssd_id", direction="backward")
    c = c[c["charter"] == "commercial"]

    # drop missing values, one report per bank-quarter (latest report_date wins)
    c = c.dropna(subset=["value"]).sort_values(["rssd_id", "q", "report_date"])
    c = c.drop_duplicates(["rssd_id", "q"], keep="last")
    if c.empty:
        return empty

    ids_out, q_out, y_out = [], [], []
    ids = c["rssd_id"].to_numpy()
    qs = c["q"].to_numpy()
    vs = c["value"].to_numpy(dtype=float)
    bounds = np.flatnonzero(np.diff(ids)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(ids)]
    for s, e in zip(starts, ends):
        q, v = qs[s:e], vs[s:e]
        n = q[-1] - q[0] + 1
        grid = np.full(n, np.nan)
        grid[q - q[0]] = v
        obs = ~np.isnan(grid)
        # fill short interior gaps linearly (no extrapolation, no look past last obs)
        miss = ~obs
        if miss.any():
            pos = np.flatnonzero(obs)
            gap = np.diff(pos) - 1
            for i in np.flatnonzero((gap > 0) & (gap <= MAX_FILL)):
                a0, b0 = pos[i], pos[i + 1]
                grid[a0 + 1:b0] = np.interp(np.arange(a0 + 1, b0), [a0, b0], [grid[a0], grid[b0]])
        filled = ~np.isnan(grid)
        # keep the most recent contiguous run (the part that gets forecast)
        nan_pos = np.flatnonzero(~filled)
        lo = nan_pos[-1] + 1 if len(nan_pos) else 0
        run = slice(lo, n)
        L = n - lo
        if L < MIN_LEN or obs[run].mean() < MIN_OBS_FRAC:
            continue
        ids_out.append(np.full(L, ids[s]))
        q_out.append(np.arange(q[0] + lo, q[0] + n))
        y_out.append(grid[run])

    if not ids_out:
        return empty
    qq = np.concatenate(q_out)
    year, qtr = qq // 4, qq % 4
    date = (pd.to_datetime(pd.DataFrame({"year": year, "month": qtr * 3 + 3, "day": 1}))
            + pd.offsets.MonthEnd(0))
    out = pd.DataFrame({"id": np.concatenate(ids_out).astype("int64"),
                        "date": date.to_numpy(), "y": np.concatenate(y_out)})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
