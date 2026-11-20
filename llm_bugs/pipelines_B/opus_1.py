"""Build a clean quarterly panel of commercial banks from Call Report data.

Rules (all vectorised; runs in seconds on ~2M rows):

1. Dates: report_date (YYYYMMDD) is mapped to its calendar quarter. Unparseable
   dates and +/-inf values are dropped / treated as missing. If a bank files more
   than once in a quarter, the latest non-missing value in that quarter is kept.
2. Charter (point in time): each bank-quarter gets the charter from the latest
   attrs snapshot with as_of <= quarter end (quarters before the first snapshot
   take the first snapshot). Banks with no attrs are dropped. A bank's history is
   split into charter spells and only its most recent *commercial* spell is kept,
   so the series is contiguous and savings-institution periods never enter.
3. Regular frequency: within the spell, the series is laid on a complete
   quarterly grid from its first to its last non-missing value. Missing
   quarters (absent rows or NaN values) are runs of gap quarters.
   - Runs longer than MAX_GAP quarters break the series; only the most recent
     unbroken segment is kept (the benchmark needs the series' end).
   - Shorter runs are filled by linear interpolation, except where the next
     observed value lies in the 4-quarter holdout window; those are
     forward-filled instead so no holdout information leaks into training.
   Gaps inside the holdout window itself are interpolated from holdout values.
4. Coverage filter: a bank is kept only if (a) at least 80% of the quarters of
   its commercial spell were observed, (b) its kept segment has at least
   MIN_QUARTERS quarters with at most 20% imputed, and (c) at most
   MAX_HOLDOUT_IMPUTED of its last HOLDOUT quarters are imputed (the final
   quarter is always a real observation).
"""

import numpy as np
import pandas as pd

HOLDOUT = 4             # quarters held out at the end of each series
MAX_GAP = 2             # longest run of missing quarters we impute
MIN_QUARTERS = 20       # minimum series length (16 training + 4 holdout)
MAX_IMPUTED_SHARE = 0.2 # maximum share of missing quarters (spell and kept series)
MAX_HOLDOUT_IMPUTED = 1 # at most this many holdout quarters may be interpolated


def _to_qindex(yyyymmdd: pd.Series) -> pd.Series:
    """YYYYMMDD ints -> integer quarter index year*4 + (quarter-1); NaN if bad."""
    d = pd.to_datetime(
        pd.to_numeric(yyyymmdd, errors="coerce").astype("Int64").astype("string"),
        format="%Y%m%d", errors="coerce",
    )
    return d.dt.year * 4 + (d.dt.month - 1) // 3, d


def _qindex_to_quarter_end(qi: np.ndarray) -> pd.DatetimeIndex:
    year = qi // 4
    month = (qi % 4) * 3 + 3
    first = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))
    return (first + pd.offsets.MonthEnd(0)).dt.normalize()


def _empty() -> pd.DataFrame:
    return pd.DataFrame({
        "id": pd.Series(dtype="int64"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "y": pd.Series(dtype="float64"),
    })


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- 1. calls -> one row per bank-quarter -------------------------------
    c = pd.DataFrame({
        "id": pd.to_numeric(calls["rssd_id"], errors="coerce"),
        "value": pd.to_numeric(calls["value"], errors="coerce").astype("float64"),
    })
    c["qi"], c["d"] = _to_qindex(calls["report_date"])
    c = c.dropna(subset=["id", "qi"])
    c["id"] = c["id"].astype("int64")
    c["qi"] = c["qi"].astype("int64")
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    # latest non-missing value within the quarter (groupby.last skips NaN)
    c = c.sort_values(["id", "qi", "d"], kind="mergesort")
    c = c.groupby(["id", "qi"], sort=True, as_index=False)["value"].last()
    if c.empty:
        return _empty()

    # ---- 2. point-in-time charter -------------------------------------------
    a = pd.DataFrame({
        "id": pd.to_numeric(attrs["rssd_id"], errors="coerce"),
        "charter": attrs["charter"].astype("string").str.strip().str.lower(),
    })
    _, a["asof"] = _to_qindex(attrs["as_of"])
    a = a.dropna(subset=["id", "asof", "charter"])
    a["id"] = a["id"].astype("int64")
    a = a.sort_values(["asof", "id"], kind="mergesort")
    a = a.drop_duplicates(["id", "asof"], keep="last")

    c["qend"] = _qindex_to_quarter_end(c["qi"].to_numpy()).to_numpy()
    c = c.sort_values("qend", kind="mergesort")
    c = pd.merge_asof(c, a, left_on="qend", right_on="asof", by="id",
                      direction="backward")
    first = a.sort_values(["id", "asof"]).groupby("id")["charter"].first()
    c["charter"] = c["charter"].fillna(c["id"].map(first))
    c = c.dropna(subset=["charter"])  # banks without any attrs
    c = c.sort_values(["id", "qi"], kind="mergesort").reset_index(drop=True)

    # charter spells; keep each bank's last commercial spell
    new_spell = (c["id"].ne(c["id"].shift())) | (c["charter"].ne(c["charter"].shift()))
    c["spell"] = np.cumsum(new_spell.to_numpy(dtype=bool))
    c = c[c["charter"].eq("commercial").to_numpy(dtype=bool)]
    last_spell = c.groupby("id")["spell"].transform("max")
    c = c[c["spell"].eq(last_spell)]
    c = c.dropna(subset=["value"])[["id", "qi", "value"]]
    if c.empty:
        return _empty()

    # ---- 3. regular quarterly grid -------------------------------------------
    rng = c.groupby("id")["qi"].agg(["min", "max"])
    n = (rng["max"] - rng["min"] + 1).to_numpy()
    gid = np.repeat(rng.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    gqi = np.repeat(rng["min"].to_numpy(), n) + offs
    g = pd.DataFrame({"id": gid, "qi": gqi})
    g = g.merge(c, on=["id", "qi"], how="left")  # keeps grid order
    obs = g["value"].notna().to_numpy()
    # coverage over the whole commercial spell, before any trimming
    spell_cov = pd.Series(obs).groupby(g["id"].to_numpy()).mean()

    # length of the missing run each grid row belongs to
    run_id = np.cumsum(obs)  # missing rows share the run id of the preceding obs
    key = pd.Series(run_id + 0)  # ids/runs nest because each series starts observed
    miss = pd.Series(~obs)
    run_len = miss.groupby(key).transform("sum").to_numpy()
    # break series at long gaps: keep rows after the last long gap per bank
    long_gap = (~obs) & (run_len > MAX_GAP)
    seg = pd.Series(long_gap).groupby(g["id"]).cumsum().to_numpy()
    last_seg = pd.Series(seg).groupby(g["id"]).transform("max").to_numpy()
    keep = (seg == last_seg) & ~long_gap
    g = g[keep].reset_index(drop=True)
    obs = obs[keep]

    # position from the end of the series (0 = last quarter)
    pos_from_end = g.groupby("id").cumcount(ascending=False).to_numpy()

    # interpolation inputs: previous/next observation (series start/end observed)
    pos = np.arange(len(g), dtype="float64")
    obs_pos = pd.Series(np.where(obs, pos, np.nan))
    prev_p = obs_pos.ffill().to_numpy()
    next_p = obs_pos.bfill().to_numpy()
    vals = g["value"].to_numpy()
    prev_v = pd.Series(vals).ffill().to_numpy()
    next_v = pd.Series(vals).bfill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(next_p > prev_p, (pos - prev_p) / (next_p - prev_p), 0.0)
    interp = prev_v + w * (next_v - prev_v)
    next_in_holdout = pos_from_end[np.nan_to_num(next_p, nan=0).astype(np.int64)] < HOLDOUT
    leak = next_in_holdout & (pos_from_end >= HOLDOUT)  # training row, holdout endpoint
    fill = np.where(leak, prev_v, interp)
    g["y"] = np.where(obs, vals, fill)

    # ---- 4. coverage filter ---------------------------------------------------
    stats = pd.DataFrame({
        "id": g["id"],
        "imputed": ~obs,
        "holdout_missing": (~obs) & (pos_from_end < HOLDOUT),
    }).groupby("id").agg(n=("imputed", "size"), imp=("imputed", "mean"),
                         hm=("holdout_missing", "sum"))
    stats["spell_cov"] = spell_cov.reindex(stats.index)
    good = stats.index[(stats["n"] >= MIN_QUARTERS)
                       & (stats["imp"] <= MAX_IMPUTED_SHARE)
                       & (stats["spell_cov"] >= 1 - MAX_IMPUTED_SHARE)
                       & (stats["hm"] <= MAX_HOLDOUT_IMPUTED)]
    g = g[g["id"].isin(good)]
    if g.empty:
        return _empty()

    out = pd.DataFrame({
        "id": g["id"].astype("int64").to_numpy(),
        "date": _qindex_to_quarter_end(g["qi"].to_numpy()).to_numpy(),
        "y": g["y"].astype("float64").to_numpy(),
    })
    return out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    calls = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sample_calls.csv")
    attrs = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "sample_attrs.csv")
    panel = build_panel(calls, attrs)
    print(panel.head())
    print(f"{len(panel)} rows, {panel['id'].nunique()} banks")
