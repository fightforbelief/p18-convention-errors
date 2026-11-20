"""Quarterly panel of Midwest banks for a forecasting benchmark.

Design decisions
----------------
* Geography is assigned point-in-time from the headquarters ``history`` (as-of
  each quarter), not from the end-of-sample ``directory``.  The directory only
  lists survivors and their *current* state, so using it would introduce
  survivorship bias and misclassify banks that moved.  The directory is only a
  fallback for banks that have no history at all.
* Each bank is placed on a regular quarterly grid (calendar quarter ends).
  A bank-quarter is usable only if the bank was headquartered in the Midwest
  in that quarter.  Quarters outside the Midwest (or outside any HQ spell,
  e.g. after the bank closed) are hard breaks and are never filled.
* Short gaps (missing report or NaN value, at most ``MAX_GAP`` consecutive
  quarters) strictly inside a series are filled by linear interpolation.
  Longer gaps split the series into segments.
* The benchmark holds out each bank's last 4 quarters, so those must be real
  observations: a segment's tail is trimmed back until its final ``HOLDOUT``
  quarters are all genuinely reported (no imputed test targets).
* For each bank we keep its most recent segment that is long enough
  (``MIN_QUARTERS``) and sufficiently observed (``MIN_OBS_SHARE``); banks
  without such a segment are dropped.
"""
import numpy as np
import pandas as pd

MIDWEST = ["IL", "IN", "IA", "MI", "MN", "OH", "WI"]
MAX_GAP = 2           # longest run of missing quarters filled by interpolation
HOLDOUT = 4           # quarters held out by the benchmark (must be observed)
MIN_QUARTERS = 20     # minimum segment length (>= 16 training + 4 holdout)
MIN_OBS_SHARE = 0.8   # minimum share of genuinely observed quarters


def _yyyymmdd_to_qidx(x):
    """int YYYYMMDD -> integer quarter index (year*4 + quarter-1)."""
    x = np.asarray(x, dtype="int64")
    year = x // 10000
    month = (x // 100) % 100
    return year * 4 + (month - 1) // 3


def _qidx_to_qend_int(q):
    q = np.asarray(q, dtype="int64")
    year = q // 4
    qq = q % 4
    month = qq * 3 + 3
    day = np.where((month == 3) | (month == 12), 31, 30)
    return year * 10000 + month * 100 + day


def _qidx_to_timestamp(q):
    q = np.asarray(q, dtype="int64")
    year = q // 4
    qq = q % 4
    per = pd.PeriodIndex.from_fields(year=year, quarter=qq + 1, freq="Q")
    return per.to_timestamp(how="end").normalize()


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame,
                history: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]

    # ---------------- clean calls ----------------
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = pd.to_numeric(c["rssd_id"], errors="coerce")
    c["report_date"] = pd.to_numeric(c["report_date"], errors="coerce")
    c["value"] = pd.to_numeric(c["value"], errors="coerce")
    c = c.dropna(subset=["rssd_id", "report_date"])
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    # Snap to calendar quarter (report_date refers to a quarter end).
    c["q"] = _yyyymmdd_to_qidx(c["report_date"].to_numpy())
    # Duplicate filings for the same bank-quarter: keep the last non-missing one.
    c = c.sort_values(["rssd_id", "q", "report_date"], kind="mergesort")
    c["has"] = c["value"].notna()
    c = c.sort_values(["rssd_id", "q", "has"], kind="mergesort")
    c = c.drop_duplicates(["rssd_id", "q"], keep="last")[["rssd_id", "q", "value"]]
    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})[out_cols]

    # ---------------- regular quarterly grid per bank ----------------
    span = c.groupby("rssd_id")["q"].agg(["min", "max"])
    lens = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), lens)
    starts = np.repeat(span["min"].to_numpy(), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    g = pd.DataFrame({"rssd_id": ids, "q": starts + offs})
    g = g.merge(c, on=["rssd_id", "q"], how="left", sort=False)
    g = g.sort_values(["rssd_id", "q"], kind="mergesort").reset_index(drop=True)
    g["qend"] = _qidx_to_qend_int(g["q"].to_numpy())

    # ---------------- point-in-time headquarters state ----------------
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h["rssd_id"] = pd.to_numeric(h["rssd_id"], errors="coerce")
    h["start_date"] = pd.to_numeric(h["start_date"], errors="coerce")
    h["end_date"] = pd.to_numeric(h["end_date"], errors="coerce")
    h = h.dropna(subset=["rssd_id", "start_date"])
    h["rssd_id"] = h["rssd_id"].astype("int64")
    h["start_date"] = h["start_date"].astype("int64")
    h["end_date"] = h["end_date"].fillna(99999999).astype("int64")
    h["state"] = h["state"].astype(str).str.strip().str.upper()
    h = h.sort_values(["start_date", "rssd_id"], kind="mergesort")

    g = g.sort_values("qend", kind="mergesort")
    g = pd.merge_asof(g, h, left_on="qend", right_on="start_date",
                      by="rssd_id", direction="backward")
    in_spell = g["start_date"].notna() & (g["qend"] <= g["end_date"])
    g["state"] = g["state"].where(in_spell)

    # Fallback: banks with no history at all use the directory state.
    d = directory[["rssd_id", "state"]].copy()
    d["rssd_id"] = pd.to_numeric(d["rssd_id"], errors="coerce")
    d = d.dropna(subset=["rssd_id"])
    d["rssd_id"] = d["rssd_id"].astype("int64")
    d["state"] = d["state"].astype(str).str.strip().str.upper()
    d = d.drop_duplicates("rssd_id", keep="last").set_index("rssd_id")["state"]
    no_hist = ~g["rssd_id"].isin(h["rssd_id"].unique())
    g.loc[no_hist, "state"] = g.loc[no_hist, "rssd_id"].map(d)

    g = g.sort_values(["rssd_id", "q"], kind="mergesort").reset_index(drop=True)
    midwest = g["state"].isin(MIDWEST).to_numpy()
    obs = (g["value"].notna().to_numpy()) & midwest
    g = g[["rssd_id", "q", "value"]]
    g["value"] = g["value"].where(obs)

    # ---------------- segment on hard breaks ----------------
    rid = g["rssd_id"].to_numpy()
    new_bank = np.r_[True, rid[1:] != rid[:-1]]
    # runs of missing-but-Midwest quarters
    miss = midwest & ~obs
    run_start = miss & (new_bank | ~np.r_[False, miss[:-1]])
    run_id = np.cumsum(run_start)
    run_len = pd.Series(np.where(miss, run_id, 0)).map(
        pd.Series(run_id[miss]).value_counts()).fillna(0).to_numpy()
    hard = ~midwest | (miss & (run_len > MAX_GAP))
    keep = ~hard
    seg_start = keep & (new_bank | ~np.r_[False, keep[:-1]])
    seg = np.cumsum(seg_start)
    g["seg"] = seg
    g["obs"] = obs
    g = g[keep].reset_index(drop=True)

    # trim leading/trailing unobserved rows of each segment
    s = g["seg"]
    o = g["obs"].astype(int)
    g = g[(o.groupby(s).cumsum() > 0) &
          (o[::-1].groupby(s[::-1]).cumsum()[::-1] > 0)].reset_index(drop=True)

    # ---------------- interpolate short interior gaps ----------------
    pos = g.groupby("seg").cumcount().astype(float)
    posobs = pos.where(g["obs"])
    s = g["seg"]
    v0 = g["value"].groupby(s).ffill()
    v1 = g["value"].groupby(s).bfill()
    p0 = posobs.groupby(s).ffill()
    p1 = posobs.groupby(s).bfill()
    w = np.where(p1 > p0, (pos - p0) / (p1 - p0), 0.0)
    g["y"] = np.where(g["obs"], g["value"], v0 + (v1 - v0) * w)

    # ---------------- holdout must be genuinely observed ----------------
    full4 = (g["obs"].astype(int).groupby(g["seg"])
             .rolling(HOLDOUT, min_periods=HOLDOUT).sum()
             .reset_index(level=0, drop=True).sort_index() == HOLDOUT)
    rowno = pd.Series(np.arange(len(g)), index=g.index)
    last_ok = rowno.where(full4).groupby(g["seg"]).transform("max")
    g = g[rowno <= last_ok].reset_index(drop=True)

    # ---------------- segment quality filter, keep latest per bank ----------------
    stats = g.groupby("seg").agg(rssd_id=("rssd_id", "first"),
                                 n=("q", "size"), nobs=("obs", "sum"),
                                 qmax=("q", "max"))
    stats = stats[(stats["n"] >= MIN_QUARTERS) &
                  (stats["nobs"] / stats["n"] >= MIN_OBS_SHARE)]
    best = stats.sort_values(["rssd_id", "qmax"]).groupby("rssd_id").tail(1).index
    g = g[g["seg"].isin(best)]

    out = pd.DataFrame({
        "id": g["rssd_id"].astype("int64").to_numpy(),
        "date": _qidx_to_timestamp(g["q"].to_numpy()),
        "y": g["y"].astype(float).to_numpy(),
    })
    out = out.dropna(subset=["y"]).sort_values(["id", "date"]).reset_index(drop=True)
    return out[out_cols]
