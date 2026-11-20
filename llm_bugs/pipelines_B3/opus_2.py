"""Quarterly panel of Midwest banks for a hold-out-last-4-quarters forecasting benchmark.

Design decisions
----------------
* Geography is point-in-time: a bank-quarter counts as Midwest if the bank's headquarters
  (per the `history` spells, end dates inclusive) was in a Midwest state *at that quarter*.
  The `directory` is an end-of-sample snapshot of surviving banks only, so using it would both
  drop failed/merged banks (survivorship bias) and mislabel banks that moved. It is used only
  as a fallback for banks with no history record at all.
* Reports with a missing value are treated as missing quarters.
* Gaps of up to MAX_INTERP_GAP consecutive missing quarters are filled by linear interpolation.
  A longer gap (or leaving / re-entering the Midwest) breaks the series; only each bank's most
  recent contiguous segment is kept, so the series ends at the bank's latest real observation.
* Coverage filter: the kept segment must span at least MIN_QUARTERS quarters, have at least
  MIN_OBS_SHARE of its quarters actually reported, and its final HOLDOUT quarters (the
  benchmark's test window) must all be real observations, not interpolated values.
"""
import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}
MAX_INTERP_GAP = 2      # longest run of missing quarters we fill
HOLDOUT = 4             # quarters held out by the benchmark
MIN_QUARTERS = 24       # >= 20 training quarters + 4 hold-out quarters
MIN_OBS_SHARE = 0.80    # share of segment quarters that must be genuinely reported


def _to_quarter_index(yyyymmdd: pd.Series) -> np.ndarray:
    v = pd.to_numeric(yyyymmdd, errors="coerce").astype("Int64")
    years = (v // 10000).astype("float")
    months = ((v // 100) % 100).astype("float")
    return (years * 4 + (months - 1) // 3).to_numpy()  # float, NaN-safe


def _state_at(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
    """Headquarters state of each call row at its report date."""
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h["rssd_id"] = h["rssd_id"].astype("int64")
    h["start_date"] = pd.to_numeric(h["start_date"], errors="coerce")
    h["end_date"] = pd.to_numeric(h["end_date"], errors="coerce").fillna(99991231)
    h = h.dropna(subset=["start_date"]).sort_values("start_date", kind="mergesort")
    h["start_date"] = h["start_date"].astype("int64")

    left = calls[["rssd_id", "report_date"]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("report_date", kind="mergesort")
    m = pd.merge_asof(left, h, left_on="report_date", right_on="start_date",
                      by="rssd_id", direction="backward")
    m.loc[m["report_date"] > m["end_date"], "state"] = np.nan  # outside any spell
    state = m.sort_values("_row")["state"].to_numpy(dtype=object)

    # Fallback: banks with no history at all -> directory (end-of-sample) state.
    no_hist = ~calls["rssd_id"].isin(h["rssd_id"]).to_numpy()
    if no_hist.any():
        dmap = directory.drop_duplicates("rssd_id", keep="last").set_index("rssd_id")["state"]
        state[no_hist] = calls["rssd_id"][no_hist].map(dmap).to_numpy(dtype=object)
    return state


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = pd.to_numeric(c["rssd_id"], errors="coerce")
    c["report_date"] = pd.to_numeric(c["report_date"], errors="coerce")
    c["value"] = pd.to_numeric(c["value"], errors="coerce")
    c = c.dropna(subset=["rssd_id", "report_date"])
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c = c[np.isfinite(c["value"])].reset_index(drop=True)
    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    # Point-in-time Midwest filter.
    st = _state_at(c, directory, history)
    c = c[pd.Series(st).isin(MIDWEST).to_numpy()]

    # Quarter index; if a bank has several reports in one quarter keep the latest-dated.
    c = c.assign(q=_to_quarter_index(c["report_date"]))
    c = c.dropna(subset=["q"])
    c["q"] = c["q"].astype("int64")
    c = (c.sort_values(["rssd_id", "q", "report_date"], kind="mergesort")
          .drop_duplicates(["rssd_id", "q"], keep="last"))

    # Split each bank's series at long gaps; keep the most recent segment.
    gap = c.groupby("rssd_id")["q"].diff() - 1
    brk = (gap.isna() | (gap > MAX_INTERP_GAP)).astype("int64")
    c["seg"] = brk.groupby(c["rssd_id"]).cumsum()
    c = c[c["seg"] == c.groupby("rssd_id")["seg"].transform("max")]

    # Coverage filters on the kept segment.
    g = c.groupby("rssd_id")["q"]
    qmin, qmax, nobs = g.transform("min"), g.transform("max"), g.transform("size")
    span = qmax - qmin + 1
    # last HOLDOUT quarters all reported <=> the HOLDOUT-th latest obs sits at qmax-HOLDOUT+1
    rank_from_end = c.groupby("rssd_id")["q"].rank(ascending=False, method="first")
    hold_ok = ((rank_from_end == HOLDOUT) & (c["q"] == qmax - HOLDOUT + 1))
    hold_ok = hold_ok.groupby(c["rssd_id"]).transform("any")
    keep = (span >= MIN_QUARTERS) & (nobs / span >= MIN_OBS_SHARE) & hold_ok
    c = c[keep]
    if c.empty:
        return build_panel(calls.iloc[0:0], directory, history)

    # Regular quarterly grid + linear interpolation of the (short) interior gaps.
    bounds = c.groupby("rssd_id")["q"].agg(["min", "max"])
    lens = (bounds["max"] - bounds["min"] + 1).to_numpy()
    ids = np.repeat(bounds.index.to_numpy(), lens)
    starts = np.repeat(bounds["min"].to_numpy(), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    grid = pd.DataFrame({"rssd_id": ids, "q": starts + offs})
    grid = grid.merge(c[["rssd_id", "q", "value"]], on=["rssd_id", "q"], how="left")

    # Linear interpolation within bank: both ends of every segment are observed,
    # so a global interpolation never bleeds across banks; still, do it by position per bank.
    v = grid["value"].to_numpy(dtype=float, copy=True)
    miss = np.isnan(v)
    if miss.any():
        x = grid["q"].to_numpy(dtype=float) + grid["rssd_id"].to_numpy(dtype=float) * 1e6
        v[miss] = np.interp(x[miss], x[~miss], v[~miss])
    grid["value"] = v

    q = grid["q"].to_numpy()
    year, qtr = q // 4, q % 4 + 1
    dates = (pd.PeriodIndex.from_fields(year=year, quarter=qtr, freq="Q")
             .to_timestamp(how="end").normalize())
    out = pd.DataFrame({"id": grid["rssd_id"].astype("int64").to_numpy(),
                        "date": np.asarray(dates, dtype="datetime64[ns]"), "y": grid["value"].astype(float).to_numpy()})
    out = out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    return out[out_cols]
