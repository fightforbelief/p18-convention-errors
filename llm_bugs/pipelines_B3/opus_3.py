"""Quarterly panel of Midwest banks for a forecasting benchmark.

Design decisions
----------------
* Geography is assigned point-in-time from the headquarters ``history`` table,
  not from ``directory``.  The directory is an end-of-sample snapshot: it only
  lists banks still alive at the end (survivorship bias) and only their *final*
  state (a bank that moved IA -> TX would be wrongly dropped / a bank that moved
  NY -> IA wrongly kept for its NY years).  The directory is used only as a
  fallback for banks that have no history rows at all.
* A bank-quarter is in the panel only while the bank is headquartered in a
  Midwest state.  Quarters spent outside the Midwest (or outside any recorded
  spell) are hard breaks in the series.
* Data are placed on a regular quarterly grid.  Short internal gaps (missing
  report or NaN value, at most ``MAX_GAP`` consecutive quarters) are filled by
  linear interpolation between the neighbouring observed values.  Longer gaps
  split the series; only the most recent contiguous segment is kept, so each
  bank yields one gap-free series ending at its latest observation.
  Nothing is extrapolated: segments start and end on observed values.
* Coverage filter: a segment is kept only if it is long enough to train on
  and still hold out the last 4 quarters (``MIN_QUARTERS``), if most of it is
  actually observed (``MIN_OBS_SHARE``), and if the 4 hold-out quarters are
  essentially real data (at most ``MAX_IMPUTED_HOLDOUT`` imputed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}
MAX_GAP = 2               # longest run of missing quarters that is interpolated
HOLDOUT = 4
MIN_QUARTERS = 24         # 20 training quarters + 4 hold-out quarters
MIN_OBS_SHARE = 0.8       # share of the kept segment that is actually observed
MAX_IMPUTED_HOLDOUT = 1   # imputed values allowed in the hold-out window


def _to_date(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    out = pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return out


def _qindex(d: pd.Series) -> np.ndarray:
    return (d.dt.year * 4 + (d.dt.month - 1) // 3).to_numpy()


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"id": pd.Series(dtype="int64"),
                         "date": pd.Series(dtype="datetime64[ns]"),
                         "y": pd.Series(dtype="float64")})


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------ calls
    c = pd.DataFrame({
        "id": pd.to_numeric(calls["rssd_id"], errors="coerce"),
        "date": _to_date(calls["report_date"]),
        "value": pd.to_numeric(calls["value"], errors="coerce").astype(float),
    })
    c = c.dropna(subset=["id", "date"])
    c["id"] = c["id"].astype("int64")
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    c["q"] = _qindex(c["date"])  # snaps any non-quarter-end date to its quarter
    # duplicates: keep the last non-missing value reported for the bank-quarter
    c = (c.sort_values(["id", "q"], kind="stable")
           .groupby(["id", "q"], sort=True)["value"].last()  # last() skips NaN
           .reset_index())
    if c.empty:
        return _empty()

    # ------------------------------------------------------------ regular grid
    span = c.groupby("id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    gid = np.repeat(span.index.to_numpy(), n)
    start = np.repeat(span["min"].to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    g = pd.DataFrame({"id": gid, "q": start + offs})
    g = g.merge(c, on=["id", "q"], how="left")
    yr, qq = np.divmod(g["q"].to_numpy(), 4)
    g["date"] = (pd.PeriodIndex.from_fields(year=yr, quarter=qq + 1, freq="Q")
                 .to_timestamp(how="end").normalize())

    # ------------------------------------------------------------ point-in-time state
    h = pd.DataFrame({
        "id": pd.to_numeric(history["rssd_id"], errors="coerce"),
        "start": _to_date(history["start_date"]),
        "end": _to_date(history["end_date"]),
        "state": history["state"].astype(str).str.strip().str.upper(),
    }).dropna(subset=["id", "start"])
    h["id"] = h["id"].astype("int64")
    h = h.sort_values(["start", "id"])

    g = g.sort_values(["date", "id"])
    g = pd.merge_asof(g, h, left_on="date", right_on="start", by="id", direction="backward")
    alive = g["start"].notna() & (g["end"].isna() | (g["date"] <= g["end"]))
    g["state"] = g["state"].where(alive)

    # fallback for banks with no history at all: end-of-sample directory state
    no_hist = ~g["id"].isin(h["id"].unique())
    if no_hist.any():
        d = pd.DataFrame({"id": pd.to_numeric(directory["rssd_id"], errors="coerce"),
                          "state": directory["state"].astype(str).str.strip().str.upper()})
        d = d.dropna(subset=["id"]).drop_duplicates("id", keep="last")
        dmap = pd.Series(d["state"].to_numpy(), index=d["id"].astype("int64").to_numpy())
        g.loc[no_hist, "state"] = g.loc[no_hist, "id"].map(dmap)

    g = g.loc[g["state"].isin(MIDWEST), ["id", "q", "date", "value"]]
    g = g.sort_values(["id", "q"]).reset_index(drop=True)
    if g.empty:
        return _empty()

    # ------------------------------------------------------------ segments
    obs = g["value"].notna().to_numpy()
    ids = g["id"].to_numpy()
    q = g["q"].to_numpy()
    new_bank = np.r_[True, ids[1:] != ids[:-1]]
    # break where quarters are not consecutive (left Midwest / no spell)
    hard = new_bank | np.r_[True, np.diff(q) != 1]
    run = np.cumsum(hard)

    # length of each run of consecutive missing values (within a hard run)
    miss_start = (~obs) & (hard | np.r_[True, obs[:-1]])
    miss_id = np.cumsum(miss_start)
    miss_len = pd.Series(np.where(obs, 0, 1)).groupby(np.where(obs, -1, miss_id)).transform("sum").to_numpy()
    long_gap = (~obs) & (miss_len > MAX_GAP)

    # drop long-gap quarters; they split the series
    keep = ~long_gap
    g = g.loc[keep].reset_index(drop=True)
    obs = obs[keep]
    q = g["q"].to_numpy()
    ids = g["id"].to_numpy()
    run = run[keep]
    brk = np.r_[True, (ids[1:] != ids[:-1]) | (run[1:] != run[:-1]) | (np.diff(q) != 1)]
    g["seg"] = np.cumsum(brk)

    # trim leading / trailing missing values inside each segment (no extrapolation)
    g["obs"] = obs
    cs = g.groupby("seg")["obs"].cumsum()
    rcs = g[::-1].groupby("seg")["obs"].cumsum()[::-1]
    g = g.loc[(cs > 0) & (rcs > 0)]

    # most recent segment per bank
    last_seg = g.groupby("id")["seg"].transform("max")
    g = g.loc[g["seg"] == last_seg].copy()

    # ------------------------------------------------------------ interpolation
    g["qo"] = g["q"].where(g["obs"])
    prev_v = g.groupby("seg")["value"].ffill()
    next_v = g.groupby("seg")["value"].bfill()
    prev_q = g.groupby("seg")["qo"].ffill()
    next_q = g.groupby("seg")["qo"].bfill()
    w = ((g["q"] - prev_q) / (next_q - prev_q)).fillna(0.0)
    g["y"] = np.where(g["obs"], g["value"], prev_v + (next_v - prev_v) * w)

    # ------------------------------------------------------------ coverage filter
    grp = g.groupby("id")
    length = grp["q"].transform("size")
    share = grp["obs"].transform("mean")
    pos_from_end = grp.cumcount(ascending=False)
    imp_hold = (~g["obs"] & (pos_from_end < HOLDOUT)).groupby(g["id"]).transform("sum")
    ok = (length >= MIN_QUARTERS) & (share >= MIN_OBS_SHARE) & (imp_hold <= MAX_IMPUTED_HOLDOUT)
    g = g.loc[ok & g["y"].notna()]

    out = g[["id", "date", "y"]].sort_values(["id", "date"]).reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    return out
