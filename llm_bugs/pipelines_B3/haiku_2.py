"""Quarterly panel of Midwest banks built from Call Report ratios.

Only pandas and numpy are used.

Design choices
--------------
* Headquarters state is time-varying. Membership in the Midwest sample is decided
  per bank-quarter from ``history`` (spell by spell). The end-of-sample
  ``directory`` is NOT used to assign quarters, because it would put a bank's
  final state onto every quarter (look-ahead) and it drops banks that failed or
  merged before the end of the sample. Spell dates are mapped to calendar
  quarters, so 1230/1231-style end dates are handled, and where two spells share
  a quarter the later spell wins.
* Reports with a missing ``value`` are treated as not reported. Duplicate
  bank-quarter reports (none in the samples) are averaged.
* Each bank is put on a regular quarterly grid over the quarters between its
  first and last report, restricted to quarters in which it was headquartered
  in a Midwest state. Quarters with no usable value are interpolated linearly
  only when the run of missing quarters is at most ``MAX_INTERP_GAP`` long and
  both neighbours lie in the same unbroken Midwest run. Nothing is extrapolated.
* Banks with too little usable coverage are dropped: fewer than ``MIN_OBS``
  reported quarters, or a share of reported quarters below ``MIN_COVERAGE``
  within their Midwest reporting span (both measured before interpolation).
* Rows that still have no ``y`` are dropped, so the output has no missing ``y``.
"""

import numpy as np
import pandas as pd

MIDWEST = ("IL", "IN", "IA", "MI", "MN", "OH", "WI")
MAX_INTERP_GAP = 2      # quarters; longer runs of missing quarters stay missing
MIN_OBS = 20            # minimum number of reported quarters per bank
MIN_COVERAGE = 0.80     # minimum share of reported quarters in the bank's Midwest span

OUT_COLUMNS = ["id", "date", "y"]


def _to_int(s):
    """Coerce a column to int64 (YYYYMMDD dates or ids); unparseable values become NaN."""
    return pd.to_numeric(s, errors="coerce")


def _quarter_index(yyyymmdd):
    """Absolute calendar-quarter index (year*4 + quarter-1) from YYYYMMDD values."""
    a = np.asarray(yyyymmdd, dtype=np.int64)
    return (a // 10000) * 4 + ((a // 100) % 100 - 1) // 3


def _quarter_end_dates(qidx):
    """Quarter-end Timestamps from absolute quarter indices."""
    qidx = np.asarray(qidx, dtype=np.int64)
    periods = pd.PeriodIndex.from_fields(
        year=qidx // 4, quarter=qidx % 4 + 1, freq="Q-DEC"
    )
    return periods.to_timestamp(how="end").normalize()


def _empty():
    return pd.DataFrame({
        "id": pd.Series(dtype="int64"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "y": pd.Series(dtype="float64"),
    })


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Clean reported values
    # ------------------------------------------------------------------
    c = pd.DataFrame({
        "id": _to_int(calls["rssd_id"]),
        "date": _to_int(calls["report_date"]),
        "value": pd.to_numeric(calls["value"], errors="coerce"),
    })
    c = c.dropna(subset=["id", "date", "value"])
    c = c[np.isfinite(c["value"])]
    if c.empty:
        return _empty()
    c["id"] = c["id"].astype("int64")
    c["q"] = _quarter_index(c["date"])
    c = c.groupby(["id", "q"], as_index=False)["value"].mean()

    # ------------------------------------------------------------------
    # 2. Headquarters spells -> quarter ranges
    # ------------------------------------------------------------------
    h = pd.DataFrame({
        "id": _to_int(history["rssd_id"]),
        "start": _to_int(history["start_date"]),
        "end": _to_int(history["end_date"]),
        "state": history["state"].astype("string").str.strip().str.upper(),
    })
    h = h.dropna(subset=["id", "start", "state"])
    h["id"] = h["id"].astype("int64")
    h["sq"] = _quarter_index(h["start"])
    # An empty end_date means the spell is still open.
    h["eq"] = np.where(h["end"].isna(), np.iinfo(np.int64).max // 8,
                       _quarter_index(h["end"].fillna(0).astype("int64")))
    h = h[h["eq"] >= h["sq"]]

    # Each bank's reporting range; quarters outside it have nothing to fill.
    span = c.groupby("id")["q"].agg(lo="min", hi="max").reset_index()
    h = h.merge(span, on="id", how="inner")
    h["sq"] = np.maximum(h["sq"], h["lo"])
    h["eq"] = np.minimum(h["eq"], h["hi"])
    h = h[h["sq"] <= h["eq"]].sort_values(["id", "sq"], kind="stable").reset_index(drop=True)
    if h.empty:
        return _empty()

    # Expand spells to bank-quarters; a later spell overrides an earlier one in
    # the same quarter (handles 1230 end dates followed by 1231 start dates).
    lengths = (h["eq"] - h["sq"] + 1).to_numpy()
    rep = np.repeat(np.arange(len(h)), lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    grid = pd.DataFrame({
        "id": h["id"].to_numpy()[rep],
        "q": h["sq"].to_numpy()[rep] + offsets,
        "state": h["state"].to_numpy()[rep],
        "order": rep,
    })
    grid = grid.sort_values(["id", "q", "order"], kind="stable").drop_duplicates(["id", "q"], keep="last")
    grid = grid[grid["state"].isin(MIDWEST)].drop(columns=["state", "order"])

    # ------------------------------------------------------------------
    # 3. Attach reported values and check coverage
    # ------------------------------------------------------------------
    grid = grid.merge(c, on=["id", "q"], how="left")
    grid = grid.sort_values(["id", "q"], kind="stable").reset_index(drop=True)

    g = grid.groupby("id")["value"]
    n_grid = g.size()
    n_obs = g.count()
    keep_ids = n_obs.index[(n_obs >= MIN_OBS) & (n_obs / n_grid >= MIN_COVERAGE)]
    grid = grid[grid["id"].isin(keep_ids)].reset_index(drop=True)
    if grid.empty:
        return _empty()

    # ------------------------------------------------------------------
    # 4. Regularise: limited linear interpolation inside unbroken Midwest runs
    # ------------------------------------------------------------------
    ids = grid["id"].to_numpy()
    q = grid["q"].to_numpy()
    y = grid["value"].to_numpy(dtype=float)
    obs = ~np.isnan(y)

    new_run = np.ones(len(grid), dtype=bool)
    new_run[1:] = (ids[1:] != ids[:-1]) | (np.diff(q) != 1)
    run = np.cumsum(new_run)

    qs = pd.Series(q, dtype="float64")
    ys = pd.Series(y)
    obs_q = qs.where(obs)
    prev_q = obs_q.groupby(run).ffill()
    next_q = obs_q.groupby(run).bfill()
    prev_y = ys.where(obs).groupby(run).ffill()
    next_y = ys.where(obs).groupby(run).bfill()

    gap = next_q - prev_q                           # spacing between surrounding observations
    fillable = (~obs) & prev_q.notna() & next_q.notna() & (gap - 1 <= MAX_INTERP_GAP)
    weight = (qs - prev_q) / gap.where(gap > 0)
    interp = prev_y + (next_y - prev_y) * weight
    y_filled = np.where(fillable.to_numpy(), interp.to_numpy(), y)

    grid["y"] = y_filled

    # ------------------------------------------------------------------
    # 5. Finalise: drop rows without y, format output
    # ------------------------------------------------------------------
    out = grid.loc[grid["y"].notna(), ["id", "q", "y"]].copy()
    if out.empty:
        return _empty()
    out["date"] = _quarter_end_dates(out["q"].to_numpy())
    out = out[OUT_COLUMNS].sort_values(["id", "date"], kind="stable").reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    return out
