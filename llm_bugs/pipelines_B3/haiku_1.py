"""Quarterly panel of Midwest bank call-report ratios.

Only pandas and numpy are used.

Design choices
--------------
* Headquarters state is classified point-in-time from ``history``. The
  ``directory`` is an end-of-sample snapshot: it omits banks that failed or
  merged before the end of the sample and it reflects headquarters moves that
  happened after the quarter being classified. Using it for selection would
  introduce survivorship and look-ahead bias, so it is not used for membership.
* A bank-quarter is in the panel only if its headquarters spell covering that
  quarter-end is in one of the Midwest states. When spells overlap (for example
  a spell ending 19971230 next to one starting 19971231), the spell that started
  latest wins.
* Quarters are mapped to a calendar quarter index, so ``report_date`` values
  anywhere in a quarter map to that quarter.
* ``value`` NaN means not reported and is treated as missing. Duplicate filings
  for the same bank-quarter keep the latest ``report_date`` with a value.
* Banks with too little data are dropped. A bank needs at least ``MIN_OBS``
  observed Midwest quarters and an observed share of at least ``MIN_COVERAGE``
  over the span between its first and last observed quarter.
* Gaps of up to ``MAX_FILL`` consecutive missing quarters inside a run of
  Midwest-membership quarters are filled by linear interpolation. Longer gaps
  are left unfilled and their quarters are dropped, so no output ``y`` is
  missing and no value is interpolated across a period outside Midwest
  membership.
"""

import numpy as np
import pandas as pd

MIDWEST = ("IL", "IN", "IA", "MI", "MN", "OH", "WI")
MIN_OBS = 20
MIN_COVERAGE = 0.75
MAX_FILL = 2

OUT_COLUMNS = ["id", "date", "y"]


def _quarter_index(dates):
    """Map YYYYMMDD integers to an absolute quarter index (year * 4 + quarter - 1)."""
    d = np.asarray(dates, dtype="int64")
    year = d // 10000
    month = (d // 100) % 100
    return year * 4 + (month - 1) // 3


def _quarter_end_dates(qidx):
    """Convert absolute quarter indices to calendar quarter-end Timestamps."""
    qidx = np.asarray(qidx, dtype="int64")
    year = qidx // 4
    month = 3 * (qidx % 4 + 1)
    frame = pd.DataFrame({"year": year, "month": month, "day": 1})
    return (pd.to_datetime(frame) + pd.offsets.MonthEnd(0)).to_numpy().astype("datetime64[ns]")


def _fill_row(y, mask, cum):
    """Return (positions, values) of the quarters to keep for one bank.

    ``y`` holds observed values (NaN where missing) and ``mask`` marks the
    quarters in which the bank is Midwest-headquartered. ``cum`` is the cumulative
    sum of ``mask`` with a leading zero. Gaps of at most MAX_FILL quarters
    between two observations are interpolated when every quarter in the gap is
    inside the mask. Quarters that are still missing are dropped.
    """
    obs = np.flatnonzero(mask & ~np.isnan(y))
    if obs.size == 0:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="float64")

    keep_pos = [obs]
    keep_val = [y[obs]]
    if obs.size > 1:
        p = obs[:-1]
        q = obs[1:]
        gap = q - p - 1
        contiguous = (cum[q + 1] - cum[p]) == (q - p + 1)
        fillable = np.flatnonzero((gap > 0) & (gap <= MAX_FILL) & contiguous)
        for i in fillable:
            a, b = p[i], q[i]
            pos = np.arange(a + 1, b)
            frac = (pos - a) / (b - a)
            keep_pos.append(pos)
            keep_val.append(y[a] + frac * (y[b] - y[a]))

    positions = np.concatenate(keep_pos)
    values = np.concatenate(keep_val)
    order = np.argsort(positions, kind="stable")
    return positions[order], values[order]


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Build the quarterly panel of Midwest-headquartered banks.

    Parameters
    ----------
    calls : DataFrame with ``rssd_id``, ``report_date`` (YYYYMMDD int) and
        ``value`` (float, NaN if not reported).
    directory : DataFrame with ``rssd_id`` and ``state``. Accepted for interface
        compatibility. It is not used for membership because it is an
        end-of-sample snapshot (see module docstring).
    history : DataFrame with ``rssd_id``, ``start_date`` (YYYYMMDD),
        ``end_date`` (YYYYMMDD, empty while ongoing) and ``state``.

    Returns
    -------
    DataFrame with columns ``id`` (int), ``date`` (quarter-end Timestamp) and
    ``y`` (float), sorted by ``id`` and ``date``.
    """
    # ---- 1. Clean the reported ratios -------------------------------------
    c = pd.DataFrame(
        {
            "id": pd.to_numeric(calls["rssd_id"], errors="coerce").astype("int64"),
            "rd": pd.to_numeric(calls["report_date"], errors="coerce").astype("int64"),
            "y": pd.to_numeric(calls["value"], errors="coerce").astype("float64"),
        }
    )
    c = c[np.isfinite(c["y"])]
    if c.empty:
        return pd.DataFrame({k: pd.Series(dtype=t) for k, t in
                             [("id", "int64"), ("date", "datetime64[ns]"), ("y", "float64")]})
    c["yq"] = _quarter_index(c["rd"].to_numpy())
    c = c.sort_values(["id", "yq", "rd"], kind="stable").drop_duplicates(["id", "yq"], keep="last")

    ids = np.sort(c["id"].unique())
    id_index = pd.Index(ids)
    nb = len(ids)
    q_lo = int(c["yq"].min())
    q_hi = int(c["yq"].max())
    nq = q_hi - q_lo + 1

    # ---- 2. Point-in-time Midwest headquarters membership -----------------
    h = pd.DataFrame(
        {
            "id": pd.to_numeric(history["rssd_id"], errors="coerce"),
            "start": pd.to_numeric(history["start_date"], errors="coerce"),
            "end": pd.to_numeric(history["end_date"], errors="coerce"),
            "state": history["state"].astype(str).str.strip().str.upper(),
        }
    ).dropna(subset=["id", "start"])
    h["id"] = h["id"].astype("int64")
    h["start"] = h["start"].astype("int64")
    h = h[h["id"].isin(ids)]
    h = h.sort_values("start", kind="stable")  # later spells overwrite earlier ones

    # An open spell (empty end_date) runs to the end of the grid.
    s_q = _quarter_index(h["start"].to_numpy())
    open_end = h["end"].isna().to_numpy()
    e_q = np.where(open_end, q_hi, _quarter_index(np.nan_to_num(h["end"].to_numpy(), nan=0).astype("int64")))
    s_q = np.clip(s_q, q_lo, q_hi + 1) - q_lo          # 0-based grid positions
    e_q = np.clip(e_q, q_lo - 1, q_hi) - q_lo          # inclusive
    is_mw = h["state"].isin(MIDWEST).to_numpy()
    rows = id_index.get_indexer(h["id"].to_numpy())

    member = np.zeros((nb, nq), dtype=bool)
    for r, a, b, flag in zip(rows, s_q, e_q, is_mw):
        if b >= a:
            member[r, a:b + 1] = flag

    # ---- 3. Observed ratios on the quarterly grid -------------------------
    Y = np.full((nb, nq), np.nan)
    Y[id_index.get_indexer(c["id"].to_numpy()), c["yq"].to_numpy() - q_lo] = c["y"].to_numpy()

    obs_in = member & ~np.isnan(Y)
    n_obs = obs_in.sum(axis=1)
    any_obs = obs_in.any(axis=1)
    first = np.where(any_obs, obs_in.argmax(axis=1), 0)
    last = np.where(any_obs, nq - 1 - obs_in[:, ::-1].argmax(axis=1), 0)
    cum = np.concatenate([np.zeros((nb, 1), dtype="int64"), np.cumsum(member, axis=1)], axis=1)
    span = cum[np.arange(nb), last + 1] - cum[np.arange(nb), first]
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(span > 0, n_obs / span, 0.0)
    keep = any_obs & (n_obs >= MIN_OBS) & (coverage >= MIN_COVERAGE)

    # ---- 4. Fill short gaps and build the long-format output --------------
    out_id, out_q, out_y = [], [], []
    for r in np.flatnonzero(keep):
        pos, val = _fill_row(Y[r], member[r], cum[r])
        if pos.size == 0:
            continue
        out_id.append(np.full(pos.size, ids[r], dtype="int64"))
        out_q.append(pos + q_lo)
        out_y.append(val)

    if not out_id:
        return pd.DataFrame({k: pd.Series(dtype=t) for k, t in
                             [("id", "int64"), ("date", "datetime64[ns]"), ("y", "float64")]})

    out = pd.DataFrame(
        {
            "id": np.concatenate(out_id),
            "date": _quarter_end_dates(np.concatenate(out_q)),
            "y": np.concatenate(out_y).astype("float64"),
        }
    )
    out = out.sort_values(["id", "date"], kind="stable").reset_index(drop=True)
    return out[OUT_COLUMNS]
