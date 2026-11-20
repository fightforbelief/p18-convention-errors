"""Quarterly panel of Midwest banks for a 4-quarter holdout forecasting benchmark.

Public entry point: build_panel(calls, directory, history) -> DataFrame[id, date, y].

Design choices
--------------
* Headquarters state is taken point-in-time from `history`, not from `directory`.
  The directory is the end-of-sample institution list, so using its state for
  earlier quarters would put banks in a region they were not yet in (look-ahead).
  `directory` is accepted for interface compatibility but not used for membership.
* A bank-quarter is a Midwest observation when a history spell with a Midwest
  state covers that quarter (spell end dates are snapped to their quarter; an
  empty end date means the spell is ongoing through the last sample quarter).
  Quarters covered by conflicting spells (Midwest and non-Midwest) are dropped.
* For each bank, the Midwest quarters are split into contiguous runs. Within a run
  the series is trimmed to its first and last reported quarter, split wherever a
  run of missing quarters longer than MAX_GAP occurs, and short internal gaps
  (<= MAX_GAP quarters) are filled by linear interpolation. No extrapolation.
* A piece is kept only if it spans at least MIN_QUARTERS quarters and at least
  MIN_COVERAGE of those quarters were actually reported. Kept pieces are
  output with no missing y.
* Values are kept as reported (negative ratios are legitimate and not winsorized).
"""

import numpy as np
import pandas as pd

MIDWEST = ("IL", "IN", "IA", "MI", "MN", "OH", "WI")
MAX_GAP = 2          # longest run of unreported quarters that is interpolated
MIN_QUARTERS = 12    # shortest usable series, in quarters
MIN_COVERAGE = 0.8   # minimum share of quarters with a reported value


def _qidx(yyyymmdd):
    """YYYYMMDD integers -> integer calendar-quarter index (year * 4 + quarter - 1)."""
    d = np.asarray(yyyymmdd, dtype=np.int64)
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def _qend(qi):
    """Integer quarter index -> pandas Timestamp at the calendar quarter end."""
    qi = np.asarray(qi, dtype=np.int64)
    periods = pd.PeriodIndex.from_fields(year=qi // 4, quarter=qi % 4 + 1, freq="Q")
    return pd.DatetimeIndex(periods.to_timestamp(how="end").normalize())


def _midwest_quarters(history, q_lo, q_hi):
    """Return a DataFrame (id, q) of quarters in which each bank was headquartered in the Midwest."""
    h = history[["rssd_id", "start_date", "end_date", "state"]]
    state = h["state"].astype(str).str.strip().str.upper().to_numpy()
    start = _qidx(h["start_date"].to_numpy())
    end_raw = pd.to_numeric(h["end_date"], errors="coerce")
    ongoing = end_raw.isna().to_numpy()
    end = np.where(
        ongoing,
        q_hi,
        _qidx(np.where(ongoing, 19000101, end_raw.fillna(0)).astype(np.int64)),
    )
    start = np.maximum(start, q_lo)
    end = np.minimum(end, q_hi)
    keep = start <= end
    ids = h["rssd_id"].to_numpy(dtype=np.int64)[keep]
    state, start, end = state[keep], start[keep], end[keep]
    is_mw = np.isin(state, MIDWEST)

    # Expand each spell into one row per covered quarter.
    lengths = end - start + 1
    total = int(lengths.sum())
    offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
    q = np.repeat(start, lengths) + (np.arange(total) - offsets)
    mem = pd.DataFrame({
        "id": np.repeat(ids, lengths),
        "q": q,
        "mw": np.repeat(is_mw, lengths),
    })
    # Keep a quarter only if every spell covering it is Midwest (drops conflicts).
    agg = mem.groupby(["id", "q"], sort=False)["mw"].all()
    agg = agg[agg].reset_index()[["id", "q"]]
    return agg


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # --- reported values, one row per bank-quarter -------------------------
    c = calls[["rssd_id", "report_date", "value"]].dropna(subset=["rssd_id", "report_date"])
    c = pd.DataFrame({
        "id": c["rssd_id"].to_numpy(dtype=np.int64),
        "q": _qidx(c["report_date"].to_numpy(dtype=np.int64)),
        "value": pd.to_numeric(c["value"], errors="coerce").to_numpy(dtype=float),
    })
    c = c.groupby(["id", "q"], sort=False, as_index=False)["value"].first()  # first non-missing

    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    # --- Midwest membership on the sample quarter grid ---------------------
    q_lo, q_hi = int(c["q"].min()), int(c["q"].max())
    mw = _midwest_quarters(history, q_lo, q_hi)

    grid = mw.merge(c, on=["id", "q"], how="left").sort_values(["id", "q"], kind="stable")
    ids = grid["id"].to_numpy(dtype=np.int64)
    qs = grid["q"].to_numpy(dtype=np.int64)
    vals = grid["value"].to_numpy(dtype=float)

    # --- per-bank cleaning ---------------------------------------------------
    out_id, out_q, out_y = [], [], []

    if len(ids):
        bank_bounds = np.flatnonzero(np.diff(ids)) + 1
        bank_starts = np.r_[0, bank_bounds]
        bank_ends = np.r_[bank_bounds, len(ids)]

        for a, b in zip(bank_starts, bank_ends):
            q_bank = qs[a:b]
            v_bank = vals[a:b]
            # contiguous Midwest runs; a move out of and back into the Midwest is a break
            run_breaks = np.flatnonzero(np.diff(q_bank) != 1) + 1
            for q_run, v_run in zip(np.split(q_bank, run_breaks), np.split(v_bank, run_breaks)):
                obs = np.flatnonzero(~np.isnan(v_run))
                if obs.size == 0:
                    continue
                # split where unreported stretches are too long to interpolate
                groups = np.split(obs, np.flatnonzero(np.diff(obs) - 1 > MAX_GAP) + 1)
                for g in groups:
                    lo, hi = int(g[0]), int(g[-1])
                    span = hi - lo + 1
                    if span < MIN_QUARTERS or g.size / span < MIN_COVERAGE:
                        continue
                    idx = np.arange(lo, hi + 1)
                    y = np.interp(idx, g, v_run[g])  # short gaps only; ends are observed
                    out_id.append(np.full(span, ids[a], dtype=np.int64))
                    out_q.append(q_run[idx])
                    out_y.append(y)

    if not out_id:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    q_out = np.concatenate(out_q)
    panel = pd.DataFrame({
        "id": np.concatenate(out_id),
        "date": _qend(q_out),
        "y": np.concatenate(out_y),
    })
    panel = panel.dropna(subset=["y"])
    panel = panel.sort_values(["id", "date"], kind="stable").reset_index(drop=True)
    return panel
