"""Quarterly panel of Midwest-headquartered banks.

Only pandas and numpy are used.

Pipeline
--------
1. Calls are reduced to one value per (bank, calendar quarter); duplicate filings
   are averaged over their non-missing values.
2. Each bank's headquarters state is taken from the history spells for each
   quarter. Banks that never appear in the history fall back to the directory
   state (the directory is only a end-of-sample snapshot, so it is used only
   where no history exists). Quarters not covered by any spell are unknown and
   are not Midwest.
3. Every bank is placed on a regular quarterly grid between its first and last
   reported quarter. Internal gaps of up to MAX_GAP quarters are linearly
   interpolated; longer gaps and leading/trailing gaps stay missing.
4. Banks are kept only if, within their Midwest-headquartered quarters inside
   their reporting span, they have at least MIN_REPORTED reported quarters and
   at least MIN_COVERAGE share of quarters reported.
5. Output rows are bank-quarters in the kept set with a non-missing y, sorted
   by id and date.
"""

import numpy as np
import pandas as pd

MIDWEST = ("IL", "IN", "IA", "MI", "MN", "OH", "WI")
MAX_GAP = 2          # longest internal gap (quarters) that is interpolated
MIN_REPORTED = 12    # min reported quarters while Midwest-headquartered
MIN_COVERAGE = 0.8   # min share of reported quarters in the bank's Midwest span


def _qidx(yyyymmdd):
    """Map YYYYMMDD integers to an integer quarter index (year*4 + quarter-1)."""
    a = np.asarray(yyyymmdd, dtype="float64")
    y = np.floor(a / 10000).astype("int64")
    m = (np.floor(a / 100) % 100).astype("int64")
    return y * 4 + (m - 1) // 3


def _quarter_end(q):
    """Map integer quarter indices back to calendar quarter-end Timestamps."""
    q = np.asarray(q, dtype="int64")
    years = q // 4
    months = 3 * (q % 4) + 3
    first = pd.to_datetime(
        pd.DataFrame({"year": years, "month": months, "day": 1})
    )
    return pd.DatetimeIndex(first + pd.offsets.MonthEnd(0))


def _state_str(s):
    return s.astype("string").str.strip().str.upper().fillna("")


def _midwest_matrix(banks, Q, history, directory):
    """Boolean bank x quarter matrix: True where the bank is Midwest-headquartered."""
    n_b, n_q = len(banks), len(Q)
    q0, q1 = Q[0], Q[-1]
    mw = np.zeros((n_b, n_q), dtype=bool)
    bank_pos = pd.Index(banks)

    # History spells -> one row per (bank, quarter) covered by a spell.
    h = history.copy()
    h["rssd_id"] = pd.to_numeric(h["rssd_id"], errors="coerce").astype("Int64")
    h["start_date"] = pd.to_numeric(h["start_date"], errors="coerce")
    h["end_date"] = pd.to_numeric(h["end_date"], errors="coerce").astype("float64")
    h["state"] = _state_str(h["state"])
    h = h.dropna(subset=["rssd_id", "start_date"])
    h["rssd_id"] = h["rssd_id"].astype("int64")
    h = h.sort_values(["rssd_id", "start_date"], kind="mergesort").reset_index(drop=True)

    s_q = np.maximum(_qidx(h["start_date"].to_numpy()), q0)
    end_raw = h["end_date"].to_numpy()
    e_q = np.where(np.isnan(end_raw), q1, _qidx(np.nan_to_num(end_raw)))
    e_q = np.minimum(e_q, q1)
    ok = e_q >= s_q
    h, s_q, e_q = h[ok].reset_index(drop=True), s_q[ok], e_q[ok]

    lengths = (e_q - s_q + 1).astype("int64")
    rep = np.repeat(np.arange(len(h)), lengths)
    starts_cum = np.cumsum(lengths) - lengths
    within = np.arange(lengths.sum()) - np.repeat(starts_cum, lengths)
    spell_q = s_q[rep] + within

    long = pd.DataFrame({
        "rssd_id": h["rssd_id"].to_numpy()[rep],
        "q": spell_q,
        "state": h["state"].to_numpy()[rep],
    })
    # Later-starting spells win on any overlap (none expected).
    long = long.drop_duplicates(["rssd_id", "q"], keep="last")
    long = long[long["state"].isin(MIDWEST)]
    bi = bank_pos.get_indexer(long["rssd_id"].to_numpy())
    ci = long["q"].to_numpy() - q0
    keep = (bi >= 0) & (ci >= 0) & (ci < n_q)
    mw[bi[keep], ci[keep]] = True

    # Banks with no history: use the directory state for every quarter.
    in_hist = set(h["rssd_id"].unique())
    d = directory.copy()
    d["rssd_id"] = pd.to_numeric(d["rssd_id"], errors="coerce")
    d = d.dropna(subset=["rssd_id"])
    d["rssd_id"] = d["rssd_id"].astype("int64")
    d["state"] = _state_str(d["state"])
    d = d.drop_duplicates("rssd_id", keep="last")
    d = d[d["state"].isin(MIDWEST) & ~d["rssd_id"].isin(in_hist)]
    bi = bank_pos.get_indexer(d["rssd_id"].to_numpy())
    bi = bi[bi >= 0]
    mw[bi, :] = True
    return mw


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # 1. One value per bank-quarter.
    c = pd.DataFrame({
        "rssd_id": pd.to_numeric(calls["rssd_id"], errors="coerce").astype("int64"),
        "q": _qidx(pd.to_numeric(calls["report_date"], errors="coerce").to_numpy()),
        "value": pd.to_numeric(calls["value"], errors="coerce").replace([np.inf, -np.inf], np.nan),
    })
    c = c.groupby(["rssd_id", "q"], as_index=False)["value"].mean()

    banks = np.sort(c["rssd_id"].unique())
    Q = np.arange(c["q"].min(), c["q"].max() + 1, dtype="int64")
    n_b, n_q = len(banks), len(Q)

    wide = (
        c.pivot(index="rssd_id", columns="q", values="value")
        .reindex(index=banks, columns=Q)
    )
    raw = wide.notna().to_numpy()

    # 2. Regular grid with short internal gaps interpolated.
    filled = wide.interpolate(axis=1, method="linear", limit=MAX_GAP, limit_area="inside")
    val = filled.to_numpy(dtype="float64")

    mw = _midwest_matrix(banks, Q, history, directory)

    # 3. Coverage: within each bank's reporting span, Midwest quarters only.
    idx = np.arange(n_q)
    first = np.where(raw, idx, n_q).min(axis=1)
    last = np.where(raw, idx, -1).max(axis=1)
    inspan = (idx[None, :] >= first[:, None]) & (idx[None, :] <= last[:, None])
    window = mw & inspan
    n_win = window.sum(axis=1)
    n_rep = (window & raw).sum(axis=1)
    bank_ok = (n_rep >= MIN_REPORTED) & (n_rep >= MIN_COVERAGE * n_win)

    # 4. Output bank-quarters.
    out = mw & bank_ok[:, None] & np.isfinite(val)
    bi, qi = np.nonzero(out)

    res = pd.DataFrame({
        "id": banks[bi].astype("int64"),
        "date": _quarter_end(Q[qi]),
        "y": val[bi, qi].astype("float64"),
    })
    res = res.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    return res


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    calls = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    directory = pd.read_csv(os.path.join(here, "sample_directory.csv"))
    history = pd.read_csv(os.path.join(here, "sample_history.csv"), dtype={"end_date": "Int64"})
    panel = build_panel(calls, directory, history)
    print(panel.head())
    print(panel.dtypes)
    print("rows", len(panel), "banks", panel["id"].nunique())
