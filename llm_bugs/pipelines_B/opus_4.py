"""Build a clean quarterly panel of commercial banks from Call Report data.

Rules (all vectorised, no per-bank Python loops):

1. Dates: report_date (YYYYMMDD) is mapped to its calendar quarter; invalid dates
   and non-finite values are dropped. If a bank has several reports in a quarter,
   the latest non-missing one is kept.
2. Charter: each bank-quarter gets the charter of the most recent attribute
   snapshot on or before that quarter (point-in-time, no look-ahead). Quarters
   before a bank's first snapshot take the earliest snapshot. Only quarters in
   which the bank is a commercial bank are kept.
3. Regular grid: every bank is put on a complete quarterly grid. A series is cut
   wherever the bank is not commercial or wherever more than MAX_GAP consecutive
   quarters are missing. Within a piece, gaps of up to MAX_GAP quarters are
   filled by linear interpolation; leading/trailing gaps are trimmed.
4. One spell per bank: the most recent piece that passes the coverage filter.
5. Coverage filter: the spell must have at least MIN_QUARTERS quarters
   (MIN_TRAIN training quarters + the HOLDOUT test quarters), at most
   MAX_IMPUTED_SHARE of its quarters imputed, and its last HOLDOUT quarters
   (the forecasting holdout) must all be genuinely reported, so the benchmark
   is never scored against interpolated values.
"""
import numpy as np
import pandas as pd

HOLDOUT = 4
MIN_TRAIN = 16
MIN_QUARTERS = MIN_TRAIN + HOLDOUT
MAX_GAP = 2
MAX_IMPUTED_SHARE = 0.20


def _quarter_index(yyyymmdd):
    d = pd.to_numeric(pd.Series(yyyymmdd), errors="coerce").to_numpy(dtype="float64")
    y = np.floor(d / 10000)
    m = np.floor(d / 100) % 100
    day = d % 100
    ok = np.isfinite(d) & (m >= 1) & (m <= 12) & (day >= 1) & (day <= 31) & (y > 1800)
    q = np.where(ok, y * 4 + (m - 1) // 3, np.nan)
    return q


def _empty():
    return pd.DataFrame({"id": pd.Series(dtype="int64"),
                         "date": pd.Series(dtype="datetime64[ns]"),
                         "y": pd.Series(dtype="float64")})


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- calls: quarter, clean, dedupe -------------------------------------
    c = pd.DataFrame({
        "id": pd.to_numeric(calls["rssd_id"], errors="coerce"),
        "q": _quarter_index(calls["report_date"]),
        "rd": pd.to_numeric(calls["report_date"], errors="coerce"),
        "v": pd.to_numeric(calls["value"], errors="coerce"),
    })
    c.loc[~np.isfinite(c["v"]), "v"] = np.nan
    c = c.dropna(subset=["id", "q", "v"])
    if c.empty:
        return _empty()
    c["id"] = c["id"].astype("int64")
    c["q"] = c["q"].astype("int64")
    c = (c.sort_values(["id", "q", "rd"], kind="mergesort")
          .drop_duplicates(["id", "q"], keep="last"))

    # ---- full quarterly grid per bank (first..last reported quarter) --------
    span = c.groupby("id", sort=True)["q"].agg(["min", "max"])
    lens = (span["max"] - span["min"] + 1).to_numpy()
    gid = np.repeat(span.index.to_numpy(), lens)
    starts = np.repeat(span["min"].to_numpy(), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    grid = pd.DataFrame({"id": gid, "q": starts + offs})
    grid = grid.merge(c[["id", "q", "v"]], on=["id", "q"], how="left", sort=False)
    grid = grid.sort_values(["id", "q"], kind="mergesort").reset_index(drop=True)

    # ---- point-in-time charter ----------------------------------------------
    a = pd.DataFrame({
        "id": pd.to_numeric(attrs["rssd_id"], errors="coerce"),
        "q": _quarter_index(attrs["as_of"]),
        "rd": pd.to_numeric(attrs["as_of"], errors="coerce"),
        "commercial": attrs["charter"].astype("string").str.strip().str.lower()
                                       .eq("commercial").fillna(False).astype(bool),
    }).dropna(subset=["id", "q"])
    a = a[a["id"].isin(span.index)]
    a["id"] = a["id"].astype("int64")
    a["q"] = a["q"].astype("int64")
    a = (a.sort_values(["id", "q", "rd"], kind="mergesort")
          .drop_duplicates(["id", "q"], keep="last")[["id", "q", "commercial"]])
    g_sorted = grid[["id", "q"]].reset_index().sort_values("q", kind="mergesort")
    a_sorted = a.sort_values("q", kind="mergesort")
    back = pd.merge_asof(g_sorted, a_sorted, on="q", by="id", direction="backward")
    first = a.drop_duplicates("id", keep="first").set_index("id")["commercial"]
    comm = back["commercial"].astype("object")
    comm = comm.where(comm.notna(), back["id"].map(first))
    comm = comm.fillna(False).astype(bool)
    grid["commercial"] = pd.Series(comm.to_numpy(), index=back["index"].to_numpy()).sort_index().to_numpy()

    # ---- cut into pieces -----------------------------------------------------
    ids = grid["id"].to_numpy()
    v = grid["v"].to_numpy()
    com = grid["commercial"].to_numpy()
    obs = ~np.isnan(v) & com                       # usable reported value
    n = len(grid)
    new_bank = np.r_[True, ids[1:] != ids[:-1]]

    # run length of consecutive non-observed quarters (per bank) ending at each row
    miss = ~obs
    run_id = np.cumsum(~miss | new_bank)
    run_len = pd.Series(miss.astype(np.int64)).groupby(run_id).transform("sum").to_numpy()
    long_gap = miss & (run_len > MAX_GAP)
    excl = ~com | long_gap
    # piece id increments at each new bank and at each excluded row; excluded
    # rows are dropped, so rows between two exclusions share one piece id
    seg = np.cumsum(new_bank | excl)
    keep = ~excl

    # trim leading/trailing missing inside each piece
    pos = np.arange(n)
    s = pd.Series(np.where(obs & keep, pos, np.nan))
    lo = s.groupby(seg).transform("min").to_numpy()
    hi = s.groupby(seg).transform("max").to_numpy()
    keep &= (pos >= lo) & (pos <= hi)

    # interpolate interior gaps (always bracketed by obs of the same piece)
    vi = v.copy()
    xp = pos[obs]
    vi[keep & ~obs] = np.interp(pos[keep & ~obs], xp, v[obs])

    d = pd.DataFrame({"id": ids, "q": grid["q"].to_numpy(), "y": vi,
                      "seg": seg, "imp": ~obs})[keep]
    if d.empty:
        return _empty()

    # ---- coverage filter per piece ------------------------------------------
    gs = d.groupby("seg", sort=False)
    length = gs["q"].transform("size")
    imp_share = gs["imp"].transform("mean")
    rank_from_end = gs.cumcount(ascending=False)
    bad_tail = (d["imp"] & (rank_from_end < HOLDOUT)).groupby(d["seg"]).transform("any")
    ok = (length >= MIN_QUARTERS) & (imp_share <= MAX_IMPUTED_SHARE) & ~bad_tail
    d = d[ok]
    if d.empty:
        return _empty()

    # most recent qualifying piece per bank
    last_seg = d.groupby("id")["seg"].transform("max")
    d = d[d["seg"] == last_seg]

    q = d["q"].to_numpy()
    year = q // 4
    month = (q % 4) * 3 + 3
    date = (pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))
            + pd.offsets.MonthEnd(0))
    out = pd.DataFrame({"id": d["id"].astype("int64").to_numpy(),
                        "date": pd.DatetimeIndex(date).astype("datetime64[ns]"),
                        "y": d["y"].astype("float64").to_numpy()})
    return out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    import time
    calls = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sample_calls.csv")
    attrs = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "sample_attrs.csv")
    t = time.time()
    p = build_panel(calls, attrs)
    print(p.head())
    print(f"{len(p)} rows, {p['id'].nunique()} banks, {time.time() - t:.2f}s")
