"""Build a clean quarterly panel of commercial banks from Call Report data.

Steps
-----
1. Map every report to its calendar quarter end; if a bank has several reports in a
   quarter keep the latest non-missing one (amended / late filings).
2. Attach the charter in force at each quarter (latest attribute snapshot on or before
   the quarter end; quarters before the first snapshot use the first snapshot) and keep
   only quarters in which the bank is a commercial bank.
3. Put each bank on a regular quarterly grid. Short gaps (at most MAX_FILL consecutive
   missing quarters, either unfiled or NaN) between two real observations are filled by
   linear interpolation. Longer gaps, and any change of charter, break the series.
4. Keep, per bank, only the most recent unbroken segment (the forecasting benchmark
   holds out the last 4 quarters, so the history must run contiguously into them).
5. Holdout targets must be genuinely reported: trim the tail of the segment back to
   the last run of HOLDOUT consecutive real (non-interpolated) quarters.
6. Drop banks with poor coverage: trimmed segment shorter than MIN_QUARTERS
   (12 training + 4 holdout quarters) or more than MAX_FILL_SHARE of it imputed.
"""
import numpy as np
import pandas as pd

HOLDOUT = 4          # quarters held out at the end of each series
MIN_TRAIN = 12       # minimum training quarters before the holdout
MIN_QUARTERS = HOLDOUT + MIN_TRAIN
MAX_FILL = 2         # longest gap (in quarters) bridged by interpolation
MAX_FILL_SHARE = 0.2 # max share of a kept series that may be interpolated


def _qindex(yyyymmdd: pd.Series) -> np.ndarray:
    """YYYYMMDD int -> integer quarter index (year*4 + quarter-1)."""
    v = pd.to_numeric(yyyymmdd, errors="coerce").to_numpy(dtype="float64")
    year = np.floor(v / 10000)
    month = np.floor(v / 100) % 100
    q = year * 4 + np.floor((month - 1) / 3)
    return q


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "date", "y"]

    # ---- 1. calls -> one value per bank-quarter ------------------------------------
    c = pd.DataFrame({
        "id": pd.to_numeric(calls["rssd_id"], errors="coerce"),
        "rd": pd.to_numeric(calls["report_date"], errors="coerce"),
        "y": pd.to_numeric(calls["value"], errors="coerce").astype("float64"),
    })
    c["y"] = c["y"].where(np.isfinite(c["y"]))
    c["q"] = _qindex(c["rd"])
    c = c.dropna(subset=["id", "q"])
    c["id"] = c["id"].astype("int64")
    c["q"] = c["q"].astype("int64")
    # latest non-missing report per quarter; if none is non-missing keep NaN
    c["has"] = c["y"].notna()
    c = c.sort_values(["id", "q", "has", "rd"], kind="mergesort")
    c = c.drop_duplicates(["id", "q"], keep="last")[["id", "q", "y"]]
    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})[out_cols]

    # ---- 2. charter as of each quarter ---------------------------------------------
    a = pd.DataFrame({
        "id": pd.to_numeric(attrs["rssd_id"], errors="coerce"),
        "aq": _qindex(attrs["as_of"]),
        "ad": pd.to_numeric(attrs["as_of"], errors="coerce"),
        "charter": attrs["charter"].astype("string").str.strip().str.lower(),
    }).dropna(subset=["id", "aq", "charter"])
    a["id"] = a["id"].astype("int64")
    a["aq"] = a["aq"].astype("int64")
    a = a.sort_values(["id", "aq", "ad"], kind="mergesort").drop_duplicates(["id", "aq"], keep="last")

    # ---- 3. regular quarterly grid per bank ----------------------------------------
    span = c.groupby("id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    gid = np.repeat(span.index.to_numpy(), n)
    start = np.repeat(span["min"].to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    g = pd.DataFrame({"id": gid, "q": start + offs})
    g = g.merge(c, on=["id", "q"], how="left")  # stays sorted by id, q

    # charter: backward as-of, falling back to the first snapshot
    g = g.sort_values("q", kind="mergesort")
    a_sorted = a.sort_values("aq", kind="mergesort")
    back = pd.merge_asof(g[["id", "q"]], a_sorted[["id", "aq", "charter"]],
                         left_on="q", right_on="aq", by="id", direction="backward")
    fwd = pd.merge_asof(g[["id", "q"]], a_sorted[["id", "aq", "charter"]],
                        left_on="q", right_on="aq", by="id", direction="forward")
    g["charter"] = back["charter"].fillna(fwd["charter"]).to_numpy()
    g = g.sort_values(["id", "q"], kind="mergesort").reset_index(drop=True)

    comm = (g["charter"] == "commercial").to_numpy()
    g.loc[~comm, "y"] = np.nan

    # ---- interpolation of short interior gaps --------------------------------------
    pos = np.arange(len(g), dtype="float64")
    obs = g["y"].notna().to_numpy()
    # charter regime id: interpolation never crosses a charter change
    chg = (g["charter"].ne(g["charter"].shift()) | g["id"].ne(g["id"].shift())).to_numpy()
    reg = np.cumsum(chg)
    key = pd.Series(reg)
    p_prev = pd.Series(np.where(obs, pos, np.nan)).groupby(key).ffill().to_numpy()
    p_next = pd.Series(np.where(obs, pos, np.nan)).groupby(key).bfill().to_numpy()
    y_prev = g["y"].groupby(key).ffill().to_numpy()
    y_next = g["y"].groupby(key).bfill().to_numpy()
    gap = p_next - p_prev - 1  # length of the gap this row sits in
    fillable = (~obs) & comm & np.isfinite(gap) & (gap <= MAX_FILL)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (pos - p_prev) / (p_next - p_prev)
    y = g["y"].to_numpy().copy()
    y[fillable] = (y_prev + w * (y_next - y_prev))[fillable]
    g["y"] = y
    g["imp"] = fillable

    # ---- 4. most recent unbroken segment -------------------------------------------
    g = g[np.isfinite(y)].reset_index(drop=True)
    brk = (g["id"].ne(g["id"].shift()) | g["q"].diff().ne(1)).to_numpy()
    g["seg"] = np.cumsum(brk)
    last_seg = g.groupby("id")["seg"].transform("max")
    g = g[g["seg"] == last_seg].reset_index(drop=True)

    # ---- 5. holdout trim + coverage filter ----------------------------------------
    # The last HOLDOUT quarters are the evaluation targets, so they must be genuinely
    # reported: trim any tail after the last run of HOLDOUT consecutive real values.
    real = ~g["imp"].to_numpy()
    run_id = np.cumsum(np.r_[True, (real[1:] != real[:-1]) | (g["id"].to_numpy()[1:] != g["id"].to_numpy()[:-1])])
    run_len = pd.Series(real.astype("int64")).groupby(run_id).cumsum().to_numpy()
    rpos = np.arange(len(g))
    ok_end = pd.Series(np.where(real & (run_len >= HOLDOUT), rpos, -1)).groupby(g["id"].to_numpy()).transform("max").to_numpy()
    g = g[rpos <= ok_end].reset_index(drop=True)

    grp = g.groupby("id")
    length = grp["q"].transform("size")
    imp_share = grp["imp"].transform("mean")
    keep = (length >= MIN_QUARTERS) & (imp_share <= MAX_FILL_SHARE)
    g = g[keep.to_numpy()]

    # ---- output ---------------------------------------------------------------------
    q = g["q"].to_numpy()
    date = pd.PeriodIndex.from_ordinals(q - 1970 * 4, freq="Q-DEC").to_timestamp(how="end").normalize()
    res = pd.DataFrame({
        "id": g["id"].to_numpy(dtype="int64"),
        "date": pd.DatetimeIndex(date).astype("datetime64[ns]"),
        "y": g["y"].to_numpy(dtype="float64"),
    })
    return res.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)


if __name__ == "__main__":
    import sys, time
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    t = time.time()
    out = build_panel(pd.read_csv(f"{d}/sample_calls.csv"), pd.read_csv(f"{d}/sample_attrs.csv"))
    print(out.head(), out.dtypes, sep="\n")
    print(f"rows={len(out)} banks={out['id'].nunique()} in {time.time() - t:.2f}s")
