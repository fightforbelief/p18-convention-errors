"""Build a clean quarterly panel of commercial banks from Call Report data.

Rules (see spec.md):
  * Report dates are mapped to calendar quarters; if a bank has several filings in
    one quarter, the latest non-missing one wins. NaN / inf values are treated as
    missing.
  * Charter is assigned point-in-time: each quarter takes the most recent attribute
    snapshot dated on or before the quarter end (quarters before a bank's first
    snapshot use that first snapshot). Only commercial-bank quarters are kept.
  * Each bank's series is put on a regular quarterly grid. Interior gaps of at most
    MAX_GAP consecutive quarters are filled by linear interpolation. A longer gap, or
    a spell under a non-commercial charter, breaks the series; only the most recent
    contiguous segment is kept (the benchmark needs one contiguous history per bank).
  * The last HOLDOUT quarters are the forecast targets, so they must be genuinely
    reported values: the segment is trimmed back to the last quarter whose trailing
    HOLDOUT quarters are all observed.
  * Banks are dropped if the final segment is shorter than MIN_QUARTERS or if more
    than MAX_IMPUTED_SHARE of it is interpolated.
"""
import numpy as np
import pandas as pd

HOLDOUT = 4             # quarters held out per bank by the benchmark
MIN_TRAIN = 16          # minimum training quarters
MIN_QUARTERS = MIN_TRAIN + HOLDOUT
MAX_GAP = 2             # longest run of missing quarters filled by interpolation
MAX_IMPUTED_SHARE = 0.10


def _quarter_index(yyyymmdd: pd.Series) -> pd.Series:
    d = pd.to_datetime(yyyymmdd.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return d.dt.year * 4 + (d.dt.month - 1) // 3


def _quarter_end(qi: np.ndarray) -> pd.DatetimeIndex:
    qi = np.asarray(qi, dtype=np.int64)
    year, q = qi // 4, qi % 4
    # first day of next quarter minus one day
    nxt_year = year + (q == 3)
    nxt_month = np.where(q == 3, 1, 3 * q + 4)
    first = pd.to_datetime({"year": nxt_year, "month": nxt_month, "day": np.ones_like(qi)})
    return pd.DatetimeIndex(first - pd.Timedelta(days=1))


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"id": pd.Series(dtype="int64"),
                         "date": pd.Series(dtype="datetime64[ns]"),
                         "y": pd.Series(dtype="float64")})


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- 1. clean filings, one value per bank-quarter --------------------------------
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["value"] = pd.to_numeric(c["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    c["rssd_id"] = pd.to_numeric(c["rssd_id"], errors="coerce")
    c["qi"] = _quarter_index(pd.to_numeric(c["report_date"], errors="coerce"))
    c = c.dropna(subset=["rssd_id", "qi", "value"])
    if c.empty:
        return _empty()
    c["rssd_id"] = c["rssd_id"].astype(np.int64)
    c["qi"] = c["qi"].astype(np.int64)
    c = (c.sort_values(["rssd_id", "qi", "report_date"], kind="mergesort")
          .drop_duplicates(["rssd_id", "qi"], keep="last"))

    # ---- 2. regular quarterly grid per bank (first to last observed quarter) ---------
    span = c.groupby("rssd_id")["qi"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    g = pd.DataFrame({"rssd_id": ids, "qi": np.repeat(span["min"].to_numpy(), n) + offs})
    g = g.merge(c[["rssd_id", "qi", "value"]], on=["rssd_id", "qi"], how="left")

    # ---- 3. point-in-time charter -------------------------------------------------
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["rssd_id"] = pd.to_numeric(a["rssd_id"], errors="coerce")
    a["qi"] = _quarter_index(pd.to_numeric(a["as_of"], errors="coerce"))
    a["charter"] = a["charter"].astype("string").str.strip().str.lower()
    a = a.dropna(subset=["rssd_id", "qi", "charter"])
    a["rssd_id"] = a["rssd_id"].astype(np.int64)
    a["qi"] = a["qi"].astype(np.int64)
    # snapshot as_of <= quarter end  <=>  snapshot quarter <= quarter
    a = (a.sort_values(["rssd_id", "qi", "as_of"], kind="mergesort")
          .drop_duplicates(["rssd_id", "qi"], keep="last"))
    g = g.sort_values(["qi", "rssd_id"], kind="mergesort")
    g = pd.merge_asof(g, a[["rssd_id", "qi", "charter"]].sort_values("qi"),
                      on="qi", by="rssd_id", direction="backward")
    g = g.sort_values(["rssd_id", "qi"], kind="mergesort").reset_index(drop=True)
    first_charter = a.groupby("rssd_id")["charter"].first()
    g["charter"] = g["charter"].fillna(g["rssd_id"].map(first_charter))
    g["comm"] = (g["charter"] == "commercial").fillna(False).to_numpy(bool)

    # ---- 4. contiguous commercial segments; keep the most recent -----------------
    g["obs"] = g["value"].notna()
    noncomm_cum = (~g["comm"]).groupby(g["rssd_id"]).cumsum()
    g["noncomm_cum"] = noncomm_cum
    o = g[g["obs"] & g["comm"]]
    new_bank = o["rssd_id"].ne(o["rssd_id"].shift())
    brk = (new_bank
           | (o["qi"].diff() > MAX_GAP + 1)
           | o["noncomm_cum"].ne(o["noncomm_cum"].shift()))
    seg = brk.cumsum()
    last_seg = seg.groupby(o["rssd_id"]).transform("max")
    o = o[seg == last_seg]
    bounds = o.groupby("rssd_id")["qi"].agg(["min", "max"])
    g = g.join(bounds, on="rssd_id", how="inner")
    g = g[(g["qi"] >= g["min"]) & (g["qi"] <= g["max"])].drop(columns=["min", "max"])
    g = g.reset_index(drop=True)

    # ---- 5. linear interpolation of short interior gaps ---------------------------
    by = g.groupby("rssd_id")
    q_obs = g["qi"].where(g["obs"])
    v0 = by["value"].ffill()
    v1 = by["value"].bfill()
    q0 = q_obs.groupby(g["rssd_id"]).ffill()
    q1 = q_obs.groupby(g["rssd_id"]).bfill()
    w = (g["qi"] - q0) / (q1 - q0)
    g["y"] = g["value"].fillna(v0 + (v1 - v0) * w)

    # ---- 6. holdout quarters must be genuinely observed --------------------------
    run4 = (g["obs"].astype(int).groupby(g["rssd_id"])
            .rolling(HOLDOUT, min_periods=HOLDOUT).sum()
            .reset_index(level=0, drop=True).sort_index())
    end_q = g["qi"].where(run4 == HOLDOUT).groupby(g["rssd_id"]).transform("max")
    g = g[g["qi"] <= end_q]

    # ---- 7. coverage filter -------------------------------------------------------
    stats = g.groupby("rssd_id")["obs"].agg(["size", "mean"])
    keep = stats.index[(stats["size"] >= MIN_QUARTERS)
                       & (1 - stats["mean"] <= MAX_IMPUTED_SHARE)]
    g = g[g["rssd_id"].isin(keep) & g["y"].notna()]
    if g.empty:
        return _empty()

    out = pd.DataFrame({"id": g["rssd_id"].astype(np.int64).to_numpy(),
                        "date": _quarter_end(g["qi"].to_numpy()).astype("datetime64[ns]"),
                        "y": g["y"].astype(float).to_numpy()})
    return out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)


if __name__ == "__main__":
    import os, time
    here = os.path.dirname(os.path.abspath(__file__))
    t = time.time()
    p = build_panel(pd.read_csv(os.path.join(here, "sample_calls.csv")),
                    pd.read_csv(os.path.join(here, "sample_attrs.csv")))
    print(p.head(), p.dtypes, sep="\n")
    print(f"{len(p)} rows, {p['id'].nunique()} banks, {time.time() - t:.2f}s")
