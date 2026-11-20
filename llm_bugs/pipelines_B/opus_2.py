"""Quarterly commercial-bank panel for a last-4-quarters holdout forecasting benchmark.

Pipeline (fully vectorised, pandas + numpy only):

1. Clean reports: parse YYYYMMDD dates (invalid dropped), map each report to its
   calendar quarter, turn +/-inf into NaN, and collapse several reports in one
   quarter to a single one (latest non-missing filing wins, i.e. amendments).
2. Put every bank on a regular quarterly grid from its first to its last
   non-missing quarter.
3. Attach charter point-in-time (latest attribute snapshot at or before the
   quarter; quarters before the first snapshot take the first snapshot).
   Banks with no attributes are dropped. Non-commercial quarters are never
   used and break the series, so a bank that converts keeps only commercial spells.
4. Split each bank into segments: a break occurs at any non-commercial quarter
   or at a run of more than MAX_GAP consecutive missing quarters.
5. In each segment, trim the end so the final HOLDOUT + 1 quarters are all
   genuinely observed (the test targets are never imputed, and imputation in
   the training part only uses training data). Start at the first observation.
6. Keep segments with at least MIN_LEN quarters and an observed share of at
   least MIN_COVERAGE; per bank keep the most recent qualifying segment.
7. Fill the remaining short internal gaps by linear interpolation in time.
"""
import numpy as np
import pandas as pd

HOLDOUT = 4          # quarters held out at the end of each series
MAX_GAP = 4          # longest run of missing quarters that is interpolated
MIN_LEN = 16         # minimum series length (12 training + 4 holdout quarters)
MIN_COVERAGE = 0.75  # minimum share of genuinely observed quarters in the series


def _quarter_ordinal(yyyymmdd: pd.Series) -> pd.Series:
    """YYYYMMDD ints -> quarter ordinal (year*4 + quarter-1); NaN if invalid."""
    d = pd.to_datetime(pd.Series(yyyymmdd).astype("Int64").astype(str),
                       format="%Y%m%d", errors="coerce")
    return d.dt.year * 4 + (d.dt.month - 1) // 3, d


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})

    # ---- 1. clean reports --------------------------------------------------
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c.dropna(subset=["rssd_id", "report_date"])
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["q"], c["d"] = _quarter_ordinal(c["report_date"])
    c = c.dropna(subset=["q"])
    c["q"] = c["q"].astype("int64")
    c["value"] = pd.to_numeric(c["value"], errors="coerce").astype("float64")
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    c["has"] = c["value"].notna()
    c = (c.sort_values(["rssd_id", "q", "has", "d"])
           .drop_duplicates(["rssd_id", "q"], keep="last"))
    c = c[c["has"]]
    if c.empty:
        return empty

    # ---- 2. regular quarterly grid ----------------------------------------
    rng = c.groupby("rssd_id")["q"].agg(["min", "max"])
    n = (rng["max"] - rng["min"] + 1).to_numpy()
    ids = np.repeat(rng.index.to_numpy(), n)
    starts = np.repeat(rng["min"].to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    g = pd.DataFrame({"rssd_id": ids, "q": starts + offs})
    g = g.merge(c[["rssd_id", "q", "value"]], on=["rssd_id", "q"], how="left")

    # ---- 3. point-in-time charter -----------------------------------------
    a = attrs[["rssd_id", "as_of", "charter"]].dropna(subset=["rssd_id", "as_of"]).copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["q"], a["d"] = _quarter_ordinal(a["as_of"])
    a = a.dropna(subset=["q"])
    a["q"] = a["q"].astype("int64")
    a["commercial"] = a["charter"].astype(str).str.strip().str.lower().eq("commercial")
    a = (a.sort_values(["rssd_id", "q", "d"])
           .drop_duplicates(["rssd_id", "q"], keep="last"))
    g = g[g["rssd_id"].isin(a["rssd_id"].unique())]
    if g.empty:
        return empty
    g = pd.merge_asof(g.sort_values("q"), a[["rssd_id", "q", "commercial"]].sort_values("q"),
                      on="q", by="rssd_id", direction="backward")
    first = a.groupby("rssd_id")["commercial"].first()
    miss = g["commercial"].isna()
    g.loc[miss, "commercial"] = g.loc[miss, "rssd_id"].map(first)
    g = g.sort_values(["rssd_id", "q"], ignore_index=True)
    comm = g["commercial"].astype(bool).to_numpy()
    obs = comm & g["value"].notna().to_numpy()
    g["obs"] = obs

    # ---- 4. segments -------------------------------------------------------
    bank = g["rssd_id"].to_numpy()
    new_bank = np.r_[True, bank[1:] != bank[:-1]]
    # length of the current run of missing (commercial) quarters
    run_id = np.cumsum(obs | new_bank | ~comm)
    miss_run = pd.Series(~obs).groupby(run_id).cumsum().to_numpy()
    brk = new_bank | ~comm | (miss_run > MAX_GAP)
    g["seg"] = np.cumsum(brk)
    g = g[comm].copy()

    # ---- 5. trim ends -------------------------------------------------------
    o = g["obs"].to_numpy()
    obs_run = pd.Series(o).groupby([g["seg"].to_numpy(), np.cumsum(~o)]).cumsum().to_numpy()  # consecutive observed
    g["ok_end"] = obs_run >= HOLDOUT + 1
    qs = g["q"].to_numpy()
    g["end_q"] = np.where(g["ok_end"], qs, -1)
    g["start_q"] = np.where(o, qs, np.iinfo(np.int64).max)
    bounds = g.groupby("seg").agg(start=("start_q", "min"), end=("end_q", "max"))
    g = g.join(bounds, on="seg")
    g = g[(g["end"] >= 0) & (g["q"] >= g["start"]) & (g["q"] <= g["end"])]

    # ---- 6. coverage filter, most recent qualifying segment per bank ------
    st = g.groupby("seg").agg(rssd_id=("rssd_id", "first"), length=("q", "size"),
                              cov=("obs", "mean"), end=("q", "max"))
    st = st[(st["length"] >= MIN_LEN) & (st["cov"] >= MIN_COVERAGE)]
    keep = st.sort_values("end").groupby("rssd_id").tail(1).index
    g = g[g["seg"].isin(keep)].copy()
    if g.empty:
        return empty

    # ---- 7. linear interpolation of internal gaps -------------------------
    qf = g["q"].astype("float64")
    prev_q = qf.where(g["obs"]).groupby(g["seg"]).ffill()
    next_q = qf.where(g["obs"]).groupby(g["seg"]).bfill()
    prev_v = g["value"].where(g["obs"]).groupby(g["seg"]).ffill()
    next_v = g["value"].where(g["obs"]).groupby(g["seg"]).bfill()
    w = ((qf - prev_q) / (next_q - prev_q)).fillna(0.0)
    y = np.where(g["obs"], g["value"], prev_v + w * (next_v - prev_v))

    q = g["q"].to_numpy()
    year, qtr = q // 4, q % 4 + 1
    date = (pd.PeriodIndex.from_fields(year=year, quarter=qtr, freq="Q")
            .to_timestamp(how="end").normalize())
    out = pd.DataFrame({"id": g["rssd_id"].astype("int64").to_numpy(),
                        "date": date.astype("datetime64[ns]"), "y": np.asarray(y, dtype="float64")})
    out = out.dropna(subset=["y"]).sort_values(["id", "date"], ignore_index=True)
    return out
