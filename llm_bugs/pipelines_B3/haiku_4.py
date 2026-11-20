"""Quarterly panel of Midwest bank ratios.

build_panel() turns raw call-report ratios, a point-in-time headquarters
history and an institution directory into a balanced-by-bank quarterly panel
of banks headquartered in IL, IN, IA, MI, MN, OH or WI.

Design choices (all tunable via the module constants below):
  * Headquarters state is point-in-time: each bank-quarter takes the state of
    the history spell covering that quarter-end date. The end-of-sample
    directory is used only for banks that have no history at all.
  * A bank-quarter is kept only if the bank is headquartered in the Midwest
    in that quarter. Quarters not covered by any spell are dropped.
  * Several filings for one bank-quarter collapse to the last non-missing value.
  * Quarters are placed on a regular grid per bank. Runs of at most MAX_GAP
    unreported quarters, bracketed by reports on both sides and within one
    continuous Midwest spell, are filled by linear interpolation. Longer gaps
    and edge gaps stay missing and their rows are dropped.
  * Banks need at least MIN_OBS reported quarters and a coverage ratio of at
    least MIN_COVERAGE (reported quarters / quarters from first to last report).
"""

import os

import numpy as np
import pandas as pd

MIDWEST = ("IL", "IN", "IA", "MI", "MN", "OH", "WI")
MAX_GAP = 4          # longest run of unreported quarters that is interpolated
MIN_OBS = 20         # minimum reported quarters per bank
MIN_COVERAGE = 0.8   # minimum share of quarters reported between first and last report
OPEN_END = 99991231  # end_date sentinel for ongoing headquarters spells


def _qidx(yyyymmdd):
    """Integer quarter index (year * 4 + quarter - 1) from YYYYMMDD integers."""
    y = yyyymmdd // 10000
    m = (yyyymmdd // 100) % 100
    return y * 4 + (m - 1) // 3


def _qend(qidx):
    """Calendar quarter-end Timestamps for integer quarter indices."""
    qidx = np.asarray(qidx, dtype=np.int64)
    parts = pd.DataFrame({
        "year": qidx // 4,
        "month": (qidx % 4 + 1) * 3,
        "day": 1,
    })
    return pd.Series(pd.to_datetime(parts) + pd.offsets.MonthEnd(0))


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # ---- reported ratios, one value per bank-quarter -------------------------
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype("int64").to_numpy(),
        "q": _qidx(calls["report_date"].astype("int64").to_numpy()),
        "value": pd.to_numeric(calls["value"], errors="coerce").astype(float).to_numpy(),
    })
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    c = c.groupby(["id", "q"], sort=True)["value"].last().reset_index()

    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    ids = c["id"].unique()
    qs = np.arange(c["q"].min(), c["q"].max() + 1, dtype=np.int64)
    qd = _qend(qs)
    qend = (qd.dt.year * 10000 + qd.dt.month * 100 + qd.dt.day).to_numpy()

    # ---- point-in-time headquarters state per bank-quarter -------------------
    hist = pd.DataFrame({
        "id": history["rssd_id"].astype("int64").to_numpy(),
        "start": pd.to_numeric(history["start_date"]).astype("int64").to_numpy(),
        "end": pd.to_numeric(history["end_date"]).fillna(OPEN_END).astype("int64").to_numpy(),
        "state": history["state"].astype(str).str.strip().str.upper().to_numpy(),
    })
    hist = hist[hist["id"].isin(ids)].reset_index(drop=True)

    if len(hist):
        s_start = hist["start"].to_numpy()[:, None]
        s_end = hist["end"].to_numpy()[:, None]
        cover = (s_start <= qend[None, :]) & (qend[None, :] <= s_end)
        si, qj = np.nonzero(cover)
        spells = pd.DataFrame({
            "id": hist["id"].to_numpy()[si],
            "q": qs[qj],
            "state": hist["state"].to_numpy()[si],
            "start": hist["start"].to_numpy()[si],
        })
        # overlapping spells on one quarter-end: the most recent start wins
        spells = (spells.sort_values(["id", "q", "start"])
                        .drop_duplicates(["id", "q"], keep="last")
                        [["id", "q", "state"]])
    else:
        spells = pd.DataFrame({"id": pd.Series(dtype="int64"),
                               "q": pd.Series(dtype="int64"),
                               "state": pd.Series(dtype=object)})

    # banks with no history at all: fall back to the directory state, applied
    # over the span of their own filings
    no_hist = c.groupby("id")["q"].agg(["min", "max"]).reset_index()
    no_hist = no_hist[~no_hist["id"].isin(hist["id"].unique())]
    if len(no_hist):
        dir_state = (directory.assign(state=directory["state"].astype(str).str.strip().str.upper())
                              .drop_duplicates("rssd_id", keep="last")
                              .set_index("rssd_id")["state"])
        no_hist = no_hist.assign(state=no_hist["id"].map(dir_state))
        lengths = (no_hist["max"] - no_hist["min"] + 1).to_numpy()
        offs = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        fallback = pd.DataFrame({
            "id": np.repeat(no_hist["id"].to_numpy(), lengths),
            "q": np.repeat(no_hist["min"].to_numpy(), lengths) + offs,
            "state": np.repeat(no_hist["state"].to_numpy(), lengths),
        })
        spells = pd.concat([spells, fallback], ignore_index=True)

    # ---- Midwest bank-quarters joined to reported values ---------------------
    panel = spells[spells["state"].isin(MIDWEST)][["id", "q"]]
    panel = panel.merge(c, on=["id", "q"], how="left").sort_values(["id", "q"]).reset_index(drop=True)

    # ---- coverage screen on raw reports --------------------------------------
    rep = panel[panel["value"].notna()]
    stats = rep.groupby("id")["q"].agg(n="count", first="min", last="max")
    stats["coverage"] = stats["n"] / (stats["last"] - stats["first"] + 1)
    keep = stats.index[(stats["n"] >= MIN_OBS) & (stats["coverage"] >= MIN_COVERAGE)]
    panel = panel[panel["id"].isin(keep)].reset_index(drop=True)

    if panel.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    # ---- contiguous Midwest segments, so no interpolation bridges a move -----
    pid = panel["id"]
    pq = panel["q"]
    new_seg = (pid.ne(pid.shift(fill_value=-1)) | (pq - pq.shift(fill_value=-10**9) != 1))
    seg = new_seg.cumsum()

    v = panel["value"]
    nan = v.isna()
    interp = v.groupby(seg).transform(lambda s: s.interpolate(limit_area="inside"))

    # runs of consecutive missing quarters within a segment
    run_start = nan.ne(nan.shift(fill_value=False)) | seg.ne(seg.shift(fill_value=-1))
    run = run_start.cumsum()
    run_len = nan.groupby(run).transform("sum")
    fillable = nan & (run_len <= MAX_GAP)
    y = v.mask(fillable, interp)

    # ---- output --------------------------------------------------------------
    out = pd.DataFrame({
        "id": panel["id"].to_numpy(),
        "q": panel["q"].to_numpy(),
        "y": y.to_numpy(dtype=float),
    })
    out = out[out["y"].notna()].copy()
    out["date"] = _qend(out["q"].to_numpy()).to_numpy()
    out = (out[["id", "date", "y"]]
           .sort_values(["id", "date"])
           .reset_index(drop=True))
    out["id"] = out["id"].astype("int64")
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    out["y"] = out["y"].astype(float)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    calls_df = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    dir_df = pd.read_csv(os.path.join(here, "sample_directory.csv"))
    hist_df = pd.read_csv(os.path.join(here, "sample_history.csv"))
    panel_df = build_panel(calls_df, dir_df, hist_df)
    print(panel_df.head())
    print(f"rows={len(panel_df)} banks={panel_df['id'].nunique()}")
