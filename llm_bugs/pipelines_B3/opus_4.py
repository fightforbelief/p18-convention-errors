"""Quarterly panel of Midwest banks built from Call Report ratios.

Design choices
--------------
* Geography is assigned point-in-time from the headquarters *history*, not from the
  end-of-sample directory. The directory only lists surviving institutions with their
  final state, so using it would drop failed/merged banks (survivorship bias) and
  mislabel banks that moved into or out of the Midwest.
* Each report is mapped to an integer quarter index; the panel is on a regular quarterly
  grid. Missing quarters and NaN values are gaps. Gaps of at most MAX_GAP quarters are
  filled by linear interpolation; longer gaps, or a spell outside the Midwest, break the
  series and only the most recent contiguous segment of each bank is kept (the benchmark
  holds out each bank's last 4 quarters, so the series must be contiguous up to its end).
* The kept segment is trimmed so that its last HOLDOUT quarters are genuinely reported
  (no interpolated values in the evaluation window).
* Banks whose kept segment is shorter than MIN_QUARTERS, or with less than MIN_OBS_FRAC
  of its quarters actually observed, are dropped.
"""
import numpy as np
import pandas as pd

MIDWEST = ("IL", "IN", "IA", "MI", "MN", "OH", "WI")
MAX_GAP = 2          # longest run of missing quarters that is interpolated
HOLDOUT = 4          # quarters held out by the benchmark; must be real observations
MIN_TRAIN = 16       # minimum training quarters before the holdout
MIN_QUARTERS = MIN_TRAIN + HOLDOUT
MIN_OBS_FRAC = 0.8   # minimum share of observed (non-interpolated) quarters


def _quarter_index(yyyymmdd: pd.Series) -> np.ndarray:
    v = pd.to_numeric(yyyymmdd, errors="coerce").to_numpy(dtype="float64")
    year = np.floor(v / 10000)
    month = np.floor(v / 100) % 100
    return year * 4 + np.floor((month - 1) / 3)  # NaN stays NaN


def _assign_state(calls: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """Point-in-time headquarters state for each call row (NaN if not covered)."""
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h["start_date"] = pd.to_numeric(h["start_date"], errors="coerce")
    h["end_date"] = pd.to_numeric(h["end_date"], errors="coerce").fillna(99991231)
    h = h.dropna(subset=["start_date"])
    h["rssd_id"] = h["rssd_id"].astype("int64")
    h["start_date"] = h["start_date"].astype("int64")
    h = h.sort_values(["start_date", "rssd_id"])

    c = calls[["rssd_id", "report_date"]].copy()
    c["_row"] = np.arange(len(c))
    c = c.sort_values(["report_date", "rssd_id"])
    m = pd.merge_asof(c, h, left_on="report_date", right_on="start_date",
                      by="rssd_id", direction="backward")
    ok = m["report_date"] <= m["end_date"]
    state = pd.Series(np.where(ok, m["state"], None), index=m["_row"].to_numpy())
    return state.sort_index().set_axis(calls.index)


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # `directory` is deliberately not used for membership (see module docstring).
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})

    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c.dropna(subset=["rssd_id", "report_date"])
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = pd.to_numeric(c["report_date"], errors="coerce")
    c = c.dropna(subset=["report_date"])
    c["report_date"] = c["report_date"].astype("int64")
    c["value"] = pd.to_numeric(c["value"], errors="coerce").astype("float64")
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    c["q"] = _quarter_index(c["report_date"]).astype("int64")
    if c.empty:
        return empty

    # One record per bank-quarter: prefer the latest non-missing report.
    c = c.sort_values(["rssd_id", "q", "report_date"])
    c["_has"] = c["value"].notna()
    c = c.sort_values(["rssd_id", "q", "_has", "report_date"])
    c = c.drop_duplicates(["rssd_id", "q"], keep="last").drop(columns="_has")
    c = c.sort_values(["rssd_id", "q"]).reset_index(drop=True)

    # Point-in-time Midwest flag; runs of consecutive reports with the same flag.
    c["mw"] = _assign_state(c, history).isin(MIDWEST).to_numpy()
    new_bank = c["rssd_id"].ne(c["rssd_id"].shift())
    run = (new_bank | c["mw"].ne(c["mw"].shift())).cumsum()

    # Observed Midwest values only; split segments on long gaps or leaving the Midwest.
    o = c.loc[c["mw"] & c["value"].notna(), ["rssd_id", "q", "value"]].copy()
    o["run"] = run[o.index].to_numpy()
    if o.empty:
        return empty
    brk = (o["rssd_id"].ne(o["rssd_id"].shift())
           | o["run"].ne(o["run"].shift())
           | (o["q"].diff() > MAX_GAP + 1))
    o["seg"] = brk.cumsum()
    last_seg = o.groupby("rssd_id")["seg"].transform("max")
    o = o[o["seg"] == last_seg]

    # Expand each bank's segment to a full quarterly grid.
    span = o.groupby("rssd_id")["q"].agg(["min", "max"])
    lens = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    qs = np.repeat(span["min"].to_numpy(), lens) + offs
    g = pd.DataFrame({"rssd_id": ids, "q": qs})
    g = g.merge(o[["rssd_id", "q", "value"]], on=["rssd_id", "q"], how="left")
    g["obs"] = g["value"].notna()
    # Every segment starts and ends on an observation, so a frame-wide linear
    # interpolation only fills interior gaps of a single bank (positions = quarters).
    g["y"] = g["value"].interpolate(method="linear", limit_area="inside")

    # Trim so the last HOLDOUT quarters are all observed.
    grp_start = g["rssd_id"].ne(g["rssd_id"].shift())
    key = ((~g["obs"]) | grp_start).cumsum()
    streak = g["obs"].astype("int64").groupby(key).cumsum()  # consecutive observed run
    g["ok_end"] = streak >= HOLDOUT
    pos = pd.Series(np.arange(len(g)))
    last_ok = pos.where(g["ok_end"]).groupby(g["rssd_id"]).transform("max")
    g = g[pos <= last_ok].copy()

    # Coverage filters.
    stats = g.groupby("rssd_id")["obs"].agg(["size", "mean"])
    keep = stats.index[(stats["size"] >= MIN_QUARTERS) & (stats["mean"] >= MIN_OBS_FRAC)]
    g = g[g["rssd_id"].isin(keep)]
    if g.empty:
        return empty

    q = g["q"].to_numpy()
    year, qtr = q // 4, q % 4
    date = (pd.to_datetime({"year": year, "month": qtr * 3 + 1, "day": 1})
            + pd.offsets.QuarterEnd(0)).dt.normalize()
    out = pd.DataFrame({"id": g["rssd_id"].astype("int64").to_numpy(),
                        "date": date.to_numpy().astype("datetime64[ns]"),
                        "y": g["y"].astype("float64").to_numpy()})
    return out.sort_values(["id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    import os
    import time
    here = os.path.dirname(os.path.abspath(__file__))
    t = time.time()
    panel = build_panel(pd.read_csv(os.path.join(here, "sample_calls.csv")),
                        pd.read_csv(os.path.join(here, "sample_directory.csv")),
                        pd.read_csv(os.path.join(here, "sample_history.csv")))
    print(panel.head(), panel.shape, panel["id"].nunique(), f"{time.time() - t:.2f}s")
