"""Quarterly panel of Midwest banks from Call Report data.

Design decisions
----------------
* Geography is point-in-time: a bank-quarter is in the sample if the bank was
  headquartered in a Midwest state *at that quarter* according to `history`.
  The end-of-sample `directory` only lists survivors (and their final state), so
  using it would introduce survivorship and look-ahead bias; it is used only as
  a fallback for banks that have no headquarters history at all.
* Report dates are snapped to the calendar quarter; duplicate filings for the
  same bank-quarter keep the last non-missing value. Non-finite values are
  treated as missing.
* Each bank is put on a regular quarterly grid. Short internal gaps (at most
  MAX_GAP consecutive missing quarters) are filled by linear interpolation in
  time. A longer gap, or a spell headquartered outside the Midwest, breaks the
  series; only the most recent unbroken segment is kept so the held-out last
  4 quarters are the bank's latest data and the series is contiguous.
  Leading/trailing missing quarters are never extrapolated.
* Banks are dropped when the kept segment is shorter than MIN_QUARTERS
  (4 holdout quarters + 12 training quarters) or when fewer than MIN_OBS_FRAC
  of its quarters are genuinely reported (rather than interpolated).
"""
import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}
MAX_GAP = 2            # max consecutive missing quarters filled by interpolation
HOLDOUT = 4
MIN_TRAIN = 12
MIN_QUARTERS = HOLDOUT + MIN_TRAIN
MIN_OBS_FRAC = 0.75


def _to_int_date(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _quarter_index(d: np.ndarray) -> np.ndarray:
    """YYYYMMDD int -> integer quarter index year*4 + (quarter-1)."""
    y = d // 10000
    m = (d // 100) % 100
    return y * 4 + (m - 1) // 3


def _quarter_end_int(qi: np.ndarray) -> np.ndarray:
    y = qi // 4
    q = qi % 4
    mmdd = np.array([331, 630, 930, 1231])[q]
    return y * 10000 + mmdd


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})

    # ---- calls: clean, snap to quarter, dedupe ---------------------------
    c = pd.DataFrame({
        "id": pd.to_numeric(calls["rssd_id"], errors="coerce"),
        "rd": _to_int_date(calls["report_date"]),
        "value": pd.to_numeric(calls["value"], errors="coerce"),
    })
    c = c.dropna(subset=["id", "rd"])
    if c.empty:
        return empty
    c["id"] = c["id"].astype("int64")
    c["rd"] = c["rd"].astype("int64")
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    c["qi"] = _quarter_index(c["rd"].to_numpy())
    c = c.sort_values(["id", "qi", "rd"], kind="mergesort")
    c = c.groupby(["id", "qi"], sort=True, as_index=False)["value"].last()  # last non-null
    c["qe"] = _quarter_end_int(c["qi"].to_numpy())

    # ---- point-in-time headquarters state --------------------------------
    h = pd.DataFrame({
        "id": pd.to_numeric(history["rssd_id"], errors="coerce"),
        "start": _to_int_date(history["start_date"]),
        "end": _to_int_date(history["end_date"]),
        "hstate": history["state"].astype(str).str.strip().str.upper(),
    }).dropna(subset=["id"])
    h["id"] = h["id"].astype("int64")
    h["start"] = h["start"].fillna(0).astype("int64")
    h["end"] = h["end"].fillna(99999999).astype("int64")

    m = c[["id", "qi", "qe"]].reset_index().merge(h, on="id", how="inner")
    m = m[(m["qe"] >= m["start"]) & (m["qe"] <= m["end"])]
    # if spells overlap, the most recently started one wins
    m = m.sort_values(["index", "start"]).drop_duplicates("index", keep="last")
    c["state"] = pd.Series(m["hstate"].to_numpy(), index=m["index"].to_numpy()).reindex(c.index)

    # fallback: banks with no history at all -> directory state
    d = pd.DataFrame({"id": pd.to_numeric(directory["rssd_id"], errors="coerce"),
                      "dstate": directory["state"].astype(str).str.strip().str.upper()})
    d = d.dropna(subset=["id"]).astype({"id": "int64"}).drop_duplicates("id", keep="last")
    no_hist = ~c["id"].isin(h["id"].unique())
    if no_hist.any():
        c.loc[no_hist, "state"] = c.loc[no_hist, "id"].map(d.set_index("id")["dstate"])
    c["mw"] = c["state"].isin(MIDWEST)

    # ---- segment each bank's Midwest series -------------------------------
    # number of known non-Midwest quarters seen so far (per bank); a change
    # between two valid observations means the bank left the Midwest in between
    c["out_cnt"] = (~c["mw"]).astype(np.int64).groupby(c["id"]).cumsum()
    v = c[c["mw"] & c["value"].notna()][["id", "qi", "value", "out_cnt"]].reset_index(drop=True)
    if v.empty:
        return empty
    new_bank = v["id"].ne(v["id"].shift())
    brk = new_bank | (v["qi"].diff() > MAX_GAP + 1) | v["out_cnt"].ne(v["out_cnt"].shift())
    v["seg"] = brk.cumsum()
    last_seg = v.groupby("id")["seg"].transform("max")
    v = v[v["seg"] == last_seg]

    # ---- regular quarterly grid + interpolation ----------------------------
    bounds = v.groupby("id")["qi"].agg(["min", "max", "count"])
    length = bounds["max"] - bounds["min"] + 1
    keep = (length >= MIN_QUARTERS) & (bounds["count"] >= MIN_OBS_FRAC * length)
    bounds, length = bounds[keep], length[keep]
    if bounds.empty:
        return empty
    v = v[v["id"].isin(bounds.index)]

    ids = np.repeat(bounds.index.to_numpy(), length.to_numpy())
    starts = np.repeat(bounds["min"].to_numpy(), length.to_numpy())
    offs = np.arange(len(ids)) - np.repeat(np.cumsum(length.to_numpy()) - length.to_numpy(),
                                           length.to_numpy())
    g = pd.DataFrame({"id": ids, "qi": starts + offs})
    g = g.merge(v[["id", "qi", "value"]], on=["id", "qi"], how="left")

    # linear interpolation in time; segments start and end with observed
    # values, so ffill/bfill never cross bank boundaries
    obs = g["value"].notna()
    q_prev = g["qi"].where(obs).ffill()
    q_next = g["qi"].where(obs).bfill()
    y_prev = g["value"].ffill()
    y_next = g["value"].bfill()
    w = ((g["qi"] - q_prev) / (q_next - q_prev)).fillna(0.0)
    g["y"] = np.where(obs, g["value"], y_prev + w * (y_next - y_prev))

    qe = _quarter_end_int(g["qi"].to_numpy())
    out = pd.DataFrame({
        "id": g["id"].astype("int64"),
        "date": pd.to_datetime(qe.astype(str), format="%Y%m%d").astype("datetime64[ns]"),
        "y": g["y"].astype("float64"),
    })
    out = out.dropna(subset=["y"]).sort_values(["id", "date"], kind="mergesort")
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    panel = build_panel(
        pd.read_csv(os.path.join(here, "sample_calls.csv")),
        pd.read_csv(os.path.join(here, "sample_directory.csv")),
        pd.read_csv(os.path.join(here, "sample_history.csv"), dtype={"end_date": "Int64"}),
    )
    print(panel.head())
    print(panel.shape, panel["id"].nunique(), "banks")
