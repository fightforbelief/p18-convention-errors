"""Quarterly panel of commercial banks built from Call Report ratios.

Only pandas and numpy are used. The panel is built on a wide bank x quarter
matrix, so the work stays vectorised for the full history (~24k banks).
"""
import os

import numpy as np
import pandas as pd

MIN_OBS = 16        # minimum reported (non-missing) quarters per bank
MIN_COVERAGE = 0.8  # share of quarters in the bank's active span that were reported
MAX_GAP = 2         # longest run of missing quarters that is linearly interpolated

_OUT_COLS = ["id", "date", "y"]


def _qidx(dates):
    """Map YYYYMMDD integers to a running calendar-quarter index (year*4 + quarter-1)."""
    d = np.asarray(dates, dtype=np.int64)
    year = d // 10000
    month = (d // 100) % 100
    return year * 4 + (month - 1) // 3


def _qidx_to_quarter_end(q):
    """Map quarter indices back to calendar quarter-end Timestamps (midnight)."""
    q = np.asarray(q, dtype=np.int64)
    frame = pd.DataFrame({"year": q // 4, "month": (q % 4 + 1) * 3, "day": 1})
    starts = pd.to_datetime(frame)
    return pd.DatetimeIndex(starts + pd.offsets.MonthEnd(0))


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # --- ratios: one value per bank-quarter -------------------------------
    c = pd.DataFrame({
        "rssd_id": calls["rssd_id"].astype(np.int64).to_numpy(),
        "report_date": calls["report_date"].astype(np.int64).to_numpy(),
        "value": pd.to_numeric(calls["value"], errors="coerce").astype(float).to_numpy(),
    })
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan   # treat +/-inf as missing
    c["q"] = _qidx(c["report_date"])
    c = c.sort_values(["rssd_id", "report_date"], kind="mergesort")
    # several filings in one quarter: keep the latest non-missing value
    vals = c.groupby(["rssd_id", "q"], sort=True)["value"].last()

    # --- charter snapshots: point-in-time classification -------------------
    a = pd.DataFrame({
        "rssd_id": attrs["rssd_id"].astype(np.int64).to_numpy(),
        "q": _qidx(attrs["as_of"]),
        "is_comm": (attrs["charter"].astype(str).str.strip().str.lower() == "commercial")
                   .astype(float).to_numpy(),
    })
    a = a.sort_values(["rssd_id", "q"], kind="mergesort").groupby(["rssd_id", "q"])["is_comm"].last()

    q_min = min(vals.index.get_level_values("q").min(), a.index.get_level_values("q").min())
    q_max = vals.index.get_level_values("q").max()
    qs = np.arange(q_min, q_max + 1, dtype=np.int64)
    banks = np.sort(np.unique(c["rssd_id"].to_numpy()))

    wide = vals.unstack("q").reindex(index=banks, columns=qs)
    charter = a.unstack("q").reindex(index=banks, columns=qs)
    # a snapshot holds from its quarter onward; quarters before the first snapshot take it too
    charter = charter.ffill(axis=1).bfill(axis=1)
    is_comm = charter.fillna(0.0).to_numpy() == 1.0

    # only commercial-bank quarters count
    X = wide.to_numpy(copy=True)
    X[~is_comm] = np.nan

    # --- coverage screen over each bank's active span -----------------------
    obs = ~np.isnan(X)
    n_obs = obs.sum(axis=1)
    has = n_obs > 0
    first = np.argmax(obs, axis=1)
    last = X.shape[1] - 1 - np.argmax(obs[:, ::-1], axis=1)
    span = np.where(has, last - first + 1, 1)
    cover = n_obs / span
    keep = has & (n_obs >= MIN_OBS) & (cover >= MIN_COVERAGE)
    if not keep.any():
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    Xk = X[keep]
    span_mask = (np.arange(X.shape[1])[None, :] >= first[keep, None]) & \
                (np.arange(X.shape[1])[None, :] <= last[keep, None])

    # --- gap filling: short interior gaps interpolated, long gaps drop the bank ----
    filled = pd.DataFrame(Xk).interpolate(axis=1, method="linear",
                                          limit=MAX_GAP, limit_area="inside").to_numpy()
    unfilled_inside = np.isnan(filled) & span_mask
    ok = ~unfilled_inside.any(axis=1)

    rows_ok = np.where(ok)[0]
    r, col = np.nonzero(span_mask[rows_ok] & ~np.isnan(filled[rows_ok]))
    r = rows_ok[r]
    bank_ids = banks[keep][r]
    quarter_dates = _qidx_to_quarter_end(qs)
    out = pd.DataFrame({
        "id": bank_ids.astype(np.int64),
        "date": pd.DatetimeIndex(quarter_dates[col]).values.astype("datetime64[ns]"),
        "y": filled[r, col].astype(float),
    })
    out = out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    return out[_OUT_COLS]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    calls_df = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    attrs_df = pd.read_csv(os.path.join(here, "sample_attrs.csv"))
    panel = build_panel(calls_df, attrs_df)
    print(panel.dtypes)
    print(panel.head())
    print("rows", len(panel), "banks", panel["id"].nunique())
