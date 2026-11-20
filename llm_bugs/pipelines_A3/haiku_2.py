"""Quarterly panel of Midwest banks, built from Call Report ratios (see spec.md).

Only pandas and numpy are used. Quarters are represented internally as integer
calendar-quarter indices q = year * 4 + (quarter - 1), which makes grids and
joins cheap on the full ~2M-row history.
"""

import numpy as np
import pandas as pd

MIDWEST = ("IL", "IN", "IA", "MI", "MN", "OH", "WI")
MISSING_SHARE_LIMIT = 0.20


def _quarter_index(yyyymmdd):
    """YYYYMMDD integers -> calendar-quarter index."""
    d = np.asarray(yyyymmdd, dtype=np.int64)
    year = d // 10000
    month = (d // 100) % 100
    return year * 4 + (month - 1) // 3


def _quarter_end_int(q):
    """Calendar-quarter index -> quarter-end date as YYYYMMDD integer."""
    q = np.asarray(q, dtype=np.int64)
    year = q // 4
    month = (q % 4) * 3 + 3
    day = np.where((month == 6) | (month == 9), 30, 31)
    return year * 10000 + month * 100 + day


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Build the Midwest bank-quarter panel.

    `directory` is accepted for interface compatibility but deliberately unused:
    it only lists institutions still operating at the end of the sample, so it
    cannot be used to select banks or quarters (rule 2).
    """
    # ---- Calls: one value per bank-quarter -------------------------------
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype("int64").to_numpy(),
        "report_date": calls["report_date"].astype("int64").to_numpy(),
        "value": calls["value"].astype("float64").to_numpy(),
    })
    c["q"] = _quarter_index(c["report_date"].to_numpy())
    # If a quarter has several reports, keep the latest non-missing value.
    c = c.sort_values(["id", "report_date"], kind="mergesort")
    vals = c.groupby(["id", "q"], sort=True)["value"].last().reset_index()

    # ---- Rule 1: regular quarterly grid over each bank's active span ------
    span = c.groupby("id", sort=True)["q"].agg(["min", "max"]).reset_index()
    lengths = (span["max"] - span["min"] + 1).to_numpy(dtype=np.int64)
    ids = np.repeat(span["id"].to_numpy(), lengths)
    first = np.repeat(span["min"].to_numpy(dtype=np.int64), lengths)
    offsets = np.arange(lengths.sum(), dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    grid = pd.DataFrame({"id": ids, "q": first + offsets})

    # ---- Rule 2: keep quarters whose HQ state (at quarter-end) is Midwest --
    h = pd.DataFrame({
        "id": history["rssd_id"].astype("int64").to_numpy(),
        "start": pd.to_numeric(history["start_date"]).astype("int64").to_numpy(),
        "end": pd.to_numeric(history["end_date"]).fillna(99991231).astype("int64").to_numpy(),
        "state": history["state"].astype(str).str.strip().str.upper().to_numpy(),
    })
    h = h[h["state"].isin(MIDWEST)][["id", "start", "end"]]
    g = grid.merge(h, on="id", how="inner")
    qe = _quarter_end_int(g["q"].to_numpy())
    in_spell = (g["start"].to_numpy() <= qe) & (qe <= g["end"].to_numpy())
    kept = g.loc[in_spell, ["id", "q"]].drop_duplicates(["id", "q"])

    # ---- Attach reported values (missing where no report or NaN) ----------
    panel = kept.merge(vals, on=["id", "q"], how="left")

    # ---- Rule 3: drop banks with more than 20% missing remaining quarters -
    missing = panel["value"].isna()
    stats = missing.groupby(panel["id"]).agg(["sum", "size"])
    bad_ids = stats.index[stats["sum"] / stats["size"] > MISSING_SHARE_LIMIT]
    panel = panel[~panel["id"].isin(bad_ids)]

    # ---- Rule 4: drop missing bank-quarters; no imputation ----------------
    panel = panel.dropna(subset=["value"])

    # ---- Output ------------------------------------------------------------
    q_arr = panel["q"].to_numpy(dtype=np.int64)
    uq = np.unique(q_arr)
    uq_dates = pd.to_datetime(pd.Series(_quarter_end_int(uq)).astype(str), format="%Y%m%d")
    dates = uq_dates.to_numpy()[np.searchsorted(uq, q_arr)]

    out = pd.DataFrame({
        "id": panel["id"].to_numpy(dtype=np.int64),
        "date": pd.to_datetime(dates),
        "y": panel["value"].to_numpy(dtype=np.float64),
    })
    out = out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    return out
