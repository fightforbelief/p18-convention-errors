"""Build a quarterly panel of Midwest banks from Call Report ratios.

Rules (see spec.md):
1. Each bank gets a regular quarterly grid from its first to its last report_date in `calls`.
2. Keep bank-quarters whose headquarters state (from `history`) is a Midwest state.
   `directory` is deliberately not used.
3. Drop banks where more than 20% of their remaining grid quarters are missing.
4. Drop missing bank-quarters; values are not imputed or transformed.

Only pandas and numpy are used.
"""

import os

import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}
MAX_MISSING_SHARE = 0.20
OPEN_END = 99991231  # sentinel for history spells with empty end_date
QEND_DAY = np.array([31, 30, 30, 31])  # day of month for Q1..Q4 ends (Mar, Jun, Sep, Dec)


def _quarter_index(yyyymmdd: np.ndarray) -> np.ndarray:
    """Map YYYYMMDD integers to an absolute quarter index (year*4 + quarter - 1)."""
    year = yyyymmdd // 10000
    month = (yyyymmdd // 100) % 100
    return year * 4 + (month - 1) // 3


def _quarter_end_int(qidx: np.ndarray) -> np.ndarray:
    """Map an absolute quarter index to the calendar quarter-end date as YYYYMMDD int."""
    year = qidx // 4
    q = qidx % 4
    return year * 10000 + (q * 3 + 3) * 100 + QEND_DAY[q]


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Return a bank-quarter panel with columns id (int), date (quarter-end Timestamp), y (float).

    `directory` is accepted for interface compatibility but intentionally not used:
    it only lists institutions still operating at the end of the sample.
    """
    # ---- Calls: normalise types, map report dates to quarter indices -------------
    c = pd.DataFrame(
        {
            "id": calls["rssd_id"].astype("int64").to_numpy(),
            "qidx": _quarter_index(calls["report_date"].astype("int64").to_numpy()),
            "value": pd.to_numeric(calls["value"], errors="coerce").astype("float64").to_numpy(),
        }
    )
    # Non-finite ratios are treated as not reported.
    c["value"] = c["value"].where(np.isfinite(c["value"]))

    # Rule 1: active span from first to last report in calls (NaN-valued reports count).
    span = c.groupby("id")["qidx"].agg(["min", "max"]).sort_index()

    # Several reports for one bank-quarter: keep the first non-missing value in input order.
    c["_na"] = c["value"].isna()
    c = (
        c.sort_values(["id", "qidx", "_na"], kind="mergesort")
        .drop_duplicates(["id", "qidx"], keep="first")
        [["id", "qidx", "value"]]
    )

    # ---- Rule 1: regular quarterly grid over each bank's active span ------------
    lo = span["min"].to_numpy(np.int64)
    hi = span["max"].to_numpy(np.int64)
    ids = span.index.to_numpy(np.int64)
    n = hi - lo + 1
    starts = np.cumsum(n) - n
    within = np.arange(n.sum(), dtype=np.int64) - np.repeat(starts, n)
    grid = pd.DataFrame(
        {
            "id": np.repeat(ids, n),
            "qidx": np.repeat(lo, n) + within,
        }
    )
    grid["date_int"] = _quarter_end_int(grid["qidx"].to_numpy())

    # ---- Rule 2: headquarters state at each quarter from history ----------------
    hist = pd.DataFrame(
        {
            "id": history["rssd_id"].astype("int64").to_numpy(),
            "h_start": history["start_date"].astype("int64").to_numpy(),
            "h_end": history["end_date"].fillna(OPEN_END).astype("int64").to_numpy(),
            "state": history["state"].astype(str).str.strip().str.upper().to_numpy(),
        }
    )
    # Spells are inclusive on both ends; a spell ending 19971230 does not cover the
    # 19971231 quarter-end, which falls in the next spell.
    kept = grid.merge(hist, on="id", how="inner")
    covered = (kept["h_start"] <= kept["date_int"]) & (kept["date_int"] <= kept["h_end"])
    kept = kept.loc[covered]
    kept = (
        kept.sort_values(["id", "qidx", "h_start"], kind="mergesort")
        .drop_duplicates(["id", "qidx"], keep="last")
    )
    kept = kept.loc[kept["state"].isin(MIDWEST), ["id", "qidx", "date_int", "state"]]

    # Attach reported values to the surviving bank-quarters (missing if not reported).
    kept = kept.merge(c, on=["id", "qidx"], how="left")

    # ---- Rule 3: drop banks with too many missing quarters after rule 2 ---------
    counts = kept.groupby("id")["value"].agg(
        n_rem="size", n_miss=lambda s: int(s.isna().sum())
    )
    bad_ids = counts.index[counts["n_miss"] > MAX_MISSING_SHARE * counts["n_rem"]]
    kept = kept.loc[~kept["id"].isin(bad_ids)]

    # ---- Rule 4: drop missing bank-quarters, no imputation ----------------------
    kept = kept.loc[kept["value"].notna()]

    out = pd.DataFrame(
        {
            "id": kept["id"].to_numpy(np.int64),
            "date": pd.to_datetime(kept["date_int"].astype("int64").astype(str), format="%Y%m%d"),
            "y": kept["value"].to_numpy(np.float64),
        }
    )
    out = out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    calls_df = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    dir_df = pd.read_csv(os.path.join(here, "sample_directory.csv"))
    hist_df = pd.read_csv(os.path.join(here, "sample_history.csv"))
    panel = build_panel(calls_df, dir_df, hist_df)
    print(panel.head())
    print(f"rows={len(panel)} banks={panel['id'].nunique()}")
