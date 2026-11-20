"""Quarterly panel of Midwest banks (see spec.md)."""
import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}
MAX_MISSING_SHARE = 0.20


def _to_qidx(yyyymmdd):
    """YYYYMMDD ints -> absolute quarter index (year*4 + quarter-1)."""
    d = np.asarray(yyyymmdd, dtype=np.int64)
    year = d // 10000
    month = (d // 100) % 100
    return year * 4 + (month - 1) // 3


def _qidx_to_qend_int(q):
    year = q // 4
    qq = q % 4
    month = (qq + 1) * 3
    day = np.where((month == 3) | (month == 12), 31, 30)
    return year * 10000 + month * 100 + day


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # directory is intentionally unused: it suffers from survivorship (rule 2).
    out_cols = ["id", "date", "y"]

    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c[c["rssd_id"].notna() & c["report_date"].notna()]
    c["rssd_id"] = c["rssd_id"].astype(np.int64)
    c["report_date"] = pd.to_numeric(c["report_date"], errors="coerce")
    c = c[c["report_date"].notna()]
    c["q"] = _to_qidx(c["report_date"].astype(np.int64))
    c["value"] = pd.to_numeric(c["value"], errors="coerce")
    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype=np.int64),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype=float)})

    # Rule 1: active span from first to last report (NaN-valued reports included).
    span = c.groupby("rssd_id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(np.int64), n)
    starts = np.repeat(span["min"].to_numpy(np.int64), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    grid = pd.DataFrame({"rssd_id": ids, "q": starts + offs})

    # Values: if a bank-quarter was filed more than once, keep the last non-NaN value.
    vals = (c.dropna(subset=["value"])
             .drop_duplicates(["rssd_id", "q"], keep="last")[["rssd_id", "q", "value"]])
    grid = grid.merge(vals, on=["rssd_id", "q"], how="left")
    grid["qend"] = _qidx_to_qend_int(grid["q"].to_numpy())

    # Rule 2: headquarters state at each quarter end, from history (spells inclusive).
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h = h[h["rssd_id"].notna()]
    h["rssd_id"] = h["rssd_id"].astype(np.int64)
    h["start_date"] = pd.to_numeric(h["start_date"], errors="coerce").fillna(0).astype(np.int64)
    h["end_date"] = (pd.to_numeric(h["end_date"].replace("", np.nan), errors="coerce")
                     .fillna(99999999).astype(np.int64))
    h["state"] = h["state"].astype(str).str.strip().str.upper()
    h = h[h["rssd_id"].isin(span.index)]

    m = grid[["rssd_id", "q", "qend"]].merge(h, on="rssd_id", how="inner")
    m = m[(m["start_date"] <= m["qend"]) & (m["qend"] <= m["end_date"])]
    # If spells overlap, the most recently started one wins.
    m = m.sort_values(["rssd_id", "q", "start_date"]).drop_duplicates(["rssd_id", "q"], keep="last")
    mw = m.loc[m["state"].isin(MIDWEST), ["rssd_id", "q"]]
    grid = grid.merge(mw, on=["rssd_id", "q"], how="inner")

    # Rule 3: drop banks with >20% missing among their remaining grid quarters.
    miss_share = grid["value"].isna().groupby(grid["rssd_id"]).mean()
    keep = miss_share.index[miss_share <= MAX_MISSING_SHARE]
    grid = grid[grid["rssd_id"].isin(keep)]

    # Rule 4: drop missing bank-quarters, no imputation.
    grid = grid[grid["value"].notna()]

    out = pd.DataFrame({
        "id": grid["rssd_id"].astype(np.int64).to_numpy(),
        "date": pd.to_datetime(grid["qend"].astype(str), format="%Y%m%d").astype("datetime64[ns]").to_numpy(),
        "y": grid["value"].astype(float).to_numpy(),
    })
    out = out.sort_values(["id", "date"]).reset_index(drop=True)
    return out[out_cols]
