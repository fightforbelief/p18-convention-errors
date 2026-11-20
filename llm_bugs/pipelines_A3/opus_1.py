"""Quarterly panel of Midwest banks built from Call Report data (see spec.md)."""
import numpy as np
import pandas as pd

MIDWEST = frozenset({"IL", "IN", "IA", "MI", "MN", "OH", "WI"})
MAX_MISSING_SHARE = 0.20


def _to_ts(s: pd.Series) -> pd.Series:
    """YYYYMMDD ints (possibly float / NaN) -> Timestamps (NaT for missing)."""
    s = pd.to_numeric(s, errors="coerce")
    out = pd.to_datetime(s.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")
    return out


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # `directory` is deliberately unused: it only lists survivors (rule 2).
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})

    # ---- 1. calls -> one value per bank-quarter -------------------------------------------
    rd = _to_ts(calls["report_date"])
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype("int64").to_numpy(),
        # quarter ordinal (same as pandas Period[Q] ordinal: 1970Q1 == 0)
        "qn": ((rd.dt.year - 1970) * 4 + (rd.dt.month - 1) // 3).to_numpy(),
        "y": pd.to_numeric(calls["value"], errors="coerce").astype("float64").to_numpy(),
    })
    c = c[rd.notna().to_numpy()]
    if c.empty:
        return empty
    c["qn"] = c["qn"].astype("int64")
    # If a bank-quarter was filed more than once, keep the last non-missing value.
    c["has"] = c["y"].notna()
    c = c.sort_values(["id", "qn", "has"], kind="stable").drop_duplicates(["id", "qn"], keep="last")

    # Regular quarterly grid over each bank's active span (first..last report_date).
    span = c.groupby("id")["qn"].agg(["min", "max"])
    lens = (span["max"] - span["min"] + 1).to_numpy()
    gid = np.repeat(span.index.to_numpy(), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    gqn = np.repeat(span["min"].to_numpy(), lens) + offs
    grid = pd.DataFrame({"id": gid, "qn": gqn})
    grid = grid.merge(c[["id", "qn", "y"]], on=["id", "qn"], how="left")
    grid["date"] = pd.PeriodIndex.from_ordinals(grid["qn"].to_numpy(), freq="Q").to_timestamp(how="end").normalize()

    # ---- 2. headquarters state at each quarter-end, from history -----------------------------
    h = pd.DataFrame({
        "id": history["rssd_id"].astype("int64").to_numpy(),
        "start": _to_ts(history["start_date"]).to_numpy(),
        "end": _to_ts(history["end_date"]).to_numpy(),
        "state": history["state"].astype("string").str.strip().str.upper().to_numpy(),
    })
    h = h[h["start"].notna()].sort_values("start", kind="stable")
    g = grid.sort_values("date", kind="stable")
    g = pd.merge_asof(g, h, left_on="date", right_on="start", by="id", direction="backward")
    in_spell = g["start"].notna() & (g["end"].isna() | (g["date"] <= g["end"]))
    g = g[in_spell & g["state"].isin(MIDWEST).fillna(False).astype(bool)]

    # ---- 3. drop banks with >20% missing among remaining quarters ----------------------------
    miss_share = g["y"].isna().groupby(g["id"]).mean()
    keep = miss_share.index[miss_share <= MAX_MISSING_SHARE]
    g = g[g["id"].isin(keep)]

    # ---- 4. drop missing bank-quarters -------------------------------------------------------
    g = g[g["y"].notna()]
    out = (g[["id", "date", "y"]].sort_values(["id", "date"], kind="stable").reset_index(drop=True))
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    out["date"] = out["date"].astype("datetime64[ns]")
    return out
