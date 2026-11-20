"""Quarterly commercial-bank panel builder (pandas + numpy only)."""
import numpy as np
import pandas as pd

MAX_GAP = 4        # interior gaps of <= MAX_GAP missing quarters are linearly interpolated
MIN_LEN = 20       # min quarters in the final regular series (4 held out + >=16 history)
MIN_OBS = 12       # min genuinely reported (non-interpolated) quarters
MIN_FILL = 0.75    # min share of the final series that is genuinely reported


def _quarter_index(yyyymmdd):
    s = pd.to_datetime(yyyymmdd.astype("int64").astype(str), format="%Y%m%d", errors="coerce")
    return s, (s.dt.year * 4 + s.dt.quarter - 1)


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c.dropna(subset=["rssd_id", "report_date"])
    d, q = _quarter_index(c["report_date"])
    c["d"], c["q"] = d, q
    c = c[d.notna()].copy()
    c["q"] = c["q"].astype("int64")
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["value"] = pd.to_numeric(c["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # ---- charter as of each report date (latest snapshot at or before; earliest if none before)
    a = attrs[["rssd_id", "as_of", "charter"]].dropna().copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["as_of_d"] = pd.to_datetime(a["as_of"].astype("int64").astype(str), format="%Y%m%d", errors="coerce")
    a = a.dropna(subset=["as_of_d"]).sort_values(["as_of_d", "rssd_id"])
    c = c.sort_values(["d", "rssd_id"])
    c = pd.merge_asof(c, a[["rssd_id", "as_of_d", "charter"]], left_on="d", right_on="as_of_d",
                      by="rssd_id", direction="backward")
    first = a.sort_values("as_of_d").groupby("rssd_id")["charter"].first()
    c["charter"] = c["charter"].fillna(c["rssd_id"].map(first))
    c = c[c["charter"].astype(str).str.lower().eq("commercial")]

    # ---- one value per bank-quarter (latest filing wins; prefer non-missing)
    c = c.sort_values(["rssd_id", "q", "value", "d"], na_position="first")
    c = c.drop_duplicates(["rssd_id", "q"], keep="last")
    obs = c.dropna(subset=["value"])  # missing reports carry no information
    if obs.empty:
        return empty
    obs = obs.sort_values(["rssd_id", "q"])

    # ---- regular quarterly grid from each bank's first to last observed quarter (no extrapolation)
    g = obs.groupby("rssd_id")["q"].agg(["min", "max"])
    n = (g["max"] - g["min"] + 1).to_numpy()
    ids = np.repeat(g.index.to_numpy(), n)
    start = np.repeat(g["min"].to_numpy(), n)
    off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    grid = pd.DataFrame({"id": ids, "q": start + off, "pos": off})
    grid = grid.merge(obs[["rssd_id", "q", "value"]].rename(columns={"rssd_id": "id", "value": "y"}),
                      on=["id", "q"], how="left")
    grid["obs"] = grid["y"].notna()

    # ---- interpolate short interior gaps; vectorised linear interpolation
    p = grid["pos"].where(grid["obs"])
    prev_p = p.groupby(grid["id"]).ffill()
    next_p = p.groupby(grid["id"]).bfill()
    prev_y = grid["y"].groupby(grid["id"]).ffill()
    next_y = grid["y"].groupby(grid["id"]).bfill()
    gap = (next_p - prev_p - 1)  # length of the surrounding gap (NaN on obs rows irrelevant)
    w = (grid["pos"] - prev_p) / (next_p - prev_p)
    interp = prev_y + w * (next_y - prev_y)
    fill = (~grid["obs"]) & (gap <= MAX_GAP)
    grid["y"] = grid["y"].where(grid["obs"], interp.where(fill))

    # ---- keep only the most recent contiguous stretch (after the last unfillable gap)
    bad_pos = grid["pos"].where(grid["y"].isna())
    last_bad = bad_pos.groupby(grid["id"]).transform("max").fillna(-1)
    grid = grid[grid["pos"] > last_bad].copy()

    # ---- coverage filter
    stat = grid.groupby("id").agg(n=("y", "size"), nobs=("obs", "sum"))
    keep = stat[(stat["n"] >= MIN_LEN) & (stat["nobs"] >= MIN_OBS) & (stat["nobs"] / stat["n"] >= MIN_FILL)].index
    grid = grid[grid["id"].isin(keep)]
    if grid.empty:
        return empty

    year, qq = grid["q"] // 4, grid["q"] % 4
    date = pd.to_datetime(pd.DataFrame({"year": year, "month": qq * 3 + 3, "day": 1})) + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"id": grid["id"].astype("int64").to_numpy(), "date": date.to_numpy(),
                        "y": grid["y"].astype(float).to_numpy()})
    out = out.dropna(subset=["y"]).sort_values(["id", "date"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    import sys
    cc = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sample_calls.csv")
    aa = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "sample_attrs.csv")
    print(build_panel(cc, aa).describe(include="all"))
