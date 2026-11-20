"""Build a clean quarterly panel of commercial banks from Call Report ratios.

Only pandas and numpy are used.

Steps
-----
1. Map every report date and attribute snapshot to an integer quarter index.
2. Assign each bank-quarter a charter status from the latest attribute snapshot
   on or before that quarter (no look-ahead). Quarters before a bank's first
   snapshot take the first snapshot. Banks with unknown charter are dropped.
3. Keep only commercial bank-quarters. Duplicate reports within a quarter keep
   the last non-missing value.
4. Within each bank's commercial span (first to last observed quarter), measure
   coverage = observed quarters / commercial quarters in that span. Keep banks
   with at least MIN_OBS observed quarters and coverage >= MIN_COVERAGE.
5. Fill interior gaps of up to MAX_GAP quarters by linear interpolation, only on
   commercial quarters. Longer gaps stay missing, and those rows are dropped so
   that the output has no missing y.
"""

import numpy as np
import pandas as pd

MAX_GAP = 2          # longest interior gap (in quarters) that gets interpolated
MIN_COVERAGE = 0.8   # min share of commercial quarters in a bank's span that are observed
MIN_OBS = 12         # min number of observed quarters for a bank to be kept


def _qindex(yyyymmdd):
    """Integer quarter index (year*4 + quarter-1) from YYYYMMDD integers."""
    s = np.asarray(yyyymmdd, dtype="int64")
    year = s // 10000
    month = (s // 100) % 100
    return year * 4 + (month - 1) // 3


def _qend(qi):
    """Calendar quarter-end Timestamps from integer quarter indices."""
    qi = np.asarray(qi, dtype="int64")
    per = pd.PeriodIndex.from_fields(year=qi // 4, quarter=qi % 4 + 1, freq="Q")
    return pd.DatetimeIndex(per.to_timestamp(how="end").normalize())


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- 1. reports: one value per bank-quarter (last non-missing filing) ----
    c = pd.DataFrame({
        "rssd_id": calls["rssd_id"].to_numpy(dtype="int64"),
        "report_date": calls["report_date"].to_numpy(dtype="int64"),
        "value": calls["value"].to_numpy(dtype="float64"),
    })
    c["qi"] = _qindex(c["report_date"])
    c = c.sort_values(["rssd_id", "qi", "report_date"], kind="stable")
    c = c.groupby(["rssd_id", "qi"], sort=True)["value"].last().reset_index()

    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    if c.empty:
        return empty

    # ---- 2. charter status per bank-quarter from attribute snapshots ----
    a = pd.DataFrame({
        "rssd_id": attrs["rssd_id"].to_numpy(dtype="int64"),
        "as_of": attrs["as_of"].to_numpy(dtype="int64"),
        "charter": attrs["charter"].astype(str).to_numpy(),
    })
    a["qa"] = _qindex(a["as_of"])
    a = a.sort_values(["rssd_id", "as_of"], kind="stable")
    a = a.drop_duplicates(["rssd_id", "qa"], keep="last")
    a = a.sort_values("qa", kind="stable")

    banks = np.sort(c["rssd_id"].unique())
    a = a[a["rssd_id"].isin(banks)]
    qs = np.arange(c["qi"].min(), c["qi"].max() + 1, dtype="int64")

    grid = pd.DataFrame({
        "rssd_id": np.repeat(banks, len(qs)),
        "qi": np.tile(qs, len(banks)),
    }).sort_values("qi", kind="stable")  # merge_asof needs the 'on' key sorted

    right = a[["rssd_id", "qa", "charter"]]
    back = pd.merge_asof(grid, right, left_on="qi", right_on="qa",
                         by="rssd_id", direction="backward")
    fwd = pd.merge_asof(grid, right, left_on="qi", right_on="qa",
                        by="rssd_id", direction="forward")
    grid["charter"] = back["charter"].fillna(fwd["charter"]).to_numpy()
    grid["comm"] = (grid["charter"] == "commercial").to_numpy()

    grid = grid.merge(c, on=["rssd_id", "qi"], how="left")

    # ---- wide bank x quarter matrices ----
    Y = grid.pivot(index="rssd_id", columns="qi", values="value")
    C = grid.pivot(index="rssd_id", columns="qi", values="comm").astype(bool)
    C = C.reindex(index=Y.index, columns=Y.columns, fill_value=False)
    Y = Y.where(C)  # non-commercial quarters are not part of the panel

    # ---- 3. coverage filter within each bank's commercial span ----
    obs = Y.notna().to_numpy()
    ncol = obs.shape[1]
    has = obs.any(axis=1)
    first = np.where(has, obs.argmax(axis=1), 0)
    last = np.where(has, ncol - 1 - obs[:, ::-1].argmax(axis=1), -1)
    colidx = np.arange(ncol)
    span = (colidx[None, :] >= first[:, None]) & (colidx[None, :] <= last[:, None])
    n_obs = obs.sum(axis=1)
    n_elig = (C.to_numpy() & span).sum(axis=1)
    cov = n_obs / np.maximum(n_elig, 1)
    keep = has & (n_obs >= MIN_OBS) & (cov >= MIN_COVERAGE)

    if not keep.any():
        return empty

    Yk = Y.loc[keep]
    Ck = C.loc[keep]

    # ---- 4. short-gap interpolation, restricted to commercial quarters ----
    Yi = Yk.interpolate(axis=1, method="linear", limit=MAX_GAP, limit_area="inside")
    Yi = Yi.where(Ck)

    arr = Yi.to_numpy(dtype="float64")
    ii, jj = np.nonzero(~np.isnan(arr))
    if ii.size == 0:
        return empty

    out = pd.DataFrame({
        "id": Yi.index.to_numpy(dtype="int64")[ii],
        "date": _qend(Yi.columns.to_numpy(dtype="int64")[jj]),
        "y": arr[ii, jj],
    })
    out = out.sort_values(["id", "date"], kind="stable").reset_index(drop=True)
    return out
