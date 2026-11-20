"""Quarterly panel of commercial banks for the forecasting benchmark.

build_panel(calls, attrs) -> DataFrame with columns id, date, y.

Steps
1. Map each report_date to its calendar quarter end. If a bank filed several
   reports for the same quarter, keep the latest non-missing value.
2. Put the data on a regular quarterly grid spanning the whole sample.
3. Assign each bank-quarter a charter: the latest attrs snapshot on or before
   the quarter end (the bank's earliest snapshot if none precedes it). Quarters
   whose charter is not "commercial" are treated as missing. Banks with no
   attrs are dropped.
4. Coverage screen over each bank's own span (first to last observed quarter):
   keep banks with at least MIN_OBS observed quarters and coverage of at least
   MIN_COVERAGE.
5. Fill interior gaps of at most MAX_GAP consecutive quarters by linear
   interpolation. Longer gaps stay missing.
6. Return only rows with a non-missing y, sorted by id and date.
"""

import numpy as np
import pandas as pd

MIN_OBS = 20         # minimum observed quarters per bank
MIN_COVERAGE = 0.8   # minimum share of quarters observed within a bank's span
MAX_GAP = 2          # longest interior gap (in quarters) that gets interpolated

OUT_COLUMNS = ["id", "date", "y"]


def _quarter_end(yyyymmdd):
    """Map YYYYMMDD integers to the normalized timestamp of their quarter end."""
    s = pd.to_numeric(pd.Series(yyyymmdd), errors="coerce").astype("Int64")
    ts = pd.to_datetime(s.astype("string"), format="%Y%m%d", errors="coerce")
    return ts.dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()


def _run_counts(mask):
    """For each cell, the number of consecutive True cells ending at it (along axis 1)."""
    out = np.zeros(mask.shape, dtype=np.int64)
    cur = np.zeros(mask.shape[0], dtype=np.int64)
    for t in range(mask.shape[1]):
        cur = np.where(mask[:, t], cur + 1, 0)
        out[:, t] = cur
    return out


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({
        "id": pd.Series(dtype="int64"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "y": pd.Series(dtype="float64"),
    })
    if len(calls) == 0:
        return empty

    # 1. Outcome per bank-quarter ------------------------------------------
    c = pd.DataFrame({
        "id": pd.to_numeric(calls["rssd_id"]).astype("int64"),
        "report_date": pd.to_numeric(calls["report_date"]).astype("int64"),
        "value": pd.to_numeric(calls["value"], errors="coerce").astype("float64"),
    })
    c["value"] = c["value"].where(np.isfinite(c["value"]))
    c["q"] = _quarter_end(c["report_date"])
    c = c.dropna(subset=["q"]).sort_values(["id", "q", "report_date"], kind="mergesort")
    # groupby.last() keeps the latest non-missing value per bank-quarter.
    obs = c.groupby(["id", "q"], sort=True)["value"].last()

    # 2. Regular quarterly grid -------------------------------------------
    wide = obs.unstack("q")
    ids = wide.index.to_numpy().astype("int64")
    qcols = pd.DatetimeIndex(wide.columns)
    quarters = (pd.period_range(qcols.min().to_period("Q"), qcols.max().to_period("Q"), freq="Q")
                .to_timestamp(how="end").normalize())
    wide = wide.reindex(columns=quarters)
    V = wide.to_numpy(dtype="float64")
    n, T = V.shape

    # 3. Charter per bank-quarter; keep commercial quarters only -----------
    a = pd.DataFrame({
        "id": pd.to_numeric(attrs["rssd_id"]).astype("int64"),
        "q": _quarter_end(attrs["as_of"]),
        "charter": attrs["charter"].astype(object).map(
            lambda x: x.strip().lower() if isinstance(x, str) else None),
    }).dropna(subset=["q"]).sort_values("q", kind="mergesort")
    a = a.drop_duplicates(["id", "q"], keep="last")

    grid = pd.DataFrame({
        "id": np.repeat(ids, T),
        "q": np.tile(quarters.to_numpy(), n),
    }).sort_values("q", kind="mergesort")
    ch = pd.merge_asof(grid, a[["id", "q", "charter"]].sort_values("q", kind="mergesort"),
                       on="q", by="id", direction="backward")
    first_charter = a.groupby("id", sort=False)["charter"].first()
    ch["charter"] = ch["charter"].where(ch["charter"].notna(), ch["id"].map(first_charter))
    ch = ch.sort_values(["id", "q"], kind="mergesort")
    charter = ch["charter"].to_numpy(dtype=object).reshape(n, T)
    is_comm = charter == "commercial"
    V = np.where(is_comm, V, np.nan)

    # 4. Coverage screen over each bank's own span -------------------------
    observed = np.isfinite(V)
    n_obs = observed.sum(axis=1)
    first = observed.argmax(axis=1)
    last = T - 1 - observed[:, ::-1].argmax(axis=1)
    span = last - first + 1
    coverage = np.where(n_obs > 0, n_obs / np.maximum(span, 1), 0.0)
    keep = (n_obs >= MIN_OBS) & (coverage >= MIN_COVERAGE)
    V = V[keep]
    ids = ids[keep]

    # 5. Interpolate short interior gaps only --------------------------------
    nan = ~np.isfinite(V)
    back = _run_counts(nan)
    fwd = _run_counts(nan[:, ::-1])[:, ::-1]
    run_len = back + fwd - 1                       # length of the NaN run containing each NaN
    short = nan & (run_len <= MAX_GAP)
    interp = pd.DataFrame(V).interpolate(axis=1, limit_area="inside").to_numpy()
    F = np.where(short, interp, V)                 # long gaps and edges stay NaN

    # 6. Long output, missing y dropped -------------------------------------
    rows, cols = np.nonzero(np.isfinite(F))
    out = pd.DataFrame({
        "id": ids[rows].astype("int64"),
        "date": quarters.to_numpy().astype("datetime64[ns]")[cols],
        "y": F[rows, cols].astype("float64"),
    })
    out = out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    return out[OUT_COLUMNS]


if __name__ == "__main__":
    from pathlib import Path

    here = Path(__file__).resolve().parent
    calls_df = pd.read_csv(here / "sample_calls.csv")
    attrs_df = pd.read_csv(here / "sample_attrs.csv")
    panel = build_panel(calls_df, attrs_df)
    print(panel.dtypes.to_string())
    print(panel.head())
    print("rows:", len(panel), "banks:", panel["id"].nunique())
