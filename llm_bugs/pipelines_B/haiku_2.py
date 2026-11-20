"""Build a clean quarterly panel of commercial banks from Call Report ratios.

Steps
-----
1. Drop missing reported values (value is NaN).
2. Map each report to its calendar quarter end.
3. Attach each report's charter as of the report date (latest attribute
   snapshot with as_of <= report_date). Keep only commercial-charter reports.
   Reports with no snapshot on or before their date cannot be classified and
   are dropped.
4. If several reports fall in the same bank-quarter, keep the latest non-NaN.
5. Pivot to a wide quarter x bank grid over the full calendar range.
6. Coverage filter (per bank, on its commercial observations): at least
   MIN_OBS observed quarters and an observed share of at least MIN_COVERAGE of
   the quarters between its first and last observation.
7. Fill interior gaps of at most MAX_GAP consecutive quarters by linear
   interpolation. Longer gaps and leading/trailing quarters stay missing.
8. Stack back to long form and drop remaining missing y.

Only pandas and numpy are used.
"""

import sys

import numpy as np
import pandas as pd

MIN_OBS = 12        # minimum observed quarters per bank
MIN_COVERAGE = 0.75  # minimum observed share of the bank's first-to-last span
MAX_GAP = 2         # longest run of missing quarters to interpolate across


def _quarter_end(yyyymmdd: pd.Series) -> pd.Series:
    """Map YYYYMMDD integers to the calendar quarter-end Timestamp."""
    dates = pd.to_datetime(yyyymmdd.astype("int64").astype(str), format="%Y%m%d")
    return dates.dt.to_period("Q").dt.end_time.dt.normalize()


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(
        {"id": pd.Series(dtype="int64"),
         "date": pd.Series(dtype="datetime64[ns]"),
         "y": pd.Series(dtype="float64")}
    )

    # 1. Drop missing observations.
    c = calls.loc[calls["value"].notna(), ["rssd_id", "report_date", "value"]].copy()
    if c.empty:
        return empty
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c["value"] = c["value"].astype("float64")

    # 3. Charter as of each report date (backward as-of join on YYYYMMDD ints).
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["as_of"] = a["as_of"].astype("int64")
    a = a.sort_values("as_of", kind="stable")
    c = c.sort_values("report_date", kind="stable")
    c = pd.merge_asof(
        c, a,
        left_on="report_date", right_on="as_of",
        by="rssd_id", direction="backward",
    )
    c = c.loc[c["charter"] == "commercial"]
    if c.empty:
        return empty

    # 2. Calendar quarter end; 4. keep latest non-NaN value per bank-quarter.
    c["date"] = _quarter_end(c["report_date"])
    c = c.sort_values(["rssd_id", "date", "report_date"], kind="stable")
    c = c.drop_duplicates(["rssd_id", "date"], keep="last")

    # 5. Wide grid over the full calendar quarter range.
    q_index = pd.period_range(
        c["date"].min(), c["date"].max(), freq="Q"
    ).to_timestamp(how="end").normalize()
    wide = c.pivot(index="date", columns="rssd_id", values="value")
    wide = wide.reindex(q_index)
    wide.columns = wide.columns.astype("int64")

    # 6. Coverage filter.
    observed = wide.notna()
    n_obs = observed.sum(axis=0)
    pos = np.arange(len(q_index))[:, None]
    first = np.where(observed.values, pos, len(q_index)).min(axis=0)
    last = np.where(observed.values, pos, -1).max(axis=0)
    span = np.where(n_obs.values > 0, last - first + 1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(span > 0, n_obs.values / span, 0.0)
    keep = (n_obs.values >= MIN_OBS) & (coverage >= MIN_COVERAGE)
    wide = wide.loc[:, keep]
    if wide.shape[1] == 0:
        return empty

    # 7. Linear interpolation, applied only to interior gaps of at most MAX_GAP
    #    consecutive missing quarters. Longer gaps stay missing.
    vals = wide.values
    n_q = vals.shape[0]
    valid = ~np.isnan(vals)
    t = np.arange(n_q)[:, None]
    prev_valid = np.maximum.accumulate(np.where(valid, t, -1), axis=0)
    next_valid = np.minimum.accumulate(np.where(valid, t, n_q)[::-1], axis=0)[::-1]
    gap_len = next_valid - prev_valid - 1
    fillable = (~valid) & (prev_valid >= 0) & (next_valid < n_q) & (gap_len <= MAX_GAP)
    interp = wide.interpolate(method="linear", limit_area="inside", axis=0).values
    wide = pd.DataFrame(
        np.where(valid | fillable, interp, np.nan),
        index=wide.index, columns=wide.columns,
    )

    # 8. Back to long form, drop anything still missing.
    long = wide.stack().rename("y").reset_index()
    long.columns = ["date", "id", "y"]
    long = long.loc[long["y"].notna(), ["id", "date", "y"]]
    long["id"] = long["id"].astype("int64")
    long["date"] = pd.to_datetime(long["date"])
    long["y"] = long["y"].astype("float64")
    long = long.sort_values(["id", "date"], kind="stable").reset_index(drop=True)
    return long


if __name__ == "__main__":
    here = sys.argv[1] if len(sys.argv) > 1 else "."
    calls_df = pd.read_csv(f"{here}/sample_calls.csv")
    attrs_df = pd.read_csv(f"{here}/sample_attrs.csv")
    panel = build_panel(calls_df, attrs_df)
    print(panel.head())
    print(panel.dtypes)
    print("rows:", len(panel), "banks:", panel["id"].nunique(),
          "y missing:", int(panel["y"].isna().sum()))
