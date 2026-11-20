"""Quarterly panel of year-over-year changes in Call Report ratios.

Only pandas and numpy are used.

Conventions
-----------
* Charter status is time-varying. Each report is assigned the charter from the
  latest attribute snapshot with ``as_of <= report_date``; reports that predate
  a bank's first snapshot take the first snapshot. Only reports whose charter is
  "commercial" are kept, so a year-over-year pair is formed only when both
  quarters were filed as a commercial bank.
* Each report is mapped to its calendar quarter. If several reports fall in the
  same bank-quarter, their non-missing values are averaged.
* ``y`` is the change in the ratio versus the same calendar quarter one year
  earlier, i.e. ``value[q] - value[q - 4]``, in ratio units. Gaps are not
  interpolated: a missing value at either end yields no ``y`` for that quarter.
* Banks with fewer than ``MIN_Y_OBS`` valid ``y`` observations are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_QUARTERS = 4   # year-over-year lag, in calendar quarters
MIN_Y_OBS = 8      # minimum number of valid y observations per bank

_OUT_COLS = ["id", "date", "y"]


def _quarter_index(dates: pd.Series) -> pd.Series:
    """Integer calendar-quarter index: year * 4 + (quarter - 1)."""
    return dates.dt.year * 4 + (dates.dt.month - 1) // 3


def _quarter_end(qidx: pd.Series) -> pd.Series:
    """Timestamp of the calendar quarter end for each quarter index."""
    year = qidx // 4
    month = (qidx % 4) * 3 + 3
    first = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": 1})
    )
    return first + pd.offsets.MonthEnd(0)


def _assign_charter(reports: pd.DataFrame, attrs: pd.DataFrame) -> pd.Series:
    """Charter in force at each report date (as-of join, with a forward fallback)."""
    snap = attrs[["rssd_id", "as_of", "charter"]].dropna(subset=["rssd_id", "as_of"]).copy()
    snap["as_of_ts"] = pd.to_datetime(snap["as_of"].astype("int64").astype(str), format="%Y%m%d")
    snap["charter"] = snap["charter"].astype("string").str.strip().str.lower()
    snap = snap.sort_values("as_of_ts")[["rssd_id", "as_of_ts", "charter"]]

    left = pd.DataFrame({
        "pos": np.arange(len(reports)),
        "rssd_id": reports["rssd_id"].to_numpy().astype("int64"),
        "rd": reports["rd"].to_numpy(),
    }).sort_values("rd")

    back = pd.merge_asof(
        left, snap, left_on="rd", right_on="as_of_ts",
        by="rssd_id", direction="backward",
    )
    missing = back["charter"].isna()
    if missing.any():
        fwd = pd.merge_asof(
            back.loc[missing, ["pos", "rssd_id", "rd"]].sort_values("rd"),
            snap, left_on="rd", right_on="as_of_ts",
            by="rssd_id", direction="forward",
        )
        back.loc[back["pos"].isin(fwd["pos"]) & missing, "charter"] = (
            fwd.set_index("pos")["charter"].reindex(back.loc[missing, "pos"]).to_numpy()
        )
    # Return charter in the original row order of ``reports``.
    return pd.Series(back.set_index("pos")["charter"].sort_index().to_numpy(), dtype="string")


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Build the bank-quarter panel of year-over-year ratio changes.

    Parameters
    ----------
    calls : DataFrame with ``rssd_id`` (int), ``report_date`` (int, YYYYMMDD)
        and ``value`` (float ratio, may be NaN).
    attrs : DataFrame with ``rssd_id``, ``as_of`` (int, YYYYMMDD) and
        ``charter`` ("commercial" or "savings").

    Returns
    -------
    DataFrame with columns ``id`` (int), ``date`` (quarter-end Timestamp) and
    ``y`` (float), one row per bank-quarter, sorted by id then date, no
    missing ``y``.
    """
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = pd.to_numeric(c["rssd_id"], errors="coerce")
    c["report_date"] = pd.to_numeric(c["report_date"], errors="coerce")
    c["value"] = pd.to_numeric(c["value"], errors="coerce").astype("float64")
    c["value"] = c["value"].where(np.isfinite(c["value"]))  # treat inf as missing
    c = c.dropna(subset=["rssd_id", "report_date"])
    c["rssd_id"] = c["rssd_id"].astype("int64")

    c["rd"] = pd.to_datetime(
        c["report_date"].astype("int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    c = c.dropna(subset=["rd"])
    if c.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    # Keep only reports filed while the bank was commercial.
    c["charter"] = _assign_charter(c[["rssd_id", "rd"]], attrs).to_numpy()
    c = c[c["charter"] == "commercial"]

    # Calendar quarter per report, then one value per bank-quarter.
    c["q"] = _quarter_index(c["rd"])
    grid = (
        c.groupby(["rssd_id", "q"], sort=False)["value"]
        .mean()                       # mean ignores NaN; all-NaN stays NaN
        .reset_index()
        .rename(columns={"rssd_id": "id"})
    )

    # Year-over-year change: match each quarter with the same quarter one year back.
    lagged = grid.assign(q=grid["q"] + LAG_QUARTERS).rename(columns={"value": "value_lag"})
    pairs = grid.merge(lagged, on=["id", "q"], how="inner")
    pairs["y"] = pairs["value"] - pairs["value_lag"]
    pairs = pairs.dropna(subset=["y"])
    pairs = pairs[np.isfinite(pairs["y"])]

    # Drop banks with too few valid observations.
    counts = pairs.groupby("id")["y"].transform("size")
    panel = pairs.loc[counts >= MIN_Y_OBS, ["id", "q", "y"]].copy()

    if panel.empty:
        return pd.DataFrame({"id": pd.Series(dtype="int64"),
                             "date": pd.Series(dtype="datetime64[ns]"),
                             "y": pd.Series(dtype="float64")})

    panel["date"] = _quarter_end(panel["q"])
    panel["id"] = panel["id"].astype("int64")
    panel["y"] = panel["y"].astype("float64")
    panel = panel.sort_values(["id", "date"]).reset_index(drop=True)
    return panel[_OUT_COLS]


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    calls_df = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    attrs_df = pd.read_csv(os.path.join(here, "sample_attrs.csv"))
    out = build_panel(calls_df, attrs_df)
    print(out.head(10).to_string())
    print(f"rows={len(out)} banks={out['id'].nunique()}")
