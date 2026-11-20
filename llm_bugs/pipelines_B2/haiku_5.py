"""Quarterly panel of year-over-year changes in Call Report ratios for commercial banks.

Only pandas and numpy are used.

Method
------
1. Clean ``calls``: parse ``report_date`` (YYYYMMDD) and map it to its calendar quarter,
   turn infinite values into NaN, and where a bank has several reports for the same
   quarter keep the latest non-missing value.
2. Put the data on a dense bank x quarter grid spanning the full history, so that a gap
   in a bank's filings stays a gap instead of shifting the comparison.
3. For every bank-quarter, ``y = value[t] - value[t - 4 quarters]``. ``y`` is missing
   whenever either leg is missing. Nothing is interpolated or forward-filled.
4. Attach the charter that was in force at each quarter end (latest ``attrs`` snapshot on
   or before that date, so no look-ahead) and keep only "commercial" bank-quarters.
5. Drop banks with fewer than ``MIN_Y_OBS`` valid ``y`` values among their commercial
   quarters (too thin to forecast from).
"""

import numpy as np
import pandas as pd

# A bank needs at least this many valid year-over-year changes to stay in the panel.
MIN_Y_OBS = 8
LAG = 4  # quarters in a year
COMMERCIAL = "commercial"

OUTPUT_COLUMNS = ["id", "date", "y"]


def _quarter_index(dates: pd.Series) -> pd.Series:
    """Map datetimes to an integer quarter number: year * 4 + (quarter - 1)."""
    return dates.dt.year * 4 + (dates.dt.month - 1) // 3


def _quarter_end(qnum: np.ndarray) -> pd.DatetimeIndex:
    """Map integer quarter numbers back to calendar quarter-end Timestamps."""
    years = qnum // 4
    quarters = qnum % 4 + 1
    periods = pd.PeriodIndex.from_fields(year=years, quarter=quarters, freq="Q")
    return periods.to_timestamp(how="end").normalize()


def _clean_calls(calls: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (rssd_id, quarter) with the latest non-missing value."""
    df = calls[["rssd_id", "report_date", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    df.loc[~np.isfinite(df["value"]), "value"] = np.nan
    dates = pd.to_datetime(df["report_date"].astype("int64").astype(str),
                           format="%Y%m%d", errors="coerce")
    df["date"] = dates
    df = df.dropna(subset=["date"])
    df["q"] = _quarter_index(df["date"])
    df["rssd_id"] = df["rssd_id"].astype("int64")
    # Latest report wins; groupby.last skips NaN, so an older valid value survives
    # if the newest filing for the quarter is blank.
    df = df.sort_values(["rssd_id", "q", "date"], kind="mergesort")
    out = df.groupby(["rssd_id", "q"], as_index=False, sort=True)["value"].last()
    return out


def _attr_snapshots(attrs: pd.DataFrame) -> pd.DataFrame:
    """One charter per (rssd_id, quarter end), taken from the latest snapshot in that quarter."""
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["charter"] = a["charter"].astype("string").str.strip().str.lower()
    a["as_of"] = pd.to_datetime(a["as_of"].astype("int64").astype(str),
                                format="%Y%m%d", errors="coerce")
    a = a.dropna(subset=["as_of"])
    a["date"] = _quarter_end(_quarter_index(a["as_of"]).to_numpy())
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a = a.sort_values(["rssd_id", "as_of"], kind="mergesort")
    a = a.drop_duplicates(["rssd_id", "date"], keep="last")
    return a[["rssd_id", "date", "charter"]].sort_values("date", kind="mergesort")


_EMPTY = pd.DataFrame({
    "id": pd.Series(dtype="int64"),
    "date": pd.Series(dtype="datetime64[ns]"),
    "y": pd.Series(dtype="float64"),
})


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Build the commercial-bank quarterly panel of year-over-year ratio changes.

    Parameters
    ----------
    calls : DataFrame with ``rssd_id`` (int), ``report_date`` (int, YYYYMMDD) and
        ``value`` (float, may be NaN).
    attrs : DataFrame with ``rssd_id``, ``as_of`` (int, YYYYMMDD) and ``charter``
        ("commercial" or "savings").

    Returns
    -------
    DataFrame with columns ``id`` (int), ``date`` (quarter-end Timestamp) and ``y``
    (float), one row per bank-quarter, sorted by ``id`` then ``date``, with no
    missing ``y``.
    """
    if calls is None or len(calls) == 0:
        return _EMPTY.copy()

    cells = _clean_calls(calls)
    if cells.empty:
        return _EMPTY.copy()

    # Dense bank x quarter grid over the whole history.
    bank_ids, bank_codes = np.unique(cells["rssd_id"].to_numpy(), return_inverse=True)
    q_min = int(cells["q"].min())
    q_max = int(cells["q"].max())
    n_q = q_max - q_min + 1
    q_codes = (cells["q"].to_numpy() - q_min).astype(np.int64)

    grid = np.full((len(bank_ids), n_q), np.nan, dtype=np.float64)
    grid[bank_codes, q_codes] = cells["value"].to_numpy()

    # y[t] = value[t] - value[t-4]; NaN propagates, so a gap on either side leaves y missing.
    yw = np.full_like(grid, np.nan)
    if n_q > LAG:
        yw[:, LAG:] = grid[:, LAG:] - grid[:, :-LAG]

    rows, cols = np.nonzero(np.isfinite(yw))
    if rows.size == 0:
        return _EMPTY.copy()

    long = pd.DataFrame({
        "id": bank_ids[rows].astype("int64"),
        "date": _quarter_end(cols.astype(np.int64) + q_min),
        "y": yw[rows, cols],
    })
    long = long.sort_values(["date", "id"], kind="mergesort").reset_index(drop=True)

    # Keep only quarters where the bank was commercial, using the charter in force then.
    snaps = _attr_snapshots(attrs)
    if snaps.empty:
        return _EMPTY.copy()
    long = pd.merge_asof(long, snaps.rename(columns={"rssd_id": "id"}),
                         on="date", by="id", direction="backward")
    long = long[long["charter"] == COMMERCIAL]

    # Drop banks too thin to forecast from.
    counts = long.groupby("id")["y"].transform("size")
    long = long[counts >= MIN_Y_OBS]

    out = long.loc[:, OUTPUT_COLUMNS].sort_values(["id", "date"], kind="mergesort")
    out = out.dropna(subset=["y"]).reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype("float64")
    return out


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    calls_df = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    attrs_df = pd.read_csv(os.path.join(here, "sample_attrs.csv"))
    panel = build_panel(calls_df, attrs_df)
    print(panel.dtypes)
    print(panel.head())
    print(f"rows={len(panel)} banks={panel['id'].nunique()}")
