"""Build a quarterly panel of year-over-year changes in a bank ratio.

Only pandas and numpy are used. See spec.md in this folder for the rules.
"""

import numpy as np
import pandas as pd

MIN_QUARTERS = 16
LAG_QUARTERS = 4


def _quarter_index(dates: pd.Series) -> pd.Series:
    """Map quarter-end timestamps to an integer index (year*4 + quarter-1)."""
    return dates.dt.year * 4 + (dates.dt.month - 1) // 3


def _quarter_end(qidx: pd.Series) -> pd.Series:
    """Map the integer quarter index back to the calendar quarter-end Timestamp."""
    year = qidx // 4
    month = (qidx % 4) * 3 + 3
    first = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": 1})
    )
    return first + pd.offsets.MonthEnd(0)


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "y": pd.Series(dtype="float64"),
        }
    )

    # --- Reported values, one row per bank-quarter ---
    c = pd.DataFrame(
        {
            "id": calls["rssd_id"].astype("int64"),
            "date": pd.to_datetime(
                calls["report_date"].astype("int64").astype(str), format="%Y%m%d"
            ),
            "value": calls["value"].astype("float64"),
        }
    )
    c["q"] = _quarter_index(c["date"])
    # Guard against repeated filings for the same bank-quarter: keep the last one.
    c = c.drop_duplicates(["id", "q"], keep="last")

    # --- Charter in force at each quarter (latest snapshot on or before quarter end) ---
    a = pd.DataFrame(
        {
            "id": attrs["rssd_id"].astype("int64"),
            "as_of": pd.to_datetime(
                attrs["as_of"].astype("int64").astype(str), format="%Y%m%d"
            ),
            "charter": attrs["charter"].astype(str).str.strip().str.lower(),
        }
    ).sort_values(["as_of", "id"])
    c = c.sort_values(["date", "id"])
    c = pd.merge_asof(
        c,
        a,
        left_on="date",
        right_on="as_of",
        by="id",
        direction="backward",
    )
    # Banks with no snapshot yet have NaN charter and are not usable.
    usable = c["value"].notna() & (c["charter"] == "commercial")
    u = c.loc[usable, ["id", "q", "date", "value"]]

    # --- Year-over-year change: both bank-quarters must be usable ---
    prior = u.assign(q=u["q"] + LAG_QUARTERS).rename(columns={"value": "value_lag"})
    panel = u.merge(
        prior[["id", "q", "value_lag"]], on=["id", "q"], how="inner"
    )
    panel["y"] = panel["value"] - panel["value_lag"]
    panel = panel[np.isfinite(panel["y"])]

    # --- Keep banks with enough defined quarters ---
    counts = panel.groupby("id")["y"].transform("size")
    panel = panel[counts >= MIN_QUARTERS]

    if panel.empty:
        return empty

    out = pd.DataFrame(
        {
            "id": panel["id"].to_numpy(),
            "date": _quarter_end(panel["q"]).to_numpy(),
            "y": panel["y"].to_numpy(),
        }
    )
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["id", "date"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    calls_df = pd.read_csv(f"{folder}/sample_calls.csv")
    attrs_df = pd.read_csv(f"{folder}/sample_attrs.csv")
    result = build_panel(calls_df, attrs_df)
    print(result.head())
    print(result.dtypes)
    print("rows:", len(result), "banks:", result["id"].nunique())
