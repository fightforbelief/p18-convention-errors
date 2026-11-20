"""Build contiguous quarterly histories for a four-quarter forecast holdout."""

import numpy as np
import pandas as pd


HOLDOUT_QUARTERS = 4
MIN_TRAIN_OBSERVATIONS = 8
MIN_TRAIN_COVERAGE = 0.80
MAX_TRAIN_GAP = 2


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Return the latest commercial spell for each bank with adequate coverage.

    Use the latest charter snapshot on or before each quarter end. A spell is
    a run of consecutive quarters classified as commercial. Ignore its leading
    quarters without a usable ratio. Require four observed holdout quarters, at
    least eight observed training quarters, and at least 80% training coverage.
    Linearly interpolate interior training gaps of up to two quarters using
    training observations only. Drop spells with any other training gaps.
    """
    if calls.empty or attrs.empty:
        return pd.DataFrame(
            {
                "id": pd.Series(dtype="int64"),
                "date": pd.Series(dtype="datetime64[ns]"),
                "y": pd.Series(dtype="float64"),
            }
        )

    reports = calls[["rssd_id", "report_date", "value"]].sort_values("report_date")
    report_dates = pd.to_datetime(reports["report_date"], format="%Y%m%d")
    reports = reports.assign(
        rssd_id=reports["rssd_id"].astype("int64"),
        quarter=report_dates.dt.year * 4 + report_dates.dt.quarter - 1,
        value=reports["value"].astype("float64").replace([np.inf, -np.inf], np.nan),
    ).drop_duplicates(["rssd_id", "quarter"], keep="last")

    spans = reports.groupby("rssd_id")["quarter"].agg(["min", "max"])
    lengths = (spans["max"] - spans["min"] + 1).to_numpy()
    quarters = (
        np.repeat(spans["min"].to_numpy() - (np.cumsum(lengths) - lengths), lengths)
        + np.arange(lengths.sum())
    )
    grid = pd.DataFrame({"rssd_id": spans.index.repeat(lengths), "quarter": quarters})
    grid["date"] = pd.to_datetime(
        {"year": quarters // 4, "month": (quarters % 4 + 1) * 3, "day": 1}
    ) + pd.offsets.MonthEnd()

    snapshots = attrs[["rssd_id", "as_of", "charter"]].assign(
        rssd_id=attrs["rssd_id"].astype("int64"),
        date=pd.to_datetime(attrs["as_of"], format="%Y%m%d"),
    )
    # merge_asof requires the date key to be globally sorted across banks.
    panel = pd.merge_asof(
        grid.sort_values("date"),
        snapshots[["rssd_id", "date", "charter"]].sort_values("date", kind="stable"),
        on="date",
        by="rssd_id",
    ).merge(reports[["rssd_id", "quarter", "value"]], on=["rssd_id", "quarter"], how="left")
    panel = panel.loc[panel["charter"].eq("commercial")].sort_values(["rssd_id", "quarter"])

    bank = panel["rssd_id"]
    quarter = panel["quarter"]
    spell = (bank.ne(bank.shift()) | quarter.diff().ne(1)).cumsum()
    panel = panel.loc[spell.eq(spell.groupby(bank).transform("last"))].copy()

    bank = panel["rssd_id"]
    quarter = panel["quarter"]
    value = panel["value"]
    first_observed = quarter.where(value.notna()).groupby(bank).transform("min")
    panel = panel.loc[quarter.ge(first_observed)].copy()

    bank = panel["rssd_id"]
    quarter = panel["quarter"]
    value = panel["value"]
    training = quarter.le(quarter.groupby(bank).transform("max") - HOLDOUT_QUARTERS)
    observed_training = training & value.notna()
    train_count = training.groupby(bank).transform("sum")
    observed_count = observed_training.groupby(bank).transform("sum")
    enough_training = (observed_count >= MIN_TRAIN_OBSERVATIONS) & (
        observed_count >= MIN_TRAIN_COVERAGE * train_count
    )

    # Locate both training endpoints before filling so neither holdout values
    # nor partial fills across longer gaps can enter the training series.
    observed_quarter = quarter.where(observed_training)
    observed_value = value.where(observed_training)
    left = observed_quarter.groupby(bank).ffill()
    right = observed_quarter.groupby(bank).bfill()
    fillable = training & value.isna() & (right - left).le(MAX_TRAIN_GAP + 1)
    fraction = (quarter[fillable] - left[fillable]) / (right[fillable] - left[fillable])
    left_value = observed_value.groupby(bank).ffill()[fillable]
    right_value = observed_value.groupby(bank).bfill()[fillable]
    panel.loc[fillable, "value"] = (1 - fraction) * left_value + fraction * right_value

    complete = panel["value"].notna().groupby(bank).transform("all")
    return (
        panel.loc[enough_training & complete, ["rssd_id", "date", "value"]]
        .rename(columns={"rssd_id": "id", "value": "y"})
        .reset_index(drop=True)
    )
