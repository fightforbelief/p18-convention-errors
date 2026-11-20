"""Prepare quarterly commercial-bank histories for a four-quarter holdout."""

import numpy as np
import pandas as pd


HOLDOUT_QUARTERS = 4
MIN_TRAIN_OBSERVATIONS = 8
MIN_TRAIN_COVERAGE = 0.80
MAX_FILL_QUARTERS = 2


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Return each bank's latest commercial spell with four observed test targets.

    Keep the holdout in the output so callers can split off the last four rows
    per bank. In training, carry the last observation forward for up to two
    quarters. Require at least eight actual training observations, at least
    80% observed training coverage, and no gaps remaining after filling.
    """
    empty = pd.DataFrame({
        "id": pd.Series(dtype="int64"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "y": pd.Series(dtype="float64"),
    })
    if calls.empty or attrs.empty:
        return empty

    reports = calls[["rssd_id", "report_date", "value"]].rename(
        columns={"rssd_id": "id", "value": "y"}
    )
    report_dates = pd.to_datetime(reports["report_date"].astype(str), format="%Y%m%d")
    quarters = report_dates.dt.to_period("Q")
    calendar = pd.period_range(quarters.min(), quarters.max(), freq="Q")
    reports["quarter"] = quarters.astype("int64") - calendar[0].ordinal
    values = reports["y"].astype("float64")
    reports["y"] = values.where(np.isfinite(values))
    # If several reports land in one quarter, use the last available ratio.
    quarterly = reports.sort_values("report_date", kind="stable").groupby(
        ["id", "quarter"]
    )["y"].last()

    bounds = quarterly.reset_index().groupby("id")["quarter"].agg(["min", "max"])
    lengths = (bounds["max"] - bounds["min"] + 1).to_numpy()
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    grid = pd.DataFrame({
        "id": np.repeat(bounds.index.to_numpy(dtype="int64"), lengths),
        "quarter": np.repeat(bounds["min"].to_numpy(), lengths) + offsets,
    })
    quarter_ends = calendar.to_timestamp(how="end").normalize()
    grid["date"] = quarter_ends.take(grid["quarter"])

    snapshots = attrs[["rssd_id", "as_of", "charter"]].rename(
        columns={"rssd_id": "id"}
    )
    snapshots["id"] = snapshots["id"].astype("int64")
    snapshots["as_of"] = pd.to_datetime(snapshots["as_of"].astype(str), format="%Y%m%d")
    grid = pd.merge_asof(
        grid.sort_values("date"),
        snapshots.sort_values("as_of", kind="stable"),
        left_on="date",
        right_on="as_of",
        by="id",
        direction="backward",
    )
    commercial = grid.loc[grid["charter"].eq("commercial"), ["id", "date", "quarter"]]
    commercial = commercial.sort_values(["id", "quarter"])
    # A quarter with a different or unknown charter breaks the commercial spell.
    spell_start = commercial.groupby("id")["quarter"].diff().ne(1)
    spell = spell_start.cumsum()
    last_spell = spell.groupby(commercial["id"]).transform("last")
    panel = commercial.loc[spell.eq(last_spell)].join(quarterly, on=["id", "quarter"])

    observed = panel["y"].notna()
    # Begin at the first usable report of the selected commercial spell.
    panel = panel.loc[observed.groupby(panel["id"]).cummax()].copy()
    by_bank = panel.groupby("id", sort=False)
    holdout = by_bank.cumcount(ascending=False).lt(HOLDOUT_QUARTERS)
    train = panel.loc[~holdout, ["id", "y"]].copy()
    training = train.groupby("id", sort=False)["y"]
    coverage = training.agg(["count", "size"])
    eligible = coverage.index[
        coverage["count"].ge(MIN_TRAIN_OBSERVATIONS)
        & coverage["count"].ge(MIN_TRAIN_COVERAGE * coverage["size"])
    ]
    train["y"] = training.ffill(limit=MAX_FILL_QUARTERS)
    panel.loc[~holdout, "y"] = train["y"]
    incomplete = panel.loc[panel["y"].isna(), "id"]
    keep = panel["id"].isin(eligible) & ~panel["id"].isin(incomplete)
    return panel.loc[keep, ["id", "date", "y"]].reset_index(drop=True)
