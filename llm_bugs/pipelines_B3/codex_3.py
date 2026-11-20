import numpy as np
import pandas as pd


MIDWEST = frozenset({"IL", "IN", "IA", "MI", "MN", "OH", "WI"})
HOLDOUT_QUARTERS = 4
MIN_TRAIN_QUARTERS = 8
MIN_TRAIN_COVERAGE = 0.8
MAX_FILL_GAP = 2


def build_panel(
    calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """Build a panel from each bank's latest Midwest run with filed reports.

    History spells include both endpoints. Directory state supplies geography
    only for banks with no history. Training gaps of up to two quarters are
    carried forward, and the four holdout quarters require reported values.
    """
    states = directory.set_index("rssd_id")["state"]
    has_history = history["rssd_id"].unique()
    directory_midwest = states.index[states.isin(MIDWEST)].difference(has_history)
    candidates = directory_midwest.union(
        pd.Index(history.loc[history["state"].isin(MIDWEST), "rssd_id"].unique())
    )
    reports = calls.loc[calls["rssd_id"].isin(candidates),
                        ["rssd_id", "report_date", "value"]].copy()
    if reports.empty:
        return pd.DataFrame({
            "id": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "y": pd.Series(dtype="float64"),
        })

    dates = pd.to_datetime(reports.pop("report_date"), format="%Y%m%d")
    reports["quarter"] = pd.PeriodIndex(dates, freq="Q").asi8
    reports["value"] = reports["value"].astype(float).replace([np.inf, -np.inf], np.nan)
    # Average duplicate filings because the inputs have no revision timestamp.
    reports = reports.groupby(["rssd_id", "quarter"], as_index=False)["value"].mean()
    reports["filed"] = True

    bounds = reports.groupby("rssd_id")["quarter"].agg(["min", "max"])
    lengths = (bounds["max"] - bounds["min"] + 1).to_numpy()
    offsets = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)
    quarters = np.repeat(bounds["min"].to_numpy(), lengths) + offsets
    dates = pd.PeriodIndex.from_ordinals(quarters, freq="Q").to_timestamp(how="end").normalize()
    grid = pd.DataFrame({
        "rssd_id": np.repeat(bounds.index.to_numpy(dtype="int64"), lengths),
        "quarter": quarters,
        "date": dates,
        "day": (dates.year * 10000 + dates.month * 100 + dates.day).astype("int64"),
    })

    spells = history.loc[history["rssd_id"].isin(candidates),
                         ["rssd_id", "start_date", "end_date", "state"]].copy()
    spells["start_date"] = spells["start_date"].astype("int64")
    spells["rssd_id"] = spells["rssd_id"].astype("int64")
    spells["end_date"] = pd.to_numeric(spells["end_date"]).fillna(np.inf)
    grid = pd.merge_asof(
        grid.sort_values("day"), spells.sort_values("start_date", kind="stable"),
        by="rssd_id", left_on="day", right_on="start_date",
    )
    midwest = (
        (grid["state"].isin(MIDWEST) & grid["day"].le(grid["end_date"]))
        | grid["rssd_id"].isin(directory_midwest)
    )
    grid = grid.loc[midwest, ["rssd_id", "quarter", "date"]].sort_values(
        ["rssd_id", "quarter"], ignore_index=True
    )
    new_run = grid["rssd_id"].ne(grid["rssd_id"].shift()) | grid["quarter"].diff().ne(1)
    grid["run"] = new_run.cumsum()
    grid = grid.merge(reports, on=["rssd_id", "quarter"], how="left", sort=False)

    filed = grid.loc[grid["filed"].eq(True)]
    latest = filed.groupby("rssd_id", sort=False)["run"].last()
    grid = grid.loc[grid["run"].eq(grid["rssd_id"].map(latest))].copy()
    first_value = grid.loc[grid["value"].notna()].groupby("run")["quarter"].min()
    last_filing = filed.groupby("run")["quarter"].max()
    grid = grid.loc[
        grid["quarter"].ge(grid["run"].map(first_value))
        & grid["quarter"].le(grid["run"].map(last_filing))
    ].copy()

    holdout = grid["quarter"].gt(grid["run"].map(last_filing) - HOLDOUT_QUARTERS)
    training = grid.loc[~holdout]
    coverage = training.groupby("run")["value"].agg(["size", "count"])
    eligible = coverage.index[
        coverage["size"].ge(MIN_TRAIN_QUARTERS)
        & coverage["count"].ge(MIN_TRAIN_COVERAGE * coverage["size"])
    ]
    grid.loc[~holdout, "value"] = training.groupby("run")["value"].ffill(limit=MAX_FILL_GAP)
    incomplete = grid.loc[grid["value"].isna(), "run"]
    result = grid.loc[grid["run"].isin(eligible) & ~grid["run"].isin(incomplete),
                      ["rssd_id", "date", "value"]]
    return result.rename(columns={"rssd_id": "id", "value": "y"}).reset_index(drop=True)
