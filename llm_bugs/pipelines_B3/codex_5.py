"""Build a point-in-time quarterly panel for a four-quarter forecast holdout."""

import numpy as np
import pandas as pd


MIDWEST = frozenset({"IL", "IN", "IA", "MI", "MN", "OH", "WI"})
HOLDOUT = 4
MIN_TRAIN_OBSERVED = 8
MIN_TRAIN_COVERAGE = 0.8
MAX_TRAIN_GAP = 2


def build_panel(
    calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """Return each bank's latest Midwest run with filings, if it passes coverage.

    Spell endpoints are inclusive. Directory state is a fallback only for banks
    with no headquarters history. The run is bounded by its first and last
    filings. Duplicate filings in a quarter are averaged over reported values.

    The final four quarters must have reported values. Training must have at
    least eight reported values and 80% coverage. Trim leading missing training
    values after checking coverage and carry values forward across gaps of up
    to two quarters. Exclude runs with longer gaps.
    """
    historical_ids = history["rssd_id"]
    fallback_ids = directory.loc[
        directory["state"].isin(MIDWEST) & ~directory["rssd_id"].isin(historical_ids),
        "rssd_id",
    ]
    candidate_ids = history.loc[history["state"].isin(MIDWEST), "rssd_id"]
    reports = calls.loc[
        calls["rssd_id"].isin(candidate_ids) | calls["rssd_id"].isin(fallback_ids),
        ["rssd_id", "report_date", "value"],
    ]
    if reports.empty:
        return pd.DataFrame({
            "id": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "y": pd.Series(dtype="float64"),
        })

    report_dates = reports["report_date"].to_numpy(dtype=np.int64)
    quarters = (report_dates // 10000) * 4 + (report_dates // 100 % 100 - 1) // 3
    filed = (
        reports.assign(quarter=quarters)
        .groupby(["rssd_id", "quarter"], as_index=False)["value"]
        .mean()
        .assign(filed=True)
    )

    bounds = filed.groupby("rssd_id")["quarter"].agg(["min", "max"])
    lengths = (bounds["max"] - bounds["min"] + 1).to_numpy()
    offsets = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)
    grid = pd.DataFrame({
        "rssd_id": np.repeat(bounds.index.to_numpy(), lengths),
        "quarter": np.repeat(bounds["min"].to_numpy(), lengths) + offsets,
    })
    calendar = pd.period_range(
        start=f"{grid['quarter'].min() // 4}Q1",
        end=f"{grid['quarter'].max() // 4}Q4",
        freq="Q",
    ).to_timestamp(how="end").normalize()
    grid["date"] = calendar.take(grid["quarter"] - (grid["quarter"].min() // 4) * 4)
    grid["report_date"] = (
        grid["date"].dt.year * 10000 + grid["date"].dt.month * 100 + grid["date"].dt.day
    ).astype("int64")

    spells = history.loc[
        history["rssd_id"].isin(bounds.index),
        ["rssd_id", "start_date", "end_date", "state"],
    ].copy()
    spells["start_date"] = spells["start_date"].astype("int64")
    grid = pd.merge_asof(
        grid.sort_values("report_date"),
        spells.sort_values("start_date", kind="stable"),
        left_on="report_date",
        right_on="start_date",
        by="rssd_id",
        direction="backward",
    )
    midwest = (
        grid["state"].isin(MIDWEST)
        & (grid["end_date"].isna() | grid["report_date"].le(grid["end_date"]))
    ) | grid["rssd_id"].isin(fallback_ids)
    panel = grid.loc[midwest, ["rssd_id", "quarter", "date"]].merge(
        filed, on=["rssd_id", "quarter"], how="left"
    ).sort_values(["rssd_id", "quarter"])

    new_run = panel["rssd_id"].ne(panel["rssd_id"].shift()) | panel["quarter"].diff().ne(1)
    panel["run"] = new_run.cumsum()
    latest = panel.loc[panel["filed"].eq(True)].groupby("rssd_id")["run"].max()
    panel = panel.loc[panel["run"].isin(latest)].copy()
    filing_quarter = panel["quarter"].where(panel["filed"].eq(True))
    first = filing_quarter.groupby(panel["run"]).transform("min")
    last = filing_quarter.groupby(panel["run"]).transform("max")
    holdout = panel["quarter"].gt(last - HOLDOUT)
    inside = panel["quarter"].between(first, last)
    panel = panel.loc[inside].copy()
    holdout = holdout.loc[inside]

    training = panel.loc[~holdout]
    coverage = training.groupby("run")["value"].agg(["count", "size"])
    eligible = coverage.index[
        coverage["count"].ge(MIN_TRAIN_OBSERVED)
        & coverage["count"].ge(MIN_TRAIN_COVERAGE * coverage["size"])
    ]
    complete_holdout = panel.loc[holdout].groupby("run")["value"].count()
    panel = panel.loc[
        panel["run"].isin(eligible.intersection(complete_holdout.index[complete_holdout.eq(HOLDOUT)]))
    ].copy()

    observed = panel["quarter"].where(panel["value"].notna())
    first_observed = observed.groupby(panel["run"]).transform("min")
    panel = panel.loc[panel["quarter"].ge(first_observed)].copy()
    panel["y"] = panel.groupby("run")["value"].ffill(limit=MAX_TRAIN_GAP)
    incomplete = panel.loc[panel["y"].isna(), "run"]
    return (
        panel.loc[~panel["run"].isin(incomplete), ["rssd_id", "date", "y"]]
        .rename(columns={"rssd_id": "id"})
        .astype({"id": "int64", "y": "float64"})
        .reset_index(drop=True)
    )
