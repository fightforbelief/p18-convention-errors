"""Build a quarterly panel for a four-quarter forecasting holdout."""

import numpy as np
import pandas as pd


_HOLDOUT_QUARTERS = 4
_MIN_TRAIN_OBSERVATIONS = 8
_MIN_TRAIN_COVERAGE = 0.8
_MAX_FORWARD_FILL = 4


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Return each eligible bank's latest continuous commercial quarterly spell.

    Charter is determined at quarter end by the latest available snapshot.
    The spell is bounded by filed quarters and starts at the first usable
    observation. Training requires eight observed quarters and 80% coverage.
    Training gaps are forward-filled for up to four quarters. Banks with
    unresolved gaps or any unobserved holdout target are dropped.
    """
    reports = calls.loc[:, ["rssd_id", "report_date", "value"]].rename(
        columns={"rssd_id": "id", "value": "y"}
    )
    reports["id"] = reports["id"].astype("int64")
    reports["y"] = reports["y"].astype("float64")
    reports["y"] = reports["y"].where(np.isfinite(reports["y"]))
    reports["report_date"] = pd.to_datetime(
        reports["report_date"].astype(str), format="%Y%m%d"
    ).astype("datetime64[ns]")
    reports["date"] = reports["report_date"] + pd.offsets.QuarterEnd(0)
    reports = (
        reports.sort_values("report_date", kind="stable")
        .drop_duplicates(["id", "date"], keep="last")
        .drop(columns="report_date")
    )

    snapshots = attrs.loc[:, ["rssd_id", "as_of", "charter"]].rename(
        columns={"rssd_id": "id"}
    )
    snapshots["id"] = snapshots["id"].astype("int64")
    snapshots["as_of"] = pd.to_datetime(
        snapshots["as_of"].astype(str), format="%Y%m%d"
    ).astype("datetime64[ns]")

    bounds = reports.groupby("id", sort=False)["date"].agg(["min", "max"])
    starts = bounds["min"].dt.to_period("Q").astype("int64").to_numpy()
    ends = bounds["max"].dt.to_period("Q").astype("int64").to_numpy()
    lengths = ends - starts + 1
    offsets = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)
    quarters = pd.PeriodIndex.from_ordinals(np.repeat(starts, lengths) + offsets, freq="Q")
    grid = pd.DataFrame(
        {
            "id": np.repeat(bounds.index.to_numpy(), lengths),
            "date": quarters.to_timestamp(how="end").normalize().astype("datetime64[ns]"),
        }
    )
    grid = pd.merge_asof(
        grid.sort_values("date"),
        snapshots.sort_values("as_of", kind="stable"),
        left_on="date",
        right_on="as_of",
        by="id",
        direction="backward",
    ).sort_values(["id", "date"], ignore_index=True)

    commercial = grid["charter"].eq("commercial").fillna(False)
    spell_starts = grid["id"].ne(grid["id"].shift()) | ~commercial.shift(fill_value=False)
    grid["spell"] = spell_starts.cumsum()
    grid = grid.loc[commercial, ["id", "date", "spell"]].merge(
        reports.assign(filed=True), on=["id", "date"], how="left", sort=False
    )
    filed = grid["filed"].eq(True)
    latest_spell = grid["spell"].where(filed).groupby(grid["id"]).transform("max")
    last_filed = grid["date"].where(filed).groupby(grid["spell"]).transform("max")
    first_observed = grid["date"].where(grid["y"].notna()).groupby(
        grid["spell"]
    ).transform("min")
    grid = grid.loc[
        grid["spell"].eq(latest_spell)
        & grid["date"].between(first_observed, last_filed),
        ["id", "date", "y"],
    ].copy()

    by_bank = grid.groupby("id", sort=False)
    training = by_bank.cumcount(ascending=False).ge(_HOLDOUT_QUARTERS)
    observed = grid["y"].notna()
    train_size = training.groupby(grid["id"], sort=False).transform("sum")
    train_observed = (training & observed).groupby(
        grid["id"], sort=False
    ).transform("sum")
    grid["y"] = grid["y"].where(
        ~training, by_bank["y"].ffill(limit=_MAX_FORWARD_FILL)
    )
    complete = grid["y"].notna().groupby(grid["id"], sort=False).transform("all")
    return grid.loc[
        complete
        & train_observed.ge(_MIN_TRAIN_OBSERVATIONS)
        & train_observed.ge(_MIN_TRAIN_COVERAGE * train_size)
    ].reset_index(drop=True)
