"""Build a commercial-bank panel for a four-quarter forecasting holdout."""

import numpy as np
import pandas as pd


HOLDOUT_QUARTERS = 4
MIN_TRAIN_OBSERVATIONS = 8
MIN_TRAIN_COVERAGE = 0.80
MAX_TRAIN_GAP = 2


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Return each eligible bank's latest commercial spell with filed data.

    Charter status is effective as of quarter end. The last four quarters
    through the final commercial filing must all have reported targets.
    Training starts at the first observed target in the spell and uses
    forward fill for gaps of at most two quarters.
    """
    reports = calls[["rssd_id", "report_date", "value"]].rename(
        columns={"rssd_id": "id", "value": "y"}
    )
    reports["id"] = reports["id"].astype("int64")
    reports["y"] = reports["y"].astype("float64").where(np.isfinite(reports["y"]))
    reports["quarter"] = pd.to_datetime(
        reports["report_date"].astype(str), format="%Y%m%d"
    ).dt.to_period("Q")
    # If multiple reports fall in one quarter, use the latest filing.
    reports = reports.sort_values("report_date", kind="stable").drop_duplicates(
        ["id", "quarter"], keep="last"
    )
    bounds = reports.groupby("id")["quarter"].agg(["min", "max"])
    starts = bounds["min"].array.asi8
    lengths = bounds["max"].array.asi8 - starts + 1
    offsets = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)
    quarters = pd.PeriodIndex.from_ordinals(np.repeat(starts, lengths) + offsets, freq="Q")
    panel = pd.DataFrame({"id": np.repeat(bounds.index.to_numpy(), lengths), "quarter": quarters})
    panel = panel.merge(
        reports[["id", "quarter", "report_date", "y"]], on=["id", "quarter"], how="left"
    )
    panel["date"] = panel.pop("quarter").dt.to_timestamp(how="end").dt.normalize().astype("datetime64[ns]")

    snapshots = attrs[["rssd_id", "as_of", "charter"]].rename(columns={"rssd_id": "id"})
    snapshots["id"] = snapshots["id"].astype("int64")
    snapshots["as_of"] = pd.to_datetime(
        snapshots["as_of"].astype(str), format="%Y%m%d"
    ).astype("datetime64[ns]")
    snapshots = snapshots.sort_values("as_of", kind="stable")
    # Track charter changes even when two snapshots fall within one quarter.
    snapshots["spell"] = snapshots["charter"].ne("commercial").groupby(snapshots["id"]).cumsum()
    panel = pd.merge_asof(
        panel.sort_values("date", kind="stable"),
        snapshots,
        left_on="date",
        right_on="as_of",
        by="id",
    )
    panel = panel.loc[panel["charter"].eq("commercial")].copy()
    filed = panel["report_date"].notna()
    latest = panel.loc[filed, ["id", "spell", "date"]].groupby("id").tail(1)
    panel = panel.join(latest.set_index("id"), on="id", rsuffix="_last")
    panel = panel.loc[
        panel["spell"].eq(panel["spell_last"])
        & panel["date"].le(panel["date_last"])
    ].copy()

    first_observed = panel.loc[panel["y"].notna()].groupby("id")["date"].min()
    panel = panel.join(first_observed.rename("first_observed"), on="id")
    panel = panel.loc[panel["date"].ge(panel["first_observed"])].copy()
    holdout = panel.groupby("id").cumcount(ascending=False).lt(HOLDOUT_QUARTERS)
    training = panel.loc[~holdout].copy()

    coverage = training.groupby("id")["y"].agg(["count", "size"])
    eligible = coverage.index[
        coverage["count"].ge(MIN_TRAIN_OBSERVATIONS)
        & coverage["count"].ge(MIN_TRAIN_COVERAGE * coverage["size"])
    ]
    training["y"] = training.groupby("id")["y"].ffill(limit=MAX_TRAIN_GAP)
    panel.loc[training.index, "y"] = training["y"]
    incomplete = panel.loc[panel["y"].isna(), "id"]
    return (
        panel.loc[panel["id"].isin(eligible) & ~panel["id"].isin(incomplete), ["id", "date", "y"]]
        .sort_values(["id", "date"])
        .reset_index(drop=True)
    )
