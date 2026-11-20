"""Build the quarterly Call Report target panel."""

import pandas as pd


HOLDOUT_QUARTERS = 4


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Return observed YoY differences for commercial bank-quarters.

    Use the latest charter snapshot available on each report date and require
    commercial status at both endpoints. Use the latest report in each quarter.
    Keep banks with usable targets in all four calendar quarters ending at their
    latest commercial report quarter and at least one earlier target. Include
    those four quarters in the panel for the benchmark to hold out.
    """
    reports = calls[["rssd_id", "report_date", "value"]].astype(
        {"rssd_id": "int64", "report_date": "int64", "value": "float64"}
    )
    snapshots = attrs[["rssd_id", "as_of", "charter"]].astype(
        {"rssd_id": "int64", "as_of": "int64"}
    )
    # merge_asof requires dates to be globally sorted, even with a bank key.
    reports = pd.merge_asof(
        reports.sort_values("report_date", kind="stable"),
        snapshots.sort_values("as_of", kind="stable"),
        by="rssd_id",
        left_on="report_date",
        right_on="as_of",
        direction="backward",
    )
    reports["quarter"] = pd.to_datetime(
        reports["report_date"], format="%Y%m%d"
    ).dt.to_period("Q-DEC")
    reports = reports.drop_duplicates(["rssd_id", "quarter"], keep="last")
    commercial = reports.loc[
        reports["charter"].eq("commercial"), ["rssd_id", "quarter", "value"]
    ]
    last_quarter = commercial.groupby("rssd_id", sort=False)["quarter"].max()
    observed = commercial.dropna(subset=["value"])

    # Shift the join key, so a missing quarter cannot move the comparison date.
    prior = observed.rename(columns={"value": "prior_value"}).assign(
        quarter=observed["quarter"] + 4
    )
    panel = observed.merge(prior, on=["rssd_id", "quarter"], validate="one_to_one")
    panel["y"] = panel["value"] - panel["prior_value"]
    panel = panel.dropna(subset=["y"])

    holdout_start = last_quarter.reindex(panel["rssd_id"]).array - (HOLDOUT_QUARTERS - 1)
    is_holdout = panel["quarter"].ge(holdout_start)
    by_bank = is_holdout.groupby(panel["rssd_id"], sort=False)
    eligible = by_bank.sum().eq(HOLDOUT_QUARTERS) & by_bank.size().gt(HOLDOUT_QUARTERS)
    panel = panel.loc[panel["rssd_id"].isin(eligible.index[eligible]), ["rssd_id", "quarter", "y"]]
    panel["date"] = panel["quarter"].dt.to_timestamp(how="end").dt.normalize()
    return (
        panel.rename(columns={"rssd_id": "id"})[["id", "date", "y"]]
        .sort_values(["id", "date"], ignore_index=True)
    )
