import pandas as pd


HOLDOUT_QUARTERS = 4


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Build a quarterly panel with at least one training target per bank.

    Both reports in a year-over-year pair must have a commercial charter
    as of their filing dates. Keep the final four calendar quarters in the
    panel for the benchmark, ending at the bank's latest commercial report.
    """
    reports = calls.loc[:, ["rssd_id", "report_date", "value"]].astype(
        {"rssd_id": "int64", "report_date": "int64", "value": "float64"}
    )
    snapshots = attrs.loc[:, ["rssd_id", "as_of", "charter"]].astype(
        {"rssd_id": "int64", "as_of": "int64"}
    )
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
    ).dt.to_period("Q")
    reports = reports.drop_duplicates(["rssd_id", "quarter"], keep="last")
    commercial = reports.loc[
        reports["charter"].eq("commercial"), ["rssd_id", "quarter", "value"]
    ]
    last_quarter = commercial.groupby("rssd_id", sort=False)["quarter"].max()

    observed = commercial.dropna(subset=["value"])
    prior = observed.rename(columns={"value": "prior_value"}).assign(
        quarter=observed["quarter"] + 4
    )
    panel = observed.merge(prior, on=["rssd_id", "quarter"], validate="one_to_one")
    panel["y"] = panel["value"] - panel["prior_value"]
    panel = panel.dropna(subset=["y"])

    final_quarter = last_quarter.reindex(panel["rssd_id"]).array
    in_holdout = panel["quarter"] > final_quarter - HOLDOUT_QUARTERS
    bank_targets = in_holdout.groupby(panel["rssd_id"], sort=False)
    eligible = bank_targets.transform("sum").eq(HOLDOUT_QUARTERS) & (
        bank_targets.transform("size") > HOLDOUT_QUARTERS
    )
    result = panel.loc[eligible, ["rssd_id", "quarter", "y"]].rename(
        columns={"rssd_id": "id"}
    )
    result["date"] = result["quarter"].dt.to_timestamp(how="end").dt.normalize()
    return result[["id", "date", "y"]].sort_values(
        ["id", "date"], ignore_index=True
    )
