import pandas as pd


HOLDOUT_QUARTERS = 4


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Build YoY targets and retain the final four quarters for benchmark splitting.

    A bank needs a usable target in each of its last four calendar quarters,
    measured from its last filing, and at least one earlier target for training.
    The charter in effect on the target report date determines eligibility.
    """
    reports = calls.loc[:, ["rssd_id", "report_date", "value"]].astype(
        {"rssd_id": "int64", "report_date": "int64", "value": "float64"}
    )
    report_dates = pd.to_datetime(reports["report_date"], format="%Y%m%d")
    reports["quarter"] = report_dates.dt.to_period("Q")
    reports = reports.sort_values("report_date", kind="stable").drop_duplicates(
        ["rssd_id", "quarter"], keep="last"
    )
    last_quarter = reports.groupby("rssd_id", sort=False)["quarter"].max()

    snapshots = attrs.loc[:, ["rssd_id", "as_of", "charter"]].astype(
        {"rssd_id": "int64", "as_of": "int64"}
    )
    classified = pd.merge_asof(
        reports,
        snapshots.sort_values("as_of", kind="stable"),
        left_on="report_date",
        right_on="as_of",
        by="rssd_id",
        direction="backward",
    )
    targets = classified.loc[
        classified["charter"].eq("commercial") & classified["value"].notna(),
        ["rssd_id", "quarter", "value"],
    ]

    prior = reports.loc[reports["value"].notna(), ["rssd_id", "quarter", "value"]]
    prior = prior.rename(columns={"value": "prior_value"})
    prior["quarter"] = prior["quarter"] + 4
    targets = targets.merge(prior, on=["rssd_id", "quarter"], how="inner")
    targets["y"] = targets["value"] - targets["prior_value"]
    targets = targets.dropna(subset=["y"])

    holdout_start = last_quarter.reindex(targets["rssd_id"]).array - (HOLDOUT_QUARTERS - 1)
    in_holdout = targets["quarter"].ge(holdout_start)
    holdout_counts = in_holdout.groupby(targets["rssd_id"], sort=False).sum()
    has_training = targets.loc[~in_holdout, "rssd_id"]
    eligible = holdout_counts.index[holdout_counts.eq(HOLDOUT_QUARTERS)]
    panel = targets.loc[
        targets["rssd_id"].isin(eligible) & targets["rssd_id"].isin(has_training),
        ["rssd_id", "quarter", "y"],
    ].rename(columns={"rssd_id": "id"})
    panel["date"] = panel.pop("quarter").dt.to_timestamp() + pd.offsets.QuarterEnd(0)
    return panel.loc[:, ["id", "date", "y"]].sort_values(
        ["id", "date"], ignore_index=True
    )
