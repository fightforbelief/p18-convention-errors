"""Build a quarterly panel of year-over-year Call Report changes."""

import pandas as pd


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    reports = calls[["rssd_id", "report_date", "value"]].dropna(subset=["value"])
    reports = reports.astype({"rssd_id": "int64", "report_date": "int64"})
    snapshots = attrs[["rssd_id", "as_of", "charter"]].astype(
        {"rssd_id": "int64", "as_of": "int64"}
    )

    # The date keys must be globally sorted for merge_asof, even with bank groups.
    matched = pd.merge_asof(
        reports.sort_values("report_date"),
        snapshots.sort_values("as_of", kind="stable"),
        left_on="report_date",
        right_on="as_of",
        by="rssd_id",
        direction="backward",
    )
    usable = matched.loc[
        matched["charter"].eq("commercial"), ["rssd_id", "report_date", "value"]
    ]

    # Calendar quarter-ends have the same month and day in consecutive years.
    prior = usable.rename(columns={"value": "prior_value"})
    prior["report_date"] = prior["report_date"] + 10_000
    paired = usable.merge(prior, on=["rssd_id", "report_date"], how="inner")
    paired["y"] = paired["value"].astype(float) - paired["prior_value"].astype(float)
    defined = paired.dropna(subset=["y"])
    counts = defined.groupby("rssd_id", sort=False)["y"].transform("size")

    panel = defined.loc[counts.ge(16), ["rssd_id", "report_date", "y"]].rename(
        columns={"rssd_id": "id", "report_date": "date"}
    )
    panel["date"] = pd.to_datetime(panel["date"], format="%Y%m%d")
    return panel.sort_values(["id", "date"], ignore_index=True)
