import pandas as pd


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Build a sorted bank-quarter panel of observed year-over-year changes."""
    reports = calls.loc[calls["value"].notna(), ["rssd_id", "report_date", "value"]]
    snapshots = attrs.loc[:, ["rssd_id", "as_of", "charter"]]

    # merge_asof requires the date key to be globally sorted, even with a bank key.
    classified = pd.merge_asof(
        reports.sort_values("report_date", kind="stable"),
        snapshots.sort_values("as_of", kind="stable"),
        left_on="report_date",
        right_on="as_of",
        by="rssd_id",
        direction="backward",
    )
    usable = classified.loc[
        classified["charter"].eq("commercial"), ["rssd_id", "report_date", "value"]
    ]

    # Adding 10000 advances a YYYYMMDD quarter-end to the same quarter next year.
    previous = usable.assign(report_date=usable["report_date"] + 10000).rename(
        columns={"value": "previous_value"}
    )
    paired = usable.merge(previous, on=["rssd_id", "report_date"], how="inner")
    paired["y"] = paired["value"] - paired["previous_value"]
    defined = paired.dropna(subset=["y"])
    counts = defined.groupby("rssd_id", sort=False)["y"].transform("size")
    panel = defined.loc[counts.ge(16), ["rssd_id", "report_date", "y"]].rename(
        columns={"rssd_id": "id", "report_date": "date"}
    )
    panel["date"] = pd.to_datetime(panel["date"].astype(str), format="%Y%m%d")
    return panel.sort_values(["id", "date"], ignore_index=True)
