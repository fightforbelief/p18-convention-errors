import pandas as pd


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Compute observed year-over-year changes for commercial bank quarters.

    Use the latest charter snapshot available at quarter end. Require four
    usable targets through the last commercial filing quarter and at least
    one earlier target for training. Keep the holdout quarters in the panel.
    """
    reports = calls[["rssd_id", "report_date", "value"]].copy()
    reports["rssd_id"] = reports["rssd_id"].astype("int64")
    reports["value"] = reports["value"].astype("float64")
    periods = pd.to_datetime(reports["report_date"], format="%Y%m%d").dt.to_period("Q")
    reports["quarter"] = periods.astype("int64")
    reports["date"] = periods.dt.to_timestamp(how="end").dt.normalize()
    reports = reports.sort_values("report_date", kind="stable").drop_duplicates(
        ["rssd_id", "quarter"], keep="last"
    )

    snapshots = attrs[["rssd_id", "as_of", "charter"]].copy()
    snapshots["rssd_id"] = snapshots["rssd_id"].astype("int64")
    snapshots["as_of"] = pd.to_datetime(snapshots["as_of"], format="%Y%m%d").astype(reports["date"].dtype)
    classified = pd.merge_asof(
        reports.sort_values("date"),
        snapshots.sort_values("as_of", kind="stable"),
        by="rssd_id",
        left_on="date",
        right_on="as_of",
        direction="backward",
    )
    commercial = classified.loc[
        classified["charter"].eq("commercial"), ["rssd_id", "quarter", "date", "value"]
    ].copy()
    commercial["last_quarter"] = commercial.groupby("rssd_id")["quarter"].transform("max")

    # Join on the exact calendar quarter so missing filings cannot shift the lag.
    previous = reports[["rssd_id", "quarter", "value"]].rename(
        columns={"value": "prior_value"}
    )
    previous["quarter"] = previous["quarter"] + 4
    panel = commercial.merge(previous, on=["rssd_id", "quarter"], how="inner")
    panel["y"] = panel["value"] - panel["prior_value"]
    panel = panel.dropna(subset=["y"])

    in_holdout = panel["quarter"].ge(panel["last_quarter"] - 3)
    holdout_counts = in_holdout.groupby(panel["rssd_id"]).transform("sum")
    target_counts = panel.groupby("rssd_id")["y"].transform("size")
    return (
        panel.loc[holdout_counts.eq(4) & target_counts.gt(4), ["rssd_id", "date", "y"]]
        .rename(columns={"rssd_id": "id"})
        .sort_values(["id", "date"], ignore_index=True)
    )
