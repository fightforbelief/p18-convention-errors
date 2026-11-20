"""Build a quarterly forecasting panel from Call Reports."""

import pandas as pd


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Return commercial-bank changes with enough data for a four-quarter holdout.

    Use the charter known at each report date and require commercial status at
    both endpoints. Keep banks with a usable target in every quarter of their
    final four calendar quarters plus at least one earlier target.
    The returned panel includes the holdout quarters.
    """
    reports = calls[["rssd_id", "report_date", "value"]].copy()
    reports["rssd_id"] = reports["rssd_id"].astype("int64")
    reports["report_date"] = reports["report_date"].astype("int64")
    reports["value"] = reports["value"].astype("float64")
    reports["quarter"] = pd.to_datetime(
        reports["report_date"], format="%Y%m%d"
    ).dt.to_period("Q")
    # Use the latest filing if more than one report falls in a quarter.
    reports = reports.sort_values("report_date", kind="stable").drop_duplicates(
        ["rssd_id", "quarter"], keep="last"
    )
    last_filed = reports.groupby("rssd_id", sort=False)["quarter"].max()

    snapshots = attrs[["rssd_id", "as_of", "charter"]].astype(
        {"rssd_id": "int64", "as_of": "int64"}
    )
    classified = pd.merge_asof(
        reports,
        snapshots.sort_values("as_of", kind="stable"),
        by="rssd_id",
        left_on="report_date",
        right_on="as_of",
        direction="backward",
    )
    commercial = classified.loc[
        classified["charter"].eq("commercial") & classified["value"].notna(),
        ["rssd_id", "quarter", "value"],
    ]
    prior = commercial.rename(columns={"value": "prior_value"}).assign(
        quarter=commercial["quarter"] + 4
    )
    changes = commercial.merge(prior, on=["rssd_id", "quarter"])
    changes["y"] = changes["value"] - changes["prior_value"]
    changes = changes.dropna(subset=["y"])

    changes = changes.join(last_filed.rename("last_filed"), on="rssd_id")
    in_holdout = changes["quarter"] > changes["last_filed"] - 4
    counts = in_holdout.groupby(changes["rssd_id"], sort=False).agg(["sum", "size"])
    eligible = counts.index[counts["sum"].eq(4) & counts["size"].gt(4)]
    result = changes.loc[changes["rssd_id"].isin(eligible), ["rssd_id", "quarter", "y"]]
    result["date"] = result.pop("quarter").dt.to_timestamp(how="end").dt.normalize()
    return (
        result.rename(columns={"rssd_id": "id"})[["id", "date", "y"]]
        .sort_values(["id", "date"], ignore_index=True)
    )
