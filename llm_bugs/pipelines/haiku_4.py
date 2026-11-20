import numpy as np
import pandas as pd


def _quarter_end(yyyymmdd: pd.Series) -> pd.Series:
    """Map YYYYMMDD integers to the calendar quarter-end Timestamp."""
    d = pd.to_datetime(yyyymmdd.astype(str), format="%Y%m%d")
    return d.dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(
        {"id": pd.Series(dtype="int64"),
         "date": pd.Series(dtype="datetime64[ns]"),
         "y": pd.Series(dtype="float64")}
    )

    # Reported values, one row per bank-quarter (report_date mapped to quarter end).
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype("int64").to_numpy(),
        "date": _quarter_end(calls["report_date"]).to_numpy(),
        "y": calls["value"].astype("float64").to_numpy(),
    })
    c = c.sort_values(["id", "date"], kind="mergesort")
    c = c.drop_duplicates(["id", "date"], keep="last")
    if c.empty:
        return empty

    # Active span: first to last report date per bank, on a regular quarterly grid.
    span = c.groupby("id")["date"].agg(["min", "max"]).reset_index()
    span["n"] = (
        (span["max"].dt.year - span["min"].dt.year) * 4
        + (span["max"].dt.month - span["min"].dt.month) // 3
        + 1
    ).astype("int64")
    ids = np.repeat(span["id"].to_numpy(), span["n"].to_numpy())
    offsets = np.arange(span["n"].sum()) - np.repeat(
        np.cumsum(span["n"].to_numpy()) - span["n"].to_numpy(), span["n"].to_numpy()
    )
    start = np.repeat(span["min"].to_numpy(), span["n"].to_numpy())
    start = pd.DatetimeIndex(start).to_period("Q")
    grid_q = start + offsets.astype("int64")
    grid = pd.DataFrame({
        "id": ids,
        "date": grid_q.to_timestamp(how="end").normalize(),
    })

    # Attach reported values (NaN where no report or value not reported).
    grid = grid.merge(c, on=["id", "date"], how="left")

    # Charter at each quarter: latest snapshot with as_of <= quarter end.
    a = pd.DataFrame({
        "id": attrs["rssd_id"].astype("int64").to_numpy(),
        "as_of": pd.to_datetime(attrs["as_of"].astype(str), format="%Y%m%d").to_numpy(),
        "charter": attrs["charter"].astype(str).str.strip().str.lower().to_numpy(),
    })
    a = a.sort_values(["as_of", "id"], kind="mergesort").drop_duplicates(
        ["id", "as_of"], keep="last"
    )
    grid = grid.sort_values("date", kind="mergesort")
    grid = pd.merge_asof(
        grid, a.sort_values("as_of"), left_on="date", right_on="as_of",
        by="id", direction="backward", allow_exact_matches=True,
    )
    grid = grid.sort_values(["id", "date"]).reset_index(drop=True)

    # Rule 2: keep only quarters where the bank was commercial (unknown charter -> drop).
    grid = grid[grid["charter"] == "commercial"]
    if grid.empty:
        return empty

    # Rule 3: drop banks with more than 20% missing among remaining quarters.
    miss = grid["y"].isna()
    stats = grid.assign(_m=miss).groupby("id").agg(n=("_m", "size"), m=("_m", "sum"))
    keep_ids = stats.index[(stats["m"] / stats["n"]) <= 0.2]
    grid = grid[grid["id"].isin(keep_ids)]

    # Rule 4: drop missing bank-quarters.
    out = grid.loc[grid["y"].notna(), ["id", "date", "y"]].reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["date"] = pd.to_datetime(out["date"])
    out["y"] = out["y"].astype("float64")
    return out.sort_values(["id", "date"]).reset_index(drop=True)
