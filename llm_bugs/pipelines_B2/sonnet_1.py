import numpy as np
import pandas as pd

MIN_OBS = 12  # >= 4 held-out quarters + 8 quarters of training history


def _to_ts(x):
    return pd.to_datetime(pd.Series(x).astype("int64").astype(str), format="%Y%m%d", errors="coerce")


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["date"] = _to_ts(c["report_date"]).values
    c = c.dropna(subset=["date"])
    # keep only calendar quarter-end dates
    c = c[c["date"].dt.is_quarter_end & c["date"].dt.is_month_end]
    c["value"] = c["value"].astype(float).replace([np.inf, -np.inf], np.nan)
    c["rssd_id"] = c["rssd_id"].astype("int64")
    # duplicate filings: keep the last one per bank-quarter (stable order)
    c = c.drop_duplicates(["rssd_id", "date"], keep="last")
    c["q"] = c["date"].dt.year * 4 + (c["date"].dt.month // 3 - 1)

    # year-over-year change vs exactly 4 quarters earlier; no filling across gaps
    prev = c[["rssd_id", "q", "value"]].rename(columns={"value": "prev"})
    prev["q"] = prev["q"] + 4
    m = c.merge(prev, on=["rssd_id", "q"], how="left")
    m["y"] = m["value"] - m["prev"]
    m = m.dropna(subset=["y"])
    m = m[np.isfinite(m["y"])]

    # point-in-time charter: latest snapshot at or before the quarter end
    # (earliest snapshot if the quarter predates all snapshots)
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["as_of"] = _to_ts(a["as_of"]).values
    a = a.dropna(subset=["as_of", "charter"])
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a = a.sort_values(["as_of"], kind="stable")
    m = m.sort_values("date", kind="stable")
    out = pd.merge_asof(m, a, left_on="date", right_on="as_of", by="rssd_id", direction="backward")
    first = a.drop_duplicates("rssd_id", keep="first").set_index("rssd_id")["charter"]
    out["charter"] = out["charter"].fillna(out["rssd_id"].map(first))
    out = out[out["charter"] == "commercial"]

    # drop thin banks
    n = out.groupby("rssd_id")["y"].transform("size")
    out = out[n >= MIN_OBS]

    res = out.rename(columns={"rssd_id": "id"})[["id", "date", "y"]]
    res = res.sort_values(["id", "date"]).reset_index(drop=True)
    res["id"] = res["id"].astype("int64")
    res["date"] = res["date"].astype("datetime64[ns]")
    res["y"] = res["y"].astype(float)
    return res


if __name__ == "__main__":
    r = build_panel(pd.read_csv("sample_calls.csv"), pd.read_csv("sample_attrs.csv"))
    print(r.shape, r.id.nunique())
    print(r.head())
