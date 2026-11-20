import numpy as np
import pandas as pd


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c[c["value"].notna()]
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c = c.sort_values(["rssd_id", "report_date"], kind="mergesort")

    # charter at each report: latest snapshot on or before report_date
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["as_of"] = a["as_of"].astype("int64")
    a = a.sort_values(["as_of", "rssd_id"], kind="mergesort")
    c = c.sort_values(["report_date", "rssd_id"], kind="mergesort")
    c = pd.merge_asof(c, a, left_on="report_date", right_on="as_of",
                      by="rssd_id", direction="backward")
    c = c[c["charter"] == "commercial"]

    # calendar quarter index; duplicates of a bank-quarter: keep last filed
    d = c["report_date"].to_numpy()
    year = d // 10000
    month = (d // 100) % 100
    c["q"] = year * 4 + (month - 1) // 3
    c = c.sort_values(["rssd_id", "q"], kind="mergesort")
    c = c.drop_duplicates(["rssd_id", "q"], keep="last")
    if c.empty:
        return empty

    cur = c[["rssd_id", "q", "value"]]
    prev = cur.assign(q=cur["q"] + 4).rename(columns={"value": "prev"})
    m = cur.merge(prev, on=["rssd_id", "q"], how="inner")
    m["y"] = m["value"] - m["prev"]
    m = m[m["y"].notna()]
    cnt = m.groupby("rssd_id")["q"].transform("size")
    m = m[cnt >= 16]
    if m.empty:
        return empty

    yr = m["q"] // 4
    mo = (m["q"] % 4) * 3 + 3
    date = pd.to_datetime(pd.DataFrame({"year": yr, "month": mo, "day": 1})) \
        + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"id": m["rssd_id"].astype("int64").to_numpy(),
                        "date": date.to_numpy(),
                        "y": m["y"].astype(float).to_numpy()})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
