import numpy as np
import pandas as pd


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c = c[c["value"].notna()]
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c = c.drop_duplicates(["rssd_id", "report_date"], keep="last")

    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["as_of"] = a["as_of"].astype("int64")
    a = a.drop_duplicates(["rssd_id", "as_of"], keep="last")

    # charter at each quarter: latest snapshot on or before the report date
    c = c.sort_values("report_date", kind="mergesort")
    a = a.sort_values("as_of", kind="mergesort")
    m = pd.merge_asof(c, a, left_on="report_date", right_on="as_of",
                      by="rssd_id", direction="backward")
    m = m[m["charter"] == "commercial"]

    # absolute quarter index so lag is exactly 4 calendar quarters (gaps are not skipped)
    yr = m["report_date"] // 10000
    mo = (m["report_date"] // 100) % 100
    m = m.assign(q=(yr * 4 + (mo - 1) // 3).astype("int64"))
    m = m[["rssd_id", "q", "report_date", "value"]]

    prev = m[["rssd_id", "q", "value"]].copy()
    prev["q"] = prev["q"] + 4
    prev = prev.rename(columns={"value": "value_prev"})
    j = m.merge(prev, on=["rssd_id", "q"], how="inner")
    j["y"] = j["value"] - j["value_prev"]
    j = j[j["y"].notna()]

    cnt = j.groupby("rssd_id")["y"].transform("size")
    j = j[cnt >= 16]

    out = pd.DataFrame({
        "id": j["rssd_id"].astype("int64").values,
        "date": pd.to_datetime(j["report_date"].astype(str), format="%Y%m%d").values,
        "y": j["y"].astype(float).values,
    })
    return out.sort_values(["id", "date"]).reset_index(drop=True)
