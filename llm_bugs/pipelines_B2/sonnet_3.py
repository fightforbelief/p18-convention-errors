import numpy as np
import pandas as pd

MIN_OBS = 12  # minimum valid YoY observations per bank (enough to hold out 4 and still train)


def _to_qidx(d):
    """YYYYMMDD int array -> integer quarter index (year*4 + quarter-1)."""
    d = np.asarray(d, dtype="int64")
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c["value"] = pd.to_numeric(c["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    c["q"] = _to_qidx(c["report_date"])

    # Duplicate filings for a bank-quarter: keep the latest report date, prefer non-missing value.
    c = c.sort_values(["rssd_id", "q", "value", "report_date"], na_position="first")
    c = c.drop_duplicates(["rssd_id", "q"], keep="last")

    # Point-in-time charter: latest snapshot at or before the report date
    # (earliest snapshot if the report precedes all snapshots).
    a = attrs[["rssd_id", "as_of", "charter"]].dropna().copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["as_of"] = a["as_of"].astype("int64")
    a["charter"] = a["charter"].astype(str).str.strip().str.lower()
    a = a.sort_values(["as_of", "rssd_id"]).drop_duplicates(["rssd_id", "as_of"], keep="last")
    c = c.sort_values(["report_date", "rssd_id"])
    m = pd.merge_asof(c, a.rename(columns={"as_of": "report_date"}),
                      on="report_date", by="rssd_id", direction="backward")
    first = a.sort_values("as_of").drop_duplicates("rssd_id").set_index("rssd_id")["charter"]
    m["charter"] = m["charter"].fillna(m["rssd_id"].map(first))
    m = m[m["charter"] == "commercial"]

    # YoY change vs. exactly the same quarter one year earlier; gaps give no value
    # (never bridge across a missing quarter).
    cur = m[["rssd_id", "q", "value"]]
    prev = cur.assign(q=cur["q"] + 4).rename(columns={"value": "prev"})
    j = cur.merge(prev, on=["rssd_id", "q"], how="inner")
    j["y"] = j["value"] - j["prev"]
    j = j[np.isfinite(j["y"])]

    # Drop banks with too little data
    n = j.groupby("rssd_id")["y"].transform("size")
    j = j[n >= MIN_OBS]

    yr, qq = j["q"] // 4, j["q"] % 4
    dates = pd.to_datetime(yr.astype(str) + "-" + ((qq + 1) * 3).astype(str).str.zfill(2) + "-01") \
        + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"id": j["rssd_id"].astype("int64").values,
                        "date": dates.values,
                        "y": j["y"].astype(float).values})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
