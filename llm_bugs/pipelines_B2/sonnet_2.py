import numpy as np
import pandas as pd

MIN_OBS = 12  # minimum number of YoY observations per bank (4 held out + >=8 for training)


def _qidx(yyyymmdd):
    """Integer quarter index (year*4 + quarter-1) from YYYYMMDD ints."""
    y = yyyymmdd // 10000
    m = (yyyymmdd // 100) % 100
    return y * 4 + (m - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["report_date"] = c["report_date"].astype("int64")
    c["q"] = _qidx(c["report_date"].to_numpy())
    # one report per bank-quarter: keep the latest-dated non-missing value
    c = c.sort_values(["rssd_id", "q", "report_date"])
    c["_nn"] = c["value"].notna()
    c = c.sort_values(["rssd_id", "q", "_nn", "report_date"]).drop_duplicates(["rssd_id", "q"], keep="last")
    c = c[c["_nn"]]

    # charter in force at the quarter's report date (as-of join on snapshots)
    a = attrs[["rssd_id", "as_of", "charter"]].dropna().copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["as_of_ts"] = pd.to_datetime(a["as_of"].astype(str), format="%Y%m%d")
    a["charter"] = a["charter"].astype(str).str.strip().str.lower()
    a = a.sort_values(["as_of_ts", "rssd_id"])
    c["rd_ts"] = pd.to_datetime(c["report_date"].astype(str), format="%Y%m%d")
    c = c.sort_values(["rd_ts", "rssd_id"])
    m = pd.merge_asof(c, a[["rssd_id", "as_of_ts", "charter"]], left_on="rd_ts",
                      right_on="as_of_ts", by="rssd_id", direction="backward")
    # before the first snapshot: use the bank's earliest known charter
    first = a.sort_values("as_of_ts").drop_duplicates("rssd_id", keep="first").set_index("rssd_id")["charter"]
    m["charter"] = m["charter"].fillna(m["rssd_id"].map(first))

    # year-over-year change against the same calendar quarter one year earlier (gaps -> no row)
    prev = c[["rssd_id", "q", "value"]].copy()
    prev["q"] += 4
    prev = prev.rename(columns={"value": "prev"})
    m = m.merge(prev, on=["rssd_id", "q"], how="left")
    m["y"] = m["value"] - m["prev"]
    m = m[(m["charter"] == "commercial") & m["y"].notna() & np.isfinite(m["y"])]

    # drop banks with too little history
    cnt = m.groupby("rssd_id")["y"].transform("size")
    m = m[cnt >= MIN_OBS]

    date = pd.to_datetime(m["rd_ts"]).dt.to_period("Q").dt.end_time.dt.normalize()
    out = pd.DataFrame({"id": m["rssd_id"].astype("int64").to_numpy(),
                        "date": date.to_numpy(),
                        "y": m["y"].astype(float).to_numpy()})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
