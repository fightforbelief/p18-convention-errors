import numpy as np
import pandas as pd

MIN_OBS = 12  # minimum YoY observations per bank (holdout takes last 4 quarters)


def _qidx(d):
    d = d.astype("int64")
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["rssd_id"] = c["rssd_id"].astype("int64")
    c["q"] = _qidx(c["report_date"])
    c = c.sort_values(["rssd_id", "q", "report_date"], kind="mergesort")
    # one row per bank-quarter (keep last filed)
    c = c.drop_duplicates(["rssd_id", "q"], keep="last")
    c["value"] = c["value"].replace([np.inf, -np.inf], np.nan)

    # YoY by exact calendar match (t vs t-4 quarters), so gaps never mis-align
    prev = c[["rssd_id", "q", "value"]].copy()
    prev["q"] += 4
    prev = prev.rename(columns={"value": "prev"})
    m = c.merge(prev, on=["rssd_id", "q"], how="inner")
    m["y"] = m["value"] - m["prev"]
    m = m.dropna(subset=["y"])

    # charter in force at report date: latest snapshot on/before it (else earliest)
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["rssd_id"] = a["rssd_id"].astype("int64")
    a["aq"] = _qidx(a["as_of"])
    a = a.sort_values(["aq", "as_of"], kind="mergesort").drop_duplicates(["rssd_id", "aq"], keep="last")
    a = a.dropna(subset=["charter"]).sort_values("aq")
    m = m.sort_values("q")
    m = pd.merge_asof(m, a[["rssd_id", "aq", "charter"]], left_on="q", right_on="aq",
                      by="rssd_id", direction="backward")
    first = a.drop_duplicates("rssd_id", keep="first").set_index("rssd_id")["charter"]
    m["charter"] = m["charter"].fillna(m["rssd_id"].map(first))
    m = m[m["charter"] == "commercial"]

    # drop thin banks
    n = m.groupby("rssd_id")["y"].transform("size")
    m = m[n >= MIN_OBS]

    year = m["q"] // 4
    qtr = m["q"] % 4
    date = pd.to_datetime(year.astype(str) + "-" + ((qtr + 1) * 3).astype(str) + "-01") + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"id": m["rssd_id"].astype("int64").to_numpy(),
                        "date": date.to_numpy(),
                        "y": m["y"].astype(float).to_numpy()})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
