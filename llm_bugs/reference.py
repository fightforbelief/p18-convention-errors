"""Reference implementation of spec.md (the 'clean' pipeline for the LLM-bug study)."""
import numpy as np
import pandas as pd


def build_panel(calls, attrs):
    c = calls.copy()
    c["date"] = pd.to_datetime(c.report_date.astype(str), format="%Y%m%d")
    c = c.sort_values("date").drop_duplicates(["rssd_id", "date"], keep="last")
    span = c.groupby("rssd_id").date.agg(["min", "max"])
    p0, p1 = span["min"].dt.to_period("Q"), span["max"].dt.to_period("Q")
    n = (p1 - p0).apply(lambda x: x.n).to_numpy() + 1
    ids = np.repeat(span.index.to_numpy(), n)
    off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    q = pd.PeriodIndex(np.repeat(p0.to_numpy(), n), freq="Q") + off
    g = pd.DataFrame({"rssd_id": ids, "date": q.to_timestamp(how="end").normalize()})
    m = g.merge(c[["rssd_id", "date", "value"]], on=["rssd_id", "date"], how="left")
    a = attrs.copy()
    a["as_of_dt"] = pd.to_datetime(a.as_of.astype(str), format="%Y%m%d")
    m = pd.merge_asof(m.sort_values("date"), a.sort_values("as_of_dt")[["rssd_id", "as_of_dt", "charter"]],
                      left_on="date", right_on="as_of_dt", by="rssd_id", direction="backward")
    m = m[m.charter == "commercial"]
    miss = m.value.isna().groupby(m.rssd_id).mean()
    m = m[m.rssd_id.isin(miss[miss <= 0.2].index)].dropna(subset=["value"])
    out = m.rename(columns={"rssd_id": "id", "value": "y"})[["id", "date", "y"]]
    return out.sort_values(["id", "date"]).reset_index(drop=True)
