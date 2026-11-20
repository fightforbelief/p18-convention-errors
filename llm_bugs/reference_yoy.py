"""Reference implementation of spec_A2.md (year-over-year change panel)."""
import pandas as pd


def build_panel(calls, attrs):
    c = calls.copy()
    c["date"] = pd.to_datetime(c.report_date.astype(str), format="%Y%m%d")
    c = c.dropna(subset=["value"]).sort_values("date").drop_duplicates(["rssd_id", "date"], keep="last")
    a = attrs.copy()
    a["as_of_dt"] = pd.to_datetime(a.as_of.astype(str), format="%Y%m%d")
    c = pd.merge_asof(c.sort_values("date"), a.sort_values("as_of_dt")[["rssd_id", "as_of_dt", "charter"]],
                      left_on="date", right_on="as_of_dt", by="rssd_id", direction="backward")
    u = c[c.charter == "commercial"][["rssd_id", "date", "value"]]
    u = u.assign(q=u.date.dt.to_period("Q"))
    lag = u[["rssd_id", "q", "value"]].assign(q=lambda d: d.q + 4).rename(columns={"value": "v4"})
    m = u.merge(lag, on=["rssd_id", "q"], how="inner")
    m["y"] = m.value - m.v4
    n = m.groupby("rssd_id").y.size()
    m = m[m.rssd_id.isin(n[n >= 16].index)]
    out = m.rename(columns={"rssd_id": "id"})[["id", "date", "y"]]
    return out.sort_values(["id", "date"]).reset_index(drop=True)
