import numpy as np
import pandas as pd

MIN_OBS = 12  # minimum number of YoY observations per bank (holdout of 4 + >=8 to train)


def _to_qend(d):
    dt = pd.to_datetime(d.astype("int64").astype(str), format="%Y%m%d")
    return dt + pd.offsets.QuarterEnd(0)  # snap to calendar quarter end (no-op if already)


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame({
        "id": calls["rssd_id"].astype("int64").values,
        "date": _to_qend(calls["report_date"]).values,
        "value": calls["value"].astype(float).values,
    })
    c = c[np.isfinite(c["value"])]
    # one report per bank-quarter (keep last filed)
    c = c.drop_duplicates(["id", "date"], keep="last")
    c["q"] = c["date"].dt.year * 4 + (c["date"].dt.month - 1) // 3

    # YoY only against the exact same quarter a year earlier; gaps give no value (no bridging)
    prev = c[["id", "q", "value"]].copy()
    prev["q"] += 4
    m = c.merge(prev, on=["id", "q"], suffixes=("", "_prev"))
    m["y"] = m["value"] - m["value_prev"]
    m = m[np.isfinite(m["y"])]

    # Charter is time-varying: use the latest snapshot at or before each date
    # (earliest snapshot for dates preceding the first one).
    a = pd.DataFrame({
        "id": attrs["rssd_id"].astype("int64").values,
        "as_of": pd.to_datetime(attrs["as_of"].astype("int64").astype(str), format="%Y%m%d").values,
        "charter": attrs["charter"].values,
    }).dropna().sort_values("as_of").drop_duplicates(["id", "as_of"], keep="last")
    m = m.sort_values("date")
    out = pd.merge_asof(m, a.rename(columns={"as_of": "date"}), on="date", by="id", direction="backward")
    first = a.sort_values("as_of").drop_duplicates("id").set_index("id")["charter"]
    out["charter"] = out["charter"].fillna(out["id"].map(first))
    out = out[out["charter"] == "commercial"]

    # drop thin banks
    n = out.groupby("id")["y"].transform("size")
    out = out[n >= MIN_OBS]

    out = out[["id", "date", "y"]].sort_values(["id", "date"]).reset_index(drop=True)
    out["id"] = out["id"].astype("int64")
    out["y"] = out["y"].astype(float)
    return out


if __name__ == "__main__":
    r = build_panel(pd.read_csv("sample_calls.csv"), pd.read_csv("sample_attrs.csv"))
    print(r.shape, r.id.nunique())
    print(r.head())
