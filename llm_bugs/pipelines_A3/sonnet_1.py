import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}


def _qidx(d):
    d = d.astype("int64")
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def _qend(q):
    y, k = q // 4, q % 4
    return pd.to_datetime(y * 10000 + (k + 1) * 300 + np.where(k == 0, 31, np.where(k == 3, 31, 30)),
                          format="%Y%m%d")


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})
    if len(calls) == 0:
        return empty
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["q"] = _qidx(c["report_date"].to_numpy())
    # one value per bank-quarter (prefer a non-NaN value)
    c = c.sort_values(["rssd_id", "q", "value"], na_position="first").drop_duplicates(["rssd_id", "q"], keep="last")

    span = c.groupby("rssd_id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    g = pd.DataFrame({"rssd_id": ids, "q": np.repeat(span["min"].to_numpy(), n) + offs})
    g = g.merge(c[["rssd_id", "q", "value"]], on=["rssd_id", "q"], how="left")

    # headquarters state at the quarter end, from history only
    qe = _qend(g["q"].to_numpy())
    g["qe"] = qe.year * 10000 + qe.month * 100 + qe.day
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h["start_date"] = pd.to_numeric(h["start_date"], errors="coerce")
    h["end_date"] = pd.to_numeric(h["end_date"], errors="coerce").fillna(99991231)
    h = h.dropna(subset=["start_date"])
    h["start_date"] = h["start_date"].astype("int64")
    h["end_date"] = h["end_date"].astype("int64")
    h = h.sort_values("start_date")
    g["rssd_id"] = g["rssd_id"].astype("int64")
    g["qe"] = g["qe"].astype("int64")
    h["rssd_id"] = h["rssd_id"].astype("int64")
    g = g.sort_values("qe")
    g = pd.merge_asof(g, h.rename(columns={"start_date": "qe"}), on="qe", by="rssd_id", direction="backward")
    ok = g["state"].isin(MIDWEST) & (g["qe"] <= g["end_date"])
    g = g[ok]

    g["miss"] = g["value"].isna()
    frac = g.groupby("rssd_id")["miss"].transform("mean")
    g = g[(frac <= 0.2) & ~g["miss"]]
    if len(g) == 0:
        return empty
    out = pd.DataFrame({"id": g["rssd_id"].astype("int64").to_numpy(),
                        "date": _qend(g["q"].to_numpy()),
                        "y": g["value"].astype(float).to_numpy()})
    return out.sort_values(["id", "date"]).reset_index(drop=True)
