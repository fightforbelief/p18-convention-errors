"""Segment-rule ablation on the task-1 reference (spec.md).
Starts from the reference grid after the charter filter and the <=20%-missing coverage filter, then applies:
  G  interpolate interior gaps of <= 2 consecutive missing quarters (linear)
  S  keep only the most recent contiguous run of non-missing quarters (after G)
  L  drop banks with fewer than L remaining quarters (0 = off)
These are the conventions the arm-B LLM pipelines chose (llm_bugs/pipelines_B, pipelines_B3)."""
import numpy as np
import pandas as pd

import reference as REF


def _grid(calls, attrs):
    """Reference steps 1-3, returning the grid WITH missing rows (value NaN)."""
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
    m = m[m.rssd_id.isin(miss[miss <= 0.2].index)]
    return m.sort_values(["rssd_id", "date"])[["rssd_id", "date", "value"]].reset_index(drop=True)


def _bank(v, G, S):
    v = v.copy()
    isn = np.isnan(v)
    if G and isn.any():
        idx = np.arange(len(v))
        # runs of NaN
        d = np.diff(np.r_[0, isn.astype(int), 0])
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if s > 0 and e < len(v) and (e - s) <= 2:
                v[s:e] = np.interp(idx[s:e], [s - 1, e], [v[s - 1], v[e]])
        isn = np.isnan(v)
    keep = ~isn
    if S and isn.any():
        last_nan = np.where(isn)[0].max()
        keep = keep & (np.arange(len(v)) > last_nan)
    return v, keep


def variant(calls, attrs, G=False, S=False, L=0):
    m = _grid(calls, attrs)
    out = []
    for i, g in m.groupby("rssd_id", sort=False):
        v, keep = _bank(g.value.to_numpy(dtype=float), G, S)
        if keep.sum() >= max(L, 1):
            out.append(pd.DataFrame({"id": i, "date": g.date.to_numpy()[keep], "y": v[keep]}))
    p = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["id", "date", "y"])
    return p.sort_values(["id", "date"]).reset_index(drop=True)


def check_reference_equivalence(calls, attrs):
    a, b = variant(calls, attrs), REF.build_panel(calls, attrs)
    return len(a) == len(b) and bool((a.id.values == b.id.values).all() and (a.date.values == b.date.values).all()
                                     and np.allclose(a.y.values, b.y.values))
