"""Entity-period ledger v1. Written after v0 failed on real data (bank_leverage):
v0 compared retention against RAW, so a legitimate quality filter already broke
'uniformity'. v1 is DIFFERENTIAL against a declared reference pipeline.

Rules (all decided before running v1 on any data; see FREEZE_LEDGER.txt):
  R1 key provenance: every processed (id, date) must exist in raw.     (values NOT checked)
  R2 absolute floor: entity-periods kept >= 0.5 x reference's.
  R3 differential uniformity: for each life-cycle coordinate (entry cohort x calendar
     quarter, exit status, age-in-years capped at 5), and for both entity and
     entity-period retention, spread(candidate) - spread(reference) <= 0.10.
Flag if any rule fails. Life-cycle coordinates come from RAW only.
"""
import numpy as np
import pandas as pd

TOL = 0.10
FLOOR = 0.5


def _coords(raw, qdates):
    first = raw.groupby("id").date.min()
    last = raw.groupby("id").date.max()
    ent = pd.Series(np.where(first == qdates[0], "inc", "ent"), index=first.index) + "_Q" + first.dt.quarter.astype(str)
    ext = pd.Series(np.where(last == qdates[-1], "alive", "exited"), index=last.index)
    return first, ent, ext


def spreads(raw, p, qdates):
    key = ["id", "date"]
    first, ent, ext = _coords(raw, qdates)
    kept_ids = set(p.id.unique())
    s = {}
    for name, c in {"entry": ent, "exit": ext}.items():
        r = c.groupby(c).apply(lambda x: x.index.isin(kept_ids).mean())
        s[f"entities_by_{name}"] = float(r.max() - r.min())
    rk = raw[key].copy()
    rk["kept"] = pd.MultiIndex.from_frame(rk[key]).isin(pd.MultiIndex.from_frame(p[key]))
    age_q = (rk.date.dt.year - rk.id.map(first).dt.year) * 4 + (rk.date.dt.quarter - rk.id.map(first).dt.quarter)
    rk["age"] = np.minimum(age_q // 4, 5)
    rk["entry"], rk["exit"] = rk.id.map(ent), rk.id.map(ext)
    for name in ["age", "entry", "exit"]:
        r = rk.groupby(name).kept.mean()
        s[f"periods_by_{name}"] = float(r.max() - r.min())
    return s, float(rk.kept.mean())


def check(raw, p, ref, qdates):
    key = ["id", "date"]
    r1 = bool(pd.MultiIndex.from_frame(p[key]).isin(pd.MultiIndex.from_frame(raw[key])).all())
    s_p, kept_p = spreads(raw, p, qdates)
    s_r, kept_r = spreads(raw, ref, qdates)
    r2 = kept_p >= FLOOR * kept_r
    diff = {k: s_p[k] - s_r[k] for k in s_p}
    r3 = all(v <= TOL for v in diff.values())
    return (r1 and r2 and r3), {"R1_keys": r1, "R2_floor": bool(r2), "R3_uniform": r3,
                                "max_diff": round(max(diff.values()), 3)}
