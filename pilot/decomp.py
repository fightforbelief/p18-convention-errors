"""Exact decomposition of a pipeline's headline shift into composition vs alteration.

Headline: Delta = mean_i d_i over evaluated banks, d_i = MASE_naive - MASE_ha.
Cells c = life-cycle cohort from RAW data (entry: incumbent/entrant x first calendar quarter;
exit: alive/exited) -> up to 10 cells.  r = reference pipeline, p = candidate.

  Delta_p - Delta_r = T1 + T2 + T3   (identity)
  T1 between-cohort composition  = sum_c (w_pc - w_rc) * mu_rc
  T2 within-cohort selection     = sum_c w_pc * (mu_r[K_pc] - mu_rc),  K_pc = banks in c kept by both
  T3 series alteration (+ banks new to p) = remainder

FORECAST = T1 + T2 uses only the reference run's per-bank results and the candidate's
eligible bank set (computable from the processed panel, no model fit on the candidate).
"""
import sys

import numpy as np
import pandas as pd

import ledger_frozen_v0 as F
import ledger_v1 as L1


def cells(raw):
    _, ent, ext = L1._coords(raw, F.QDATES)
    return ent + "|" + ext


def eligible_ids(p):
    """Banks the evaluator would score, from the panel alone (no model)."""
    g = p.groupby("id").y
    n = g.size()
    return set(n[n >= F.MIN_LEN].index)


def decompose(ev_r, ev_p, cell, elig_p=None):
    dr = (ev_r.set_index("id").naive - ev_r.set_index("id").ha)
    dp = (ev_p.set_index("id").naive - ev_p.set_index("id").ha)
    cr, cp = cell.reindex(dr.index), cell.reindex(dp.index)
    w_r = cr.value_counts(normalize=True)
    mu_r = dr.groupby(cr).mean()
    # forecast uses the candidate's eligible set; actual uses its evaluated set
    out = {}
    for label, ids in {"actual": dp.index, "forecast": pd.Index(sorted(elig_p)) if elig_p else dp.index}.items():
        c_ids = cell.reindex(ids)
        w_p = c_ids.value_counts(normalize=True)
        allc = w_r.index.union(w_p.index)
        wr, wp, mr = w_r.reindex(allc, fill_value=0), w_p.reindex(allc, fill_value=0), mu_r.reindex(allc, fill_value=0)
        T1 = float(((wp - wr) * mr).sum())
        common = ids.intersection(dr.index)
        mr_k = dr.reindex(common).groupby(cell.reindex(common)).mean().reindex(allc)
        T2 = float((wp * (mr_k - mr)).fillna(0).sum())
        out[label] = (T1, T2)
    total = float(dp.mean() - dr.mean())
    T1, T2 = out["actual"]
    het = float(np.sqrt((w_r * (mu_r - (w_r * mu_r).sum()) ** 2).sum()))
    tv = float(0.5 * (w_r.reindex(w_r.index.union(cp.value_counts(normalize=True).index), fill_value=0)
                      - cp.value_counts(normalize=True).reindex(w_r.index.union(cp.value_counts(normalize=True).index), fill_value=0)).abs().sum())
    return {"total": total, "T1_between": T1, "T2_within_sel": T2, "T3_alteration": total - T1 - T2,
            "forecast_T1T2": sum(out["forecast"]), "cohort_TV": tv, "heterogeneity_sd": het,
            "ref_delta": float(dr.mean()),
            "ref_halfwidth": float(1.96 * dr.std() / np.sqrt(len(dr))),
            "harm_ratio": abs(total) / float(1.96 * dr.std() / np.sqrt(len(dr)))}


def run(raw, pipes, tag):
    ref_p = F.regrid_clean(raw)
    ev_r = F.evaluate(ref_p)
    cell = cells(raw)
    rows = []
    for name, (fn, leak) in pipes.items():
        p = fn(raw)
        ev_p = F.evaluate(p, scale_leak=leak)
        if len(ev_p) == 0:
            continue
        r = decompose(ev_r, ev_p, cell, eligible_ids(p))
        rows.append({"panel": tag, "pipeline": name, **r})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import heldout_v0 as X
    import realdata_v0 as R
    pipes = {k: v for k, v in F.PIPELINES.items() if k != "C0_clean"}
    pipes = {**pipes, **{k: (fn, False) for k, fn in X.HELDOUT.items()}}
    frames = []
    for src in sys.argv[1:]:
        if src.startswith("synthetic"):
            seed = int(src.split(":")[1])
            F.QDATES = pd.date_range("1990-03-31", "2020-12-31", freq="QE")
            raw = F.simulate(seed=seed)
            frames.append(run(raw, pipes, src))
        else:
            raw = R.load(src, 4000, 0)
            p2 = dict(pipes); p2["B1_grid_3mo_step"] = (R.bug_grid_step_long, False)
            frames.append(run(raw, p2, src.split("/")[-1].replace("_panel.parquet", "")))
        print("done", src, flush=True)
    out = pd.concat(frames)
    out.to_csv("decomp_results.csv", index=False)
    print(out.round(3).to_string())
