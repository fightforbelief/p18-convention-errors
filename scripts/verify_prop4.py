"""Prop 4(a) check against the SAME reference the ablation used: llm_bugs/reference.py (task-1 spec, charter filter)
on the cached task-1 inputs (data/work/t12 for banks, data/work_eia/t12 for EIA). verify_props.py used the unfiltered
regrid_clean reference, which is a different population; this script fixes that comparison."""
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(ROOT, "pilot"))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(ROOT, "llm_bugs"))
import estimands as E  # noqa: E402
import reference as REF  # noqa: E402

INP = {**{p: os.path.join(ROOT, "data", "work", "t12", f"inputs_{p}") for p in
          ["bank_leverage", "bank_cash_liquidity", "bhc_leverage", "bhc_cash_liquidity"]},
       "eia_plants": os.path.join(ROOT, "data", "work_eia", "t12", "inputs_eia_plants")}
rows = []
for p, d in INP.items():
    ref = REF.build_panel(pd.read_parquet(os.path.join(d, "calls.parquet")), pd.read_parquet(os.path.join(d, "attrs.parquet")))
    R = E.per_bank(ref)
    a = R.naive - R.ha
    n = ref.groupby("id").size().reindex(a.index)
    res_dir = "results_win_eia" if p == "eia_plants" else "results_win"
    meas = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, res_dir, "llm", f"*_results_{p}.csv"))
                      if os.path.basename(f).startswith(("ABL_", "LSWEEP_"))])
    for _, r in meas[meas.pipeline.str.startswith("G0_S0_")].iterrows():
        L = int(r.pipeline.split("_L")[1])
        pi = float((n >= L).mean())
        pred = 0.0 if (L <= 16 or pi == 1) else (1 - pi) * (float(a[n >= L].mean()) - float(a[n < L].mean()))
        rows.append({"panel": p, "L": L, "predicted": pred, "measured": float(r.E1_total), "abs_err": abs(pred - r.E1_total)})
    print(p, "done", flush=True)
d = pd.DataFrame(rows).sort_values(["panel", "L"])
d.to_csv(os.path.join(ROOT, "results_win", "proofs", "prop4_checks_taskref.csv"), index=False)
print("max |predicted - measured| =", d.abs_err.max())
print(d.round(6).to_string(index=False))
