"""Grade AB0-AB6 (FREEZE_LEDGER.txt) from ABL_results_*.csv; reference LLM shifts from B_results_*.csv."""
import glob

import pandas as pd

d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("ABL_results_*.csv")))
cfg = d.pipeline.str.extract(r"G(\d)_S(\d)_L(\d+)").astype(int)
d["G"], d["S"], d["L"] = cfg[0], cfg[1], cfg[2]
llm = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("B_results_*.csv")))
son = llm[llm.pipeline == "sonnet_4"].set_index("panel").E1_total
g = lambda name: d[d.pipeline == name].set_index("panel")
res = {}
res["AB0"] = bool((g("G0_S0_L0").key_dev == 0).all())
res["AB1"] = bool((g("G1_S0_L0").E1_rel <= 0.01).all())
res["AB2"] = bool(all((g(f"G0_S0_L{L}").E1_rel <= 0.03).all() for L in (16, 20, 24)))
a, b = g("G1_S1_L0").E1_total, g("G1_S1_L24").E1_total
res["AB3"] = bool((a < 0).sum() >= 3 and ((a / b) >= 0.6).all())
ratio = b / son.reindex(b.index)
res["AB4"] = (ratio.round(2).to_dict(), bool(((ratio - 1).abs() <= 0.3).all()))
s = d[(d.S == 1) & (d.E1_rel >= 0.02)]
sel = (s.E1_total - s.E1_T3).abs() / s.E1_total.abs()
res["AB5"] = (round(float(sel.min()), 2) if len(s) else None, bool((sel >= 0.7).all()))
res["AB6"] = bool(d[d.S == 1].ledger_flag.all() and not d[(d.G == 0) & (d.S == 0) & (d.L <= 16)].ledger_flag.any())
for k, v in res.items():
    print(k, v)
pd.set_option("display.width", 250)
piv = d.pivot_table(index="pipeline", columns="panel", values="E1_total").round(4)
print("\nE1 shift (total) by config x panel:\n", piv.to_string())
print("\nE1 rel:\n", d.pivot_table(index="pipeline", columns="panel", values="E1_rel").round(3).to_string())
print("\nselection share:\n", d.assign(sel=(d.E1_total - d.E1_T3).abs() / d.E1_total.abs())
      .pivot_table(index="pipeline", columns="panel", values="sel").round(2).to_string())
print("\nledger flag:\n", d.pivot_table(index="pipeline", columns="panel", values="ledger_flag").to_string())
print("\nkey_dev:\n", d.pivot_table(index="pipeline", columns="panel", values="key_dev").round(3).to_string())
print("\nLLM sonnet_4 (task-1 arm B) E1 shift:", son.round(4).to_dict())
