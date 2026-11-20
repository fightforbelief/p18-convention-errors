"""Grade N1-N6 (FREEZE_LEDGER.txt) from results_win_eia/. Usage: python scripts/grade_eia.py <results_win_eia>"""
import glob
import os
import sys

import pandas as pd

R = sys.argv[1]
pd.set_option("display.width", 230)
FLAG = ["B1_grid_3mo_step", "B2_survivor_join", "B3_ffill_balanced", "B4_warmup_dropna",
        "x1_step_90d", "x2_quarterend_rollforward", "x3_active_at_end", "x4_long_history"]
NOFLAG = ["B5_mase_scale_leak", "x5_next_target_dropna", "x6_many_to_many_link", "x7_winsorize", "x8_global_zscore",
          "x9_benign_reshuffle"]
res = {}

rd = pd.read_csv(glob.glob(os.path.join(R, "realdata", "realdata_results_eia_plants_all.csv"))[0]).set_index("pipeline")
hit, fa = int(rd.loc[FLAG, "v1_flag"].sum()), int(rd.loc[NOFLAG, "v1_flag"].sum())
res["N1"] = (f"{hit}/8 flagged, {fa}/6 false", hit >= 7 and fa <= 1)

es = pd.read_csv(glob.glob(os.path.join(R, "estimands", "estimands_eia_plants_full.csv"))[0])
e1 = es[es.estimand == "E1"].set_index("pipeline")
res["N2"] = ({k: round(float(e1.loc[k, "rel_shift"]), 4) for k in ["B1_grid_3mo_step", "B2_survivor_join", "x3_active_at_end"]},
             bool(not e1.loc["B1_grid_3mo_step", "material"] and e1.loc["B2_survivor_join", "material"] and e1.loc["x3_active_at_end", "material"]))
sel = es[es.pipeline.isin(["B1_grid_3mo_step", "B2_survivor_join", "x3_active_at_end", "x4_long_history"])]
res["N3"] = bool((sel[sel.estimand.isin(["E1", "E2"])].T3.abs() < 1e-9).all() and sel[sel.estimand == "E3"].forecast_exact.all())

L = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "llm", "ABL_results_eia_plants.csv")) + glob.glob(os.path.join(R, "llm", "LSWEEP_results_eia_plants.csv"))])
L = L[L.pipeline.str.startswith("G0_S0_")].assign(L=lambda d: d.pipeline.str.extract(r"L(\d+)")[0].astype(int)).sort_values("L")
c = L[L.L >= 16].set_index("L").E1_total
mono = (c.diff().dropna() <= 1e-12).all() or (c.diff().dropna() >= -1e-12).all()
signs = set((c[c.index > 16] > 0).tolist())
res["N4"] = ({int(k): round(float(v), 4) for k, v in L.set_index("L").E1_rel.items()},
             bool(abs(c.loc[16]) < 1e-12 and mono and len(signs) == 1 and L.set_index("L").E1_rel.loc[48] >= 0.05))

A = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(R, "llm", "A_results_eia_plants.csv")))
B = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(R, "llm", "B_results_eia_plants.csv")))
a_exact = int(((A.key_dev == 0) & (A.value_mismatch == 0)).sum())
b_div = int(((B.key_dev >= 0.001) | (B.value_mismatch > 0)).sum())
res["N5"] = (f"A exact {a_exact}/{len(A)}, B diverge {b_div}/{len(B)}, B ledger {int(B.ledger_flag.sum())}/{len(B)}",
             bool(a_exact == len(A) and b_div == len(B) and B.ledger_flag.mean() >= 0.9))

allr = []
for pre in ["A", "B", "A2", "B2", "A3", "B3"]:
    for f in glob.glob(os.path.join(R, "llm", f"{pre}_results_eia_plants.csv")):
        allr.append(pd.read_csv(f).assign(arm=pre))
allr = pd.concat(allr)
mat = allr[allr.E1_rel >= 0.10][["arm", "pipeline", "E1_rel", "key_dev"]]
res["N6"] = ("material LLM cells (check min length by code review):\n" + (mat.round(4).to_string(index=False) if len(mat) else "none"), None)

for k, v in res.items():
    print(k, v)
print("\nE1 rel by arm/family (median, max):")
allr["fam"] = allr.pipeline.str.split("_").str[0]
print(allr.groupby(["arm", "fam"]).E1_rel.agg(["median", "max"]).round(4).to_string())
print("\nestimands E1 (design bugs):")
print(e1[["total", "T1", "T2", "T3", "rel_shift", "material", "harm_ratio", "cohort_TV", "ledger_v1_flag"]].round(4).to_string())
