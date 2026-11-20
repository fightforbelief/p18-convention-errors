"""Task 3 grading: survivorship diagnostic + S1-S6 (see FREEZE_LEDGER.txt).
Survivorship diagnostic: share of the reference's entity-periods from EXITED banks (last report < sample end)
that the pipeline's output drops. Usage: python grade3.py <work_dir_B3> <work_dir_A3>"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import reference_region as REF

PANELS = ["bank_leverage", "bank_cash_liquidity", "bhc_leverage", "bhc_cash_liquidity"]


def surv_diag(work, prefix):
    rows = []
    for t in PANELS:
        inp = f"{work}/inputs_{t}"
        ins = {os.path.basename(f)[:-8]: pd.read_parquet(f) for f in glob.glob(f"{inp}/*.parquet")}
        ref = REF.build_panel(**ins)
        last = ins["calls"].groupby("rssd_id").report_date.max()
        exited = set(last[last < last.max()].index)
        rx = ref[ref.id.isin(exited)]
        rk = pd.MultiIndex.from_frame(rx[["id", "date"]])
        for f in sorted(glob.glob(f"{work}/out_{t}_*.parquet")):
            name = os.path.basename(f)[len(f"out_{t}_"):-8]
            p = pd.read_parquet(f)
            p["date"] = pd.to_datetime(p["date"])
            pk = pd.MultiIndex.from_frame(p[["id", "date"]].astype({"id": "int64"}))
            rows.append({"panel": t, "pipeline": name, "exited_dropped": 1 - rk.isin(pk).mean() if len(rk) else np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    wB, wA = sys.argv[1], sys.argv[2]
    d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("B3_results_*.csv")))
    a = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("A3_results_*.csv")))
    d = d.merge(surv_diag(wB, "B3"), on=["panel", "pipeline"], how="left")
    d["fam"] = d.pipeline.str.split("_").str[0]
    d["surv"] = d.exited_dropped >= 0.5
    sp = d.groupby("pipeline").surv.any()
    d.to_csv("B3_graded.csv", index=False)
    res = {}
    res["S1"] = (int(sp.sum()), bool(sp.sum() >= 5))
    s = d[d.pipeline.isin(sp[sp].index)]
    res["S2"] = bool(len(s) and s.groupby("pipeline").apply(lambda g: (g.E1_rel >= 0.10).sum() >= 3).all())
    res["S3"] = bool(len(s) and s.ledger_flag.all())
    a_exact = a.groupby("pipeline").apply(lambda g: ((g.key_dev == 0) & (g.value_mismatch == 0)).all())
    res["S4"] = (int(a_exact.sum()), bool(a_exact.sum() >= 7))
    fam = sp.groupby(sp.index.str.split("_").str[0]).sum()
    res["S5"] = (fam.to_dict(), bool((fam >= 2).any() and (fam <= 1).any()))
    sel = (s.E1_total - s.E1_T3).abs() / s.E1_total.abs()
    res["S6"] = (round(float(sel.median()), 3) if len(s) else None, bool(len(s) and sel.median() >= 0.7))
    for k, v in res.items():
        print(k, v)
    pd.set_option("display.width", 250)
    print(d[["panel", "pipeline", "status", "key_dev", "missing_keys", "extra_keys", "value_mismatch", "exited_dropped",
             "ledger_flag", "E1_total", "E1_T3", "E1_rel", "E2_rel", "E3_top1_changed"]].round(4).to_string(index=False))
    if "err" in d:
        print(d[d.status != "ok"][["panel", "pipeline", "err"]])
