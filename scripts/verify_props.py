"""Numerical checks of paper/PROOFS.md on all panels (run on hanqing-win from the project root).
Prop 2(a)/(b), Prop 3(a): inequalities for the selection bugs B1, B2, x3, x4 under E1 with entry x exit cells.
Prop 4(a): closed-form min-length shift vs the measured G0_S0_L* results (results_win*/llm/{ABL,LSWEEP}_results_<panel>.csv).
Prop 4(b): binned m(n) = mean a by history length, reported for the monotonicity premise."""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(ROOT, "pilot"))
sys.path.insert(0, ".")
import estimands as E  # noqa: E402
import heldout_v0 as X  # noqa: E402
import ledger_frozen_v0 as F  # noqa: E402
import ledger_v1 as L1  # noqa: E402
import realdata_v0 as RD  # noqa: E402

PANELS = ["bank_leverage", "bank_cash_liquidity", "bhc_leverage", "bhc_cash_liquidity", "eia_plants"]
BUGS = {"B1_grid_3mo_step": RD.bug_grid_step_long, "B2_survivor_join": F.bug_survivor_join,
        "x3_active_at_end": X.HELDOUT["x3_active_at_end"], "x4_long_history": X.HELDOUT["x4_long_history"]}
TOL = 1e-12


def wmean(a, w):
    return float((a * w).sum() / w.sum())


def props_2_3(a, cell, keep):
    """a: reference contributions (index = S_r), unit weights; keep: K subset."""
    w = pd.Series(1.0, index=a.index)
    c = cell.reindex(a.index)
    Wr = w.groupby(c).sum() / w.sum()
    mur = a.groupby(c).mean()
    k = pd.Series(a.index.isin(keep), index=a.index).astype(float)
    Wk = (k.groupby(c).sum() / k.sum()).reindex(Wr.index, fill_value=0)
    muk = (a[k > 0].groupby(c[k > 0]).mean()).reindex(Wr.index).fillna(mur)
    T1 = float(((Wk - Wr) * mur).sum())
    T2 = float((Wk * (muk - mur)).sum())
    TV = float(0.5 * (Wk - Wr).abs().sum())
    rng = float(mur.max() - mur.min())
    chi2 = float((Wr * (Wk / Wr - 1) ** 2).sum())
    sB = float(np.sqrt((Wr * (mur - wmean(mur, Wr)) ** 2).sum()))
    pi = k.groupby(c).mean()
    sd = a.groupby(c).std(ddof=0)
    b3 = float((Wk * sd * np.sqrt((1 - pi) / pi.where(pi > 0))).fillna(0).sum())
    return dict(T1=T1, T2=T2, TV=TV, range=rng, bound2a=TV * rng, chi2=chi2, sigmaB=sB, bound2b=np.sqrt(chi2) * sB,
                bound3a=b3, ok2a=abs(T1) <= TV * rng + TOL, ok2b=abs(T1) <= np.sqrt(chi2) * sB + TOL,
                ok3a=abs(T2) <= b3 + TOL, sel_check=abs(wmean(a[k > 0], w[k > 0]) - wmean(a, w) - T1 - T2) < 1e-9)


def main():
    rows2, rows4, rowsm = [], [], []
    for p in PANELS:
        raw = RD.load(os.path.join(ROOT, "data", "raw", f"{p}_panel.parquet"))
        ref = F.regrid_clean(raw)
        R = E.per_bank(ref)
        a = (R.naive - R.ha)
        cell = (lambda cc: cc[1] + "|" + cc[2])(L1._coords(raw, F.QDATES))
        for name, fn in BUGS.items():
            keep = E.per_bank(fn(raw)).index.intersection(a.index)
            rows2.append({"panel": p, "bug": name, **props_2_3(a, cell, keep)})
        # Prop 4: lengths of the reference series (what the ablation's L filter acts on)
        n = ref.groupby("id").size().reindex(a.index)
        theta_r = float(a.mean())
        res_dir = "results_win_eia" if p == "eia_plants" else "results_win"
        meas = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, res_dir, "llm", f"*_results_{p}.csv"))
                          if os.path.basename(f).startswith(("ABL_", "LSWEEP_"))])
        meas = meas[meas.pipeline.str.startswith("G0_S0_")]
        for _, r in meas.iterrows():
            L = int(r.pipeline.split("_L")[1])
            if L <= 16:
                pred = 0.0
            else:
                pi = float((n >= L).mean())
                pred = (1 - pi) * (float(a[n >= L].mean()) - float(a[n < L].mean())) if pi < 1 else 0.0
            rows4.append({"panel": p, "L": L, "predicted": pred, "measured": float(r.E1_total),
                          "abs_err": abs(pred - float(r.E1_total))})
        bins = pd.cut(n, [15, 19, 23, 27, 31, 39, 47, 63, 95, 400])
        m = a.groupby(bins, observed=True).agg(["count", "mean"])
        for b, rr in m.iterrows():
            rowsm.append({"panel": p, "len_bin": str(b), "count": int(rr["count"]), "m_n": float(rr["mean"])})
        print(p, "done", flush=True)
    out = os.path.join(ROOT, "results_win", "proofs")
    os.makedirs(out, exist_ok=True)
    d2, d4, dm = pd.DataFrame(rows2), pd.DataFrame(rows4).sort_values(["panel", "L"]), pd.DataFrame(rowsm)
    d2.to_csv(os.path.join(out, "prop2_3_checks.csv"), index=False)
    d4.to_csv(os.path.join(out, "prop4_checks.csv"), index=False)
    dm.to_csv(os.path.join(out, "prop4_m_of_n.csv"), index=False)
    pd.set_option("display.width", 220)
    print(d2[["panel", "bug", "T1", "bound2a", "bound2b", "T2", "bound3a", "TV", "range", "ok2a", "ok2b", "ok3a", "sel_check"]].round(4).to_string(index=False))
    print("\nProp 2a/2b/3a hold in all rows:", bool(d2.ok2a.all()), bool(d2.ok2b.all()), bool(d2.ok3a.all()), "| identity:", bool(d2.sel_check.all()))
    print("\nProp 4(a) max |predicted - measured| =", d4.abs_err.max())
    print(d4.round(5).to_string(index=False))
    print("\nm(n) by history-length bin:")
    print(dm.pivot(index="len_bin", columns="panel", values="m_n").round(3).to_string())


if __name__ == "__main__":
    main()
