"""Why is entry calendar quarter outcome-relevant for power plants (FREEZE_LEDGER M1-M5)?
Two candidate mechanisms: (T) technology mix differs by entry quarter; (H) entry quarter proxies history length / entry year.
d_i = MASE(naive) - MASE(HA) per plant on the reference pipeline (regrid_clean), as in E1.
Plant technology = EIA-860 2024 'Technology' of the plant's largest generator (Operable + Retired sheets).
Writes results_win_eia/mechanism/*.csv and prints the tests."""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(ROOT, "pilot"))
sys.path.insert(0, ".")
import estimands as E  # noqa: E402
import ledger_frozen_v0 as F  # noqa: E402
import realdata_v0 as RD  # noqa: E402

OUT = os.path.join(ROOT, "results_win_eia", "mechanism")
os.makedirs(OUT, exist_ok=True)

GROUPS = [("solar", ["Solar Photovoltaic", "Solar Thermal"]), ("wind", ["Onshore Wind", "Offshore Wind"]),
          ("gas_cc", ["Natural Gas Fired Combined Cycle"]),
          ("gas_ct_ic", ["Natural Gas Fired Combustion Turbine", "Natural Gas Internal Combustion Engine",
                         "Landfill Gas", "Other Natural Gas"]),
          ("gas_steam", ["Natural Gas Steam Turbine"]), ("coal", ["Conventional Steam Coal", "Coal Integrated Gasification Combined Cycle"]),
          ("hydro", ["Conventional Hydroelectric", "Hydroelectric Pumped Storage"]), ("nuclear", ["Nuclear"]),
          ("petroleum", ["Petroleum Liquids", "Petroleum Coke"]), ("storage", ["Batteries", "Flywheels"])]


def tech_map():
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "eia860", "eia8602024.zip"))
    xl = pd.ExcelFile(io.BytesIO(z.read("3_1_Generator_Y2024.xlsx")))
    g = pd.concat([xl.parse(s, header=1) for s in ("Operable", "Retired and Canceled")], ignore_index=True)
    g = g[pd.to_numeric(g["Plant Code"], errors="coerce").notna()]
    g["plant"] = g["Plant Code"].astype(int)
    g["cap"] = pd.to_numeric(g["Nameplate Capacity (MW)"], errors="coerce").fillna(0)
    top = g.sort_values("cap").groupby("plant").tail(1).set_index("plant")["Technology"].astype(str)
    lookup = {t: k for k, ts in GROUPS for t in ts}
    return top.map(lambda t: lookup.get(t, "other"))


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, float(r @ r)


def entry_q_test(df, controls):
    """Entrants only. d ~ entry-quarter dummies (Q1 base) + controls. Returns spread of Q coefs and joint F p-value."""
    q = pd.get_dummies(df.entry_q, prefix="q", dtype=float).drop(columns="q_1")
    C = [pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float) for c in controls]
    Xr = pd.concat([pd.Series(1.0, index=df.index, name="const"), *C], axis=1).to_numpy()
    Xf = np.column_stack([Xr, q.to_numpy()])
    y = df.d.to_numpy()
    bf, rssf = ols(y, Xf)
    _, rssr = ols(y, Xr)
    k, n, p = q.shape[1], len(y), Xf.shape[1]
    Fst = ((rssr - rssf) / k) / (rssf / (n - p))
    coefs = np.r_[0.0, bf[-k:]]
    return {"controls": "+".join(controls) or "none", "n": n, "spread": float(coefs.max() - coefs.min()),
            "q2": float(bf[-3]), "q3": float(bf[-2]), "q4": float(bf[-1]), "F": float(Fst),
            "p": float(stats.f.sf(Fst, k, n - p))}


def main():
    raw = RD.load(os.path.join(ROOT, "data", "raw", "eia_plants_panel.parquet"))
    ref = F.regrid_clean(raw)
    R = E.per_bank(ref)
    d = (R.naive - R.ha).rename("d")
    first = raw.groupby("id").date.min()
    nq = ref.groupby("id").size()
    ids = pd.read_parquet(os.path.join(ROOT, "data", "raw", "eia_plants_panel.parquet")).unique_id.drop_duplicates()
    plant_of = dict(zip(pd.factorize(ids)[0], ids.to_numpy()))  # RD.load factorizes unique_id in order of appearance
    tech = tech_map()
    df = pd.DataFrame({"d": d})
    df["plant"] = df.index.map(plant_of)
    df["tech"] = df.plant.map(tech).fillna("unmatched")
    df["incumbent"] = df.index.map(first) == F.QDATES[0]
    df["entry_q"] = df.index.map(first).quarter
    df["entry_year"] = df.index.map(first).year
    df["len_bin"] = pd.cut(df.index.map(nq), [0, 20, 24, 32, 40, 60, 200], labels=False)
    df.to_csv(os.path.join(OUT, "plant_level.csv"))
    print("tech matched share", round((df.tech != "unmatched").mean(), 3))
    ent = df[~df.incumbent]
    # M1: technology composition by entry quarter
    ct = pd.crosstab(ent.tech, ent.entry_q)
    chi2, p_chi, *_ = stats.chi2_contingency(ct)
    ct.to_csv(os.path.join(OUT, "tech_by_entry_quarter.csv"))
    print("\nM1 tech x entry quarter (entrants), chi2 =", round(chi2, 1), "p =", p_chi)
    print((ct / ct.sum()).round(3).to_string())
    # M2-M4: entry-quarter effect with controls
    rows = [entry_q_test(ent, c) for c in ([], ["tech"], ["entry_year"], ["len_bin"], ["tech", "entry_year"],
                                           ["tech", "len_bin"])]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "entry_quarter_tests.csv"), index=False)
    print("\nentry-quarter effect on d (entrants; Q1 = 0):")
    print(t.round(4).to_string(index=False))
    print("\nmean d by tech x entry quarter (entrants):")
    print(ent.pivot_table(index="tech", columns="entry_q", values="d", aggfunc="mean").round(3).to_string())
    # M5: B1 decomposition with technology cells
    p_b1 = RD.bug_grid_step_long(raw)
    P = E.per_bank(p_b1)
    cell = df.tech.reindex(R.index)
    cell_all = pd.Series(df.tech, index=df.index).reindex(sorted(set(R.index) | set(P.index))).fillna("unmatched")
    one = pd.Series(1.0, index=R.index)
    dec = E.decompose(d, one, P.naive - P.ha, pd.Series(1.0, index=P.index), cell_all)
    dec_entry = E.decompose(d, one, P.naive - P.ha, pd.Series(1.0, index=P.index),
                            pd.Series(np.where(df.incumbent, "inc", "ent_Q" + df.entry_q.astype(str)), index=df.index))
    m5 = pd.DataFrame([{"cells": "technology", **dec}, {"cells": "entry_cohort", **dec_entry}])
    m5.to_csv(os.path.join(OUT, "b1_decomposition_by_cells.csv"), index=False)
    print("\nM5 B1 decomposition, technology cells vs entry-cohort cells:")
    print(m5[["cells", "total", "T1", "T2", "T3"]].round(4).to_string(index=False))
    print("tech share of selection (T1/(T1+T2)) =", round(dec["T1"] / (dec["T1"] + dec["T2"]), 3))


if __name__ == "__main__":
    main()
