"""Generate paper/generated/{values.tex, tab_*.tex} from the hanqing-win results pulled into results_win*/.
Every number in paper/main.tex that comes from data is set here (\\setval) or in a generated table.
Usage (report collation, no experiments): python scripts/make_paper_values.py"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RW, RE = os.path.join(ROOT, "results_win"), os.path.join(ROOT, "results_win_eia")
GEN = os.path.join(ROOT, "paper", "generated")
os.makedirs(GEN, exist_ok=True)
BANKS = ["bank_leverage", "bank_cash_liquidity", "bhc_leverage", "bhc_cash_liquidity"]
ALL = BANKS + ["eia_plants"]
NICE = {"bank_leverage": "Bank leverage", "bank_cash_liquidity": "Bank cash", "bhc_leverage": "BHC leverage",
        "bhc_cash_liquidity": "BHC cash", "eia_plants": "Power plants"}
V = {}


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def rng(xs, d=1):
    xs = [float(x) for x in xs]
    return f"{100 * min(xs):.{d}f}--{100 * max(xs):.{d}f}\\%"


def est(p):
    base = RE if p == "eia_plants" else RW
    return pd.read_csv(os.path.join(base, "estimands", f"estimands_{p}_full.csv"))


def llm(prefix, p):
    base = RE if p == "eia_plants" else RW
    return pd.read_csv(os.path.join(base, "llm", f"{prefix}_results_{p}.csv"))


# ---------------------------------------------------------------- panel sizes
for p in ALL:
    e = est(p)
    V[f"n-{p}"] = f"{int(e.n_ref.iloc[0]):,}"
man_b = json.load(open(os.path.join(RW, "run_manifest_pilot_estimands_llm.json")))
man_e = json.load(open(os.path.join(RE, "run_manifest_realdata_estimands_llm.json")))
V["py-version"] = man_b["python"]
V["pandas-version"] = man_b["versions"]["pandas"]
V["runtime-bank-min"] = f"{man_b['total_seconds'] / 60:.0f}"
V["runtime-eia-min"] = f"{man_e['total_seconds'] / 60:.0f}"

# ---------------------------------------------------------------- design bugs table (E1, full panels)
SHOW = [("B1_grid_3mo_step", "B1 grid step (motivating)", "entry quarter"),
        ("B2_survivor_join", "B2 latest-snapshot join", "exit status"),
        ("x3_active_at_end", "x3 active at sample end", "exit status"),
        ("x4_long_history", "x4 long-history filter", "history length"),
        ("B4_warmup_dropna", "B4 warm-up \\texttt{dropna}", "age (alteration)"),
        ("B3_ffill_balanced", "B3 balanced + ffill", "alteration"),
        ("B5_mase_scale_leak", "B5 scale leak", "alteration"),
        ("x7_winsorize", "x7 winsorize", "alteration")]
lines = ["\\begin{tabular}{llrrrrr}", "\\toprule",
         "Pipeline & Selects on & " + " & ".join(NICE[p] for p in ALL) + " \\\\", "\\midrule"]
lines[0] = "\\begin{tabular}{ll" + "r" * len(ALL) + "}"
for key, lab, sel in SHOW:
    cells = []
    for p in ALL:
        r = est(p)
        r = r[(r.pipeline == key) & (r.estimand == "E1")].iloc[0]
        s = f"{100 * r.rel_shift:.1f}"
        if bool(r.material):
            s = "\\textbf{" + s + "}"
        if bool(r.ledger_v1_flag):
            s += "$^{\\ell}$"
        cells.append(s)
    lines.append(f"{lab} & {sel} & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(GEN, "tab_design.tex"), "w").write("\n".join(lines) + "\n")

b1 = {p: est(p).query("pipeline=='B1_grid_3mo_step' and estimand=='E1'").iloc[0] for p in ALL}
V["b1-bank-range"] = rng([b1[p].rel_shift for p in BANKS])
V["b1-bank-tv-range"] = f"{min(b1[p].cohort_TV for p in BANKS):.2f}--{max(b1[p].cohort_TV for p in BANKS):.2f}"
V["b1-eia"] = pct(b1["eia_plants"].rel_shift)
V["b1-eia-harm"] = f"{b1['eia_plants'].harm_ratio:.2f}"
V["b1-eia-tv"] = f"{b1['eia_plants'].cohort_TV:.2f}"
surv = [est(p).query("pipeline==@k and estimand=='E1'").iloc[0].rel_shift for p in BANKS for k in ("B2_survivor_join", "x3_active_at_end")]
V["surv-bank-range"] = rng(surv)
V["surv-eia-range"] = rng([est("eia_plants").query("pipeline==@k and estimand=='E1'").iloc[0].rel_shift
                           for k in ("B2_survivor_join", "x3_active_at_end")])

dev = pd.read_csv(os.path.join(RW, "pilot", "decomp_dev_results.csv"))
_sub = dev[(dev.panel == "bank_leverage") & (dev.pipeline == "B1_grid_3mo_step")].harm_ratio.iloc[0]
V["b1-harm-sub"] = f"{_sub:.2f}"
V["b1-harm-full"] = f"{b1['bank_leverage'].harm_ratio:.2f}"
V["b1-rel-full-bl"] = pct(b1["bank_leverage"].rel_shift)
V["b1-bank-tv-max"] = pct(max(b1[p].cohort_TV for p in BANKS), 0)
# selection bugs: T3 exactly zero, ranking forecast exact
selbugs = ["B1_grid_3mo_step", "B2_survivor_join", "x3_active_at_end", "x4_long_history"]
t3max, rk_ok, rk_n = 0.0, 0, 0
for p in ALL:
    e = est(p)
    s = e[e.pipeline.isin(selbugs)]
    t3max = max(t3max, float(s[s.estimand.isin(["E1", "E2"])].T3.abs().max()))
    r3 = s[s.estimand == "E3"]
    rk_ok += int(r3.forecast_exact.astype(bool).sum())
    rk_n += len(r3)
V["t3-max-sel"] = "0" if t3max == 0 else f"{t3max:.1e}"
V["rank-forecast-exact"] = f"{rk_ok}/{rk_n}"
# rankings: selection vs alteration flips
flip_sel = flip_alt = 0
for p in ALL:
    e = est(p)
    r3 = e[e.estimand == "E3"]
    flip_sel += int(((r3.pipeline.isin(selbugs)) & (r3.harm_ratio > 0)).sum())
    flip_alt += int(((r3.pipeline.isin(["B3_ffill_balanced", "B5_mase_scale_leak", "x5_next_target_dropna", "x7_winsorize"])) & (r3.harm_ratio > 0)).sum())
V["rank-flips-sel"] = str(flip_sel)
V["rank-flips-alt"] = str(flip_alt)

# ledger: pre-registered test (bhc_leverage, 4000 subsample) and full panels
FLAG = ["B1_grid_3mo_step", "B2_survivor_join", "B3_ffill_balanced", "B4_warmup_dropna", "x1_step_90d",
        "x2_quarterend_rollforward", "x3_active_at_end", "x4_long_history"]
NOFLAG = ["B5_mase_scale_leak", "x5_next_target_dropna", "x6_many_to_many_link", "x7_winsorize", "x8_global_zscore",
          "x9_benign_reshuffle"]
rd = pd.read_csv(os.path.join(RW, "pilot", "realdata_results_bhc_leverage_4000.csv")).set_index("pipeline")
V["ledger-prereg"] = f"{int(rd.loc[FLAG, 'v1_flag'].sum())}/8 and {int(rd.loc[NOFLAG, 'v1_flag'].sum())}/6"
V["ledger-prereg-hit"] = f"{int(rd.loc[FLAG, 'v1_flag'].sum())} of 8"
V["ledger-prereg-fa"] = f"{int(rd.loc[NOFLAG, 'v1_flag'].sum())} of 6"
V["schema-fires"] = str(int(rd.schema_flag.sum()))
hits = fas = 0
for p in ALL:
    e = est(p).query("estimand=='E1'").set_index("pipeline")
    hits += int(e.loc[[k for k in FLAG if k in e.index], "ledger_v1_flag"].astype(bool).sum())
    fas += int(e.loc[NOFLAG, "ledger_v1_flag"].astype(bool).sum())
V["ledger-full-hits"] = f"{hits}/{8 * len(ALL)}"
V["ledger-full-fa"] = f"{fas}/{6 * len(ALL)}"

# ---------------------------------------------------------------- LLM study
ARMS = [("1", "explicit", "A"), ("1", "under-specified", "B"), ("2", "explicit", "A2"), ("2", "under-specified", "B2"),
        ("3", "explicit", "A3"), ("3", "under-specified", "B3")]
lines = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
         "Task & Spec & Pipelines & Exact & Diverge & Ledger flags & Max shift & Material \\\\", "\\midrule"]
tot = {"exact_pipes": 0, "explicit_pipes": 0, "div_pipes": 0, "under_pipes": 0}
for task, spec, pre in ARMS:
    d = pd.concat([llm(pre, p) for p in ALL])
    ok = d[d.status == "ok"]
    pipes = d.pipeline.nunique()
    exact_p = int(ok.groupby("pipeline").apply(lambda g: ((g.key_dev == 0) & (g.value_mismatch == 0)).all()).sum())
    div_p = int(ok.groupby("pipeline").apply(lambda g: ((g.key_dev >= 0.001) | (g.value_mismatch > 0)).any()).sum())
    flags = f"{int(ok.ledger_flag.astype(bool).sum())}/{len(ok)}"
    mx = ok.E1_rel.max()
    mat = int((ok.E1_rel >= 0.10).sum())
    lines.append(f"{task} & {spec} & {pipes} & {exact_p} & {div_p} & {flags} & {100 * mx:.1f}\\% & {mat}/{len(ok)} \\\\")
    if spec == "explicit":
        tot["exact_pipes"] += exact_p; tot["explicit_pipes"] += pipes
    else:
        tot["div_pipes"] += div_p; tot["under_pipes"] += pipes
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(GEN, "tab_llm.tex"), "w").write("\n".join(lines) + "\n")
V["llm-explicit-exact"] = f"{tot['exact_pipes']}/{tot['explicit_pipes']}"
V["llm-under-diverge"] = f"{tot['div_pipes']}/{tot['under_pipes']}"
V["llm-total-pipes"] = str(tot["explicit_pipes"] + tot["under_pipes"])
b3 = pd.concat([llm("B3", p) for p in BANKS])
V["b3-crash-panels"] = str(int((b3.status != "ok").sum()))

# ---------------------------------------------------------------- min-length dose response
rows = {}
for p in ALL:
    d = pd.concat([llm("ABL", p), llm("LSWEEP", p)])
    d = d[d.pipeline.str.startswith("G0_S0_")].copy()
    d["L"] = d.pipeline.str.extract(r"L(\d+)")[0].astype(int)
    rows[p] = d.set_index("L").E1_rel
Ls = [16, 20, 24, 32, 48]
lines = ["\\begin{tabular}{r" + "r" * len(ALL) + "}", "\\toprule",
         "Minimum length $L$ & " + " & ".join(NICE[p] for p in ALL) + " \\\\", "\\midrule"]
for L in Ls:
    lines.append(f"{L} & " + " & ".join(f"{100 * rows[p].loc[L]:.1f}\\%" for p in ALL) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(GEN, "tab_minlen.tex"), "w").write("\n".join(lines) + "\n")
V["minlen-l24-bank-range"] = rng([rows[p].loc[24] for p in BANKS])
V["minlen-l24-eia"] = pct(rows["eia_plants"].loc[24])
V["minlen-l48-eia"] = pct(rows["eia_plants"].loc[48])
# ablation: segment and interpolation alone
seg = []
for p in BANKS:
    a = llm("ABL", p).set_index("pipeline")
    seg += [a.loc["G1_S1_L0", "E1_rel"], a.loc["G1_S0_L0", "E1_rel"]]
V["seg-interp-max"] = pct(max(seg))
ratios = []
for p in BANKS:
    a = llm("ABL", p).set_index("pipeline").loc["G1_S1_L24", "E1_total"]
    s4 = llm("B", p).set_index("pipeline").loc["sonnet_4", "E1_total"]
    ratios.append(a / s4)
V["abl-replication"] = f"{min(ratios):.2f}--{max(ratios):.2f}"

# ---------------------------------------------------------------- propositions
pc = pd.read_csv(os.path.join(RW, "proofs", "prop2_3_checks.csv"))
V["prop-cases"] = str(len(pc))
V["prop-all-hold"] = "all" if bool(pc[["ok2a", "ok2b", "ok3a", "sel_check"]].all().all()) else "NOT all"
p4 = pd.read_csv(os.path.join(RW, "proofs", "prop4_checks_taskref.csv"))
V["prop4-cases"] = str(len(p4))
V["prop4-maxerr"] = f"{p4.abs_err.max():.0e}"
e1 = pc.query("panel=='eia_plants' and bug=='B1_grid_3mo_step'").iloc[0]
V["b1-eia-T1"] = f"{abs(e1.T1):.3f}"
V["b1-eia-bound2b"] = f"{e1.bound2b:.3f}"
V["b1-bank-bound2b-max"] = f"{pc[(pc.bug=='B1_grid_3mo_step') & pc.panel.isin(BANKS)].bound2b.max():.3f}"

# ---------------------------------------------------------------- mechanism (EIA-860)
mt = pd.read_csv(os.path.join(RE, "mechanism", "entry_quarter_tests.csv")).set_index("controls")
V["mech-spread-none"] = f"{mt.loc['none', 'spread']:.3f}"
V["mech-spread-tech"] = f"{mt.loc['tech', 'spread']:.3f}"
V["mech-spread-year"] = f"{mt.loc['entry_year', 'spread']:.3f}"
V["mech-reduction"] = pct(1 - mt.loc["tech", "spread"] / mt.loc["none", "spread"], 0)
V["mech-p-techyear"] = f"{mt.loc['tech+entry_year', 'p']:.3f}"
ct = pd.read_csv(os.path.join(RE, "mechanism", "tech_by_entry_quarter.csv"), index_col=0)
sh = ct / ct.sum()
V["solar-q1"] = pct(sh.loc["solar", "1"], 0)
V["solar-q4"] = pct(sh.loc["solar", "4"], 0)
bd = pd.read_csv(os.path.join(RE, "mechanism", "b1_decomposition_by_cells.csv")).set_index("cells")
V["mech-b1-tech-share"] = pct(bd.loc["technology", "T1"] / (bd.loc["technology", "T1"] + bd.loc["technology", "T2"]), 0)
pl = pd.read_csv(os.path.join(RE, "mechanism", "plant_level.csv"))
V["tech-matched"] = pct((pl.tech != "unmatched").mean(), 0)
ent = pl[~pl.incumbent.astype(bool)]
V["d-inc-eia"] = f"{pl[pl.incumbent.astype(bool)].d.mean():.2f}"
V["d-q12-eia"] = f"{ent[ent.entry_q.isin([1, 2])].d.mean():.2f}"
V["d-q34-eia"] = f"{ent[ent.entry_q.isin([3, 4])].d.mean():.2f}"

# ---------------------------------------------------------------- figure data (pgfplots reads these .dat files)
with open(os.path.join(GEN, "fig_minlen.dat"), "w") as f:
    Lall = sorted(set().union(*[set(rows[p].index) for p in ALL]))
    f.write("L " + " ".join(ALL) + "\n")
    for L in Lall:
        if L >= 16:
            f.write(f"{L} " + " ".join(f"{100 * rows[p].loc[L]:.4f}" for p in ALL) + "\n")
with open(os.path.join(GEN, "fig_tv.dat"), "w") as f:
    f.write("panel bug tv shift isb1 iseia\n")
    for p in ALL:
        e = est(p).query("estimand=='E1'").set_index("pipeline")
        for k in selbugs:
            f.write(f"{p} {k} {e.loc[k, 'cohort_TV']:.4f} {100 * e.loc[k, 'rel_shift']:.4f} "
                    f"{int(k == 'B1_grid_3mo_step')} {int(p == 'eia_plants')}\n")

# ---------------------------------------------------------------- E2 degeneracy on leverage panels (needs data/raw; run on hanqing-win)
RAWD = os.path.join(ROOT, "data", "raw")
if os.path.exists(os.path.join(RAWD, "bhc_leverage_panel.parquet")):
    import sys
    _cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, "pilot"))
    sys.path.insert(0, os.path.join(ROOT, "pilot"))
    import estimands as E  # noqa: E402
    import ledger_frozen_v0 as F  # noqa: E402
    import realdata_v0 as RD  # noqa: E402
    os.chdir(_cwd)
    shares = {}
    for p in ("bank_leverage", "bhc_leverage"):
        R = E.per_bank(F.regrid_clean(RD.load(os.path.join(RAWD, f"{p}_panel.parquet"))))
        shares[p] = float(R.sxx.max() / R.sxx.sum())
    V["e2-top1-share-bhc"] = pct(shares["bhc_leverage"], 0)
    V["e2-top1-share-bank"] = pct(shares["bank_leverage"], 0)
else:
    print("data/raw not found: E2 weight shares not computed (run on hanqing-win)")
e2 = est("bhc_leverage").query("pipeline=='B1_grid_3mo_step' and estimand=='E2'").iloc[0]
V["e2-b1-tv-bhc"] = f"{e2.cohort_TV:.2f}"
V["e1-b1-tv-bhc"] = f"{b1['bhc_leverage'].cohort_TV:.2f}"

# ---------------------------------------------------------------- pipeline catalogue (Appendix B)
CAT = [("B1", "Grid built as first date + $k\\times$3 months (the benchmark's \\texttt{3mo} step)", "entry quarter", "design"),
       ("B2", "Attributes inner-joined from the latest snapshot only", "exit status", "design"),
       ("B3", "Panel balanced on the full calendar, then forward-filled", "alteration", "design"),
       ("B4", "Lag and rolling features built, then \\texttt{dropna()} on the frame", "age / alteration", "design"),
       ("B5", "MASE scale computed on the full series incl.\\ test (leak)", "alteration", "design"),
       ("x1", "Grid built with a 90-day step", "calendar (empties panel)", "held-out"),
       ("x2", "Dates rolled forward with \\texttt{QuarterEnd()} (off by one quarter)", "calendar keys", "held-out"),
       ("x3", "Keep entities active in the last year (current constituents)", "exit status", "held-out"),
       ("x4", "Keep entities with at least 40 quarters of history", "history length", "held-out"),
       ("x5", "Next-period target via \\texttt{shift(-1)}, then \\texttt{dropna}", "alteration", "held-out"),
       ("x6", "Many-to-many link table with a stale copy", "alteration", "held-out"),
       ("x7", "Winsorize at the 1st/99th percentile", "alteration", "held-out"),
       ("x8", "Global z-score", "none (benign)", "held-out"),
       ("x9", "Row reshuffle", "none (benign)", "held-out")]
lines = ["\\begin{tabular}{llp{0.55\\linewidth}l}", "\\toprule", "ID & Set & Operation & Selects on / class \\\\", "\\midrule"]
for i, d, s, g in CAT:
    lines.append(f"{i} & {g} & {d} & {s} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(GEN, "tab_catalogue.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- full pre-registration table (Appendix C), from FREEZE_LEDGER.txt outcomes
PRED = [
    ("Ledger v1", "v1", "Held-out bhc\\_leverage: $\\ge$7/8 flags, $\\le$1/6 false", "pass"),
    ("Decomposition", "D1", "B1 harmful on both cash panels", "fail"),
    ("Decomposition", "D2", "$T_3\\approx0$ for selection bugs, dominant for alterations", "pass"),
    ("Decomposition", "D3", "B1 harm on cash panels exceeds bank\\_leverage", "fail"),
    ("Full panels", "P1", "Selection bugs: $T_3=0$, ranking forecast exact", "pass"),
    ("Full panels", "P2", "B1 CI-harmful on full bank\\_leverage but not material", "fail"),
    ("Full panels", "P3", "B2, x3 E1-material on all panels", "pass"),
    ("Full panels", "P4", "x4 E2-material on $\\ge$3 panels", "fail"),
    ("Full panels", "P5", "B1 never changes top-1; B3 flips a significant pair", "pass"),
    ("Full panels", "P6", "No-refit forecast misses alteration bugs", "pass"),
    ("LLM task 1 explicit", "Q1", "$\\ge$4/15 pipelines silently deviate", "fail"),
    ("LLM task 1 explicit", "Q2", "Ledger: $\\ge$60\\% sensitivity, no false flags", "pass"),
    ("LLM task 1 explicit", "Q3", "Stats check cannot discriminate", "vacuous"),
    ("LLM task 1 explicit", "Q4", "Deviations trace to charter rules", "vacuous"),
    ("LLM task 1 explicit", "Q5", "Material deviations select on exit/history", "vacuous"),
    ("LLM task 1 explicit", "Q6", "Haiku deviates more than Opus", "fail"),
    ("LLM task 1 explicit", "Q7", "$T_3=0$ for composition-only deviations", "vacuous"),
    ("LLM task 1 under-spec.", "QB1", "$\\ge$8/15 diverge", "pass"),
    ("LLM task 1 under-spec.", "QB2", "$\\ge$3/15 impute", "pass"),
    ("LLM task 1 under-spec.", "QB3", "$\\ge$3/15 single-snapshot charter", "fail"),
    ("LLM task 1 under-spec.", "QB4", "Ledger flags $\\ge$60\\% of divergences", "pass"),
    ("LLM task 1 under-spec.", "QB5", "Some pipeline material on $\\ge$2 panels", "fail"),
    ("LLM task 1 under-spec.", "QB6", "$T_3=0$ when no value mismatch", "fail"),
    ("LLM task 2", "T2A", "$\\ge$7/8 explicit pipelines exact", "pass"),
    ("LLM task 2", "R1", "$\\ge$15/20 under-specified diverge", "pass"),
    ("LLM task 2", "R2", "$\\ge$4/20 lag by rows", "fail"),
    ("LLM task 2", "R3", "$\\ge$3/20 use percent change", "fail"),
    ("LLM task 2", "R4", "Some pipeline material on $\\ge$2 panels", "fail"),
    ("LLM task 2", "R5", "Same-sign shift in $\\ge$75\\% of cells", "fail"),
    ("LLM task 2", "R6", "Ledger blind to pure value changes", "vacuous"),
    ("LLM task 2", "C1", "Codex task 1: diverges and interpolates", "fail"),
    ("LLM task 2", "C2", "Codex and Claude shift signs agree", "fail"),
    ("LLM task 3", "S1", "$\\ge$5/20 fall for the survivorship trap", "fail"),
    ("LLM task 3", "S2", "Survivorship pipelines are material", "vacuous"),
    ("LLM task 3", "S3", "Ledger flags survivorship pipelines", "vacuous"),
    ("LLM task 3", "S4", "$\\ge$7/8 explicit pipelines exact", "pass"),
    ("LLM task 3", "S5", "Survivorship rate differs by family", "fail"),
    ("LLM task 3", "S6", "Selection $\\ge$70\\% of survivorship shift", "vacuous"),
    ("Ablation", "AB0", "Baseline config equals reference", "pass"),
    ("Ablation", "AB1", "Interpolation alone $\\le$1\\%", "pass"),
    ("Ablation", "AB2", "Minimum length alone $\\le$3\\%", "fail"),
    ("Ablation", "AB3", "Segment rule drives $\\ge$60\\% of the shift", "fail"),
    ("Ablation", "AB4", "Full config reproduces the LLM shift ($\\pm$30\\%)", "pass"),
    ("Ablation", "AB5", "Shift is $\\ge$70\\% selection", "pass"),
    ("Ablation", "AB6", "Ledger flags segment rule, not L16", "fail"),
    ("Power plants", "N1", "Ledger v1: $\\ge$7/8 flags, $\\le$1/6 false", "pass"),
    ("Power plants", "N2", "B1 not material; B2, x3 material", "fail"),
    ("Power plants", "N3", "Selection bugs: $T_3=0$, ranking exact", "pass"),
    ("Power plants", "N4", "Monotone minimum-length dose-response", "pass"),
    ("Power plants", "N5", "LLM explicit exact, under-specified diverge", "pass"),
    ("Power plants", "N6", "Material LLM cells all have min length $>16$", "pass"),
    ("Mechanism", "M1", "Technology mix differs by entry quarter", "pass"),
    ("Mechanism", "M2", "Technology FE cut the gap by $\\ge$50\\%", "fail"),
    ("Mechanism", "M3", "Technology explains more than history", "pass"),
    ("Mechanism", "M4", "No residual entry-quarter effect ($p>0.05$)", "fail"),
    ("Mechanism", "M5", "Between-technology share of B1 selection $\\ge$50\\%", "pass"),
]
assert len({p[1] for p in PRED}) == len(PRED)
lines = ["\\begin{longtable}{llp{0.58\\linewidth}l}", "\\toprule", "Experiment & ID & Prediction & Outcome \\\\",
         "\\midrule", "\\endhead"]
for ex, i, txt, out in PRED:
    o = {"pass": "pass", "fail": "\\textbf{fail}", "vacuous": "vacuous"}[out]
    lines.append(f"{ex} & {i} & {txt} & {o} \\\\")
lines += ["\\bottomrule", "\\end{longtable}"]
open(os.path.join(GEN, "tab_prereg.tex"), "w").write("\n".join(lines) + "\n")
for k in ("pass", "fail", "vacuous"):
    V[f"prereg-{k}"] = str(sum(1 for p in PRED if p[3] == k))
V["prereg-total"] = str(len(PRED))

with open(os.path.join(GEN, "values.tex"), "w") as f:
    for k, v in V.items():
        f.write(f"\\setval{{{k}}}{{{v}}}\n")
print(f"{len(V)} values written;", sorted(os.listdir(GEN)))
for k, v in V.items():
    print(f"  {k} = {v}")
