"""Grade the frozen P1-P6 predictions against estimands_*_full.csv (see FREEZE_LEDGER.txt)."""
import glob

import pandas as pd

d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("estimands_*_full.csv"))])
SEL, ALT = ["B1_grid_3mo_step", "B2_survivor_join", "x3_active_at_end", "x4_long_history"], \
    ["B3_ffill_balanced", "B4_warmup_dropna", "x5_next_target_dropna", "x7_winsorize"]
g = lambda pipe, est: d[(d.pipeline == pipe) & (d.estimand == est)].set_index("panel")
res = {}

s = d[d.pipeline.isin(SEL)]
res["P1"] = bool((s[s.estimand.isin(["E1", "E2"])].T3.abs() < 1e-9).all()
                 and s[s.estimand == "E3"].forecast_exact.all())
b1 = g("B1_grid_3mo_step", "E1")
res["P2"] = bool(b1.loc["bank_leverage", "harm_ratio"] > 1 and b1.loc["bank_leverage", "rel_shift"] < 0.10
                 and (~b1.material.astype(bool)).sum() >= 3)
res["P3"] = bool(g("B2_survivor_join", "E1").material.all() and g("x3_active_at_end", "E1").material.all())
res["P4"] = bool(g("x4_long_history", "E2").material.sum() >= 3
                 and all((g(p, "E2").total > 0).sum() >= 3 for p in ["x4_long_history", "B2_survivor_join", "x3_active_at_end"]))
res["P5"] = bool((~g("B1_grid_3mo_step", "E3").top1_changed.astype(bool)).all()
                 and (g("B3_ffill_balanced", "E3").harm_ratio >= 1).sum() >= 3)
a = d[d.pipeline.isin(ALT) & d.estimand.isin(["E1", "E2"]) & (d.harm_ratio > 1)]
res["P6"] = bool(((a.total - a.forecast).abs() > 0.5 * a.total.abs()).mean() >= 0.75) if len(a) else None
print(res)

pd.set_option("display.width", 250)
cols = ["panel", "pipeline", "estimand", "n_p", "ledger_v1_flag", "total", "T1", "T2", "T3", "forecast",
        "harm_ratio", "rel_shift", "material", "cohort_TV"]
print(d[d.estimand != "E3"][cols].round(4).to_string())
print(d[d.estimand == "E3"][["panel", "pipeline", "rank_ref", "rank_p", "forecast_exact", "top1_changed", "sig_flips"]].to_string())
