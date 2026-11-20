"""Run the frozen ledger + design bugs (B1-B5) + held-out bugs (x1-x9) on a REAL panel:
FTSFR bank_leverage (NYU Call Report, P12 confirm run panel.parquet: unique_id, ds, y).

Only change vs the frozen module: QDATES is set to the real panel's quarter-end calendar
(it defines 'incumbent' = present at the first quarter, and the sample end).
Usage: python realdata_v0.py <panel.parquet> [n_banks_subsample] [seed]
"""
import sys

import numpy as np
import pandas as pd

import heldout_v0 as X  # asserts the frozen hash
import ledger_frozen_v0 as F
import ledger_v1 as L1
import hashlib
assert hashlib.sha256(open('ledger_v1.py','rb').read()).hexdigest().startswith('ba7ac1a6')


def load(path, n_sub=None, seed=0):
    d = pd.read_parquet(path).rename(columns={"unique_id": "id", "ds": "date"})
    d["date"] = pd.to_datetime(d["date"])
    d["id"] = pd.factorize(d["id"])[0]
    d = d[["id", "date", "y"]].dropna().drop_duplicates(["id", "date"])
    if n_sub:
        ids = np.random.default_rng(seed).choice(d.id.unique(), n_sub, replace=False)
        d = d[d.id.isin(ids)]
    F.QDATES = pd.date_range(d.date.min(), d.date.max(), freq="QE")
    d["incumbent"] = d.id.map(d.groupby("id").date.min()) == F.QDATES[0]
    return d.sort_values(["id", "date"]).reset_index(drop=True)


def bug_grid_step_long(df):
    """B1 identical to frozen version except k covers 60 years (real span is 45y;
    frozen code capped at 40y, which would truncate incumbents). Deviation D1."""
    out = []
    for i, g in df.groupby("id", sort=False):
        first, last = g.date.min(), g.date.max()
        grid = pd.DatetimeIndex([first + pd.DateOffset(months=3 * j) for j in range(240)])
        out.append(pd.DataFrame({"id": i, "date": grid[grid <= last]}))
    m = pd.concat(out, ignore_index=True).merge(df[["id", "date", "y"]], on=["id", "date"], how="left")
    return F.quality_filter(m)


def main(path, n_sub=None, seed=0):
    tag = path.split('/')[-1].replace('_panel.parquet', '')
    raw = load(path, n_sub, seed)
    print(f"raw rows={len(raw)} banks={raw.id.nunique()} span={F.QDATES[0].date()}..{F.QDATES[-1].date()} "
          f"incumbent share={raw.groupby('id').incumbent.first().mean():.2f}")
    rng = np.random.default_rng(123)
    clean_p = F.regrid_clean(raw)
    clean = F.conclusion(F.evaluate(clean_p), rng)
    print("clean:", clean)
    pipes = {k: v for k, v in F.PIPELINES.items() if k != "C0_clean"}
    pipes["B1_grid_3mo_step"] = (bug_grid_step_long, False)
    pipes = {**{k: (v[0], v[1]) for k, v in pipes.items()}, **{k: (fn, False) for k, fn in X.HELDOUT.items()}}
    rows = []
    for name, (fn, leak) in pipes.items():
        p = fn(raw)
        ev = F.evaluate(p, scale_leak=leak)
        c = F.conclusion(ev, rng) if len(ev) else {"delta": np.nan, "n_banks": 0}
        ok, info = F.check_ledger(raw, p)
        ent = raw.groupby("id").incumbent.first()
        rows.append({"pipeline": name, "rows_kept": len(p) / len(raw),
                     "banks_kept": p.id.nunique() / raw.id.nunique(),
                     "entrant_share_eval": 1 - ent.reindex(ev.id).mean() if len(ev) else np.nan,
                     "delta": c["delta"], "shift": (c["delta"] - clean["delta"]) / abs(clean["delta"]),
                     "harmful": not (clean["lo"] <= c["delta"] <= clean["hi"]),
                     "schema_flag": not F.check_schema(raw, p), "stats_flag": not F.check_stats(raw, p),
                     "ledger_flag": not ok, "prov_ok": info["provenance_ok"],
                     "max_spread": max(info["spreads"].values()),
                     **{f"v1_{k}": v for k, v in zip(["flag", "info"], (lambda r: (not r[0], r[1]))(L1.check(raw, p, clean_p, F.QDATES)))}})
        print(name, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"realdata_results_{tag}_{n_sub or 'all'}.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None,
         int(sys.argv[3]) if len(sys.argv) > 3 else 0)
