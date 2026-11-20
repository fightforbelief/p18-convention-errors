"""Held-out bug set, written AFTER the ledger was frozen (see FREEZE_LEDGER.txt).
Each pattern comes from an externally documented pitfall, not from the ledger design.

Ground truth is NOT our judgment: a pipeline is 'harmful' if its headline estimate
leaves the clean pipeline's 95% bootstrap CI.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

import ledger_frozen_v0 as F

FROZEN = "39b156e65fe1c690eee8fe0ad92b912f1b9d1fb6d9bc6b1fd663c5d2264f1d0f"
assert hashlib.sha256(open("ledger_frozen_v0.py", "rb").read()).hexdigest() == FROZEN


def x1_step_90d(df):
    """Quarterly grid as a fixed 90-day step (common StackOverflow idiom)."""
    out = []
    for i, g in df.groupby("id", sort=False):
        grid = pd.date_range(g.date.min(), g.date.max(), freq="90D")
        out.append(pd.DataFrame({"id": i, "date": grid}))
    m = pd.concat(out).merge(df[["id", "date", "y"]], on=["id", "date"], how="left")
    return F.quality_filter(m)


def x2_quarterend_rollforward(df):
    """'Snap to quarter end' with date + QuarterEnd(): n=1 always rolls forward,
    so dates already on a quarter end move one quarter (pandas-documented)."""
    d = df[["id", "date", "y"]].copy()
    d["date"] = d.date + pd.offsets.QuarterEnd()
    return F.regrid_clean(d)


def x3_active_at_end(df):
    """Keep banks that still report in the last sample year (current-constituent pitfall)."""
    alive = df[df.date >= df.date.max() - pd.DateOffset(years=1)].id.unique()
    return F.regrid_clean(df[df.id.isin(alive)])


def x4_long_history(df):
    """Undisclosed 'stable history' filter: >= 40 quarters."""
    n = df.groupby("id").size()
    return F.regrid_clean(df[df.id.isin(n[n >= 40].index)])


def x5_next_target_dropna(df):
    """Build y_next = shift(-1) for a supervised frame, then dropna() (drops last obs)."""
    d = F.regrid_clean(df).sort_values(["id", "date"])
    d["y_next"] = d.groupby("id").y.shift(-1)
    return d.dropna()[["id", "date", "y"]].reset_index(drop=True)


def x6_many_to_many_link(df, seed=7):
    """Link table with overlapping validity ranges for 10% of banks -> duplicated rows
    (CCM-style link pitfall); deduplicated by keeping the first."""
    rng = np.random.default_rng(seed)
    ids = df.id.unique()
    dup = set(rng.choice(ids, int(0.1 * len(ids)), replace=False))
    link = pd.DataFrame({"id": ids, "link": 0})
    link = pd.concat([link, pd.DataFrame({"id": sorted(dup), "link": 1})])
    m = df.merge(link, on="id")
    # second link carries a stale copy shifted by one quarter (old filing vintage)
    stale = m.link == 1
    m.loc[stale, "y"] = m[stale].groupby("id").y.shift(1)
    m = m.dropna(subset=["y"]).sort_values(["id", "date", "link"], ascending=[True, True, False])
    m = m.drop_duplicates(["id", "date"], keep="first")
    return F.regrid_clean(m[["id", "date", "y"]])


def x7_winsorize(df):
    """Benign, common: pooled 1/99 winsorization of the target."""
    d = F.regrid_clean(df)
    lo, hi = d.y.quantile([0.01, 0.99])
    d["y"] = d.y.clip(lo, hi)
    return d


def x8_global_zscore(df):
    """Normalize the target on the full sample before splitting (leakage pattern)."""
    d = F.regrid_clean(df)
    d["y"] = (d.y - d.y.mean()) / d.y.std()
    return d


def x9_benign_reshuffle(df):
    """Benign: shuffle rows, round-trip ids through str."""
    d = F.regrid_clean(df).sample(frac=1.0, random_state=1)
    d["id"] = d.id.astype(str).astype(int)
    return d.reset_index(drop=True)


HELDOUT = {k: v for k, v in globals().items() if k.startswith("x") and callable(v)}


def run(seed):
    rng = np.random.default_rng(123)
    raw = F.simulate(seed=seed)
    clean = F.conclusion(F.evaluate(F.regrid_clean(raw)), rng)
    rows = []
    for name, fn in HELDOUT.items():
        p = fn(raw)
        c = F.conclusion(F.evaluate(p), rng)
        led_ok, info = F.check_ledger(raw, p)
        rows.append({
            "seed": seed, "pipeline": name,
            "shift": (c["delta"] - clean["delta"]) / abs(clean["delta"]),
            "harmful": not (clean["lo"] <= c["delta"] <= clean["hi"]),
            "schema_flag": not F.check_schema(raw, p), "stats_flag": not F.check_stats(raw, p),
            "ledger_flag": not led_ok, "provenance_ok": info["provenance_ok"],
            "max_spread": max(info["spreads"].values()),
        })
    return rows


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    allrows = [r for s in seeds for r in run(s)]
    pd.DataFrame(allrows).to_csv("heldout_results.csv", index=False)
    print(pd.DataFrame(allrows).groupby("pipeline").agg(
        shift=("shift", "mean"), harmful=("harmful", "mean"), schema=("schema_flag", "mean"),
        stats=("stats_flag", "mean"), ledger=("ledger_flag", "mean"),
        prov_ok=("provenance_ok", "mean"), spread=("max_spread", "mean")).round(3).to_string())
