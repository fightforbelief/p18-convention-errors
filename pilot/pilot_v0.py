"""P18 pilot v0: do silent calendar/composition bugs in a panel pipeline pass
standard checks, change the conclusion, and get caught by an entity-period ledger?

Synthetic quarterly bank panel with known structure:
  incumbents (present at the first quarter): mean-reverting  -> HistoricAverage should win
  entrants: random walk with drift                               -> Naive should win
Task conclusion: mean per-bank MASE(Naive) - MASE(HistoricAverage), h = 4.

Each bug is one realistic pandas idiom. Nothing raises an error.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

H = 4
MIN_LEN = 16
QDATES = pd.date_range("1990-03-31", "2020-12-31", freq="QE")


# ---------------------------------------------------------------- data
def simulate(n_banks: int = 4000, frac_incumbent: float = 0.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = len(QDATES)
    rows = []
    for i in range(n_banks):
        inc = rng.random() < frac_incumbent
        start = 0 if inc else int(rng.integers(4, T - MIN_LEN - 4))
        end = T - 1
        if rng.random() < 0.3:  # exit
            end = int(rng.integers(start + MIN_LEN + 2, T))
        n = end - start + 1
        mu = rng.normal(0.12, 0.03)
        # incumbents mean-revert (HA favoured); entrants are near-unit-root (Naive favoured)
        phi = rng.uniform(0.3, 0.7) if inc else rng.uniform(0.93, 1.0)
        sd = 0.01
        y = np.empty(n)
        y[0] = mu + rng.normal(0, 0.03 if not inc else 0.01)
        for t in range(1, n):
            y[t] = mu + phi * (y[t - 1] - mu) + rng.normal(0, sd)
        rows.append(pd.DataFrame({"id": i, "date": QDATES[start : end + 1], "y": y,
                                  "incumbent": inc}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- pipelines
def regrid_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for i, g in df.groupby("id", sort=False):
        grid = pd.date_range(g.date.min(), g.date.max(), freq="QE")
        out.append(pd.DataFrame({"id": i, "date": grid}))
    grid = pd.concat(out, ignore_index=True)
    m = grid.merge(df[["id", "date", "y"]], on=["id", "date"], how="left")
    return quality_filter(m)


def quality_filter(m: pd.DataFrame, max_missing: float = 0.2) -> pd.DataFrame:
    miss = m.y.isna().groupby(m.id).mean()
    keep = miss[miss <= max_missing].index
    return m[m.id.isin(keep)].dropna(subset=["y"]).reset_index(drop=True)


def bug_grid_step(df):
    """B1: grid built as first + k*DateOffset(months=3) (the P12/FTSFR '3mo' step).
    From 30 Jun / 30 Sep this lands on 30 Dec, never matching 31 Dec."""
    out = []
    for i, g in df.groupby("id", sort=False):
        first, last = g.date.min(), g.date.max()
        k = np.arange(0, 4 * 40)
        grid = pd.DatetimeIndex([first + pd.DateOffset(months=3 * int(j)) for j in k])
        grid = grid[grid <= last]
        out.append(pd.DataFrame({"id": i, "date": grid}))
    grid = pd.concat(out, ignore_index=True)
    m = grid.merge(df[["id", "date", "y"]], on=["id", "date"], how="left")
    return quality_filter(m)


def bug_survivor_join(df):
    """B2: attach bank attributes from the latest call-report snapshot (inner join)."""
    snap = df[df.date == df.date.max()][["id"]].assign(size_class="x")
    return regrid_clean(df.merge(snap, on="id", how="inner").drop(columns="size_class"))


def bug_ffill_balanced(df):
    """B3: balance the panel on the full calendar, then ffill missing values."""
    full = pd.MultiIndex.from_product([df.id.unique(), QDATES], names=["id", "date"])
    p = df.set_index(["id", "date"])[["y"]].reindex(full).sort_index()
    p["y"] = p.groupby(level="id").y.ffill()
    return p.dropna().reset_index()


def bug_warmup_dropna(df):
    """B4: build lag/rolling features, then dropna() the whole frame."""
    d = regrid_clean(df).sort_values(["id", "date"])
    g = d.groupby("id").y
    d["roll8"] = g.transform(lambda s: s.rolling(8, min_periods=8).mean())
    d["lag4"] = g.shift(4)
    d["yoy"] = d.y - d.lag4
    d["roll_yoy"] = d.groupby("id").yoy.transform(lambda s: s.rolling(8, min_periods=8).mean())
    return d.dropna()[["id", "date", "y"]].reset_index(drop=True)


PIPELINES = {
    "C0_clean": (regrid_clean, False),
    "B1_grid_3mo_step": (bug_grid_step, False),
    "B2_survivor_join": (bug_survivor_join, False),
    "B3_ffill_balanced": (bug_ffill_balanced, False),
    "B4_warmup_dropna": (bug_warmup_dropna, False),
    "B5_mase_scale_leak": (regrid_clean, True),  # non-composition control bug
}


# ---------------------------------------------------------------- evaluation
def evaluate(p: pd.DataFrame, scale_leak: bool = False) -> pd.DataFrame:
    res = []
    for i, g in p.sort_values(["id", "date"]).groupby("id", sort=False):
        y = g.y.to_numpy()
        if len(y) < MIN_LEN:
            continue
        tr, te = y[:-H], y[-H:]
        scale = np.mean(np.abs(np.diff(y if scale_leak else tr)))
        if scale <= 0:
            continue
        res.append((i, np.mean(np.abs(te - tr[-1])) / scale, np.mean(np.abs(te - tr.mean())) / scale))
    return pd.DataFrame(res, columns=["id", "naive", "ha"])


def conclusion(ev: pd.DataFrame, rng) -> dict:
    d = (ev.naive - ev.ha).to_numpy()
    boots = [rng.choice(d, d.size).mean() for _ in range(500)]
    return {"n_banks": int(d.size), "delta": float(d.mean()),
            "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "winner": "Naive" if d.mean() < 0 else "HistoricAverage"}


# ---------------------------------------------------------------- checks
def check_schema(raw, p):
    ok = (set(["id", "date", "y"]) <= set(p.columns) and p.y.notna().all()
          and not p.duplicated(["id", "date"]).any() and bool(p.date.dt.is_quarter_end.all()))
    return bool(ok)


def check_stats(raw, p):
    rows_ok = len(p) >= 0.8 * len(raw)
    mean_ok = abs(p.y.mean() - raw.y.mean()) / raw.y.std() < 0.1
    ks = stats.ks_2samp(raw.y, p.y).statistic < 0.05
    return bool(rows_ok and mean_ok and ks)


def lifecycle_coords(raw):
    """Entity life-cycle coordinates taken from RAW data only."""
    first = raw.groupby("id").date.min()
    last = raw.groupby("id").date.max()
    ent = pd.Series(np.where(first == QDATES[0], "inc", "ent"), index=first.index)
    ent = ent + "_Q" + first.dt.quarter.astype(str)
    ext = pd.Series(np.where(last == QDATES[-1], "alive", "exited"), index=last.index)
    return first, ent, ext


def check_ledger(raw, p, tol=0.10):
    """Entity-period ledger, computed against RAW:
    (a) provenance: every processed (id,date,y) exists in raw with the same value;
    (b) retention of entities and entity-periods is uniform across life-cycle
        coordinates: entry cohort, exit status, and age (periods since entry)."""
    key = ["id", "date"]
    j = p.merge(raw[key + ["y"]], on=key, how="left", suffixes=("", "_raw"))
    provenance = bool(j.y_raw.notna().all() and np.allclose(j.y, j.y_raw))
    first, ent, ext = lifecycle_coords(raw)
    kept_ids = set(p.id.unique())
    spreads = {}
    for name, c in {"entry": ent, "exit": ext}.items():
        r = c.groupby(c).apply(lambda s: s.index.isin(kept_ids).mean())
        spreads[f"entities_by_{name}"] = float(r.max() - r.min())
    rk = raw[key].assign(kept=pd.MultiIndex.from_frame(raw[key]).isin(pd.MultiIndex.from_frame(p[key])))
    rk["age"] = np.minimum(((rk.date.dt.to_period("Q") - rk.id.map(first).dt.to_period("Q"))
                            .apply(lambda x: x.n) // 4), 5)
    rk["entry"] = rk.id.map(ent)
    rk["exit"] = rk.id.map(ext)
    for name in ["age", "entry", "exit"]:
        r = rk.groupby(name).kept.mean()
        spreads[f"periods_by_{name}"] = float(r.max() - r.min())
    uniform = all(v <= tol for v in spreads.values())
    return provenance and uniform, {"provenance_ok": provenance,
                                    "spreads": {k: round(v, 3) for k, v in spreads.items()}}


def main(seed=0):
    rng = np.random.default_rng(123)
    raw = simulate(seed=seed)
    out = {}
    for name, (fn, leak) in PIPELINES.items():
        p = fn(raw)
        ev = evaluate(p, scale_leak=leak)
        ev_inc = ev[ev.id.isin(raw[raw.incumbent].id)]
        led_ok, led_info = check_ledger(raw, p)
        out[name] = {
            "rows_kept": round(len(p) / len(raw), 3),
            "banks_kept": round(p.id.nunique() / raw.id.nunique(), 3),
            "entrant_share_eval": round(1 - len(ev_inc) / max(len(ev), 1), 3),
            "conclusion": conclusion(ev, rng),
            "checks": {"schema": check_schema(raw, p), "stats": check_stats(raw, p),
                       "ledger": led_ok},
            "ledger_info": led_info,
        }
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(json.dumps(main(seed), indent=1, default=str))
