"""Decomposition v2: three estimands, full panels.

E1  mean per-bank MASE(Naive) - MASE(HistoricAverage)       (h=4, MIN_LEN=16; identical to frozen evaluate)
E2  pooled within-bank AR(1) persistence rho = sum_i Sxy_i / sum_i Sxx_i  (FE estimator on the full series)
    = weighted mean of per-bank rho_i with weights Sxx_i
E3  ranking of 6 forecasters by mean MASE: Naive, HA, SNaive(4), Drift, Window(4), SES(alpha grid)

Every estimand is a weighted mean theta(S) = sum_S w_i a_i / sum_S w_i (E3 = 15 such pairwise gaps).
For candidate pipeline p and reference r (C0), with S_k = S_p n S_r and reference values (a_i, w_i):
  theta_sel = theta_r-values over S_k                      (computable WITHOUT refitting on p)
  total = theta_p - theta_r = T1 + T2 + T3
  T1 = sum_c (W_kc - W_rc) mu_rc                         between-cohort composition
  T2 = sum_c W_kc (mu_kc - mu_rc)                        within-cohort selection
  T3 = theta_p - theta_sel                               alteration of kept series + banks new to p
Cells c = life-cycle cohort from RAW (entry cohort x first calendar quarter, exit status), as in decomp.py.

Materiality (n-free, population-level), used to classify coverage vs bias:
  E1 material if |total| >= 0.10 |theta_r|
  E2 material if |total| >= 0.10 (1 - rho_r)     (10% change in mean-reversion speed ~ half-life)
  E3 material if the top-1 model changes
Harm (n-dependent): |total| > reference 95% half-width (E1: normal; E2: bank-cluster bootstrap);
  E3 harmful if any pairwise gap whose clean CI excludes 0 flips sign.
"""
import sys

import numpy as np
import pandas as pd

import ledger_frozen_v0 as F
import ledger_v1 as L1

MODELS = ["naive", "ha", "snaive", "drift", "win4", "ses"]
ALPHAS = np.linspace(0.1, 0.9, 9)


def _ses_level(tr):
    lv = np.full(ALPHAS.size, tr[0])
    sse = np.zeros(ALPHAS.size)
    for x in tr[1:]:
        e = x - lv
        sse += e * e
        lv = lv + ALPHAS * e
    return lv[np.argmin(sse)]


def per_bank(p, scale_leak=False):
    H = F.H
    rows = []
    for i, g in p.sort_values(["id", "date"]).groupby("id", sort=False):
        y = g.y.to_numpy(dtype=float)
        n = len(y)
        if n < F.MIN_LEN:
            continue
        tr, te = y[:-H], y[-H:]
        scale = np.mean(np.abs(np.diff(y if scale_leak else tr)))
        if scale <= 0:
            continue
        k = np.arange(1, H + 1)
        fc = {"naive": np.full(H, tr[-1]), "ha": np.full(H, tr.mean()),
              "snaive": tr[-4:], "drift": tr[-1] + k * (tr[-1] - tr[0]) / (len(tr) - 1),
              "win4": np.full(H, tr[-4:].mean()), "ses": np.full(H, _ses_level(tr))}
        yc = y - y.mean()
        rows.append((i, *[np.mean(np.abs(te - fc[m])) / scale for m in MODELS],
                     float(yc[1:] @ yc[:-1]), float(yc[:-1] @ yc[:-1])))
    return pd.DataFrame(rows, columns=["id", *MODELS, "sxy", "sxx"]).set_index("id")


def _theta(a, w):
    return float((a * w).sum() / w.sum())


def decompose(a_r, w_r, a_p, w_p, cell):
    """a_*, w_*: per-bank contribution and weight Series indexed by id."""
    theta_r, theta_p = _theta(a_r, w_r), _theta(a_p, w_p)
    keep = a_p.index.intersection(a_r.index)
    cr, ck = cell.reindex(a_r.index), cell.reindex(keep)
    Wr = w_r.groupby(cr).sum() / w_r.sum()
    mur = (a_r * w_r).groupby(cr).sum() / w_r.groupby(cr).sum()
    wk, ak = w_r.reindex(keep), a_r.reindex(keep)
    if len(keep) == 0:
        return dict(theta_r=theta_r, theta_p=theta_p, total=theta_p - theta_r, T1=np.nan, T2=np.nan,
                    T3=np.nan, forecast=np.nan, cohort_TV=np.nan, kept_share=0.0)
    Wk = (wk.groupby(ck).sum() / wk.sum()).reindex(Wr.index, fill_value=0)
    muk = ((ak * wk).groupby(ck).sum() / wk.groupby(ck).sum()).reindex(Wr.index)
    T1 = float(((Wk - Wr) * mur).sum())
    T2 = float((Wk * (muk - mur)).fillna(0).sum())
    theta_sel = _theta(ak, wk)
    return dict(theta_r=theta_r, theta_p=theta_p, total=theta_p - theta_r, T1=T1, T2=T2,
                T3=theta_p - theta_sel, forecast=theta_sel - theta_r,
                cohort_TV=float(0.5 * (Wk - Wr).abs().sum()), kept_share=len(keep) / len(a_r))


def ranking(df):
    return list(df[MODELS].mean().sort_values().index)


def run(raw, pipes, tag, seed=0):
    rng = np.random.default_rng(seed)
    ref_p = F.regrid_clean(raw)
    R = per_bank(ref_p)
    cell = (lambda c: c[1] + "|" + c[2])(L1._coords(raw, F.QDATES))
    n = len(R)
    one = pd.Series(1.0, index=R.index)
    # reference uncertainty
    d1 = R.naive - R.ha
    hw1 = 1.96 * d1.std() / np.sqrt(n)
    rho_i = R.sxy / R.sxx
    cnt = rng.multinomial(n, np.full(n, 1 / n), size=300)
    boot = (cnt @ R.sxy.to_numpy()) / (cnt @ R.sxx.to_numpy())
    hw2 = float((np.quantile(boot, 0.975) - np.quantile(boot, 0.025)) / 2)
    gaps = {(a, b): R[a] - R[b] for i, a in enumerate(MODELS) for b in MODELS[i + 1:]}
    sig = {k: (abs(v.mean()) > 1.96 * v.std() / np.sqrt(n)) for k, v in gaps.items()}
    rank_r = ranking(R)
    rows = []
    for name, (fn, leak) in pipes.items():
        p = fn(raw)
        P = per_bank(p, scale_leak=leak)
        flag = (not L1.check(raw, p, ref_p, F.QDATES)[0]) if len(p) else True
        base = dict(panel=tag, pipeline=name, n_ref=n, n_p=len(P), ledger_v1_flag=flag)
        if len(P) == 0:
            rows.append({**base, "estimand": "E1"})
            continue
        e1 = decompose(d1, one, P.naive - P.ha, pd.Series(1.0, index=P.index), cell)
        rows.append({**base, "estimand": "E1", **e1, "halfwidth": hw1, "harm_ratio": abs(e1["total"]) / hw1,
                     "rel_shift": abs(e1["total"]) / abs(e1["theta_r"]),
                     "material": abs(e1["total"]) >= 0.10 * abs(e1["theta_r"])})
        ok = P.sxx > 0
        e2 = decompose(rho_i, R.sxx, (P.sxy / P.sxx)[ok], P.sxx[ok], cell)
        rows.append({**base, "estimand": "E2", **e2, "halfwidth": hw2, "harm_ratio": abs(e2["total"]) / hw2,
                     "rel_shift": abs(e2["total"]) / (1 - e2["theta_r"]),
                     "material": abs(e2["total"]) >= 0.10 * (1 - e2["theta_r"])})
        rank_p = ranking(P)
        keep = P.index.intersection(R.index)
        rank_f = ranking(R.loc[keep])
        flips = [f"{a}>{b}" for (a, b), v in gaps.items()
                 if sig[(a, b)] and np.sign((P[a] - P[b]).mean()) != np.sign(v.mean())]
        t3_max = max(abs(decompose(v, one, P[a] - P[b], pd.Series(1.0, index=P.index), cell)["T3"])
                     for (a, b), v in gaps.items())
        rows.append({**base, "estimand": "E3", "rank_ref": ">".join(rank_r), "rank_p": ">".join(rank_p),
                     "rank_forecast": ">".join(rank_f), "forecast_exact": rank_f == rank_p,
                     "top1_changed": rank_p[0] != rank_r[0], "material": rank_p[0] != rank_r[0],
                     "sig_flips": ";".join(flips), "harm_ratio": float(len(flips)), "T3": t3_max})
        print(tag, name, "done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import heldout_v0 as X
    import realdata_v0 as RD
    src = sys.argv[1]
    n_sub = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pipes = {k: v for k, v in F.PIPELINES.items() if k != "C0_clean"}
    pipes["B1_grid_3mo_step"] = (RD.bug_grid_step_long, False)
    pipes = {**pipes, **{k: (fn, False) for k, fn in X.HELDOUT.items()}}
    raw = RD.load(src, n_sub, 0)
    tag = src.split("/")[-1].replace("_panel.parquet", "")
    out = run(raw, pipes, tag)
    out.to_csv(f"estimands_{tag}_{n_sub or 'full'}.csv", index=False)
    print(out.round(4).to_string())
