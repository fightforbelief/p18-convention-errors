"""LLM-written pipelines vs the reference, on one real panel.
Usage: python harness.py <panel.parquet> <pipelines_dir> <work_dir> [reference_module] [output_prefix]
Ground truth is the reference implementation of spec.md; the ledger and checks never see it except as
ledger v1's declared reference (the same role C0 played before)."""
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pilot"))
_cwd = os.getcwd()
os.chdir(os.path.join(HERE, "..", "pilot"))  # pilot modules hash-check their files by relative path
import estimands as E  # noqa: E402  (frozen decomposition v2)
import ledger_frozen_v0 as F  # noqa: E402
import ledger_v1 as L1  # noqa: E402
import realdata_v0 as RD  # noqa: E402
os.chdir(_cwd)
import make_inputs as M  # noqa: E402
import importlib  # noqa: E402

TIMEOUT = 1800


def coerce(p):
    """Minimal, non-semantic coercion so a column-dtype slip does not hide a semantic result."""
    p = p.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["id"] = p["id"].astype(np.int64)
    p["y"] = p["y"].astype(float)
    return p[["id", "date", "y"]]


def main(panel, pdir, work, ref_mod="reference", prefix="llm_results"):
    REF = importlib.import_module(ref_mod)
    tag = panel.split("/")[-1].replace("_panel.parquet", "")
    inp = f"{work}/inputs_{tag}"
    os.makedirs(inp, exist_ok=True)
    raw = RD.load(panel)  # sets F.QDATES; raw = every filed bank-quarter
    if not os.path.exists(f"{inp}/calls.parquet"):
        M.to_calls(raw).to_parquet(f"{inp}/calls.parquet")
        M.make_attrs(raw).to_parquet(f"{inp}/attrs.parquet")
    inputs = {os.path.basename(f)[:-8]: pd.read_parquet(f) for f in sorted(glob.glob(f"{inp}/*.parquet"))}
    ref = REF.build_panel(**inputs)
    R = E.per_bank(ref)
    cell = (lambda c: c[1] + "|" + c[2])(L1._coords(raw, F.QDATES))
    one = pd.Series(1.0, index=R.index)
    d1, rho = R.naive - R.ha, R.sxy / R.sxx
    rk = pd.MultiIndex.from_frame(ref[["id", "date"]])
    rows = []
    for f in sorted(glob.glob(f"{pdir}/*.py")):
        name = os.path.basename(f)[:-3]
        out = f"{work}/out_{tag}_{name}.parquet"
        row = {"panel": tag, "pipeline": name, "model": name.split("_")[0]}
        try:
            r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_one.py"),
                                f, inp, out], capture_output=True, text=True, timeout=TIMEOUT)
            if r.returncode != 0:
                rows.append({**row, "status": "crash", "err": r.stderr.strip().splitlines()[-1][:200]})
                continue
        except subprocess.TimeoutExpired:
            rows.append({**row, "status": "timeout"})
            continue
        p0 = pd.read_parquet(out)
        try:
            schema = F.check_schema(raw, p0)
        except Exception:
            schema = False
        try:
            p = coerce(p0)
        except Exception as e:
            rows.append({**row, "status": "bad_output", "err": str(e)[:200]})
            continue
        pk = pd.MultiIndex.from_frame(p[["id", "date"]])
        inter = rk.intersection(pk)
        mm = ref.set_index(["id", "date"]).y.reindex(inter) - p.drop_duplicates(["id", "date"]).set_index(["id", "date"]).y.reindex(inter)
        row.update(status="ok", schema_ok=schema, stats_ok=F.check_stats(raw, p), ref_stats_ok=F.check_stats(raw, ref),
                   n_ref=len(ref), n_p=len(p), dup_keys=int(p.duplicated(["id", "date"]).sum()),
                   missing_keys=len(rk.difference(pk)), extra_keys=len(pk.difference(rk)),
                   key_dev=(len(rk.difference(pk)) + len(pk.difference(rk))) / len(rk),
                   value_mismatch=int((mm.abs() > 1e-9).sum()),
                   ids_ref=ref.id.nunique(), ids_p=p.id.nunique())
        ok, info = L1.check(raw, p.drop_duplicates(["id", "date"]), ref, F.QDATES)
        row.update(ledger_flag=not ok, **{f"ledger_{k}": v for k, v in info.items()})
        P = E.per_bank(p.drop_duplicates(["id", "date"]))
        if len(P):
            e1 = E.decompose(d1, one, P.naive - P.ha, pd.Series(1.0, index=P.index), cell)
            okk = P.sxx > 0
            e2 = E.decompose(rho, R.sxx, (P.sxy / P.sxx)[okk], P.sxx[okk], cell)
            rr, rp, rf = E.ranking(R), E.ranking(P), E.ranking(R.loc[P.index.intersection(R.index)])
            row.update(**{f"E1_{k}": v for k, v in e1.items()}, E1_rel=abs(e1["total"]) / abs(e1["theta_r"]),
                       **{f"E2_{k}": v for k, v in e2.items()}, E2_rel=abs(e2["total"]) / (1 - e2["theta_r"]),
                       E3_rank_ref=">".join(rr), E3_rank_p=">".join(rp), E3_forecast_exact=rf == rp,
                       E3_top1_changed=rp[0] != rr[0])
        rows.append(row)
        print(tag, name, row["status"], f"key_dev={row['key_dev']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{prefix}_{tag}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
