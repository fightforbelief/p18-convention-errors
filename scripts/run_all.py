"""Reproduce every P18 result from the project root (hanqing-win, .\\.venv\\Scripts\\python.exe).

Usage:  python scripts/run_all.py [stage ...]      stages: pilot realdata estimands llm all   (default: all = pilot estimands llm)
Env: P18_PANELS (comma list), P18_OUT, P18_WORK select another panel set / output and work dirs.
Inputs: data/raw/{panel}_panel.parquet  (copies of the P12 confirm-run panels and the bank-cash build)
Outputs: results_win/<stage>/... plus results_win/run_manifest.json (versions, input and code hashes, timings).
Work files (pipeline outputs, cached inputs) go to data/work/.
"""
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PANELS = os.environ.get("P18_PANELS", "bank_leverage,bank_cash_liquidity,bhc_leverage,bhc_cash_liquidity").split(",")
RAW = {p: os.path.join(ROOT, "data", "raw", f"{p}_panel.parquet").replace("\\", "/") for p in PANELS}  # frozen scripts tag panels via split("/")
OUT = os.path.join(ROOT, os.environ.get("P18_OUT", "results_win"))
WORK = os.path.join(ROOT, "data", os.environ.get("P18_WORK", "work"))
LOG = []


def sh(args, cwd, tag):
    t = time.time()
    r = subprocess.run([PY, *args], cwd=cwd, capture_output=True, text=True)
    LOG.append({"tag": tag, "args": args, "cwd": os.path.relpath(cwd, ROOT), "rc": r.returncode,
                "seconds": round(time.time() - t, 1)})
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    with open(os.path.join(OUT, "logs", f"{tag}.log"), "w", encoding="utf-8") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"{tag} failed (rc={r.returncode}); see results_win/logs/{tag}.log")
    return r


def collect(pattern, cwd, dest):
    os.makedirs(os.path.join(OUT, dest), exist_ok=True)
    for f in glob.glob(os.path.join(cwd, pattern)):
        shutil.copy2(f, os.path.join(OUT, dest, os.path.basename(f)))


def stage_pilot():
    pdir = os.path.join(ROOT, "pilot")
    for s in (0, 1, 2):
        r = sh(["pilot_v0.py", str(s)], pdir, f"pilot_seed{s}")
        os.makedirs(os.path.join(OUT, "pilot"), exist_ok=True)
        open(os.path.join(OUT, "pilot", f"out_seed{s}.json"), "w").write(r.stdout)
    sh(["heldout_v0.py", "0", "1", "2"], pdir, "heldout")
    for p in ("bank_leverage", "bhc_leverage"):
        sh(["realdata_v0.py", RAW[p], "4000", "0"], pdir, f"realdata_{p}")
    sh(["decomp.py", "synthetic:0", RAW["bank_leverage"]], pdir, "decomp_dev")
    shutil.copy2(os.path.join(pdir, "decomp_results.csv"), os.path.join(pdir, "decomp_dev_results.csv"))
    sh(["decomp.py", RAW["bhc_cash_liquidity"], RAW["bank_cash_liquidity"]], pdir, "decomp_test")
    shutil.copy2(os.path.join(pdir, "decomp_results.csv"), os.path.join(pdir, "decomp_test_results.csv"))
    for pat in ("heldout_results.csv", "realdata_results_*_4000.csv", "decomp_dev_results.csv", "decomp_test_results.csv"):
        collect(pat, pdir, "pilot")


def stage_realdata():
    """Design bugs (B1-B5, x1-x9) through ledger v0/v1 on full panels (no subsample)."""
    pdir = os.path.join(ROOT, "pilot")
    with cf.ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda p: sh(["realdata_v0.py", RAW[p], "all"], pdir, f"realdata_{p}_all"), PANELS))
    collect("realdata_results_*_all.csv", pdir, "realdata")


def stage_estimands():
    pdir = os.path.join(ROOT, "pilot")
    with cf.ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda p: sh(["estimands.py", RAW[p]], pdir, f"estimands_{p}"), PANELS))
    collect("estimands_*_full.csv", pdir, "estimands")


# harness arms: (pipelines dir, reference module, output prefix, work subdir)
ARMS = [("pipelines", "reference", "A_results", "t12"), ("pipelines_B", "reference", "B_results", "t12"),
        ("pipelines_A2", "reference_yoy", "A2_results", "t12"), ("pipelines_B2", "reference_yoy", "B2_results", "t12"),
        ("pipelines_A3", "reference_region", "A3_results", "t3"), ("pipelines_B3", "reference_region", "B3_results", "t3"),
        ("pipelines_abl", "reference", "ABL_results", "t12"), ("pipelines_Lsweep", "reference", "LSWEEP_results", "t12")]


def build_t3_inputs(p):
    code = ("import sys,os; sys.path.insert(0,'../pilot'); os.chdir('../pilot'); import realdata_v0 as RD; os.chdir('../llm_bugs');"
            "import make_inputs3 as M3; raw=RD.load(sys.argv[1]); d=sys.argv[2]; os.makedirs(d,exist_ok=True);"
            "M3.to_calls(raw).to_parquet(d+'/calls.parquet'); cur,h=M3.make(raw);"
            "cur.to_parquet(d+'/directory.parquet'); h.to_parquet(d+'/history.parquet')")
    sh(["-c", code, RAW[p], os.path.join(WORK, "t3", f"inputs_{p}")], os.path.join(ROOT, "llm_bugs"), f"t3inputs_{p}")


def stage_llm():
    ldir = os.path.join(ROOT, "llm_bugs")

    def one_panel(p):
        build_t3_inputs(p)
        for pdir, ref, prefix, sub in ARMS:
            wd = os.path.join(WORK, f"{sub}_{prefix}")
            os.makedirs(wd, exist_ok=True)
            src = os.path.join(WORK, sub, f"inputs_{p}")
            dst = os.path.join(wd, f"inputs_{p}")
            if sub == "t12" and not os.path.exists(os.path.join(src, "calls.parquet")):
                os.makedirs(src, exist_ok=True)  # harness builds calls/attrs here on first use
            if not os.path.exists(dst):
                shutil.copytree(src, dst) if os.path.exists(os.path.join(src, "calls.parquet")) else os.makedirs(dst, exist_ok=True)
            sh(["harness.py", RAW[p], pdir, wd, ref, prefix], ldir, f"{prefix}_{p}")
            if sub == "t12" and not os.path.exists(os.path.join(src, "calls.parquet")):
                for f in ("calls.parquet", "attrs.parquet"):
                    shutil.copy2(os.path.join(dst, f), os.path.join(src, f))

    with cf.ThreadPoolExecutor(4) as ex:
        list(ex.map(one_panel, PANELS))
    for _, _, prefix, _ in ARMS:
        collect(f"{prefix}_*.csv", ldir, "llm")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    stages = sys.argv[1:] or ["all"]
    if "all" in stages:
        stages = ["pilot", "estimands", "llm"]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for s in stages:
        {"pilot": stage_pilot, "realdata": stage_realdata, "estimands": stage_estimands, "llm": stage_llm}[s]()
    import numpy, pandas, pyarrow, scipy
    code = sorted(glob.glob(os.path.join(ROOT, "pilot", "*.py")) + glob.glob(os.path.join(ROOT, "llm_bugs", "*.py")))
    manifest = {"stages": stages, "python": platform.python_version(), "platform": platform.platform(),
                "versions": {m.__name__: m.__version__ for m in (pandas, numpy, scipy, pyarrow)},
                "inputs": {p: sha(f) for p, f in RAW.items()},
                "code": {os.path.relpath(f, ROOT): sha(f) for f in code}, "steps": LOG,
                "total_seconds": round(time.time() - t0, 1)}
    json.dump(manifest, open(os.path.join(OUT, f"run_manifest_{'_'.join(stages)}.json"), "w"), indent=1)
    print("done", stages, round(time.time() - t0, 1), "s")
