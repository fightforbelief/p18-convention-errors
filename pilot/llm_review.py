"""Pilot LLM-reviewer probe: give a reviewer the pipeline code (docstrings and
bug-revealing names stripped) plus the summary a reviewer would normally see,
ask whether anything could bias the conclusion. Graded later by hand/keyword."""
import ast
import inspect
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pilot_v0 as P

TASK = """You are reviewing a colleague's data pipeline before results go into a paper.
Goal: compare two forecasting baselines (Naive vs HistoricAverage), horizon 4 quarters,
on a quarterly panel of U.S. bank cash-liquidity ratios (Call Reports, 1990Q1-2020Q4,
dates are calendar quarter-ends). Raw input `df` has columns id, date, y; banks enter
and exit over time. Metric: per-bank MASE on the last 4 observations, averaged over banks.

Pipeline code:
```python
{code}
```

Run summary: raw rows={raw_rows}, processed rows={rows}, banks kept={banks:.1%} of raw,
banks evaluated={n_eval}. Result: mean MASE(Naive) - MASE(HA) = {delta:+.3f}
(95% CI {lo:+.3f} to {hi:+.3f}). Checks passed: schema (no nulls, unique id-date,
all dates quarter-end) and distribution (row retention >= 80%, mean/KS shift small).

Is this pipeline sound for the stated goal? Reply in JSON only:
{{"verdict": "OK" or "PROBLEM", "issues": [short strings, most important first]}}
"""


def strip_docstrings(src: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Module)) and node.body and \
           isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def pipeline_code(name):
    fn, leak = P.PIPELINES[name]
    parts = [inspect.getsource(P.quality_filter)]
    if fn is not P.regrid_clean:
        parts.append(inspect.getsource(fn))
    if fn in (P.regrid_clean, P.bug_survivor_join, P.bug_warmup_dropna):
        parts.append(inspect.getsource(P.regrid_clean))
    ev = inspect.getsource(P.evaluate)
    if not leak:
        ev = ev.replace("np.diff(y if scale_leak else tr)", "np.diff(tr)")
    else:
        ev = ev.replace("np.diff(y if scale_leak else tr)", "np.diff(y)")
    ev = ev.replace(", scale_leak: bool = False", "")
    parts.append(ev)
    code = strip_docstrings("\n\n".join(parts))
    code = code.replace(fn.__name__, "build_panel").replace("regrid_clean", "make_grid")
    code += "\n\npanel = build_panel(df)\nresults = evaluate(panel)\n"
    return code


def ask(prompt, model):
    r = subprocess.run(["claude", "-p", "--model", model, prompt], capture_output=True,
                       text=True, timeout=600)
    return r.stdout.strip()


if __name__ == "__main__":
    summary = json.load(open("out_seed0.json"))
    model = sys.argv[1] if len(sys.argv) > 1 else "sonnet"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    raw_rows = len(P.simulate(seed=0))
    jobs = []
    for name, s in summary.items():
        c = s["conclusion"]
        prompt = TASK.format(code=pipeline_code(name), raw_rows=raw_rows,
                             rows=int(round(s["rows_kept"] * raw_rows)), banks=s["banks_kept"],
                             n_eval=c["n_banks"], delta=c["delta"], lo=c["lo"], hi=c["hi"])
        open(f"prompt_{name}.txt", "w").write(prompt)
        jobs += [(name, k, prompt) for k in range(reps)]
    with ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(lambda j: ask(j[2], model), jobs))
    res = [{"pipeline": n, "rep": k, "model": model, "reply": o} for (n, k, _), o in zip(jobs, outs)]
    json.dump(res, open(f"llm_review_{model}.json", "w"), indent=1)
    print("done", len(res))
