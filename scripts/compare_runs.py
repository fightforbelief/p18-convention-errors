"""Compare the Mac pilot results with the hanqing-win re-run (results_win/).
Usage: python scripts/compare_runs.py <results_win_dir>
For each result CSV produced on both sides, align rows on their identifying columns and report the largest absolute and
relative difference over numeric columns, and any mismatch in non-numeric columns."""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAC = {"pilot": os.path.join(ROOT, "pilot"), "estimands": os.path.join(ROOT, "pilot"), "llm": os.path.join(ROOT, "llm_bugs")}
KEYS = ["panel", "pipeline", "estimand", "seed"]


def compare(a, b):
    keys = [k for k in KEYS if k in a.columns and k in b.columns]
    if keys:
        a, b = a.set_index(keys).sort_index(), b.set_index(keys).sort_index()
    if len(a) != len(b) or not a.index.equals(b.index):
        return {"rows_mac": len(a), "rows_win": len(b), "aligned": False}
    num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
           and not pd.api.types.is_bool_dtype(a[c]) and not pd.api.types.is_bool_dtype(b[c])]
    oth = [c for c in a.columns if c in b.columns and c not in num and c != "err"]
    diff = (a[num] - b[num]).abs()
    rel = diff / a[num].abs().where(a[num].abs() > 1e-12)
    mism = sum(int((a[c].astype(str) != b[c].astype(str)).sum()) for c in oth)
    return {"rows": len(a), "aligned": True, "max_abs": float(np.nanmax(diff.to_numpy())) if len(num) else 0.0,
            "max_rel": float(np.nanmax(rel.to_numpy())) if len(num) and np.isfinite(rel.to_numpy()).any() else 0.0,
            "worst_col": diff.max().idxmax() if len(num) else None, "nonnumeric_mismatches": mism}


if __name__ == "__main__":
    win = sys.argv[1]
    rows = []
    for stage, mac_dir in MAC.items():
        for f in sorted(glob.glob(os.path.join(win, stage, "*.csv"))):
            name = os.path.basename(f)
            m = os.path.join(mac_dir, name)
            if not os.path.exists(m):
                rows.append({"file": f"{stage}/{name}", "note": "no Mac counterpart"})
                continue
            rows.append({"file": f"{stage}/{name}", **compare(pd.read_csv(m), pd.read_csv(f))})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
