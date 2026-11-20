"""Render the primary comparison from the paired bootstrap output."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
labels = {"bank_leverage": "Bank leverage", "bank_cash_liquidity": "Bank cash",
          "bhc_leverage": "BHC leverage", "bhc_cash_liquidity": "BHC cash",
          "eia_plants": "Power plants"}
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Panel & Retained / reference & Reference & Candidate & Shift & Relative shift [95\% interval] \\",
         r"\midrule"]
with (root / "results_grid/paired_shift.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        shifts = [float(row[k]) for k in ("signed_pct", "pct_lower", "pct_upper")]
        lines.append(" & ".join([labels[row["panel"]], f'{int(row["n_kept"]):,} / {int(row["n_ref"]):,}',
                                 *[f'${float(row[k]):.3f}$' for k in ("reference", "candidate", "shift")],
                                 f'${shifts[0]:+.1f}\\;[{shifts[1]:+.1f}, {shifts[2]:+.1f}]$']) + r" \\")
lines.extend([r"\bottomrule", r"\end{tabular}"])
(root / "paper/generated/tab_grid.tex").write_text("\n".join(lines) + "\n")
