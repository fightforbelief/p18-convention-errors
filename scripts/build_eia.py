"""Build the non-bank replication panel from EIA-923 (data/raw/eia923/*.zip, 2001-2024; see MANIFEST.csv).

Unit = power plant (Plant Id). y = quarterly net generation, GWh (sum of the 3 monthly Netgen values, summed over
all prime-mover / fuel rows of the plant). Rows with Plant Id 99999 (state-fuel imputation increments) are dropped.
Entry / exit: within each plant, leading and trailing quarters with zero or missing generation are trimmed, so the
panel starts at the first quarter with non-zero generation and ends at the last one (opening / retirement).
Interior zero quarters are kept (legitimate for peaking plants); interior quarters absent from a year file stay absent.
Output: data/raw/eia_plants_panel.parquet with columns unique_id, ds (quarter end), y  (same layout as the bank panels).
"""
import glob
import io
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "eia923")
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def page1(zpath):
    z = zipfile.ZipFile(zpath)
    names = [n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx", ".xlsm"))]
    best = None
    for n in sorted(names, key=lambda n: -z.getinfo(n).file_size):
        xl = pd.ExcelFile(io.BytesIO(z.read(n)))
        sh = [s for s in xl.sheet_names if s.lower().startswith("page 1 generation and fuel")]
        if sh:
            best = (n, xl, sh[0])
            break
    if best is None:
        raise RuntimeError(f"no Page 1 sheet in {zpath}")
    n, xl, sh = best
    head = xl.parse(sh, header=None, nrows=15)
    hrow = next(i for i in range(15) if any(norm(v) == "plantid" for v in head.iloc[i].tolist()))
    df = xl.parse(sh, header=hrow)
    cols = {norm(c): c for c in df.columns}
    pid = cols["plantid"]
    gen = []
    for m in MONTHS:
        cand = [c for k, c in cols.items() if k.startswith("netgen") and m in k[6:]]
        if len(cand) != 1:
            raise RuntimeError(f"{zpath}: netgen column for {m}: {cand}")
        gen.append(cand[0])
    out = df[[pid, *gen]].copy()
    out.columns = ["plant", *MONTHS]
    out["plant"] = pd.to_numeric(out.plant, errors="coerce")
    out = out[out.plant.notna() & (out.plant != 99999)]
    for m in MONTHS:
        out[m] = pd.to_numeric(out[m], errors="coerce")
    g = out.groupby("plant")[MONTHS].sum(min_count=1)
    return g, n, sh, hrow


def main():
    frames, log = [], []
    for f in sorted(glob.glob(os.path.join(SRC, "*.zip"))):
        year = int(re.search(r"(\d{4})\.zip$", f).group(1))
        g, member, sheet, hrow = page1(f)
        q = pd.DataFrame({qq: g[MONTHS[3 * qq:3 * qq + 3]].sum(axis=1, min_count=1) for qq in range(4)})
        q = q.stack().rename("mwh").reset_index().rename(columns={"level_1": "q"})
        q["ds"] = pd.PeriodIndex.from_fields(year=np.full(len(q), year), quarter=q.q.to_numpy() + 1, freq="Q").to_timestamp(how="end").normalize()
        frames.append(q[["plant", "ds", "mwh"]])
        log.append((year, os.path.basename(f), member, sheet, hrow, len(g)))
        print(year, member, "plants", len(g), flush=True)
    d = pd.concat(frames, ignore_index=True).dropna(subset=["mwh"])
    d = d.sort_values(["plant", "ds"])
    nz = d.mwh != 0
    first = d[nz].groupby("plant").ds.min()
    last = d[nz].groupby("plant").ds.max()
    d = d[d.plant.isin(first.index)]
    d = d[(d.ds >= d.plant.map(first)) & (d.ds <= d.plant.map(last))]
    panel = pd.DataFrame({"unique_id": d.plant.astype(np.int64), "ds": d.ds, "y": d.mwh / 1000.0})
    out = os.path.join(ROOT, "data", "raw", "eia_plants_panel.parquet")
    panel.to_parquet(out, index=False)
    pd.DataFrame(log, columns=["year", "zip", "member", "sheet", "header_row", "plants"]).to_csv(
        os.path.join(ROOT, "data", "raw", "eia_build_log.csv"), index=False)
    span = panel.groupby("unique_id").ds.agg(["min", "max"])
    print("rows", len(panel), "plants", panel.unique_id.nunique(), "span", panel.ds.min().date(), panel.ds.max().date(),
          "| present at start", int((span["min"] == panel.ds.min()).sum()), "| alive at end", int((span["max"] == panel.ds.max()).sum()))


if __name__ == "__main__":
    main()
