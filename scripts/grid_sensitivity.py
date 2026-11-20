"""Post hoc paired sensitivity analysis of B1. Run from the P18 project root.

--extract builds anonymous reference-entity scores from the frozen raw snapshots.
Without that option, only the published sufficient data and result tables are needed.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
PANELS = ("bank_leverage", "bank_cash_liquidity", "bhc_leverage", "bhc_cash_liquidity", "eia_plants")
OUT = ROOT / "results_grid"
REPS = 4000
SEED = 20261120


def load_scores(path):
    # Python's float reader round-trips the published decimal scores.
    with gzip.open(path, "rt", newline="") as stream:
        rows = [(float(r["gap"]), int(r["kept"])) for r in csv.DictReader(stream)]
    return pd.DataFrame(rows, columns=["gap", "kept"])


def extract(panel, base, published):
    pilot = ROOT / "pilot"
    sys.path.insert(0, str(pilot))
    # The frozen modules check their historical file hashes relative to pilot/.
    os.chdir(pilot)
    try:
        import estimands
        import ledger_frozen_v0
        import realdata_v0

        path = ROOT / "data" / "raw" / f"{panel}_panel.parquet"
        manifest_path, = base.glob("run_manifest*.json")
        expected = json.loads(manifest_path.read_text())["inputs"][panel]
        with path.open("rb") as stream:
            same_file = hashlib.file_digest(stream, "sha256").hexdigest() == expected
        raw = realdata_v0.load(str(path))
        ref = estimands.per_bank(ledger_frozen_v0.regrid_clean(raw))
        # A 30-day quarter-end anchor misses both subsequent 31-day quarters.
        # The 20% missingness screen excludes these series at evaluation lengths.
        starts = raw.groupby("id").date.min().reindex(ref.index)
        scores = pd.DataFrame({"gap": ref.naive - ref.ha, "kept": starts.dt.day.eq(31).astype(int)})
        scores = scores.reset_index(drop=True)
        if not same_file:
            # A rebuilt Parquet can use another encoding or row order. Compare all
            # scored observations, then use the archive order for identical RNG draws.
            archive = load_scores(published)
            keys = ["kept", "gap"]
            pd.testing.assert_frame_equal(scores.sort_values(keys).reset_index(drop=True),
                                          archive.sort_values(keys).reset_index(drop=True),
                                          check_exact=True)
            return archive
        return scores
    finally:
        os.chdir(ROOT)


def estimate(gap, kept):
    reference = gap.mean()
    candidate = gap[kept].mean()
    return reference, candidate, 100 * (candidate - reference) / abs(reference)


def analyze(panel, scores, base, rng):
    gap = scores.gap.to_numpy()
    kept = scores.kept.to_numpy(bool)
    reference, candidate, signed = estimate(gap, kept)
    measured = pd.read_csv(base / "estimands" / f"estimands_{panel}_full.csv")
    row, = measured.query("estimand == 'E1' and pipeline == 'B1_grid_3mo_step'").itertuples()
    assert (len(gap), kept.sum()) == (row.n_ref, row.n_p)
    np.testing.assert_allclose((reference, candidate), (row.theta_r, row.theta_p), rtol=1e-12)
    shifts = np.empty(REPS)
    ratios = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(len(gap), size=len(gap))
        r, p, ratios[i] = estimate(gap[idx], kept[idx])
        shifts[i] = p - r
    lo, hi = np.quantile(shifts, [0.025, 0.975])
    rlo, rhi = np.quantile(ratios, [0.025, 0.975])
    return dict(panel=panel, n_ref=len(gap), n_kept=kept.sum(), reference=reference,
                candidate=candidate, shift=candidate-reference, lower=lo, upper=hi,
                signed_pct=signed, pct_lower=rlo, pct_upper=rhi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    for panel, seed in zip(PANELS, np.random.SeedSequence(SEED).spawn(len(PANELS))):
        base = ROOT / ("results_win_eia" if panel == "eia_plants" else "results_win")
        path = OUT / f"{panel}.csv.gz"
        if args.extract:
            scores = extract(panel, base, path)
        else:
            scores = load_scores(path)
        rows.append(analyze(panel, scores, base, np.random.default_rng(seed)))
        if args.extract:
            path.write_bytes(gzip.compress(scores.to_csv(index=False, lineterminator="\n").encode(), mtime=0))
        print(panel, "verified", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "paired_shift.csv", index=False)


if __name__ == "__main__":
    main()
