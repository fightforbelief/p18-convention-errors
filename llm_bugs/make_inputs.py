"""Turn a panel (id, date, y) into the spec's raw inputs (calls, attrs). Charter is synthetic,
assigned from the bank id only (independent of y): 85% commercial throughout, 9% savings
throughout, 4% savings->commercial and 2% commercial->savings at a random year-end in the span.
Snapshots: one at the first report date, then every 31 Dec inside the active span."""
import numpy as np
import pandas as pd


def to_calls(raw):
    return pd.DataFrame({"rssd_id": raw.id.astype(int),
                         "report_date": raw.date.dt.strftime("%Y%m%d").astype(int),
                         "value": raw.y.astype(float)})


def make_attrs(raw, seed=11):
    span = raw.groupby("id").date.agg(["min", "max"])
    rng = np.random.default_rng(seed)
    u = rng.random(len(span))
    kind = np.select([u < 0.85, u < 0.94, u < 0.98], ["C", "S", "S2C"], "C2S")
    rows = []
    for (i, (a, b)), k, s in zip(span.iterrows(), kind, rng.random(len(span))):
        dec = pd.date_range(pd.Timestamp(a.year, 12, 31), b, freq="YE")
        dates = [a] + [d for d in dec if d > a]
        sw = dates[int(s * len(dates))] if k in ("S2C", "C2S") else None
        for d in dates:
            if k == "C":
                ch = "commercial"
            elif k == "S":
                ch = "savings"
            elif k == "S2C":
                ch = "commercial" if sw is not None and d >= sw else "savings"
            else:
                ch = "savings" if sw is not None and d >= sw else "commercial"
            rows.append((i, int(d.strftime("%Y%m%d")), ch))
    return pd.DataFrame(rows, columns=["rssd_id", "as_of", "charter"])


def sample(seed=3, n=300):
    """Synthetic sample for the pipeline authors (no real data)."""
    rng = np.random.default_rng(seed)
    q = pd.date_range("1990-03-31", "2020-12-31", freq="QE")
    rows = []
    for i in range(n):
        s = 0 if rng.random() < 0.5 else int(rng.integers(1, len(q) - 20))
        e = len(q) - 1 if rng.random() < 0.7 else int(rng.integers(s + 12, len(q)))
        y = 0.1 + np.cumsum(rng.normal(0, 0.005, e - s + 1))
        keep = rng.random(e - s + 1) > (0.05 if rng.random() < 0.8 else 0.3)
        keep[0] = keep[-1] = True
        for t, (d, v) in enumerate(zip(q[s:e + 1], y)):
            if keep[t]:
                rows.append((1000 + i, d, v if rng.random() > 0.02 else np.nan))
    raw = pd.DataFrame(rows, columns=["id", "date", "y"])
    return to_calls(raw), make_attrs(raw)
