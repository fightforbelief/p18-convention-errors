"""Task 3 inputs from a panel (id, date, y): calls, directory (current institutions only), history (HQ spells).
State is synthetic and assigned from the bank id only (independent of y): 50% Midwest, 50% other;
4% of banks relocate once (to the other group) at a random year-end inside their span."""
import numpy as np
import pandas as pd

MIDWEST = ["IL", "IN", "IA", "MI", "MN", "OH", "WI"]
OTHER = ["CA", "TX", "NY", "PA", "FL", "GA", "NC"]


def make(raw, seed=17):
    rng = np.random.default_rng(seed)
    span = raw.groupby("id").date.agg(["min", "max"])
    end_all = raw.date.max()
    rows = []
    for (i, (a, b)), u, m, s, k in zip(span.iterrows(), rng.random(len(span)), rng.random(len(span)),
                                       rng.random(len(span)), rng.integers(0, 7, (len(span), 2)).tolist()):
        st0 = (MIDWEST if m < 0.5 else OTHER)[k[0]]
        st1 = (OTHER if m < 0.5 else MIDWEST)[k[1]]
        dec = [d for d in pd.date_range(pd.Timestamp(a.year, 12, 31), b, freq="YE") if a < d < b]
        fmt = lambda d: int(d.strftime("%Y%m%d"))
        last_end = None if b == end_all else fmt(b)
        if u < 0.04 and dec:
            sw = dec[int(s * len(dec))]
            rows += [(i, fmt(a), fmt(sw - pd.Timedelta(days=1)), st0), (i, fmt(sw), last_end, st1)]
        else:
            rows.append((i, fmt(a), last_end, st0))
    hist = pd.DataFrame(rows, columns=["rssd_id", "start_date", "end_date", "state"])
    hist["end_date"] = hist.end_date.astype("Int64")
    cur = hist[hist.end_date.isna()][["rssd_id", "state"]].reset_index(drop=True)
    return cur, hist


def to_calls(raw):
    return pd.DataFrame({"rssd_id": raw.id.astype(int), "report_date": raw.date.dt.strftime("%Y%m%d").astype(int),
                         "value": raw.y.astype(float)})
