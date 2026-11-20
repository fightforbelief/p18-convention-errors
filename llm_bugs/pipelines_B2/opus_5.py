"""Quarterly panel of year-over-year changes in a Call Report ratio, commercial banks only.

Design decisions
----------------
* Dates: each report_date is mapped to its calendar quarter (so off-quarter-end filing dates
  still land in the right quarter); output `date` is the calendar quarter-end Timestamp.
* Duplicates: if a bank files more than once in a quarter, the latest non-missing filing wins
  (amendments supersede originals).
* YoY change: y(t) = value(t) - value(t - 4 quarters), matched on the *calendar* quarter, not
  on row position, so gaps in filing never pair non-adjacent years. A quarter whose same-quarter
  value a year earlier is missing (gap or NaN) gets no y. The ratio is differenced (not a
  percentage change) because ratios can be near zero or negative.
* Charter: point-in-time. Each quarter takes the charter from the most recent attribute snapshot
  on or before its quarter end (falling back to the earliest snapshot for quarters before the
  first one). A y is kept only if the bank was commercial in both quarters being compared, so
  changes never straddle a charter conversion. Banks with no attributes are dropped.
* Thin data: banks with fewer than MIN_OBS valid y rows are dropped, so every kept bank has at
  least MIN_OBS - 4 training quarters after the last 4 quarters are held out.
"""
import numpy as np
import pandas as pd

MIN_OBS = 12  # 4 held-out quarters + at least 8 training quarters


def _quarter_index(yyyymmdd: pd.Series) -> np.ndarray:
    d = pd.to_datetime(yyyymmdd.astype("int64").astype(str), format="%Y%m%d", errors="coerce")
    qi = d.dt.year * 4 + (d.dt.month - 1) // 3
    return qi.to_numpy(dtype="float64")  # NaN for unparseable dates


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame({"id": pd.Series(dtype="int64"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "y": pd.Series(dtype="float64")})

    # --- calls: one value per bank-quarter -------------------------------------------------
    c = pd.DataFrame({
        "id": pd.to_numeric(calls["rssd_id"], errors="coerce"),
        "rd": pd.to_numeric(calls["report_date"], errors="coerce"),
        "value": pd.to_numeric(calls["value"], errors="coerce"),
    })
    c = c[np.isfinite(c["value"]) & c["id"].notna() & c["rd"].notna()]
    c["qi"] = _quarter_index(c["rd"])
    c = c[c["qi"].notna()]
    c["id"] = c["id"].astype("int64")
    c["qi"] = c["qi"].astype("int64")
    c = c.sort_values(["id", "qi", "rd"], kind="mergesort")
    c = c.drop_duplicates(["id", "qi"], keep="last")[["id", "qi", "value"]]
    if c.empty:
        return empty

    # --- YoY difference by exact calendar quarter ------------------------------------------
    prev = c.rename(columns={"value": "value_prev"})
    prev["qi"] = prev["qi"] + 4
    p = c.merge(prev, on=["id", "qi"], how="inner")
    p["y"] = p["value"] - p["value_prev"]

    # --- point-in-time charter -------------------------------------------------------------
    a = pd.DataFrame({
        "id": pd.to_numeric(attrs["rssd_id"], errors="coerce"),
        "asof": pd.to_numeric(attrs["as_of"], errors="coerce"),
        "commercial": attrs["charter"].astype(str).str.strip().str.lower().eq("commercial"),
    })
    a = a[a["id"].notna() & a["asof"].notna()]
    a["qi"] = _quarter_index(a["asof"])
    a = a[a["qi"].notna()]
    a["id"] = a["id"].astype("int64")
    a["qi"] = a["qi"].astype("int64")
    a = a.sort_values(["id", "qi", "asof"], kind="mergesort").drop_duplicates(["id", "qi"], keep="last")
    if a.empty:
        return empty

    def charter_at(ids: np.ndarray, qis: np.ndarray) -> np.ndarray:
        # vectorised as-of lookup on a combined (id, quarter) key
        key_a = a["id"].to_numpy() * 100000 + a["qi"].to_numpy()
        key_q = ids * 100000 + qis
        pos = np.searchsorted(key_a, key_q, side="right") - 1
        pos_c = np.clip(pos, 0, len(key_a) - 1)
        a_id = a["id"].to_numpy()
        hit = (pos >= 0) & (a_id[pos_c] == ids)
        # fallback: earliest snapshot for that bank (quarters before the first snapshot)
        first = np.searchsorted(a_id, ids, side="left")
        first_c = np.clip(first, 0, len(a_id) - 1)
        has_bank = a_id[first_c] == ids
        idx = np.where(hit, pos_c, first_c)
        comm = a["commercial"].to_numpy()[idx]
        return comm & (hit | has_bank)

    ids = p["id"].to_numpy()
    qis = p["qi"].to_numpy()
    keep = charter_at(ids, qis) & charter_at(ids, qis - 4)
    p = p[keep & np.isfinite(p["y"].to_numpy())]

    # --- drop thin banks -------------------------------------------------------------------
    n = p.groupby("id")["y"].transform("size")
    p = p[n >= MIN_OBS]
    if p.empty:
        return empty

    year = p["qi"] // 4
    q = p["qi"] % 4 + 1
    date = pd.PeriodIndex.from_fields(year=year.to_numpy(), quarter=q.to_numpy(), freq="Q") \
        .to_timestamp(how="end").normalize().astype("datetime64[ns]")

    out = pd.DataFrame({"id": p["id"].astype("int64").to_numpy(),
                        "date": date,
                        "y": p["y"].astype("float64").to_numpy()})
    return out.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)


if __name__ == "__main__":
    import sys, time
    calls_path = sys.argv[1] if len(sys.argv) > 1 else "sample_calls.csv"
    attrs_path = sys.argv[2] if len(sys.argv) > 2 else "sample_attrs.csv"
    t = time.time()
    res = build_panel(pd.read_csv(calls_path), pd.read_csv(attrs_path))
    print(res.head(10))
    print(res.dtypes)
    print(f"{len(res)} rows, {res['id'].nunique()} banks, {time.time() - t:.2f}s")
