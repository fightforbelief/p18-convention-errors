"""Quarterly panel of year-over-year changes in Call Report ratios.

build_panel(calls, attrs) returns one row per commercial bank-quarter with
    id   : rssd_id (int)
    date : calendar quarter-end Timestamp (midnight)
    y    : value[t] - value[t - 4 quarters] (float, never missing)

Design choices:
- Reports are mapped to calendar quarters by integer quarter index
  (year * 4 + quarter-of-year), so the year-ago comparison is a fixed 4-quarter
  offset rather than a row shift. Missing quarters therefore stay missing.
- y is produced only where both the quarter and its year-ago quarter have a
  non-missing value. Nothing is interpolated or filled.
- Charter status is point-in-time: the most recent attribute snapshot on or
  before each quarter. Both the quarter and its year-ago quarter must be
  commercial, so a charter switch does not create a fake change.
- Banks with fewer than MIN_OBS valid y values are dropped as too thin.
"""

import numpy as np
import pandas as pd

MIN_OBS = 8  # minimum number of valid y observations per bank


def _quarter_index(yyyymmdd: pd.Series) -> pd.Series:
    """Map YYYYMMDD integers to an integer quarter index (year*4 + q0)."""
    d = yyyymmdd.astype("int64")
    year = d // 10000
    month = (d // 100) % 100
    return year * 4 + (month - 1) // 3


def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    # ---- clean ratio observations and collapse to bank-quarter ----
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["value"] = pd.to_numeric(c["value"], errors="coerce").astype("float64")
    c.loc[~np.isfinite(c["value"]), "value"] = np.nan
    c = c.dropna(subset=["value"])
    c["q"] = _quarter_index(c["report_date"])
    cq = c.groupby(["rssd_id", "q"], sort=False)["value"].mean().reset_index()

    # ---- year-ago pairs: quarter t and quarter t-4 must both be observed ----
    lag = cq.assign(q=cq["q"] + 4).rename(columns={"value": "v_lag"})
    pairs = cq.rename(columns={"value": "v_t"}).merge(lag, on=["rssd_id", "q"], how="inner")
    if pairs.empty:
        return pd.DataFrame(columns=["id", "date", "y"]).astype(
            {"id": "int64", "date": "datetime64[ns]", "y": "float64"}
        )
    pairs["y"] = pairs["v_t"] - pairs["v_lag"]
    pairs["q_lag"] = pairs["q"] - 4

    # ---- point-in-time charter status per (bank, quarter) ----
    a = attrs[["rssd_id", "as_of", "charter"]].copy()
    a["q"] = _quarter_index(a["as_of"])
    a["commercial"] = a["charter"].astype(str).str.strip().str.lower().eq("commercial")
    a = a.sort_values(["q", "as_of"], kind="mergesort")[["rssd_id", "q", "commercial"]]

    keys = pd.concat(
        [pairs[["rssd_id", "q"]], pairs[["rssd_id"]].assign(q=pairs["q_lag"])],
        ignore_index=True,
    ).drop_duplicates()
    keys = keys.sort_values("q", kind="mergesort")
    charter = pd.merge_asof(
        keys, a, on="q", by="rssd_id", direction="backward"
    )
    charter["commercial"] = charter["commercial"].fillna(False).astype(bool)

    pairs = pairs.merge(
        charter.rename(columns={"commercial": "com_t"}), on=["rssd_id", "q"], how="left"
    )
    pairs = pairs.merge(
        charter.rename(columns={"q": "q_lag", "commercial": "com_lag"}),
        on=["rssd_id", "q_lag"],
        how="left",
    )
    keep = pairs["com_t"].fillna(False).astype(bool) & pairs["com_lag"].fillna(False).astype(bool)
    out = pairs.loc[keep, ["rssd_id", "q", "y"]]

    # ---- drop banks with too little data ----
    counts = out.groupby("rssd_id")["y"].transform("size")
    out = out.loc[counts >= MIN_OBS]

    # ---- format output ----
    q = out["q"].to_numpy()
    dates = pd.PeriodIndex.from_fields(
        year=q // 4, quarter=q % 4 + 1, freq="Q"
    ).to_timestamp(how="end").normalize()
    result = pd.DataFrame(
        {
            "id": out["rssd_id"].to_numpy().astype("int64"),
            "date": pd.DatetimeIndex(dates).astype("datetime64[ns]"),
            "y": out["y"].to_numpy().astype("float64"),
        }
    )
    result = result.sort_values(["id", "date"], kind="mergesort").reset_index(drop=True)
    return result


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    calls_df = pd.read_csv(os.path.join(here, "sample_calls.csv"))
    attrs_df = pd.read_csv(os.path.join(here, "sample_attrs.csv"))
    panel = build_panel(calls_df, attrs_df)
    print(panel.dtypes)
    print(panel.head(10))
    print("rows:", len(panel), "banks:", panel["id"].nunique())
