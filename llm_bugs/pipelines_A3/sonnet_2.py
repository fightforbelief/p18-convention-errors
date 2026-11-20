import numpy as np
import pandas as pd

MIDWEST = {"IL", "IN", "IA", "MI", "MN", "OH", "WI"}


def _qindex(d):
    d = np.asarray(d, dtype="int64")
    return (d // 10000) * 4 + ((d // 100) % 100 - 1) // 3


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    c = calls[["rssd_id", "report_date", "value"]].copy()
    c["q"] = _qindex(c["report_date"])
    # one report per bank-quarter; prefer a non-missing value
    c = c.sort_values(["rssd_id", "q", "value"]).drop_duplicates(["rssd_id", "q"], keep="first")

    span = c.groupby("rssd_id")["q"].agg(["min", "max"])
    n = (span["max"] - span["min"] + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    q = np.repeat(span["min"].to_numpy(), n) + offs
    grid = pd.DataFrame({"id": ids, "q": q})
    # quarter-end date as YYYYMMDD
    y, m = q // 4, (q % 4 + 1) * 3
    day = np.where(np.isin(m, [3, 12]), 31, 30)
    grid["d"] = y * 10000 + m * 100 + day

    grid = grid.merge(c[["rssd_id", "q", "value"]].rename(columns={"rssd_id": "id"}),
                      on=["id", "q"], how="left")

    # headquarters state at each quarter from history spells
    h = history[["rssd_id", "start_date", "end_date", "state"]].copy()
    h["start_date"] = pd.to_numeric(h["start_date"], errors="coerce").fillna(0)
    h["end_date"] = pd.to_numeric(h["end_date"], errors="coerce").fillna(99999999)
    h = h[h["state"].isin(MIDWEST)].rename(columns={"rssd_id": "id"})
    # spells are only needed for Midwest; a quarter in a Midwest spell is kept
    # (a non-Midwest spell cannot overlap a Midwest one for the same date in clean data)
    allh = history.rename(columns={"rssd_id": "id"})
    m_ = grid[["id", "d"]].reset_index().merge(
        h[["id", "start_date", "end_date"]], on="id", how="inner")
    m_ = m_[(m_["start_date"] <= m_["d"]) & (m_["d"] <= m_["end_date"])]
    keep = grid.index.isin(m_["index"].unique())
    grid = grid[keep]

    miss = grid["value"].isna()
    frac = miss.groupby(grid["id"]).mean()
    good = frac[frac <= 0.2].index
    grid = grid[grid["id"].isin(good) & grid["value"].notna()]

    out = pd.DataFrame({
        "id": grid["id"].astype("int64").to_numpy(),
        "date": pd.to_datetime(grid["d"].astype("int64").astype(str), format="%Y%m%d"),
        "y": grid["value"].astype(float).to_numpy(),
    })
    return out.sort_values(["id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    r = build_panel(pd.read_csv("sample_calls.csv"), pd.read_csv("sample_directory.csv"),
                    pd.read_csv("sample_history.csv"))
    print(r.shape, r.id.nunique())
    print(r.head())
