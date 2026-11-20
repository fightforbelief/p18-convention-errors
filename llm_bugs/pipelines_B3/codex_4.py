"""Build a quarterly forecasting panel using point-in-time headquarters."""

import numpy as np
import pandas as pd


MIDWEST = frozenset({"IL", "IN", "IA", "MI", "MN", "OH", "WI"})
HOLDOUT_QUARTERS = 4
MIN_TRAIN_OBSERVATIONS = 8
MIN_TRAIN_COVERAGE = 0.80
MAX_TRAIN_GAP = 2


def _quarter_number(dates):
    return (dates // 10000) * 4 + (dates // 100 % 100 - 1) // 3


def build_panel(
    calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """Use the latest contiguous Midwest stint with filings for each bank.

    Headquarters spell endpoints are inclusive. The directory supplies a
    location only for banks with no history. The final four quarters through
    the last Midwest filing must all have reported values. Training starts
    at its first reported value and must meet the observation, coverage,
    and gap limits above. Internal training gaps are linearly interpolated,
    and trailing training gaps use the last observed training value.
    Multiple reports for a quarter are averaged over their available values.
    """
    history_ids = history["rssd_id"]
    directory_midwest = set(
        directory.loc[
            directory["state"].isin(MIDWEST)
            & ~directory["rssd_id"].isin(history_ids),
            "rssd_id",
        ]
    )
    history_midwest = set(history.loc[history["state"].isin(MIDWEST), "rssd_id"])
    candidates = directory_midwest | history_midwest
    spells = history.loc[history_ids.isin(history_midwest)].sort_values(
        "start_date", kind="stable"
    )
    locations = {
        bank_id: (
            group["start_date"].to_numpy(dtype=np.int64),
            group["end_date"].fillna(99991231).to_numpy(dtype=np.int64),
            group["state"].isin(MIDWEST).to_numpy(),
        )
        for bank_id, group in spells.groupby("rssd_id", sort=False)
    }

    reports = calls.loc[
        calls["rssd_id"].isin(candidates), ["rssd_id", "report_date", "value"]
    ].copy()
    reports["quarter"] = _quarter_number(reports["report_date"])
    reports["value"] = reports["value"].where(np.isfinite(reports["value"]))
    quarterly = reports.groupby(["rssd_id", "quarter"])["value"].mean().reset_index()

    ids, dates, values = [], [], []
    quarter_end = np.array([331, 630, 930, 1231])
    for bank_id, group in quarterly.groupby("rssd_id", sort=False):
        reported_quarters = group["quarter"].to_numpy(dtype=np.int64)
        quarters = np.arange(reported_quarters[0], reported_quarters[-1] + 1)
        calendar_dates = quarters // 4 * 10000 + quarter_end[quarters % 4]
        report_positions = reported_quarters - quarters[0]

        start, stop = 0, len(quarters)
        if bank_id in locations:
            starts, ends, in_region = locations[bank_id]
            spell = np.searchsorted(starts, calendar_dates, side="right") - 1
            eligible = (
                (spell >= 0)
                & (calendar_dates <= ends[spell])
                & in_region[spell]
            )
            eligible_reports = report_positions[eligible[report_positions]]
            if not eligible_reports.size:
                continue
            stop = eligible_reports[-1] + 1
            breaks = np.flatnonzero(~eligible[:stop])
            if breaks.size:
                start = breaks[-1] + 1

        if stop - start < MIN_TRAIN_OBSERVATIONS + HOLDOUT_QUARTERS:
            continue
        series = np.full(len(quarters), np.nan)
        series[report_positions] = group["value"].to_numpy(dtype=float, na_value=np.nan)
        series = series[start:stop]
        holdout = series[-HOLDOUT_QUARTERS:]
        if not np.isfinite(holdout).all():
            continue

        training = series[:-HOLDOUT_QUARTERS]
        observed = np.flatnonzero(np.isfinite(training))
        if observed.size < MIN_TRAIN_OBSERVATIONS:
            continue
        first = observed[0]
        if observed.size / (len(training) - first) < MIN_TRAIN_COVERAGE:
            continue
        if np.diff(observed, append=len(training)).max() - 1 > MAX_TRAIN_GAP:
            continue

        training[first:] = np.interp(np.arange(first, len(training)), observed, training[observed])
        retained = series[first:]
        ids.append(np.full(len(retained), bank_id, dtype=np.int64))
        dates.append(calendar_dates[start + first:stop])
        values.append(retained)

    if not ids:
        return pd.DataFrame(
            {"id": pd.Series(dtype="int64"), "date": pd.Series(dtype="datetime64[ns]"),
             "y": pd.Series(dtype="float64")}
        )
    return pd.DataFrame(
        {
            "id": np.concatenate(ids),
            "date": pd.to_datetime(
                np.concatenate(dates), format="%Y%m%d", cache=True
            ).astype("datetime64[ns]"),
            "y": np.concatenate(values),
        }
    )
