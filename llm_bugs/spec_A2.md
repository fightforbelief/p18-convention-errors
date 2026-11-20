# Task: build a quarterly panel of year-over-year changes

Write a Python module `pipeline.py` that defines

```python
def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame: ...
```

using only pandas and numpy. It will be run on the full US Call Report history
(about 24,000 banks, 1976-2020, ~2 million rows), so it should finish in a few minutes.

## Inputs

`calls`: one row per filed regulatory report: `rssd_id` (int), `report_date` (int, `YYYYMMDD`,
the calendar quarter-end the report refers to), `value` (float, the reported ratio; NaN if not reported).

`attrs`: bank attribute snapshots: `rssd_id`, `as_of` (int, `YYYYMMDD`), `charter`
(`"commercial"` or `"savings"`). A snapshot is valid from its `as_of` date until the bank's next snapshot.

## Output

A DataFrame with columns `id` (int, the rssd_id), `date` (pandas Timestamp at the calendar quarter
end) and `y` (float): one row per bank-quarter, sorted by `id`, `date`, with no missing `y`.

## Rules

1. A bank-quarter is usable if the bank reported a non-NaN value for that calendar quarter and its
   charter at that quarter (its latest snapshot on or before the quarter) is `"commercial"`.
2. `y` at quarter t is the value at t minus the value at the calendar quarter exactly four quarters
   earlier (e.g. 2003-06-30 minus 2002-06-30). It is defined only if both bank-quarters are usable.
   Do not fill, interpolate or otherwise impute anything.
3. Keep only bank-quarters where `y` is defined, then drop banks with fewer than 16 such quarters.

The panel feeds a forecasting benchmark that holds out each bank's last 4 quarters.

Sample inputs are in `sample_calls.csv` and `sample_attrs.csv` in this folder.
