# Task: build a quarterly panel of Midwest banks

Write a Python module `pipeline.py` that defines

```python
def build_panel(calls: pd.DataFrame, directory: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame: ...
```

using only pandas and numpy. It will be run on the full US Call Report history
(about 24,000 banks, 1976-2020, ~2 million rows), so it should finish in a few minutes.

## Inputs

`calls`: one row per filed regulatory report: `rssd_id` (int), `report_date` (int, `YYYYMMDD`, the
calendar quarter-end the report refers to), `value` (float, the reported ratio; NaN if not reported).

`directory`: the regulator's institution directory as published at the end of the sample period:
`rssd_id` (int), `state` (two-letter headquarters state).

`history`: headquarters history, one row per bank per headquarters spell: `rssd_id` (int),
`start_date` (int, `YYYYMMDD`), `end_date` (int `YYYYMMDD`, empty while the spell is ongoing), `state`.

## Output

A DataFrame with columns `id` (int, the rssd_id), `date` (pandas Timestamp at the calendar quarter
end) and `y` (float): one row per bank-quarter, sorted by `id`, `date`, with no missing `y`.

## Rules

The Midwest states are IL, IN, IA, MI, MN, OH and WI.

1. A bank's active span runs from its first to its last `report_date` in `calls`. Place each bank on a
   regular quarterly grid covering its active span. Quarters with no report or a NaN value count as missing.
2. Keep only bank-quarters in which the bank's headquarters state at that quarter, taken from `history`,
   is a Midwest state. The `directory` lists only institutions still operating at the end of the sample,
   so it must not be used to decide which banks or quarters to keep.
3. Drop banks for which more than 20% of their remaining grid quarters (after rule 2) are missing.
4. Drop the missing bank-quarters. Do not impute or transform values.

The panel feeds a forecasting benchmark that holds out each bank's last 4 quarters.

Sample inputs are in `sample_calls.csv`, `sample_directory.csv` and `sample_history.csv` in this folder.
