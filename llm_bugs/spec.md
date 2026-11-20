# Task: build a quarterly bank panel

Write a Python module `pipeline.py` that defines

```python
def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame: ...
```

using only pandas and numpy. It will be run on the full US Call Report history
(about 24,000 banks, 1976-2020, ~2 million rows), so it should finish in a few minutes.

## Inputs

`calls`: one row per filed regulatory report.

| column | type | meaning |
|---|---|---|
| `rssd_id` | int | bank identifier |
| `report_date` | int, `YYYYMMDD` | the calendar quarter-end the report refers to |
| `value` | float | the reported ratio; NaN if the item was not reported |

`attrs`: bank attribute snapshots.

| column | type | meaning |
|---|---|---|
| `rssd_id` | int | bank identifier |
| `as_of` | int, `YYYYMMDD` | date of the snapshot |
| `charter` | str | `"commercial"` or `"savings"` |

A snapshot is valid from its `as_of` date until the bank's next snapshot.

## Output

A DataFrame with columns `id` (int, the rssd_id), `date` (pandas Timestamp at the
calendar quarter end, e.g. 2003-06-30) and `y` (float): one row per bank-quarter, sorted by
`id`, `date`, with no missing `y`.

## Rules

1. A bank's active span runs from its first to its last `report_date` in `calls`. Place each
   bank on a regular quarterly grid covering its active span. Quarters with no report or a NaN
   value count as missing.
2. Keep only bank-quarters in which the bank's charter at that quarter (its latest snapshot on
   or before the quarter) is `"commercial"`.
3. Drop banks for which more than 20% of their remaining grid quarters (after rule 2) are missing.
4. Drop the missing bank-quarters. Do not impute or transform values.

The panel feeds a forecasting benchmark that holds out each bank's last 4 quarters.

Sample inputs are in `sample_calls.csv` and `sample_attrs.csv` in this folder.
