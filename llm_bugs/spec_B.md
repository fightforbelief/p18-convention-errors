# Task: build a quarterly bank panel

Write a Python module `pipeline.py` that defines

```python
def build_panel(calls: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame: ...
```

using only pandas and numpy. It will be run on the full US Call Report history
(about 24,000 banks, 1976-2020, ~2 million rows), so it should finish in a few minutes.

`calls` has one row per filed report: `rssd_id` (int), `report_date` (int, YYYYMMDD) and
`value` (float, the reported ratio, may be NaN). `attrs` has bank attribute snapshots:
`rssd_id`, `as_of` (int, YYYYMMDD) and `charter` ("commercial" or "savings").

We need a clean quarterly panel of commercial banks for a forecasting benchmark that holds
out each bank's last 4 quarters. Put the data on a regular quarterly frequency, handle gaps
sensibly and leave out banks whose data coverage is too poor to be useful.

Return a DataFrame with columns `id` (int, the rssd_id), `date` (pandas Timestamp at the
calendar quarter end) and `y` (float), one row per bank-quarter, sorted, with no missing `y`.

Sample inputs are in `sample_calls.csv` and `sample_attrs.csv` in this folder.
