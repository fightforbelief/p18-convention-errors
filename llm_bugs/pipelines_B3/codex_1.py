import numpy as np
import pandas as pd


MIDWEST = frozenset(('IL', 'IN', 'IA', 'MI', 'MN', 'OH', 'WI'))
HOLDOUT = 4
MIN_TRAIN_OBS = 8
MIN_TRAIN_COVERAGE = 0.8
MAX_GAP = 2


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame,
                history: pd.DataFrame) -> pd.DataFrame:
    """Build one contiguous Midwest run per bank with four observed test targets.

    History is authoritative (inclusive spell ends); the directory is a proxy
    only for banks with no history. Use the latest Midwest run containing a
    filing, bounded by its first and last filings. Average duplicate ratios.
    Require eight observed training quarters, 80% training coverage, and no
    unfillable gaps. Carry values forward at most two quarters, never backward.
    """
    reports = calls.groupby(['rssd_id', 'report_date'], as_index=False)['value'].mean()
    reports['value'] = reports['value'].replace([np.inf, -np.inf], np.nan)
    dates = pd.to_datetime(reports['report_date'], format='%Y%m%d')
    reports['quarter'] = dates.dt.to_period('Q').astype('int64')
    bounds = reports.groupby('rssd_id')['quarter'].agg(['min', 'max'])
    sizes = (bounds['max'] - bounds['min'] + 1).to_numpy()
    offsets = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    grid = pd.DataFrame({
        'rssd_id': np.repeat(bounds.index.to_numpy(dtype='int64'), sizes),
        'quarter': np.repeat(bounds['min'].to_numpy(), sizes) + offsets,
    })
    dates = pd.PeriodIndex.from_ordinals(grid['quarter'], freq='Q').to_timestamp(how='end').normalize()
    grid['date'] = dates
    grid['report_date'] = (dates.year * 10000 + dates.month * 100 + dates.day).astype('int64')
    grid = grid.merge(reports[['rssd_id', 'report_date', 'value']].assign(filed=True),
                      on=['rssd_id', 'report_date'], how='left')

    spells = history[['rssd_id', 'start_date', 'end_date', 'state']].copy()
    spells['rssd_id'] = spells['rssd_id'].astype('int64')
    spells['start_date'] = spells['start_date'].astype('int64')
    spells['end_date'] = pd.to_numeric(spells['end_date']).fillna(99991231)
    grid = pd.merge_asof(grid.sort_values('report_date'), spells.sort_values('start_date'),
                         left_on='report_date', right_on='start_date', by='rssd_id')
    fallback = grid['rssd_id'].map(directory.drop_duplicates('rssd_id').set_index('rssd_id')['state'])
    historical = grid['rssd_id'].isin(spells['rssd_id'])
    midwest = (grid['state'].isin(MIDWEST) & grid['report_date'].le(grid['end_date'])) | (
        ~historical & fallback.isin(MIDWEST))
    grid = grid.loc[midwest].sort_values(['rssd_id', 'quarter']).copy()
    grid['run'] = grid.groupby('rssd_id')['quarter'].diff().ne(1).cumsum()
    filed = grid.loc[grid['filed'].eq(True)]
    spans = filed.groupby('run')['quarter'].agg(first='min', last='max')
    latest = filed.drop_duplicates('rssd_id', keep='last')['run']
    grid = grid.join(spans.loc[latest], on='run')
    grid = grid.loc[grid['quarter'].between(grid['first'], grid['last'])].copy()

    training = grid['quarter'].le(grid['last'] - HOLDOUT)
    observed = grid['value'].notna()
    train_count = (training & observed).groupby(grid['run']).transform('sum')
    train_size = grid['last'] - grid['first'] + 1 - HOLDOUT
    valid_test = (training | observed).groupby(grid['run']).transform('all')
    grid['value'] = grid.groupby('run')['value'].ffill(limit=MAX_GAP)
    complete = grid['value'].notna().groupby(grid['run']).transform('all')
    keep = (train_count.ge(MIN_TRAIN_OBS) & train_count.ge(MIN_TRAIN_COVERAGE * train_size)
            & valid_test & complete)
    return (grid.loc[keep, ['rssd_id', 'date', 'value']]
            .rename(columns={'rssd_id': 'id', 'value': 'y'})
            .astype({'y': 'float64'}).reset_index(drop=True))
