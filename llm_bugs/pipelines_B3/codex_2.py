import numpy as np
import pandas as pd


MIDWEST = frozenset({'IL', 'IN', 'IA', 'MI', 'MN', 'OH', 'WI'})
HOLDOUT = 4
MIN_TRAIN_OBS = 8
MIN_TRAIN_COVERAGE = 0.8
MAX_GAP = 2


def _dates(values: pd.Series) -> pd.Series:
    encoded = values.replace('', pd.NA).astype('Int64').astype('string')
    return pd.to_datetime(encoded, format='%Y%m%d').astype('datetime64[ns]')


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame,
                history: pd.DataFrame) -> pd.DataFrame:
    """Build contiguous Midwest series with four observed holdout quarters.

    History dates are inclusive; the directory is a fallback only for banks
    without history. Use the latest Midwest run containing an observation,
    trimming after its last observation and restarting after its last long gap.
    Training must have at least eight observations and 80% coverage. Fill gaps
    of up to two quarters using only prior training observations.
    """
    reports = calls[['rssd_id', 'value']].copy()
    reports['quarter'] = _dates(calls['report_date']).dt.to_period('Q').astype('int64')
    reports['value'] = reports['value'].astype('float64').where(np.isfinite(reports['value']))
    reports = reports.groupby(['rssd_id', 'quarter'], as_index=False)['value'].mean()

    spans = reports.groupby('rssd_id')['quarter'].agg(['min', 'max'])
    lengths = (spans['max'] - spans['min'] + 1).to_numpy()
    offsets = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)
    quarters = np.repeat(spans['min'].to_numpy(), lengths) + offsets
    grid = pd.DataFrame({
        'rssd_id': np.repeat(spans.index.to_numpy(dtype='int64'), lengths),
        'quarter': quarters,
        'date': pd.PeriodIndex.from_ordinals(quarters, freq='Q').to_timestamp(
            how='end').normalize().astype('datetime64[ns]'),
    })

    spells = history[['rssd_id', 'state']].copy()
    spells['rssd_id'] = spells['rssd_id'].astype('int64')
    spells['start'] = _dates(history['start_date'])
    spells['end'] = _dates(history['end_date'])
    grid = pd.merge_asof(grid.sort_values('date'), spells.sort_values('start', kind='stable'),
                         left_on='date', right_on='start', by='rssd_id')
    historical = grid['rssd_id'].isin(spells['rssd_id'])
    active = grid['start'].notna() & (grid['end'].isna() | grid['date'].le(grid['end']))
    directory_states = directory.drop_duplicates('rssd_id', keep='last').set_index('rssd_id')['state']
    eligible = (active & grid['state'].isin(MIDWEST)) | (
        ~historical & grid['rssd_id'].map(directory_states).isin(MIDWEST)
    )
    grid = grid.loc[eligible, ['rssd_id', 'quarter', 'date']].merge(
        reports, on=['rssd_id', 'quarter'], how='left'
    ).sort_values(['rssd_id', 'quarter'])

    bank = grid['rssd_id']
    run = (bank.ne(bank.shift()) | grid['quarter'].diff().ne(1)).cumsum()
    observed = grid.loc[grid['value'].notna(), ['rssd_id', 'quarter']].assign(
        run=run[grid['value'].notna()]
    )
    latest = observed.drop_duplicates('rssd_id', keep='last').set_index('rssd_id')
    observed = observed.loc[observed['run'].eq(observed['rssd_id'].map(latest['run']))]
    gap = observed.groupby('rssd_id')['quarter'].diff()
    starts = observed.loc[gap.isna() | gap.gt(MAX_GAP + 1)].groupby('rssd_id')['quarter'].last()
    last = bank.map(latest['quarter'])
    grid = grid.loc[run.eq(bank.map(latest['run'])) & grid['quarter'].between(bank.map(starts), last)].copy()

    bank = grid['rssd_id']
    holdout = grid['quarter'].gt(bank.map(latest['quarter']) - HOLDOUT)
    present = grid['value'].notna()
    train_obs = (present & ~holdout).groupby(bank).transform('sum')
    train_size = (~holdout).groupby(bank).transform('sum')
    holdout_obs = (present & holdout).groupby(bank).transform('sum')
    keep = (holdout_obs.eq(HOLDOUT) & train_obs.ge(MIN_TRAIN_OBS)
            & train_obs.ge(MIN_TRAIN_COVERAGE * train_size))
    grid = grid.loc[keep].copy()
    grid['value'] = grid.groupby('rssd_id')['value'].ffill(limit=MAX_GAP)
    return grid.rename(columns={'rssd_id': 'id', 'value': 'y'})[['id', 'date', 'y']].reset_index(drop=True)
