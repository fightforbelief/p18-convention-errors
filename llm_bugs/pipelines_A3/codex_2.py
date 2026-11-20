import numpy as np
import pandas as pd


MIDWEST = frozenset(('IL', 'IN', 'IA', 'MI', 'MN', 'OH', 'WI'))


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame,
                history: pd.DataFrame) -> pd.DataFrame:
    """Build the panel using historical headquarters spells, with inclusive ends."""
    reports = calls[['rssd_id', 'report_date', 'value']].copy()
    reports['rssd_id'] = reports['rssd_id'].astype('int64')
    reports['report_date'] = reports['report_date'].astype('int64')
    report_dates = reports['report_date'].to_numpy()
    reports['quarter'] = (report_dates // 10000) * 4 + (report_dates // 100 % 100 // 3 - 1)

    spans = reports.groupby('rssd_id', sort=False)['quarter'].agg(['min', 'max'])
    lengths = (spans['max'] - spans['min'] + 1).to_numpy()
    offsets = lengths.cumsum() - lengths
    quarters = np.repeat(spans['min'].to_numpy() - offsets, lengths) + np.arange(lengths.sum())
    year, quarter = np.divmod(quarters, 4)
    grid = pd.DataFrame({
        'rssd_id': np.repeat(spans.index.to_numpy(), lengths),
        'report_date': year * 10000 + np.array([331, 630, 930, 1231])[quarter],
    })

    spells = history[['rssd_id', 'start_date', 'end_date', 'state']].copy()
    spells['rssd_id'] = spells['rssd_id'].astype('int64')
    spells['start_date'] = spells['start_date'].astype('int64')
    spells['end_date'] = pd.to_numeric(spells['end_date'])
    located = pd.merge_asof(
        grid.sort_values('report_date'), spells.sort_values('start_date', kind='stable'),
        left_on='report_date', right_on='start_date', by='rssd_id',
    )
    midwest = located.loc[
        located['state'].isin(MIDWEST)
        & (located['end_date'].isna() | located['report_date'].le(located['end_date'])),
        ['rssd_id', 'report_date'],
    ]
    panel = midwest.merge(reports[['rssd_id', 'report_date', 'value']],
                          how='left', on=['rssd_id', 'report_date'], validate='one_to_one')
    counts = panel.groupby('rssd_id', sort=False)['value'].agg(['size', 'count'])
    eligible = counts.index[5 * counts['count'] >= 4 * counts['size']]
    panel = panel.loc[panel['rssd_id'].isin(eligible) & panel['value'].notna()]
    panel = panel.sort_values(['rssd_id', 'report_date']).reset_index(drop=True)
    return pd.DataFrame({
        'id': panel['rssd_id'],
        'date': pd.to_datetime(panel['report_date'], format='%Y%m%d').astype('datetime64[ns]'),
        'y': panel['value'].astype(float),
    })
