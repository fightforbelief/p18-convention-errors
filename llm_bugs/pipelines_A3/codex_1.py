import numpy as np
import pandas as pd


MIDWEST = frozenset(('IL', 'IN', 'IA', 'MI', 'MN', 'OH', 'WI'))
QUARTER_ENDS = np.array([331, 630, 930, 1231])


def build_panel(calls: pd.DataFrame, directory: pd.DataFrame,
                history: pd.DataFrame) -> pd.DataFrame:
    """Return observed Midwest bank-quarters for banks meeting the coverage rule."""
    report_dates = calls['report_date'].to_numpy(dtype=np.int64)
    reports = calls[['rssd_id', 'report_date', 'value']].copy()
    reports['rssd_id'] = reports['rssd_id'].astype(np.int64)
    reports['report_date'] = report_dates
    reports['quarter'] = (report_dates // 10000 * 4
                          + (report_dates // 100 % 100 - 1) // 3)
    spans = reports.groupby('rssd_id')['quarter'].agg(['min', 'max'])
    lengths = (spans['max'] - spans['min'] + 1).to_numpy(dtype=np.int64)
    offsets = np.cumsum(lengths) - lengths
    quarters = (np.repeat(spans['min'].to_numpy(), lengths)
                + np.arange(lengths.sum()) - np.repeat(offsets, lengths))
    grid = pd.DataFrame({
        'rssd_id': np.repeat(spans.index.to_numpy(), lengths),
        'report_date': quarters // 4 * 10000 + QUARTER_ENDS[quarters % 4],
    })

    spells = history[['rssd_id', 'start_date', 'end_date', 'state']].copy()
    spells['rssd_id'] = spells['rssd_id'].astype(grid['rssd_id'].dtype)
    spells['start_date'] = spells['start_date'].astype(np.int64)
    spells['end_date'] = pd.to_numeric(spells['end_date']).fillna(99991231)
    located = pd.merge_asof(
        grid.sort_values('report_date'), spells.sort_values('start_date'),
        left_on='report_date', right_on='start_date', by='rssd_id',
    )
    eligible = located.loc[
        located['state'].isin(MIDWEST)
        & (located['report_date'] <= located['end_date']),
        ['rssd_id', 'report_date'],
    ]
    panel = eligible.merge(
        reports.drop(columns='quarter'), on=['rssd_id', 'report_date'],
        how='left', validate='one_to_one',
    )
    coverage = panel.groupby('rssd_id')['value'].agg(['size', 'count'])
    retained = coverage.index[5 * coverage['count'] >= 4 * coverage['size']]
    panel = panel.loc[panel['rssd_id'].isin(retained) & panel['value'].notna()]
    panel = panel.sort_values(['rssd_id', 'report_date']).reset_index(drop=True)
    return pd.DataFrame({
        'id': panel['rssd_id'].astype(np.int64),
        'date': pd.to_datetime(panel['report_date'].astype(str),
                               format='%Y%m%d').astype('datetime64[ns]'),
        'y': panel['value'].astype(float),
    })
