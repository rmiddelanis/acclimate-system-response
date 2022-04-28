import tqdm
import numpy as np
import pandas as pd


def region_impact_matrix(data: dict, variable, aggregation='rel_diff', t_max=365):
    regions = sorted([k[0] for k in list(data.keys())])
    res = pd.DataFrame(data=np.zeros((len(regions), len(regions))), columns=regions, index=regions)
    res.index.name = 'r_direct'
    res.columns.name = 'r_indirect'
    for (r, _), dataset in tqdm.tqdm(data.items()):
        aggregate = dataset[variable].isel(time=np.arange(t_max)).sum(dim='time').groupby('agent_region').sum()
        baseline = dataset[variable].isel(time=0).groupby('agent_region').sum() * t_max
        row = (aggregate / baseline).sel(agent_region=regions)
        res.loc[r, row.agent_region.values] = row.values
    return res


def correlation_matrix(impact_matrix, how='direct'):
    res = pd.DataFrame(columns=impact_matrix.columns, index=impact_matrix.index)
    for i in tqdm.tqdm(range(len(res))):
        res.iloc[i, i] = np.corrcoef(impact_matrix.iloc[i, :], impact_matrix.iloc[:, i])[0, 1]
        for j in range(i + 1, len(res.columns)):
            if how == 'direct':
                res.iloc[i, j] = np.corrcoef(impact_matrix.iloc[[i, j], :])[0, 1]
            elif how == 'indirect':
                res.iloc[i, j] = np.corrcoef(impact_matrix.iloc[:, [i, j]])[0, 1]
            res.iloc[j, i] = res.iloc[i, j]
    return res
