import tqdm
import numpy as np
import pandas as pd


def region_impact_matrix(data: dict, variable, aggregation='rel_diff', t_max=365):
    regions = sorted([k[0] for k in list(data.keys())])
    res = pd.DataFrame(columns=regions, index=regions)
    res.index.name = 'r_direct'
    res.columns.name = 'r_indirect'
    for (r, _), dataset in tqdm.tqdm(data.items()):
        aggregate = dataset[variable].isel(time=np.arange(t_max)).sum(dim='time').groupby('agent_region').sum()
        baseline = dataset[variable].isel(time=0).groupby('agent_region').sum() * t_max
        row = aggregate / baseline
        res.loc[r, row.agent_region.values] = row.values
    return res
