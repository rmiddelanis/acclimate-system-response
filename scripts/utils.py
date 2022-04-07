from netCDF4 import Dataset
import numpy as np
import tqdm


EORA_CHN_USA_REGIONS = ['AFG', 'ALB', 'DZA', 'AND', 'AGO', 'ATG', 'ARG', 'ARM', 'ABW',
                        'AUS', 'AUT', 'AZE', 'BHS', 'BHR', 'BGD', 'BRB', 'BLR', 'BEL',
                        'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BIH', 'BWA', 'BRA', 'VGB',
                        'BRN', 'BGR', 'BFA', 'BDI', 'KHM', 'CMR', 'CAN', 'CPV', 'CYM',
                        'CAF', 'TCD', 'CHL', 'CN.AH', 'CN.BJ', 'CN.CQ', 'CN.FJ', 'CN.GS',
                        'CN.GD', 'CN.GX', 'CN.GZ', 'CN.HA', 'CN.HB', 'CN.HL', 'CN.HE',
                        'CN.HU', 'CN.HN', 'CN.JS', 'CN.JX', 'CN.JL', 'CN.LN', 'CN.NM',
                        'CN.NX', 'CN.QH', 'CN.SA', 'CN.SD', 'CN.SH', 'CN.SX', 'CN.SC',
                        'CN.TJ', 'CN.XJ', 'CN.XZ', 'CN.YN', 'CN.ZJ', 'COL', 'COG', 'CRI',
                        'HRV', 'CUB', 'CYP', 'CZE', 'CIV', 'PRK', 'COD', 'DNK', 'DJI',
                        'DOM', 'ECU', 'EGY', 'SLV', 'ERI', 'EST', 'ETH', 'FJI', 'FIN',
                        'FRA', 'PYF', 'GAB', 'GMB', 'GEO', 'DEU', 'GHA', 'GRC', 'GRL',
                        'GTM', 'GIN', 'GUY', 'HTI', 'HND', 'HKG', 'HUN', 'ISL', 'IND',
                        'IDN', 'IRN', 'IRQ', 'IRL', 'ISR', 'ITA', 'JAM', 'JPN', 'JOR',
                        'KAZ', 'KEN', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR',
                        'LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MDG', 'MWI', 'MYS', 'MDV',
                        'MLI', 'MLT', 'MRT', 'MUS', 'MEX', 'MCO', 'MNG', 'MNE', 'MAR',
                        'MOZ', 'MMR', 'NAM', 'NPL', 'NLD', 'ANT', 'NCL', 'NZL', 'NIC',
                        'NER', 'NGA', 'NOR', 'PSE', 'OMN', 'PAK', 'PAN', 'PNG', 'PRY',
                        'PER', 'PHL', 'POL', 'PRT', 'QAT', 'KOR', 'MDA', 'ROU', 'RUS',
                        'RWA', 'WSM', 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC', 'SLE',
                        'SGP', 'SVK', 'SVN', 'SOM', 'ZAF', 'SDS', 'ESP', 'LKA', 'SDN',
                        'SUR', 'SWZ', 'SWE', 'CHE', 'SYR', 'TWN', 'TJK', 'THA', 'MKD',
                        'TGO', 'TTO', 'TUN', 'TUR', 'TKM', 'UGA', 'UKR', 'ARE', 'GBR',
                        'TZA', 'US.AL', 'US.AK', 'US.AZ', 'US.AR', 'US.CA', 'US.CO',
                        'US.CT', 'US.DE', 'US.DC', 'US.FL', 'US.GA', 'US.HI', 'US.ID',
                        'US.IL', 'US.IN', 'US.IA', 'US.KS', 'US.KY', 'US.LA', 'US.ME',
                        'US.MD', 'US.MA', 'US.MI', 'US.MN', 'US.MS', 'US.MO', 'US.MT',
                        'US.NE', 'US.NV', 'US.NH', 'US.NJ', 'US.NM', 'US.NY', 'US.NC',
                        'US.ND', 'US.OH', 'US.OK', 'US.OR', 'US.PA', 'US.RI', 'US.SC',
                        'US.SD', 'US.TN', 'US.TX', 'US.UT', 'US.VT', 'US.VA', 'US.WA',
                        'US.WV', 'US.WI', 'US.WY', 'URY', 'UZB', 'VUT', 'VEN', 'VNM',
                        'YEM', 'ZMB', 'ZWE']

EORA_SECTORS = ['FCON', 'AGRI', 'FISH', 'MINQ', 'FOOD', 'TEXL', 'WOOD', 'OILC',
                'METL', 'MACH', 'TREQ', 'MANU', 'RECY', 'ELWA', 'CONS', 'REPA',
                'WHOT', 'RETT', 'GAST', 'TRAN', 'COMM', 'FINC', 'ADMI', 'EDHE',
                'HOUS', 'OTHE', 'REXI']


def write_ncdf_output(_forcing_curves, _sector_list, _out_file, max_len=365 * 3):
    max_len = min(max([len(curve) for curve in _forcing_curves.values()]), max_len)
    with Dataset(_out_file, 'w') as outfile:
        timedim = outfile.createDimension("time")
        timevar = outfile.createVariable("time", "f", "time")
        timevar[:] = np.arange(0, max_len)
        timevar.units = "days since 2009-01-01"
        timevar.calendar = "standard"

        regions = list(_forcing_curves.keys())
        regiondim = outfile.createDimension("region", len(regions))
        regionvar = outfile.createVariable("region", str, "region")
        for i, r in enumerate(regions):
            regionvar[i] = r

        sectors = _sector_list
        sectordim = outfile.createDimension("sector", len(sectors))
        sectorvar = outfile.createVariable("sector", str, "sector")
        for i, s in enumerate(sectors):
            sectorvar[i] = s

        forcing = outfile.createVariable("forcing", "f", ("time", "sector", "region"), zlib=True, complevel=7,
                                         fill_value=0)
        forcing_data = np.zeros((max_len, len(sectors), len(regions)))

        for reg, forcing_ts in tqdm.tqdm(_forcing_curves.items()):
            for sec in sectors:
                forcing_data[:len(forcing_ts), sectors.index(sec), regions.index(reg)] = forcing_ts
        forcing_data[forcing_data < 0] = 0
        forcing[:] = forcing_data
