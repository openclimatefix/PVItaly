"""

Idea is to make an image of when the pv systems are available for the italy data
This is data taken from pvoutput.org

Data is currently here on GCP:
- https://storage.cloud.google.com/solar-pv-nowcasting-data/PV/PVOutput.org/Italy/Italy_PV_timeseries_batch.hdf
Not public

"""

import pandas as pd
import os

data_filename = './data/Italy_PV_timeseries_batch_with_SV.hdf'

os.path.exists(data_filename)

with pd.HDFStore(data_filename) as hdf:
    keys = hdf.keys()

# keys.pop('/statistics')
keys.pop(0)
# keys.pop('/missing_dates')
keys.pop(0)


df_all = None
for i, key in enumerate(keys):
    print(f'{key}: {i} out of {len(keys)}')
    df = pd.read_hdf(data_filename, key)
    df = df.resample('5T').mean()
    if len(df) > 288*10: # more than 10 days of data

        print(len(df))

        name = key.split('/')[-1]

        # resample to 15 minutes
        df = df.resample('15T').mean()

        # take difference so its ~power not cumulative
        if 'cumulative_energy_gen_Wh' in df.columns:
            df.rename(columns={'cumulative_energy_gen_Wh':name},inplace=True)
            df = df.diff()
            df[df < 0] = 0
        elif 'energy_gen_Wh' in df.columns:
            df.rename(columns={'energy_gen_Wh': name}, inplace=True)
        else:
            raise Exception('Data does not contain "cumulative_energy_gen_Wh" or "energy_gen_Wh"')

        if df_all is None:
            df_all = df[[name]]
        else:
            df_all = df_all.join(df[[name]],how='outer')

# save to .parquet, this means we can skip the step above if we need to
df_all.to_parquet('PVItaly.parquet')

# make netcdf
xr_all = df_all.to_xarray()
xr_all = xr_all.rename({'index':'datetime'})
xr_all.to_netcdf('PVItaly.netcdf',engine='h5netcdf')
