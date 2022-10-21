"""

Idea is to make an image of when the pv systems are available for the italy data
This is data taken from pvoutput.org

Data is currently here on GCP:
- https://storage.cloud.google.com/solar-pv-nowcasting-data/PV/PVOutput.org/Italy/Italy_PV_timeseries_batch.hdf
Not public

"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

data_filename = './data/Italy_PV2.hdf'

os.path.exists(data_filename)

with pd.HDFStore(data_filename) as hdf:
    keys = hdf.keys()

# keys.pop('/statistics')
keys.pop(0)
# keys.pop('/missing_dates')
keys.pop(0)


df_all = None
for i, key in enumerate(keys[0:10]):
    print(f'{key}: {i} out of {len(keys)}')
    df = pd.read_hdf(data_filename, key)
    df = df.resample('5T').mean()
    if len(df) > 288*10: # more than 10 days of data

        print(len(df))

        name = key.split('/')[-1]

        df.rename(columns={'cumulative_energy_gen_Wh':name},inplace=True)

        if df_all is None:
            df_all = df[[name]]
        else:
            df_all = df_all.join(df[[name]],how='outer')

# take the diff so its power, not cumulative energy
df_all = df_all.diff()
df_all[df_all<0] = 0

# save to .parquet, this means we can skip the step above if we need to
df_all.to_parquet('PVItaly.parquet')

# load parquet file
df_all = pd.read_parquet('PVItaly.parquet')

# plot one pv system
system_id = df_all.columns[5]
N=1000
df_one = df_all[system_id]
df_one.dropna(inplace=True)
fig = px.scatter(x=df_one.index[-N:], y=df_one[-N:])
fig.update_xaxes(title="datetime")
fig.update_yaxes(title="power [Wh]")
fig.show(renderer='browser')

# plot all data
df_daily = df_all.resample('1D').mean()
nans = df_daily.isna()
img_bw = np.array(nans, dtype=np.uint8)
# example to 3 dims
img_bw = np.tile(img_bw[:, :, np.newaxis], (1, 1, 3))
# put time in first axis
img_bw = np.transpose(img_bw, (1, 0, 2))

x = [str(x) for x in list(df_daily.index)]

fig = go.Figure(data=go.Heatmap(z=img_bw[:,:,0], x=x, colorscale=["white", "black"]))
fig.show(renderer='browser')

# focus just in on 2020 and 2021
df_daily = df_all.resample('1D').mean()
df_daily = df_daily[df_daily.index > '2021-01-01']
nans = df_daily.isna()
img_bw = np.array(nans, dtype=np.uint8)
# example to 3 dims
img_bw = np.tile(img_bw[:, :, np.newaxis], (1, 1, 3))
# put time in first axis
img_bw = np.transpose(img_bw, (1, 0, 2))

x = [str(x) for x in list(df_daily.index)]
fig = go.Figure(data=go.Heatmap(z=img_bw[:,:,0], x=x, colorscale=["white", "black"]))
fig.show(renderer='browser')

# histogram of how man non nans
df_daily = df_all.resample('1D').mean()
nans_daily = (~df_daily.isna()).sum()
fig = go.Figure(data=[go.Histogram(x=nans_daily,nbinsx=50)])
fig.show(renderer='browser')


#

