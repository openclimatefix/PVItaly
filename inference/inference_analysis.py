import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta


# load file
df = pd.read_parquet("inference/dfl_001.parquet")
dfw = pd.read_parquet("inference/dfw_001.parquet")

# option to save to csv
# dfw.to_csv("inference/dfw_001.csv")

colour_hex = ["#ffd053", "#ff9736", "#7bcdf3", "#4c9a8e", "#14120e"]


# get mae and rmse
mae = df["abs_err"].mean()
rmse = df["squ_err"].mean() ** 0.5

print("Overall MAE and MSE")
print(f"{mae=}")
print(f"{rmse=}")


# now lets group the data by system
stats_system = df.groupby(["system_name"]).mean()
print("MAE by system")
print(stats_system["abs_err"])

# now lets group by step and suste,m
stats = df.groupby(["system_name", "step"]).mean()
stats_abs_err = stats["abs_err"].reset_index()

# lets make a plit
fig = go.Figure()
traces = []
system_names = stats_abs_err["system_name"].unique()
for i in range(0, 4):
    tmp = stats_abs_err[stats_abs_err["system_name"] == system_names[i]]
    trace = go.Scatter(
        x=tmp["step"], y=tmp["abs_err"], name=system_names[i], line=dict(color=colour_hex[i])
    )
    traces.append(trace)

fig.add_traces(traces)
fig.update_layout(
    title="MAE with time horizon per system",
    xaxis_title="Time horizon [minutes]",
    yaxis_title="MAE [k]",
)
fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=60))
fig.show(renderer="browser")


#
# # would be good to explore if there are the corretc amount of data in our forecasts
#
# dfw['date'] = dfw['datetime'].dt.date
# df_day = dfw.groupby('date').count()
# df_day_inccomplete = df_day[df_day['datetime'] != 384]
# print(df_day_inccomplete)
# # problems with '2022-08-20' to '2022-09-07'\
#
# # lets check in the data we were given
# import xarray as xr
# sw_xr = xr.open_dataset('experiments/e004/SV_Italy_PV_timeseries_batch.netcdf', engine='h5netcdf')
# sw_df = sw_xr.to_pandas()
# sw_df['date'] = sw_df.index.date
# sw_day_inccomplete = sw_df.groupby('date').count()
#
# # general feeling is that the data missing in the forecast,
# # is the same data missing in input data
# # e.g. '2022-07-17' doesnt have any daata in it, and we have no forecasts either


# plot good predictions

datetimes = [
    datetime(2022, 6, 5, 11, 30),
    datetime(2022, 5, 20, 11),
    datetime(2022, 1, 5, 12),
    datetime(2022, 1, 4, 12),
    datetime(2022, 1, 9, 12),
]
system_ids = [2, 1, 3, 0, 1, 3]
ylimits = [5000, 5000, 2000, 5000, 5000, 2000]


for i in range(0, 6):
    fig = go.Figure()
    traces = []

    day = np.random.randint(0, 30)

    d = datetimes[i]
    d = d.replace(day=day)
    system_id = system_ids[i]

    dfw_temp = dfw[dfw["datetime"] == d].iloc[system_id]
    pred = dfw_temp[17:34]
    true1 = dfw_temp[0:17]

    pred.index = [d + timedelta(minutes=15 * i) for i in range(0, 17)]
    true1.index = [d + timedelta(minutes=15 * i) for i in range(0, 17)]

    dfw_temp = dfw[dfw["datetime"] == d - timedelta(hours=4)].iloc[system_id]
    true2 = dfw_temp[0:17]
    true2.index = [d - timedelta(hours=4) + timedelta(minutes=15 * i) for i in range(0, 17)]
    true = true2.append(true1)

    name = dfw_temp.system_name

    trace = go.Scatter(x=true.index, y=true, name="truth", line=dict(color=colour_hex[0]))
    traces.append(trace)

    trace = go.Scatter(x=pred.index, y=pred, name="forecast", line=dict(color=colour_hex[1]))
    traces.append(trace)

    fig.add_traces(traces)

    fig.update_layout(
        title=f"Forecast and Prediction - {name} ",
        xaxis_title="Time",
        yaxis_title="Generation [W]",
    )
    fig.update(layout_yaxis_range=[0, ylimits[i]])

    fig.show(renderer="browser")
