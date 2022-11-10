import pandas as pd

# load file
df = pd.read_parquet('inference/dfl_001.parquet')

mae = df['abs_err'].mean()
rmse = df['squ_err'].mean()**0.5

print(f'{mae=}')
print(f'{rmse=}')

stats_system = df.groupby(['system_name']).mean()

stats = df.groupby(['system_name','step']).mean()
stats_abs_err = stats['abs_err'].reset_index()

import plotly.graph_objects as go

fig = go.Figure()
traces=[]
system_names = stats_abs_err['system_name'].unique()
for i in range(0,4):
    tmp = stats_abs_err[stats_abs_err['system_name'] == system_names[i]]
    trace = go.Scatter(x=tmp['step'],y=tmp['abs_err'],name=system_names[i])
    traces.append(trace)

fig.add_traces(traces)
fig.update_layout(
    title="MAE with time horizon per system",
    xaxis_title="Time horizon [minutes]",
    yaxis_title="MAE [k]",
)
fig.show(renderer='browser')





