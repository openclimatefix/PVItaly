import pandas as pd
import xarray as xr
import numpy as np

# load file
dfw = pd.read_parquet("dfw_000.parquet")
dfl = pd.read_parquet("dfl_000.parquet")

dfw.sort_values(by=["datetime", "system"])

# load metadata
metadata = pd.read_csv("experiments/e004/SV_Italy_systems_metadata.csv")
metadata["system_name"] = metadata["name"]

pv_power_ds = xr.open_dataset(
    "experiments/e004/SV_Italy_PV_timeseries_batch.netcdf", engine="h5netcdf"
)
pv_capacity_watt_power = pv_power_ds.max().to_pandas().astype(np.float32)

# add system name to output file
dfw["system_name"] = ""
dfl["system_name"] = ""
for i, row in metadata.iterrows():
    print(i, row)

    # add system name
    mask = dfw.index[dfw["system"] == i]
    dfw.loc[mask, "system_name"] = row.system_name

    # renormalize
    capacity = pv_capacity_watt_power.iloc[i]
    dfw.loc[mask, dfw.columns[:-2]] *= capacity

    # add system name - dfl
    mask = dfl.index[dfl["system"] == i]
    dfl.loc[mask, "system_name"] = row.system_name

    # renormalize
    capacity = pv_capacity_watt_power.iloc[i]
    columns = ["true", "pred", "abs_err"]
    dfl.loc[mask, columns] *= capacity
    dfl.loc[mask, "squ_err"] *= capacity ** 2

# save
dfw.to_parquet("inference/dfw_001.parquet")
dfl.to_parquet("inference/dfl_001.parquet")
dfw.sort_values(by=["datetime", "system_name"])


print(dfw.head(50))


true = pv_power_ds["2"].to_pandas()
true = true[true.index > "2022-01-01"]
