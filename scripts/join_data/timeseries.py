"""
Idea is to load both PVoutput.org sites and SV sites.

After looking at the SV data, we need to filter out date by
1. Remove any negative value
2. Remove any values above 10^6
3. Remove any values above 5 stds
4. remove specific datetimes for site 4, 2021-06-09 to 2021-07-12

"""
import pandas as pd
from datetime import datetime
import os
import shutil


# Load data from
# https://storage.cloud.google.com/solar-pv-nowcasting-data/PV/PVOutput.org/Italy/Italy_PV_timeseries_batch.hdf

# Client data is stored here
# https://storage.cloud.google.com/solar-pv-nowcasting-data/PV/Siram Veolia

pv_filename = "./data/Italy_PV_timeseries_batch.hdf"
pv_filename_new = "./data/Italy_PV_timeseries_batch_with_SV.hdf"

with pd.HDFStore(pv_filename) as hdf:
    keys = hdf.keys()

if os.path.exists(pv_filename_new):
    os.remove(pv_filename_new)
shutil.copyfile(pv_filename, pv_filename_new)

data_example = pd.read_hdf(pv_filename, key=keys[3])
# This has the following columns
# ['cumulative_energy_gen_Wh', 'instantaneous_power_gen_W',
#        'temperature_C', 'voltage', 'datetime_of_API_request', 'query_date']

pv_filename_sv_1 = "./data/SV/PV_Siram Veolia_Ospedale di Belluno.xlsm"
pv_filename_sv_2 = "./data/SV/PV_Siram Veolia_Policlinico di Bari FV.xlsx"
pv_filename_sv_3 = "./data/SV/PV_Siram Veolia_Universita Ca Foscari - Campus.xlsx"

# site 1
pv_data_1 = pd.read_excel(pv_filename_sv_1)
pv_data_1["To Timestamp"] = pv_data_1["From Timestamp"] + pd.Timedelta(minutes=15)
pv_data_1.set_index("To Timestamp", drop=True, inplace=True)

pv_data_belluno_1 = pv_data_1[["Energia Elettrica ATTIVA prodotta fotovoltaico (kWh)"]]
pv_data_belluno_1["energy_gen_Wh"] = (
    pv_data_belluno_1["Energia Elettrica ATTIVA prodotta fotovoltaico (kWh)"] * 1000
)
pv_data_belluno_1 = pv_data_belluno_1[["energy_gen_Wh"]]

# site 2 and 3
pv_data_2 = pd.read_excel(pv_filename_sv_2)
pv_data_2["To Timestamp"] = pv_data_2["From Timestamp"] + pd.Timedelta(minutes=15)
pv_data_2.set_index("To Timestamp", drop=True, inplace=True)
pv_data_bari_1 = pv_data_2[["Energia Elettrica Prodotta da Impianto Fotovoltaico ASCLEPIOS (kWh)"]]
pv_data_bari_2 = pv_data_2[
    ["Energia Elettrica Prodotta da Impianto Fotovoltaico CONVITTO ALLIEVI (kWh)"]
]

pv_data_bari_1["energy_gen_Wh"] = (
    pv_data_bari_1["Energia Elettrica Prodotta da Impianto Fotovoltaico ASCLEPIOS (kWh)"] * 1000
)
pv_data_bari_2["energy_gen_Wh"] = (
    pv_data_bari_2["Energia Elettrica Prodotta da Impianto Fotovoltaico CONVITTO ALLIEVI (kWh)"]
    * 1000
)

pv_data_bari_1 = pv_data_bari_1[["energy_gen_Wh"]]
pv_data_bari_2 = pv_data_bari_2[["energy_gen_Wh"]]

# site 3
pv_data_3 = pd.read_excel(pv_filename_sv_3)
pv_data_3["To Timestamp"] = pv_data_3["From Timestamp"] + pd.Timedelta(minutes=15)
pv_data_3.set_index("To Timestamp", drop=True, inplace=True)

pv_data_forscari_1 = pv_data_3[["Energia Elettrica Fotovoltaico Edificio Beta (kWh)"]]
pv_data_forscari_1["energy_gen_Wh"] = (
    pv_data_forscari_1["Energia Elettrica Fotovoltaico Edificio Beta (kWh)"] * 1000
)
pv_data_forscari_1 = pv_data_forscari_1[["energy_gen_Wh"]]


data = {"1": pv_data_belluno_1, "2": pv_data_bari_1, "3": pv_data_bari_2, "4": pv_data_forscari_1}

# Filtering
# 1. Remove any negative value
for key, value in data.items():
    print(f"Formatting {key}")
    # 1. Remove any negative value
    value = value[value["energy_gen_Wh"] >= 0]

    # 2. Remove any values above 10^6
    value = value[value["energy_gen_Wh"] < 10 ** 6]

    # 3. Remove any values above 5 stds
    mean = value["energy_gen_Wh"].mean()
    std = value["energy_gen_Wh"].std()
    data[key] = value[value["energy_gen_Wh"] < mean + 5 * std]

    # remove specific things for site 4
    if key == "4":
        value = data[key]
        data[key] = value[
            (pd.to_datetime(value.index) < datetime(2021, 6, 9))
            | (pd.to_datetime(value.index) > datetime(2021, 7, 12))
        ]

with pd.HDFStore(pv_filename_new) as hdf:
    for key, value in data.items():
        print(f"adding {key} to {pv_filename_new}")
        hdf.append(value=value, key=f"/timeseries/{key}")


with pd.HDFStore(pv_filename_new) as hdf:
    keys = hdf.keys()
