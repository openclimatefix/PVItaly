import pandas as pd


pv_metadata_filename_new = "./data/SV_Italy_systems_metadata.csv"

site_1 = {
    "name": "Belluno Hospital",
    "system_DC_capacity_W": 35000,
    "latitude": "46.138667",
    "longitude": "12.202361",
    "system_id": 1,
}

site_2 = {
    "name": "Bari Hospital 1",
    "system_DC_capacity_W": 25000,
    "latitude": "41.114139",
    "longitude": "16.860861",
    "system_id": 2,
}

site_3 = {
    "name": "Bari Hospital 2",
    "system_DC_capacity_W": 20000,
    "latitude": "41.112778",
    "longitude": "16.857528",
    "system_id": 3,
}

site_4 = {
    "name": "Ca' Foscari University",
    "system_DC_capacity_W": 20000,
    "latitude": "45.4788382",
    "longitude": "12.254642",
    "system_id": 4,
}

sites = [site_1,site_2,site_3,site_4]
sites_df = pd.DataFrame(sites)

pv_metadata = pd.concat([sites_df])
pv_metadata.reset_index(drop=True,inplace=True)

pv_metadata.to_csv(pv_metadata_filename_new)