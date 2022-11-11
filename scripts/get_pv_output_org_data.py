"""

Get all the PV data from Italy
This assumes the metadata data has already been collected


"""
import logging
import os

import pandas as pd
from pvoutput import *

# setup
BASE_PATH = os.path.expanduser("./data/")

OUTPUT_TIMESERIES_FILENAME = os.path.join(BASE_PATH, "Italy_PV_timeseries_batch.hdf")

INPUT_PV_LIST_FILENAME = os.path.join(BASE_PATH, "PVOutput_Italy_systems.csv")
METADATA_FILENAME = os.path.join(BASE_PATH, "PVOutput_Italy_systems_metadata.csv")

START_DATE = pd.Timestamp("1950-01-01")
END_DATE = pd.Timestamp("2022-10-01")

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)


logger = logging.getLogger(__name__)

# get metadata
pv_systems = pd.read_csv(INPUT_PV_LIST_FILENAME, index_col="system_id")
pv_metadata = pd.read_csv(METADATA_FILENAME, index_col="system_id")
pv_systems_joined = pv_systems.join(
    pv_metadata[["status_interval_minutes", "install_date"]], how="left"
)

# Filter 'bad' systems
pv_systems_filtered = pv_systems_joined.query("status_interval_minutes <= 60")
pv_systems_filtered = pv_systems_filtered.dropna(subset=["latitude", "longitude"])

# sort by Capacity
pv_systems_filtered.sort_values("system_DC_capacity_W", ascending=False, inplace=True)

bad_systems = [25703]
pv_systems_filtered = pv_systems_filtered[~pv_systems_filtered.index.isin(bad_systems)]

# download load data

# need to set API_KEY and SYSTEM_ID
pv = PVOutput()
logger.info("\n******* STARTING UP ************")

try:
    pv.download_multiple_systems_to_disk(
        system_ids=pv_systems_filtered.index,
        start_date=START_DATE,
        end_date=END_DATE,
        output_filename=OUTPUT_TIMESERIES_FILENAME,
        timezone="Europe/Rome",
    )
except Exception as e:
    logger.exception("Exception! %s", e)
    raise
