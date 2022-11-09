# PVItaly
Forecast for PV energy systems

## Data downloading
Create a `.env` file with your [UCAR](https://rda.ucar.edu/datasets/ds084.1/) login details:
```bash
UCAR_EMAIL='you@email.com'
UCAR_PASS='your-password'
```

Then:
```bash
sudo apt-get install libeccodes-dev libeccodes-tools
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

Then this will output individual zarrs for each timestamp and time-step into the provided output dir:
```bash
python scripts/download.py 2021-01-01 2021-02-01 zarrs/
```

These zarrs can then be merged:
```bash
python scripts/merge.py zarrs/ merged/
```

## Running inference
Install requirements:
```bash
pip install -r requirements-ml.txt
```

Check the help on the infer script:
```bash
python infer.py --help
```

It will output a wide- and a long- DF to the specified output_dir.
