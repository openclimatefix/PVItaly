# PVItaly
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/PVItaly/commits?author=peterdudfield" title="Code">💻</a></td>
      <td align="center"><a href="https://rdrn.me/"><img src="https://avatars.githubusercontent.com/u/19817302?v=4?s=100" width="100px;" alt="Chris Arderne"/><br /><sub><b>Chris Arderne</b></sub></a><br /><a href="https://github.com/openclimatefix/PVItaly/commits?author=carderne" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!