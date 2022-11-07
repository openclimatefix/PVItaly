import os
import tempfile
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import typer
import xarray as xr
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def create_ds(path: Path) -> xr.Dataset:
    ds_t = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "cfVarName": "t",
                "typeOfLevel": "surface",
                "stepType": "instant",
            }
        },
    )
    ds_t = ds_t.t

    ds_dp = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "surface", "stepType": "avg"}
        },
    )
    ds_d = ds_dp.dlwrf
    ds_p = ds_dp.prate

    ds_v = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"cfVarName": "v", "typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
        },
    )
    ds_v = ds_v.isel(isobaricInhPa=0).v

    ds_u = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"cfVarName": "u", "typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
        },
    )
    ds_u = ds_u.isel(isobaricInhPa=0).u

    ds_merged = xr.merge([ds_t, ds_u, ds_v, ds_d, ds_p])

    return ds_merged


def download_file(
    url: str,
    headers: Optional[dict] = None,
    cookies: Optional[httpx.Cookies] = None,
) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as file:
        with httpx.stream(
            "GET", url, headers=headers, cookies=cookies, timeout=30
        ) as res:
            total = int(res.headers["Content-Length"])
            with tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                num_bytes_downloaded = res.num_bytes_downloaded
                for chunk in res.iter_bytes():
                    file.write(chunk)
                    progress.update(res.num_bytes_downloaded - num_bytes_downloaded)
                    num_bytes_downloaded = res.num_bytes_downloaded
    return file.name


def build_url(date: datetime, fc: int) -> str:
    base_url = "https://rda.ucar.edu/data/ds084.1"
    ymd = date.strftime("%Y%m%d")
    return f"{base_url}/{date.year}/{ymd}/gfs.0p25.{ymd}{date.hour:02d}.f{fc:03d}.grib2"


class UcarDownload:
    def __init__(self) -> None:
        self.cookies = self.auth()

    def auth(self) -> httpx.Cookies:
        auth_url = "https://rda.ucar.edu/cgi-bin/login"
        auth_data = {
            "email": os.environ["UCAR_EMAIL"],
            "passwd": os.environ["UCAR_PASS"],
            "action": "login",
        }
        res = httpx.post(auth_url, data=auth_data)
        assert res.status_code == 200
        return res.cookies

    def download_range(
        self, start_date: datetime, end_date: datetime
    ) -> Iterable[tuple[datetime, int, Path]]:
        date = start_date
        delta = timedelta(hours=6)

        while date < end_date:
            for fc in [3, 6]:
                url = build_url(date, fc)
                try:
                    file = Path(download_file(url, cookies=self.cookies))
                    yield date, fc, file
                except KeyError as e:
                    print(f"Failed for {date=}, {fc=} with {e}")
            date += delta


def main(
    start_date: datetime,
    end_date: datetime,
    dest_dir: Path,
) -> None:
    downloader = UcarDownload()
    files = downloader.download_range(start_date, end_date)
    for date, fc, path in files:
        ds = create_ds(path)
        ymdh = date.strftime("%Y%m%d_%H")
        fname = f"{ymdh}_f{fc:03d}"
        ds.to_zarr(dest_dir / fname, mode="w")
        print(f"Saved zarr to 'dest/{fname}'")
        path.unlink()


if __name__ == "__main__":
    typer.run(main)
