from pathlib import Path

import typer
import xarray as xr
from tqdm import tqdm


def merge_zarrs(path: Path, out_path: Path) -> None:
    print(f"Merging zarrs from '{path}'")
    zarrs_f3 = []
    zarrs_f6 = []
    for d in tqdm(sorted(path.iterdir())):
        ds = xr.open_zarr(d)
        if "f003" in d.stem:
            zarrs_f3.append(ds)
        else:
            zarrs_f6.append(ds)

    ds_f3 = xr.concat(zarrs_f3, dim="time")
    ds_f6 = xr.concat(zarrs_f6, dim="time")
    ds_out = xr.concat([ds_f3, ds_f6], dim="step")
    print(f"Saving single zarr to '{out_path}'")
    ds_out.to_zarr(
        out_path,
        encoding={
            "time": {"dtype": "int64", "_FillValue": -1},
            "valid_time": {"dtype": "int64", "_FillValue": -1},
        },
    )


def main(path_in: Path, path_out: Path) -> None:
    merge_zarrs(path_out, path_out)


if __name__ == "__main__":
    typer.run(main)
