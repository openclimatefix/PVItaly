import importlib
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import typer
from ocf_datapipes.training.simple_pv import simple_pv_datapipe  # type: ignore[import]
from ocf_datapipes.utils.consts import BatchKey  # type: ignore[import]
from torch.utils.data import DataLoader
from tqdm import tqdm

Batch = dict[BatchKey, np.ndarray]


def exp_factory(module: str) -> tuple[type[pl.LightningModule], Callable]:
    exp = importlib.import_module(module)
    return exp.Model, exp.batch_to_x


def infer(  # type: ignore[no-any-unimported]
    config: Path, module: str, ckpt_path: Path, batches: int
) -> tuple[np.ndarray, Batch, int]:
    """Run inference and return y_hat and batches.

    Returns:
        y_hat: np.ndarray shape [num_preds, forward_time_steps]
        truth: dict of np.ndarray, shaped [num_preds, ...]
        pv_t0_idx: int
    """
    pv_data_pipeline = simple_pv_datapipe(config, tag="test")
    dl = DataLoader(pv_data_pipeline, batch_size=None, num_workers=0)

    Model, batch_to_x = exp_factory(module)

    batch = next(iter(dl))
    x = batch_to_x(batch)
    pv_t0_idx = batch[BatchKey.pv_t0_idx]
    y = batch[BatchKey.pv][:, pv_t0_idx:, 0]
    input_length = x.shape[1]
    output_length = y.shape[1]

    model: pl.LightningModule = Model.load_from_checkpoint(  # type: ignore[assignment]
        str(ckpt_path), input_length=input_length, output_length=output_length
    )

    all_batch = []
    all_y_hat = []
    for i, batch in tqdm(enumerate(iter(dl)), total=batches):
        if i >= batches:
            break

        y_hat = model(batch)

        b = {k: v.detach().numpy() for k, v in batch.items() if type(v) == torch.Tensor}
        all_y_hat.append(y_hat.detach().numpy())
        all_batch.append(b)

    y_hat = np.array(all_y_hat).reshape(-1, y_hat.shape[-1])
    batch = {
        key: np.squeeze(np.array([b[key] for b in all_batch]).reshape(y_hat.shape[0], -1))
        for key in all_batch[0].keys()
    }

    return y_hat, batch, pv_t0_idx


def post_process(  # type: ignore[no-any-unimported]
    y_hat: np.ndarray, batch: Batch, pv_t0_idx: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the y_hat and batch data to DataFrames.

    Returns
        dfw: wide-format DataFrame with true0-240, pred0-240, system, datetime columns
        dfl: long-format DataFrame, with columns system, datetime, step, true, abs_err, squ_err
    """
    data = np.hstack((batch[BatchKey.pv][:, pv_t0_idx:], y_hat))
    dfw = pd.DataFrame(data)
    r = range(y_hat.shape[-1])
    dfw.columns = pd.Index([f"true{x*15}" for x in r] + [f"pred{x*15}" for x in r])

    dfw = dfw.assign(
        system=batch[BatchKey.pv_system_row_number],
        datetime=[datetime.utcfromtimestamp(ts[pv_t0_idx]) for ts in batch[BatchKey.pv_time_utc]],
    )

    dfw = dfw.sort_values(by=["datetime", "system"])
    dfw = dfw.drop_duplicates(subset=["system", "datetime"])

    dfl = pd.wide_to_long(
        dfw,
        stubnames=["true", "pred"],
        i=["system", "datetime"],
        j="step",
    ).reset_index()
    dfl = dfl.assign(
        abs_err=(dfl["pred"] - dfl["true"]).abs(),
        squ_err=(dfl["pred"] - dfl["true"]) ** 2,
    )

    return dfw, dfl


def main(
    config: Path = Path("infer.yaml"),
    module: str = "experiments.e001.experiment_001",
    ckpt_dir: Path = Path("../lightning_logs/version_0/checkpoints"),
    batches: int = int((1.2 * 4 * 365 * 24 * 60) / (15 * 32)),
    output_dir: Path = Path(""),
    output_suffix: str = "000",
) -> None:
    ckpt_path = list(sorted(ckpt_dir.glob("*.ckpt")))[-1]
    y_hat, batch, pv_t0_idx = infer(config, module, ckpt_path, batches)
    dfw, dfl = post_process(y_hat, batch, pv_t0_idx)

    dfw_path = output_dir / f"dfw_{output_suffix}.parquet"
    dfl_path = output_dir / f"dfl_{output_suffix}.parquet"
    dfw.to_parquet(dfw_path)
    dfl.to_parquet(dfl_path)
    print(f"\nCSVs saved to {dfw_path} and {dfl_path}")


if __name__ == "__main__":
    main()
