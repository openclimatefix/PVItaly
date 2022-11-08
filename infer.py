import importlib
from pathlib import Path
from typing import Callable

import joblib  # type: ignore[import]
import numpy as np
import pytorch_lightning as pl
import torch
from ocf_datapipes.training.simple_pv import simple_pv_datapipe  # type: ignore[import]
from ocf_datapipes.utils.consts import BatchKey  # type: ignore[import]
from torch.utils.data import DataLoader
from tqdm import tqdm


def exp_factory(
    name: str = "experiments.e001.experiment_001",
) -> tuple[type[pl.LightningModule], Callable]:
    exp = importlib.import_module(name)
    return exp.Model, exp.batch_to_x


def infer(batch_limit: int) -> None:
    pv_data_pipeline = simple_pv_datapipe("inference/infer.yaml", tag="test")
    dl = DataLoader(pv_data_pipeline, batch_size=None, num_workers=0)
    batch = next(iter(dl))

    Model, batch_to_x = exp_factory()

    x = batch_to_x(batch)
    y = batch[BatchKey.pv][:, batch[BatchKey.pv_t0_idx] :, 0]
    input_length = x.shape[1]
    output_length = y.shape[1]

    ckpts = sorted(Path("lightning_logs/version_0/checkpoints").glob("*.ckpt"))
    ckpt = str(list(ckpts)[-1])
    model: pl.LightningModule = Model.load_from_checkpoint(  # type: ignore[assignment]
        ckpt, input_length=input_length, output_length=output_length
    )

    dump: dict[str, list] = {"batch": [], "x": [], "y": [], "y_hat": []}
    for i, batch in tqdm(enumerate(iter(dl)), total=batch_limit):
        if i >= batch_limit:
            break
        y_hat = model(batch)

        b = {
            k.name: v.detach().numpy()
            for k, v in batch.items()
            if type(v) == torch.Tensor
        }
        dump["batch"].append(b)
        dump["y_hat"].append(y_hat.detach().numpy())
    joblib.dump({k: np.array(v) for k, v in dump.items()}, "inference/dump.joblib")


if __name__ == "__main__":
    batch_limit = int((1.2 * 4 * 365 * 24 * 60) / (15 * 32))
    infer(batch_limit)
