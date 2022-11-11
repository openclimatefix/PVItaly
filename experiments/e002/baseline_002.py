import logging

import pandas as pd
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
from ocf_datapipes.training.simple_pv import simple_pv_datapipe
from ocf_datapipes.utils.consts import BatchKey
from plotly.subplots import make_subplots
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredLogError

baseline = "zero"
baseline = "persist"


logger = logging.getLogger(__name__)
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="pv-italy", name=f"exp-2-{baseline}")

# set up logging
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)

pv_data_pipeline = simple_pv_datapipe("experiments/e002/exp_002.yaml", tag="validation")

dl = DataLoader(dataset=pv_data_pipeline, batch_size=None)
pv_iter = iter(dl)

# get a batch
batch = next(pv_iter)


def plot(batch, y_hat):
    y = batch[BatchKey.pv][:, :, 0]
    time_y_hat = batch[BatchKey.pv_time_utc][:, batch[BatchKey.pv_t0_idx] :]
    time = batch[BatchKey.pv_time_utc]
    ids = batch[BatchKey.pv_id].detach().numpy()
    ids = [str(id) for id in ids]

    fig = make_subplots(rows=4, cols=4, subplot_titles=ids)

    for i in range(0, 16):
        row = i % 4 + 1
        col = i // 4 + 1
        time_i = pd.to_datetime(time[i], unit="s")
        time_y_hat_i = pd.to_datetime(time_y_hat[i], unit="s")
        trace_1 = go.Scatter(
            x=time_i, y=y[i].detach().numpy(), name="truth", line=dict(color="blue")
        )
        trace_2 = go.Scatter(
            x=time_y_hat_i,
            y=y_hat[i].detach().numpy(),
            name="predict",
            line=dict(color="red"),
        )

        fig.add_trace(trace_1, row=row, col=col)
        fig.add_trace(trace_2, row=row, col=col)

    fig.update_yaxes(range=[0, 1])

    try:
        fig.show(renderer="browser")
    except:
        pass


def batch_to_x(batch):

    pv_t0_idx = batch[BatchKey.pv_t0_idx]
    # nwp_t0_idx = batch[BatchKey.nwp_t0_idx]

    # history pv
    pv = batch[BatchKey.pv][:, :pv_t0_idx, 0].nan_to_num(0.0)

    return pv


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def _training_or_validation_step(self, x, return_model_outputs: bool = False, tag="train"):
        """
        batch: The batch data
        tag: either 'Train', 'Validation' , 'Test'
        """

        # put the batch data through the model
        y_hat = self(x)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        y = x[BatchKey.pv][:, x[BatchKey.pv_t0_idx] :, 0].nan_to_num(0.0)

        # calculate mse, mae
        mse_loss = torch.nn.MSELoss()(y_hat, y)
        # TODO, why does this work better? SOmething to do with sigmoid
        mae_loss = (y_hat - y).abs().mean()
        bce_loss = torch.nn.BCELoss()(y_hat, y)
        msle_loss = MeanSquaredLogError()(y_hat, y)

        loss = mse_loss + mae_loss + 0.1 * bce_loss
        if tag == "val":
            on_step = True
        else:
            on_step = True

        self.log(f"mse_{tag}", mse_loss, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"msle_{tag}", msle_loss, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"mae_{tag}", mae_loss, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"bce_{tag}", bce_loss, on_step=on_step, on_epoch=True, prog_bar=True)

        if return_model_outputs:
            return loss, y_hat
        else:
            return loss

    def training_step(self, x, batch_idx):

        if batch_idx < 1:
            plot(x, self(x))

        return self._training_or_validation_step(x, tag="tra")

    def validation_step(self, x, batch_idx):

        if batch_idx < 1:
            plot(x, self(x))

        return self._training_or_validation_step(x, tag="val")

    def predict_step(self, x, batch_idx, dataloader_idx=0):
        return x, self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        return optimizer

    def on_epoch_start(self):
        print("\n")


class Model(BaseModel):
    def __init__(self, input_length, output_length):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length

    def forward(self, x):
        x = batch_to_x(x)

        if baseline == "persist":
            out = x[:, -1:].repeat((1, self.output_length))
        elif baseline == "zero":
            out = torch.zeros((x.shape[0], self.output_length))

        out = x[:, -1:].repeat((1, self.output_length))
        # out = torch.zeros((x.shape[0], self.output_length))

        return out


# Initialize a trainer
trainer = Trainer(
    accelerator="auto",
    devices=None,
    max_epochs=1,
    limit_val_batches=50,
    log_every_n_steps=5,
    logger=wandb_logger,
)

x = batch_to_x(batch)
y = batch[BatchKey.pv][:, batch[BatchKey.pv_t0_idx] :, 0]
input_length = x.shape[1]
output_length = y.shape[1]


def main():
    # train_loader = DataLoader(pv_data_pipeline, batch_size=None, num_workers=0)
    val_loader = DataLoader(pv_data_pipeline, batch_size=None, num_workers=4)
    # predict_loader = DataLoader(pv_data_pipeline, batch_size=None, num_workers=0)

    model = Model(input_length=input_length, output_length=output_length)

    trainer.validate(model, dataloaders=val_loader)

    # predict model for some plots
    batch = next(pv_iter)
    y_hat = model(batch)

    plot(batch, y_hat)


if __name__ == "__main__":
    main()


# results
# 1. zeros
# mse = 0.047
# mae = 0.104

# 2. persistance
# mse = 0.024
# mae = 0.078
