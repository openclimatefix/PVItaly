"""
Train on both SV and pvoutout.org sites
"""
import logging
import os

import pandas as pd
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ocf_datapipes.training.simple_pv import simple_pv_datapipe
from ocf_datapipes.utils.consts import BatchKey
from plotly.subplots import make_subplots
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredLogError
# from neptune.new.integrations.pytorch_lightning import NeptuneLogger
# import neptune.new as neptune

from pytorch_lightning.loggers import WandbLogger


logger = logging.getLogger(__name__)

# set up logging
# logging.basicConfig(
#     level=getattr(logging, "INFO"),
#     format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
# )
wandb_logger = WandbLogger(project="pv-italy",name='exp-3-pv-10')


pv_data_pipeline = simple_pv_datapipe("experiments/003/exp_003.yaml")
pv_data_pipeline_validation = simple_pv_datapipe("experiments/003/exp_003_validation.yaml", tag='validation')

# train_loader = DataLoader(dataset=pv_data_pipeline, batch_size=None)
# pv_iter = iter(train_loader)
#
# # get a batch
# batch = next(pv_iter)


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
            x=time_y_hat_i, y=y_hat[i].detach().numpy(), name="predict", line=dict(color="red")
        )

        fig.add_trace(trace_1, row=row, col=col)
        fig.add_trace(trace_2, row=row, col=col)

    fig.update_yaxes(range=[0, 1])

    fig.show(renderer="browser")


def batch_to_x(batch):

    pv_t0_idx = batch[BatchKey.pv_t0_idx]
    # nwp_t0_idx = batch[BatchKey.nwp_t0_idx]

    # x,y locations
    x_osgb = batch[BatchKey.pv_x_osgb] / 10 ** 6
    y_osgb = batch[BatchKey.pv_y_osgb] / 10 ** 6

    # add pv capacity
    pv_capacity = batch[BatchKey.pv_capacity_watt_power] / 1000

    # future sun
    sun = batch[BatchKey.pv_solar_elevation][:, pv_t0_idx:]
    sun_az = batch[BatchKey.pv_solar_azimuth][:, pv_t0_idx:]

    # future nwp
    # nwp = batch[BatchKey.nwp][:, nwp_t0_idx:]
    # nwp = nwp.reshape([nwp.shape[0], nwp.shape[1] * nwp.shape[2]])

    # fourier features on pv time
    pv_time_utc_fourier = batch[BatchKey.pv_time_utc_fourier]
    pv_time_utc_fourier = pv_time_utc_fourier.reshape(
        [pv_time_utc_fourier.shape[0], pv_time_utc_fourier.shape[1] * pv_time_utc_fourier.shape[2]]
    )

    # history pv
    pv = batch[BatchKey.pv][:, :pv_t0_idx, 0].nan_to_num(0.0)
    x = torch.concat(
        (pv, sun, sun_az, x_osgb, y_osgb, pv_capacity, pv_time_utc_fourier), dim=1
    )

    return x


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def _training_or_validation_step(self, x, return_model_outputs: bool = False, tag='train'):
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

        loss = mse_loss + mae_loss + 0.1*bce_loss

        on_step = True

        self.log(f"mse_{tag}", mse_loss, on_step=on_step, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"msle_{tag}", msle_loss, on_step=on_step, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"mae_{tag}", mae_loss, on_step=on_step, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"bce_{tag}", bce_loss, on_step=on_step, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"loss_{tag}", loss, on_step=on_step, on_epoch=True, prog_bar=True, sync_dist=True)

        if return_model_outputs:
            return loss, y_hat
        else:
            return loss

    def training_step(self, x, batch_idx):

        if batch_idx < 1:
            plot(x, self(x))

        return self._training_or_validation_step(x, tag='tra')

    def validation_step(self, x, batch_idx):

        if batch_idx < 1:
            plot(x, self(x))

        return self._training_or_validation_step(x,tag='val')

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
        features = 128
        self.input_length = input_length
        self.fc1 = nn.Linear(in_features=self.input_length, out_features=features)
        self.fc2 = nn.Linear(in_features=features, out_features=features)
        self.fc3 = nn.Linear(in_features=features, out_features=features)

        # move embedding up a layer (or top)
        self.pv_system_id_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.fc4 = nn.Linear(in_features=features + 16, out_features=output_length)

    def forward(self, x):

        id_embedding = self.pv_system_id_embedding(x[BatchKey.pv_system_row_number].type(torch.IntTensor))

        x = batch_to_x(x)

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        id_embedding = id_embedding.squeeze(1)
        out = torch.concat([out, id_embedding], dim=1)

        out = torch.sigmoid(self.fc4(out))
        # out = self.fc4(out)

        # TODO add mixture density model
        # TODO PVNET, colapse 7 days of data
        # TODO full shebang, Grid NWP, mutiple PV, Satellite

        return out

# Initialize a trainer
trainer = Trainer(
    accelerator="auto",
    devices=None,
    max_epochs=10,
    limit_train_batches=90,
    limit_val_batches=10,
    val_check_interval=90,
    reload_dataloaders_every_n_epochs=10,
    logger=wandb_logger,
    log_every_n_steps=5,
)

# x = batch_to_x(batch)
# y = batch[BatchKey.pv][:, batch[BatchKey.pv_t0_idx] :, 0]
# input_length = x.shape[1]
# output_length = y.shape[1]

print('******')
# print(f'{input_length}')
# print(f'{output_length}')
print('******')

input_length = 317
output_length = 17


def main():
    train_loader = DataLoader(pv_data_pipeline, batch_size=None, num_workers=4)
    val_loader = DataLoader(pv_data_pipeline_validation, batch_size=None, num_workers=2)
    predict_loader = DataLoader(pv_data_pipeline, batch_size=None, num_workers=0)

    model = Model(input_length=input_length, output_length=output_length)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # predict model for some plots
    batch = next(val_loader)
    y_hat = model(batch)

    plot(batch, y_hat)

if __name__ == "__main__":
    main()


# results
# 1. All sites, after 1, 2,3  epochs
# mse = 0.023, 0.0106, 0.0122
# mae = 0.0737, 0.0498, 0.0532

# results
# 2. Adding embedding, after 1, 2,3 epochs
# mse = 0.0253, 0.0205
# mae = 0.0741, 0.0737

# 3. only using 10 pv output.org sites
# mse_val = 0.0217, 0.123, 0.00682, 0.0114,0.0067, 0.00689, 0.00489,
# mae_val = 0.0735, 0.564 ,0.0404, 0.524, 0.0386, 0.0359, 0.0293,

# 4. only using 0 pv output.org sites
# mse_val =
# mae_val =


