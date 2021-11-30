import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

log_dir = "lightning_logs"


def main(dataset, n_classes, n_channels, log_dir="lightning_logs"):
    model = Unet(dataset, n_classes, n_channels)

    os.makedirs(log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(log_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        # save_best_only=False,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=True,
    )
    trainer = Trainer(
        gpus=1,
        callbacks=[checkpoint_callback, stop_callback]
        # checkpoint_callback=checkpoint_callback,
        # early_stop_callback=stop_callback,
    )

    trainer.fit(model)


if __name__ == '__main__':
    # parent_parser = ArgumentParser(add_help=False)
    # parent_parser.add_argument('--dataset', required=True)
    # parent_parser.add_argument('--log_dir', default='lightning_logs')
    log_dir = "lightning_logs"
    # parser = Unet.add_model_specific_args(parent_parser)
    # hparams = parser.parse_args()

    main("cells", 1, 1)
    # model = Unet("cells", 1, 1)

    # os.makedirs(log_dir, exist_ok=True)
    # try:
    #     log_dir = sorted(os.listdir(log_dir))[-1]
    # except IndexError:
    #     log_dir = os.path.join(log_dir, 'version_0')
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join(log_dir, 'checkpoints'),
    #     # save_best_only=False,
    #     verbose=True,
    # )
    # stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     mode='min',
    #     patience=5,
    #     verbose=True,
    # )
    # trainer = Trainer(
    #     gpus=1,
    #     callbacks=[checkpoint_callback, stop_callback]
    #     # checkpoint_callback=checkpoint_callback,
    #     # early_stop_callback=stop_callback,
    # )

    # trainer.fit(model)
