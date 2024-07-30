import os
import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from rich.console import Console

from maverick.data.pl_data_modules import BasePLDataModule
from maverick.models.pl_modules import BasePLModule

torch.set_printoptions(edgeitems=100)


def train(conf: omegaconf.DictConfig) -> None:
    # fancy logger
    console = Console()
    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_determinism_the_old_way(conf.train.pl_trainer.deterministic)
    conf.train.pl_trainer.deterministic = True

    console.log(f"Starting training for [bold cyan]{conf.train.model_name}[/bold cyan] model")
    if conf.train.pl_trainer.fast_dev_run:
        console.log(f"Debug mode {conf.train.pl_trainer.fast_dev_run}. Forcing debugger configuration")
        # Debuggers don't like GPUs nor multiprocessing
        conf.train.pl_trainer.accelerator = "cpu"

        conf.train.pl_trainer.precision = 32
        conf.data.datamodule.num_workers = {k: 0 for k in conf.data.datamodule.num_workers}
        # Switch wandb to offline mode to prevent online logging
        conf.logging.log = None
        # remove model checkpoint callback
        conf.train.model_checkpoint_callback = None

    # data module declaration
    console.log(f"Instantiating the Data Module")
    pl_data_module: BasePLDataModule = hydra.utils.instantiate(conf.data.datamodule, _recursive_=False)
    # force setup to get labels initialized for the model
    pl_data_module.prepare_data()
    pl_data_module.setup("fit")

    # main module declaration
    console.log(f"Instantiating the Model")

    pl_module: BasePLModule = hydra.utils.instantiate(conf.model.module, _recursive_=False)

    # pl_module = BasePLModule.load_from_checkpoint(conf.evaluation.checkpoint, _recursive_=False, map_location="cuda:0")
    experiment_logger: Optional[WandbLogger] = None
    experiment_path: Optional[Path] = None
    if conf.logging.log:
        console.log(f"Instantiating Wandb Logger")
        experiment_logger = hydra.utils.instantiate(conf.logging.wandb_arg)
        experiment_logger.watch(pl_module, **conf.logging.watch)
        experiment_path = Path(experiment_logger.experiment.dir)
        # Store the YaML config separately into the wandb dir
        yaml_conf: str = OmegaConf.to_yaml(cfg=conf)
        (experiment_path / "hparams.yaml").write_text(yaml_conf)

        # callbacks declaration
    callbacks_store = [RichProgressBar()]

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback,
            dirpath=experiment_path / "checkpoints",
        )
        callbacks_store.append(model_checkpoint_callback)

    if conf.train.learning_rate_callback is not None and not conf.train.pl_trainer.fast_dev_run:
        lr: LearningRateMonitor = hydra.utils.instantiate(conf.train.learning_rate_callback)
        callbacks_store.append(lr)
    # trainer
    console.log(f"Instantiating the Trainer")
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store, logger=experiment_logger)

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    # module test
    trainer.test(pl_module, datamodule=pl_data_module)


def set_determinism_the_old_way(deterministic: bool):
    # determinism for cudnn
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # fixing non-deterministic part of horovod
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
        os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    print(OmegaConf.to_yaml(conf))
    train(conf)


if __name__ == "__main__":
    main()
