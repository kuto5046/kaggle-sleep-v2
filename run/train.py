import logging
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import polars as pl
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.datamodule.seg import SegDataModule
from src.modelmodule.seg import SegModel
from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)


def evaluate(cfg: DictConfig):
    """
    best modelで最終的なスコアを計算する
    """
    preds = np.load("preds.npy")
    # labels = np.load("labels.npy")
    keys = np.load("keys.npy")

    gt_df = pd.read_csv(Path(cfg.dir.data_dir) / "train_events.csv")
    gt_df = (
        gt_df[gt_df["series_id"].isin(cfg.split.valid_series_ids)].dropna().reset_index(drop=True)
    )

    pred_df: pl.DataFrame = post_process_for_seg(
        keys,
        preds[:, :, [1, 2]],
        score_th=cfg.post_process.score_th,
        distance=cfg.post_process.distance,
        low_pass_filter_hour=cfg.post_process.low_pass_filter_hour,
    )
    score = event_detection_ap(gt_df, pred_df.to_pandas())
    wandb.log({"score": score})


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    seed_everything(cfg.seed)

    # init lightning model
    datamodule = SegDataModule(cfg)
    LOGGER.info("Set Up DataModule")
    num_warmup_steps = int(
        cfg.epoch * len(datamodule.train_dataloader()) * cfg.scheduler.warmup_step_rate
    )
    model = SegModel(
        cfg,
        datamodule.valid_event_df,
        len(cfg.features),
        len(cfg.labels),
        cfg.duration,
        num_warmup_steps,
    )

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.monitor,
        mode=cfg.monitor_mode,
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    # init experiment logger
    pl_logger = WandbLogger(
        project="kaggle-sleep",
        entity="kuto5046",
        group=cfg.exp_name,
        tags=["tubo_code"],
        mode="disabled" if cfg.debug else "online",
        notes=cfg.notes,
    )

    limit_train_batches: Optional[int] = None

    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.accelerator,
        precision=16 if cfg.use_amp else 32,
        # training
        fast_dev_run=cfg.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.epoch,
        max_steps=cfg.epoch * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        limit_train_batches=limit_train_batches,
    )

    all_training = len(cfg.split.valid_series_ids) == 0
    if all_training:
        trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
    else:
        trainer.fit(model, datamodule=datamodule)

    if all_training:
        return
    # load best weights
    if not cfg.debug:
        model = model.load_from_checkpoint(
            checkpoint_cb.best_model_path,
            cfg=cfg,
            val_event_df=datamodule.valid_event_df,
            feature_dim=len(cfg.features),
            num_classes=len(cfg.labels),
            duration=cfg.duration,
        )

    evaluate(cfg)
    weights_path = str("model_weights.pth")  # type: ignore
    LOGGER.info(f"Extracting and saving best weights: {weights_path}")
    torch.save(model.model.state_dict(), weights_path)
    return


if __name__ == "__main__":
    main()
