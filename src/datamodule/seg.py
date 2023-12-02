import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

from src.augmentation.local_shuffle import LocalShuffleAug
from src.augmentation.mixup import SwapMixup
from src.augmentation.swap_event import SwapEvent
from src.utils.common import pad_if_needed


###################
# Load Functions
###################
def load_features(
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        features[series_dir.name] = np.stack(this_feature, axis=1)

    return features


def load_chunk_features(
    duration: int,
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
    slide_tta: bool,
) -> dict[str, np.ndarray]:
    """
    TTA的な感じで予測区間をスライドさせる
    重複する部分は予測後に平均をとる
    """
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        this_feature = np.stack(this_feature, axis=1)

        if slide_tta:
            slide_duration = duration // 2
            num_chunks = max(1, (len(this_feature) - duration) // slide_duration + 1)
        else:
            slide_duration = duration
            num_chunks = (len(this_feature) // duration) + 1

        for i in range(num_chunks):
            start = i * slide_duration
            end = start + duration
            chunk_feature = this_feature[start:end]
            chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)  # type: ignore
            features[f"{series_id}_{i:07}"] = chunk_feature
    return features  # type: ignore


###################
# Augmentation
###################
def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    end = start + duration
    return start, end


###################
# Label
###################
def get_label(
    this_event_df: pd.DataFrame, num_frames: int, duration: int, start: int, end: int
) -> np.ndarray:
    # # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    label = np.zeros((num_frames, 3))
    # onset, wakeup, sleepのラベルを作成
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset = int((onset - start) / duration * num_frames)
        wakeup = int((wakeup - start) / duration * num_frames)
        if onset >= 0 and onset < num_frames:
            label[onset, 1] = 1
        if wakeup < num_frames and wakeup >= 0:
            label[wakeup, 2] = 1

        onset = max(0, onset)
        wakeup = min(num_frames, wakeup)
        label[onset:wakeup, 0] = 1  # sleep

    return label


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")

    return label


def laplace_kernel(length: int, scale: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-abs(x) / scale)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def laplace_label(label: np.ndarray, offset: int, scale: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], laplace_kernel(offset, scale), mode="same")

    return label


def add_soft_label(label, cfg):
    if cfg.label_type == "gaussian":
        label[:, [1, 2]] = gaussian_label(label[:, [1, 2]], offset=cfg.offset, sigma=cfg.sigma)
    elif cfg.label_type == "laplace":
        label[:, [1, 2]] = laplace_label(label[:, [1, 2]], offset=cfg.offset, scale=cfg.scale)
    else:
        raise NotImplementedError
    return label


def negative_sampling(this_event_df: pd.DataFrame, num_steps: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    # onsetとwakupを除いた範囲からランダムにサンプリング
    positive_positions = set(this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist())
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]


###################
# Dataset
###################
def nearest_valid_size(input_size: int, downsample_rate: int) -> int:
    """
    (x // hop_length) % 32 == 0
    を満たすinput_sizeに最も近いxを返す
    """

    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0

    return input_size


def event_relabeling(gt_df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
    relabeled_events = pl.read_csv(Path(cfg.dir.data_dir) / "relabeled_train_events.csv")
    relabeled_gt_df = gt_df.join(relabeled_events, on=["series_id", "step", "event"], how="left")
    relabeled_gt_df = relabeled_gt_df.with_columns(
        pl.when(pl.col("relabeled_step").is_null())
        .then(pl.col("step"))
        .otherwise(pl.col("relabeled_step"))
        .alias("step"),
    ).select(["series_id", "night", "event", "step", "timestamp"])
    assert gt_df.shape == relabeled_gt_df.shape
    changed_label_count = np.sum(gt_df["step"].to_numpy() != relabeled_gt_df["step"].to_numpy())
    relabeled_count = len(
        relabeled_events.filter(pl.col("series_id").is_in(gt_df["series_id"].unique().to_list()))
    )
    assert changed_label_count == relabeled_count
    return relabeled_gt_df


def add_noisy_event_flag(gt_df: pl.DataFrame) -> pl.DataFrame:
    """
    ノイジーなラベルにフラグをつけて後ほどソフトラベルにする
    """
    relabeled_events = pl.read_csv(
        "/home/kuto/kaggle/kaggle-sleep-v2/data/child-mind-institute-detect-sleep-states/relabeled_train_events.csv"
    )
    relabeled_gt_df = gt_df.join(relabeled_events, on=["series_id", "step", "event"], how="left")
    relabeled_gt_df = relabeled_gt_df.with_columns(
        pl.when(pl.col("relabeled_step").is_null())
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("is_noisy_event"),
    ).select(["series_id", "night", "event", "step", "timestamp", "is_noisy_event"])
    return relabeled_gt_df


class TrainDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        event_df: pl.DataFrame,
        features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        # trainのみrelabelingしてvalidの評価は元のラベルを使う
        if cfg.relabeling:
            event_df = event_relabeling(event_df, cfg)
        if cfg.use_noisy_event_label:
            event_df = add_noisy_event_flag(event_df)
        self.event_df: pd.DataFrame = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step")
            .drop_nulls()
            .to_pandas()
        )
        self.features = features
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

        self.local_shuffle_aug = LocalShuffleAug(self.cfg.augmentation.local_shuffle_window_size)
        self.swap_event_aug = SwapEvent(
            self.cfg.augmentation.swap_event_window_size, self.cfg.augmentation.swap_channels
        )
        self.swap_mixup = SwapMixup(
            self.cfg.augmentation.swap_event_window_size,
            self.cfg.augmentation.swap_channels,
            self.cfg.augmentation.swap_mixup_alpha,
        )

    def __len__(self):
        return len(self.event_df)

    def __getitem__(self, idx):
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate

        event = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])
        pos = self.event_df.at[idx, event]
        series_id = self.event_df.at[idx, "series_id"]
        this_event_df = self.event_df.query("series_id == @series_id").reset_index(drop=True)
        # extract data matching series_id
        this_feature = self.features[series_id]  # (n_steps, num_features)
        n_steps = this_feature.shape[0]

        # sample background
        if random.random() < self.cfg.bg_sampling_rate:
            pos = negative_sampling(this_event_df, n_steps)

        # crop
        start, end = random_crop(pos, self.cfg.duration, n_steps)
        feature = this_feature[start:end]  # (duration, num_features)

        # shuffle aug
        if random.random() < self.cfg.augmentation.local_shuffle_prob:
            feature = self.local_shuffle_aug(feature)

        # from hard label to gaussian label
        label = get_label(this_event_df, num_frames, self.cfg.duration, start, end)
        label = add_soft_label(label, self.cfg)

        if random.random() < self.cfg.augmentation.swap_event_prob:
            # 同一series_idの同一eventからランダムに1箇所をサンプリング
            swap_pos = this_event_df[event].sample(1).to_numpy()[0]
            swap_start, swap_end = random_crop(swap_pos, self.cfg.duration, n_steps)
            swap_feature = this_feature[swap_start:swap_end]  # (duration, num_features)
            swap_label = get_label(
                this_event_df, num_frames, self.cfg.duration, swap_start, swap_end
            )
            swap_label = add_soft_label(swap_label, self.cfg)
            feature = self.swap_event_aug(feature, label, swap_feature, swap_label, event)

        if random.random() < self.cfg.augmentation.swap_mixup_prob:
            # 同一series_idの同一eventからランダムに1箇所をサンプリング
            swap_pos = this_event_df[event].sample(1).to_numpy()[0]
            swap_start, swap_end = random_crop(swap_pos, self.cfg.duration, n_steps)
            swap_feature = this_feature[swap_start:swap_end]  # (duration, num_features)
            swap_label = get_label(
                this_event_df, num_frames, self.cfg.duration, swap_start, swap_end
            )
            swap_label = add_soft_label(swap_label, self.cfg)
            feature = self.swap_mixup(feature, label, swap_feature, swap_label, event)
        # upsample
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "series_id": series_id,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (pred_length, num_classes)
        }


class ValidDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        chunk_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.event_df = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step")
            .drop_nulls()
            .to_pandas()
        )
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        if self.cfg.slide_tta:
            slide_duration = self.cfg.duration // 2
            start = chunk_id * slide_duration
            end = start + self.cfg.duration
        else:
            start = chunk_id * self.cfg.duration
            end = start + self.cfg.duration
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            num_frames,
            self.cfg.duration,
            start,
            end,
        )
        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "label": torch.FloatTensor(label),  # (duration, num_classes)
        }


class TestDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        chunk_features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
        }


def drop_data(labels: pl.DataFrame):
    # mismatch eventを除外
    drop_event_ids = labels.group_by(["series_id", "night"]).count().filter(pl.col("count") != 2)
    # target_eventsからdrop_event_idsのseries_idとnightの組み合わせを除外
    for series_id, night in drop_event_ids[["series_id", "night"]].to_numpy():
        labels = labels.filter(~((pl.col("series_id") == series_id) & (pl.col("night") == night)))

    # stepには欠損があったことでtrainとlabelsで型が異なるのでu32に変換する
    return labels


###################
# DataModule
###################
class SegDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.processed_dir = Path(cfg.dir.processed_dir)
        self.event_df = pl.read_csv(self.data_dir / "train_events.csv").drop_nulls()
        # データを除くと悪化するのでコメントアウト
        # self.event_df = drop_data(self.event_df)
        self.train_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.train_series_ids)
        )
        self.valid_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.valid_series_ids)
        )
        # train data
        self.train_features = load_features(
            feature_names=self.cfg.features,
            series_ids=self.cfg.split.train_series_ids,
            processed_dir=self.processed_dir,
            phase="train",
        )

        # valid data
        self.valid_chunk_features = load_chunk_features(
            duration=self.cfg.duration,
            feature_names=self.cfg.features,
            series_ids=self.cfg.split.valid_series_ids,
            processed_dir=self.processed_dir,
            phase="train",
            slide_tta=self.cfg.slide_tta,
        )

    def train_dataloader(self):
        train_dataset = TrainDataset(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = ValidDataset(
            cfg=self.cfg,
            chunk_features=self.valid_chunk_features,
            event_df=self.valid_event_df,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return valid_loader
