import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "enmo",
    "step",
    "anglez_diff",
    "enmo_diff",
    "hour_sin",
    "hour_cos",
    "duplicate",
    # "duplicate_len",
    # "duplicate_nearest",
    # "month_sin",
    # "month_cos",
    # "minute_sin",
    # "minute_cos",
    # "anglez_yesterday",
    # "enmo_yesterday",
    # "anglez_smoothed_avg_12",
    # "anglez_smoothed_max_12",
    # "anglez_smoothed_min_12",
    # "anglez_smoothed_std_12",
    # "enmo_smoothed_avg_12",
    # "enmo_smoothed_max_12",
    # "enmo_smoothed_min_12",
    # "enmo_smoothed_std_12",
    # "anglez_smoothed_avg_60",
    # "anglez_smoothed_max_60",
    # "anglez_smoothed_min_60",
    # "anglez_smoothed_std_60",
    # "enmo_smoothed_avg_60",
    # "enmo_smoothed_max_60",
    # "enmo_smoothed_min_60",
    # "enmo_smoothed_std_60",
    # "anglez_smoothed_avg_360",
    # "anglez_smoothed_max_360",
    # "anglez_smoothed_min_360",
    # "anglez_smoothed_std_360",
    # "enmo_smoothed_avg_360",
    # "enmo_smoothed_max_360",
    # "enmo_smoothed_min_360",
    # "enmo_smoothed_std_360",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829
DUPLICATE_LENGTH_MEAN = 30.610896
DUPLICATE_LENGTH_STD = 111.852321
DUPLICATE_NEAREST_MEAN = 42.321609
DUPLICATE_NEAREST_STD = 134.247448


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def duplicate_feature(indices):
    nearest_distances = []
    for i in range(len(indices)):
        current_element = indices[i]
        # 他の要素との距離を格納する一時的なリスト
        distances = []
        for j in range(len(indices)):
            if i != j:
                # 絶対距離を計算してリストに追加
                distance = abs(current_element - indices[j])
                distances.append(distance)
        # 最小の距離を見つけてリストに追加
        nearest_distance = min(distances)
        nearest_distances.append(nearest_distance)
    length = len(indices)
    return nearest_distances, length


def add_duplicate(df: pl.DataFrame):
    # NumPy配列への変換
    array = df[["enmo", "anglez"]].to_numpy()

    # 180行ごとに分割
    subsets = [array[i : i + 180] for i in range(0, len(array), 180)]

    subsets_dict = {}
    for i, subset in enumerate(subsets):
        # サブセットをタプルのリストに変換
        subset_key = tuple(map(tuple, subset))
        if subset_key in subsets_dict:
            subsets_dict[subset_key].append(i)
        else:
            subsets_dict[subset_key] = [i]

    # 完全に一致するサブセットのインデックスペアを探す
    matching_subsets_len = {}
    matching_subsets_nearest = {}
    for indices in subsets_dict.values():
        if len(indices) > 1:
            nearest_distances, length = duplicate_feature(indices)
            for j, idx in enumerate(indices):
                matching_subsets_nearest[idx] = nearest_distances[j]
                matching_subsets_len[idx] = length

    duplicate_array = np.zeros(len(df), dtype=int)
    # duplicate_array_len = np.zeros(len(df), dtype=int)
    # duplicate_array_nearest = np.zeros(len(df), dtype=int)

    # matching_subsetsに含まれるサブセットに対応する配列の範囲を1に更新
    for index in matching_subsets_len.keys():
        if np.mean(subsets[index][:, 0]) == 0:
            continue
        start_row = index * 180
        end_row = start_row + 180
        duplicate_array[start_row:end_row] = 1
        # duplicate_array_len[start_row:end_row] = matching_subsets_len[index]
        # duplicate_array_nearest[start_row:end_row] = matching_subsets_nearest[index]

    duplicate_array_len = (duplicate_array_len - DUPLICATE_LENGTH_MEAN) / DUPLICATE_LENGTH_STD
    duplicate_array_nearest = (
        duplicate_array_nearest - DUPLICATE_NEAREST_MEAN
    ) / DUPLICATE_NEAREST_STD
    df = df.with_columns(pl.Series(duplicate_array).alias("duplicate"))
    df = df.with_columns(pl.Series(duplicate_array_len).alias("duplicate_len"))
    df = df.with_columns(pl.Series(duplicate_array_nearest).alias("duplicate_nearest"))
    return df


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = add_duplicate(series_df)
    series_df = series_df.with_columns(
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
        *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
        pl.col("anglez").diff().fill_null(0).alias("anglez_diff"),
        pl.col("enmo").diff().fill_null(0).alias("enmo_diff"),
        # 前日の全く同じ時間のanglezとenmoをshiftして特徴量にする
        pl.col("anglez").shift(12 * 60 * 24).fill_null(0).alias("anglez_yesterday"),
        pl.col("enmo").shift(12 * 60 * 24).fill_null(0).alias("enmo_yesterday"),
    )
    series_df = series_df.select("series_id", *FEATURE_NAMES)
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("step"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df)

            # 特徴量をそれぞれnpyで保存
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()
