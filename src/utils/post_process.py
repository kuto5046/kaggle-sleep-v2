import numpy as np
import pandas as pd
import polars as pl
from scipy import signal
from scipy.signal import find_peaks

from src.utils.common import pad_if_needed


def low_path_filter(wave: np.ndarray, hour: int, fe: int = 60, n: int = 3):
    fs = 12 * 60 * hour
    nyq = fs / 2.0
    b, a = signal.butter(1, fe / nyq, btype="low")
    for i in range(0, n):
        wave = signal.filtfilt(b, a, wave)
    return wave


def post_process_for_seg(
    keys: list[str],
    preds: np.ndarray,
    score_th: float = 0.01,
    distance: int = 5000,
    low_pass_filter_hour: int = 1,
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            this_event_preds = low_path_filter(this_event_preds, hour=low_pass_filter_hour)
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df


def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def event_score_from_sleep_prediction(this_series_preds: np.ndarray, window_size: int):
    """
    this_series_preds: sleepの予測値(seq, 1)
    window_size: スコア算出のためのwindowサイズ
    """
    df = pl.DataFrame({"pred": this_series_preds.reshape(-1)})
    df = (
        df.with_columns(
            pl.col("pred")
            .rolling_mean(window_size, min_periods=1, center=False)
            .alias("before_pred_mean"),
            pl.col("pred")
            .shift(-window_size + 1)
            .rolling_mean(window_size, min_periods=1, center=False)
            .alias("after_pred_mean"),
        )
        .with_columns(
            (pl.col("after_pred_mean") - pl.col("before_pred_mean")).alias("pred_diff"),
        )
        .with_columns(
            # 0以上の場合
            pl.when(pl.col("pred_diff") > 0)
            .then(pl.col("pred_diff"))
            .otherwise(0)
            .alias("onset_score"),
            # 0以下の場合
            pl.when(pl.col("pred_diff") < 0)
            .then(pl.col("pred_diff").abs())
            .otherwise(0)
            .alias("wakeup_score"),
        )
    )
    this_series_preds = df.select(["onset_score", "wakeup_score"]).to_numpy()
    return this_series_preds


def ensemble_sleep_and_event_prediction(
    this_series_preds: np.ndarray, window_size: int, event_weight: float
):
    """
    sleep予測から計算したevent予測結果を元々のevent予測結果とアンサンブルする
    入出力は3次元の予測値
    """
    this_series_sleep = this_series_preds[:, 0].reshape(-1, 1)  # 1次元
    this_series_events = this_series_preds[:, 1:]  # 2次元
    this_series_events_by_sleep = event_score_from_sleep_prediction(
        this_series_sleep, window_size=window_size
    )  # 2次元
    this_series_events = (
        event_weight * this_series_events + (1 - event_weight) * this_series_events_by_sleep
    )  # 2次元
    output_series_preds = np.concatenate([this_series_sleep, this_series_events], axis=1)  # 3次元
    return output_series_preds


def post_process_asleep_to_event(
    preds: np.ndarray, keys: list[str], window_size: int, event_weight: float
):
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    # 順番を保持したいためpandasのuniqueを利用
    # unique_series_ids = np.unique(series_ids)
    unique_series_ids = pd.Series(series_ids).unique()
    results = []
    duration = preds.shape[1]
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 3)
        this_series_preds = ensemble_sleep_and_event_prediction(
            this_series_preds, window_size, event_weight
        )
        this_series_preds = this_series_preds.reshape(-1, duration, 3)
        results.append(this_series_preds)
    preds = np.concatenate(results)
    return preds


def post_process_for_asleep(
    keys: list[str],
    preds: np.ndarray,
    score_th: float = 0.5,
    distance: int = 40,
    window_size: int = 100,
    low_pass_filter_hour: int = 1,
) -> pl.DataFrame:
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)
    # sleep
    preds = min_max_normalize(preds)
    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1)
        this_series_preds = event_score_from_sleep_prediction(
            this_series_preds, window_size=window_size
        )

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            this_event_preds = low_path_filter(this_event_preds, hour=low_pass_filter_hour)
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )
    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df


def post_process_for_asleep_and_event(
    keys: list[str],
    preds: np.ndarray,
    score_th: float = 0.5,
    distance: int = 40,
    window_size: int = 100,
    low_pass_filter_hour: int = 1,
    event_weight=0.5,
) -> pl.DataFrame:
    if event_weight == 0:
        return post_process_for_asleep(
            keys,
            preds[:, :, 0],
            score_th=score_th,
            distance=distance,
            window_size=window_size,
            low_pass_filter_hour=low_pass_filter_hour,
        )
    elif event_weight == 1:
        return post_process_for_seg(
            keys,
            preds[:, :, 1:],
            score_th=score_th,
            distance=distance,
            low_pass_filter_hour=low_pass_filter_hour,
        )
    else:
        series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
        unique_series_ids = np.unique(series_ids)
        records = []
        for series_id in unique_series_ids:
            series_idx = np.where(series_ids == series_id)[0]
            this_series_all = preds[series_idx].reshape(-1, 3)
            this_series_sleep = this_series_all[:, 0]
            this_series_events = this_series_all[:, 1:]
            this_series_events_by_sleep = event_score_from_sleep_prediction(
                this_series_sleep, window_size=window_size
            )
            this_series_preds = (
                event_weight * this_series_events
                + (1 - event_weight) * this_series_events_by_sleep
            )
            for i, event_name in enumerate(["onset", "wakeup"]):
                this_event_preds = this_series_preds[:, i]
                this_event_preds = low_path_filter(this_event_preds, hour=low_pass_filter_hour)
                steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
                scores = this_event_preds[steps]

                for step, score in zip(steps, scores):
                    records.append(
                        {
                            "series_id": series_id,
                            "step": step,
                            "event": event_name,
                            "score": score,
                        }
                    )

        if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
            records.append(
                {
                    "series_id": series_id,
                    "step": 0,
                    "event": "onset",
                    "score": 0,
                }
            )
        sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
        row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
        sub_df = sub_df.with_columns(row_ids).select(
            ["row_id", "series_id", "step", "event", "score"]
        )
        return sub_df


def post_process_for_sliding_data(values: np.ndarray, keys: list[str], duration: int):
    """
    slideさせて作成したデータの重複したstepの箇所を平均する
    valuesにはpredsやlabelsが入る
    """
    all_values = []
    all_keys = []
    series_ids = np.array([key.split("_")[0] for key in keys])
    # seriesの順番が変わってしまうがkeyも合わせて変わるため問題ない
    unique_series_ids = np.unique(series_ids)
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        series_values = values[series_idx]
        # shuffleされていないので、series_predの順番は変わらない
        df_list = []
        for i in range(series_values.shape[0]):
            start = i * (duration // 2)
            end = start + duration
            _df = pl.DataFrame(
                {
                    "sleep": series_values[i][:, 0],
                    "onset": series_values[i][:, 1],
                    "wakeup": series_values[i][:, 2],
                    "step": np.arange(start, end),
                }
            )
            df_list.append(_df)
        # スライドさせた結果を平均する
        series_df = (
            pl.concat(df_list)
            .group_by("step")
            .agg(
                pl.col("sleep").mean(),
                pl.col("onset").mean(),
                pl.col("wakeup").mean(),
            )
        ).sort(by="step")

        # series_dfをdurationごとに割り切れるようにpaddingする
        # 必要長さを計算
        num_chunks = len(series_df) // (duration)
        if len(series_df) % duration > 0:
            num_chunks += 1
        need_length = num_chunks * duration
        all_values.append(
            pad_if_needed(
                series_df[["sleep", "onset", "wakeup"]].to_numpy(), need_length, pad_value=0
            ).reshape(num_chunks, duration, 3)
        )
        all_keys.extend([series_id] * num_chunks)  # ここがデータ分作られるのでメモリ使用量が増える

    all_values = np.concatenate(all_values)
    return all_values, all_keys
