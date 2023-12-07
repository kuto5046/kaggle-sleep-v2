import numpy as np
import pandas as pd
import polars as pl
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import norm
from tqdm import tqdm

from src.utils.common import pad_if_needed


def low_pass_filter(wave: np.ndarray, hour: int, fe: int = 60, n: int = 3):
    fs = 12 * 60 * hour
    nyq = fs / 2.0
    b, a = signal.butter(1, fe / nyq, btype="low")
    for i in range(0, n):
        wave = signal.filtfilt(b, a, wave)
    return wave


def score_decay_by_null_prob(this_series_preds: np.ndarray, k: float = 0.5, m: int = 100):
    max_steps = this_series_preds.shape[0]
    step_rate = np.arange(0, max_steps) / max_steps
    discount_rate = 1 - (1 - k) * step_rate**m
    this_series_preds *= discount_rate
    return this_series_preds


def get_prob_dict(mu, sigma=10):
    direct_distance = np.abs(np.arange(0, 24) - mu)
    circular_distance = np.minimum(direct_distance, 24 - direct_distance)
    prob = norm.pdf(circular_distance, 0, sigma)
    prob_dict = {}
    for i in range(24):
        prob_dict[i] = prob[i]

    # min-max scaling
    prob_dict = {k: v / max(prob_dict.values()) for k, v in prob_dict.items()}
    return prob_dict


def post_process_for_seg(
    keys: list[str],
    preds: np.ndarray,
    input_df: pl.DataFrame,
    score_th: float = 0.01,
    distance: int = 5000,
    low_pass_filter_hour: int = 1,
    use_hour_prob: bool = False,
    sigma=5,
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
    df = input_df.with_columns(
        pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z").dt.hour().alias("hour")
    )
    records = []
    for series_id in tqdm(unique_series_ids):
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)
        if use_hour_prob:
            series_df = df.filter(pl.col("series_id") == series_id)
            series_df = series_df.with_columns(
                pl.lit(this_series_preds[: len(series_df), 0]).alias("onset"),
                pl.lit(this_series_preds[: len(series_df), 1]).alias("wakeup"),
            )
        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]

            if use_hour_prob:
                peak_thr = 0.1
                # sigma = 5
                mu = series_df.filter(pl.col(event_name) > peak_thr)["hour"].mean()
                prob_dict = get_prob_dict(mu, sigma)
                this_event_preds = (
                    series_df.with_columns(
                        (pl.col(event_name) * pl.col("hour").apply(lambda x: prob_dict[x])).alias(
                            "prob"
                        )
                    )
                    .select("prob")
                    .to_numpy()
                    .flatten()
                )

            # this_event_preds = score_decay_by_null_prob(this_event_preds)

            this_event_preds = low_pass_filter(this_event_preds, hour=low_pass_filter_hour)
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


def post_process_asleep_to_event(
    preds: np.ndarray, keys: list[str], window_size: int, event_weight: float
):
    """
    与えらえたweightをもとに内部でsleepとeventを混ぜる
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    # 順番を保持したいためpandasのuniqueを利用
    # unique_series_ids = np.unique(series_ids)
    unique_series_ids = pd.Series(series_ids).unique()
    results = []
    duration = preds.shape[1]
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 3)
        this_series_preds_by_sleep = get_event_prediction_from_sleep(
            this_series_preds, window_size
        )  # 3次元

        # weighted average
        this_series_preds = (
            event_weight * this_series_preds + (1 - event_weight) * this_series_preds_by_sleep
        )  # 3次元
        this_series_preds = this_series_preds.reshape(-1, duration, 3)
        results.append(this_series_preds)
    preds = np.concatenate(results)
    return preds


def get_event_prediction_from_sleep(this_series_preds: np.ndarray, window_size: int):
    """
    sleep予測から計算したevent予測結果を出力する
    入出力は3次元の予測値
    """
    this_series_sleep = this_series_preds[:, 0].reshape(-1, 1)  # 1次元
    # this_series_events = this_series_preds[:, 1:]  # 2次元
    this_series_events_by_sleep = event_score_from_sleep_prediction(
        this_series_sleep, window_size=window_size
    )  # 2次元
    output_series_preds = np.concatenate(
        [this_series_sleep, this_series_events_by_sleep], axis=1
    )  # 3次元
    return output_series_preds


def post_process_asleep_to_event_v2(preds: np.ndarray, keys: list[str], window_size: int):
    """
    内部では混ぜずsleepをeventに変換した3次元の予測値を返す(sleepは元の値で埋めておく)
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    # 順番を保持したいためpandasのuniqueを利用
    # unique_series_ids = np.unique(series_ids)
    unique_series_ids = pd.Series(series_ids).unique()
    results = []
    duration = preds.shape[1]
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 3)
        this_series_preds = get_event_prediction_from_sleep(this_series_preds, window_size)
        this_series_preds = this_series_preds.reshape(-1, duration, 3)
        results.append(this_series_preds)
    preds = np.concatenate(results)
    return preds  # 3次元


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
            this_event_preds = low_pass_filter(this_event_preds, hour=low_pass_filter_hour)
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
                this_event_preds = low_pass_filter(this_event_preds, hour=low_pass_filter_hour)
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
        n = series_values.shape[0]
        for i in range(n):
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
            # # seriesの端以外はデータの最初と最後の1/8を削除
            # k = 16
            # if 0 < i < n - 1:
            #     # データの最初と最後の1/8を削除
            #     _df = _df.filter(pl.col("step").is_in(np.arange(start+duration//k, end-duration//k)))

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


def find_nonpaired_events(df: pl.DataFrame, min_hour: int = 0, max_hour: int = 24) -> pl.DataFrame:
    """
    ピークの高い予測を対象に一定範囲内にペアとなるピークがないnonpairの予測イベントを取得する
    usage:
    # 閾値を高めに設定しピークの高い予測を抽出(thr=0.01(通常使っている0.001ではない点に注意))
    ```python
    high_score_sub_df = sub_df.filter(pl.col('score')>0.01).select('series_id','step','event','score')
    # 対象の予測内でペアを作りペアができないイベントを取得
    nonpair_event_df = find_nonpaired_events(high_score_sub_df, min_hour=1, max_hour=24)
    # ペア無しイベントをフィルタする
    sub_df = sub_df.join(nonpair_event_df, on=["series_id", "step"], how="left")
    sub_df = sub_df.filter(pl.col("non_pair_step").is_null()).drop('row_id').with_row_count("row_id").select(["row_id", "series_id", "step", "event", "score"])
    ```
    """
    # ペアの想定する最小時間と最大時間をステップに変換
    min_step, max_step = min_hour * 60 * 60 // 5, max_hour * 60 * 60 // 5

    # onsetイベントを持つ行を選択
    onsets = (
        df.filter(pl.col("event") == "onset")
        .select(["series_id", "step"])
        .rename({"step": "onset_step"})
    )
    # wakeupイベントを持つ行を選択
    wakeups = (
        df.filter(pl.col("event") == "wakeup")
        .select(["series_id", "step"])
        .rename({"step": "wakeup_step"})
    )

    # 全ての'onset'に対し、それに続く'wakeup'を検索
    paired_events = (
        onsets.join(wakeups, on="series_id", how="left")
        .with_columns((pl.col("wakeup_step") - pl.col("onset_step")).alias("duration_in_steps"))
        .filter(
            (pl.col("duration_in_steps") >= min_step) & (pl.col("duration_in_steps") <= max_step)
        )
    )

    # ペアになったイベントのIDを取得
    paired_onsets = paired_events.select("onset_step").to_series()
    paired_wakeups = paired_events.select("wakeup_step").to_series()

    # ペアにならなかった'onset'と'wakeup'を抽出
    non_paired_onsets = onsets.filter(~pl.col("onset_step").is_in(paired_onsets)).rename(
        {"onset_step": "step"}
    )
    non_paired_wakeups = wakeups.filter(~pl.col("wakeup_step").is_in(paired_wakeups)).rename(
        {"wakeup_step": "step"}
    )

    # 結果を結合
    non_paired_events = pl.concat([non_paired_onsets, non_paired_wakeups]).sort("step")
    non_paired_events = non_paired_events.with_columns(pl.lit(True).alias("non_pair_step"))
    return non_paired_events
