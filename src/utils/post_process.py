import numpy as np
import polars as pl
from scipy import signal
from scipy.signal import find_peaks


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


def post_process_for_asleep(
    keys: list[str], preds: np.ndarray, score_thr: float = 0.5
) -> pl.DataFrame:
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1)

        this_series_preds_norm = min_max_normalize(this_series_preds)
        this_series_preds_class = (this_series_preds_norm > score_thr) * 1
        pred_onset_steps = np.where(np.diff(this_series_preds_class) > 0)[0]
        pred_wakeup_steps = np.where(np.diff(this_series_preds_class) < 0)[0]

        if len(pred_onset_steps) > 0:
            try:
                # Ensuring all predicted sleep periods begin and end
                if min(pred_wakeup_steps) < min(pred_onset_steps):
                    pred_wakeup_steps = pred_wakeup_steps[1:]
                if max(pred_onset_steps) > max(pred_wakeup_steps):
                    pred_onset_steps = pred_onset_steps[:-1]
            except:
                pass

        sleep_periods = [
            (onset, wakeup)
            for onset, wakeup in zip(pred_onset_steps, pred_wakeup_steps)
            if wakeup - onset > 0
        ]  # 5sおきなので120step=10min
        for onset, wakeup in sleep_periods:
            # Scoring using mean probability over period
            pair_score = this_series_preds[onset:wakeup].mean()

            records.append(
                {
                    "series_id": series_id,
                    "step": onset,
                    "event": "onset",
                    "score": pair_score,
                }
            )
            records.append(
                {
                    "series_id": series_id,
                    "step": wakeup,
                    "event": "wakeup",
                    "score": pair_score,
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
