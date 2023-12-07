import polars as pl

from src.utils.metrics import event_detection_ap


def prepare_gt(series_id="7476c0bd18d2"):
    # Load labels
    labels = pl.read_csv(
        "/home/kuto/kaggle/kaggle-sleep-v2/data/child-mind-institute-detect-sleep-states/train_events.csv"
    ).drop_nulls()
    labels = labels.filter(pl.col("series_id") == series_id)
    return labels


def test_there_are_two_prediction():
    """
    gtからdiff離れたところに最もscoreの高いピーク(0.6)があり、その次に高いscore(0.5)がGTのstepにある場合
    diff分離れたピークがまずgtとのtoleranceに基づいて評価される。その後scoreが低いピークも評価されそれはscore=1になって平均が取られる。
    scoreが高い順に見る→そのstepに近い未マッチのgtを見る→予測とGTをマッチさせる
    """
    target_events = prepare_gt()

    # 予測のサンプルを作成
    diff = 36
    pred_events = target_events.clone().select(["series_id", "event", "step"])
    pred_events = pred_events.with_columns(
        pl.lit(0.5).alias("score"),
    )
    pred_events = pl.concat(
        [
            pred_events,
            pred_events.with_columns(
                pl.col("step") + diff,
                pl.lit(0.6).alias("score"),
            ),
        ]
    )
    score = event_detection_ap(target_events.to_pandas(), pred_events.to_pandas())
    print(f"score:{score}")
