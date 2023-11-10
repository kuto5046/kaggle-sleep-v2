import numpy as np
import torch


class SwapEvent:
    """
    onset, wakeupイベントを含むwindowを取得し、それを別の日のイベントのwindowと入れ替える
    前後関係が変化するためそこに多様性を出す
    入れ替えるのはlabelとセンサの特徴量

    """

    def __init__(self, window_size: int = 100, swap_channels=[0, 1]):
        self.window_size = window_size
        self.swap_channels = swap_channels

    def __call__(
        self,
        feature: np.ndarray,
        label: np.ndarray,
        swap_feature: torch.Tensor,
        swap_label: np.ndarray,
        event: str,
    ) -> np.ndarray:
        """Swap Event augmentation.

        Args:
            feature: (n_timesteps, n_channels)
            label: (n_timesteps, n_classes)

        Returns:
            np.array feature: (n_timesteps, n_channels)
        """
        event = 1 if event == "onset" else 2

        idxes = np.where(label[:, event] == 1)[0]
        swap_idxes = np.where(swap_label[:, event] == 1)[0]
        if len(idxes) == 0 or len(swap_idxes) == 0:
            return feature

        idx = idxes[0]
        swap_idx = swap_idxes[0]
        feature[
            idx - self.window_size : idx + self.window_size, self.swap_channels
        ] = swap_feature[
            swap_idx - self.window_size : swap_idx + self.window_size, self.swap_channels
        ]
        return feature
