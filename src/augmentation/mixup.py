import numpy as np
import torch


class Mixup:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mixup augmentation.

        Args:
            imgs (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (torch.Tensor): (batch_size, n_timesteps, n_classes)

        Returns:
            tuple[torch.Tensor]: mixed_imgs (batch_size, n_channels, n_timesteps)
                                 mixed_labels (batch_size, n_timesteps, n_classes)
        """
        batch_size = imgs.size(0)
        idx = torch.randperm(batch_size)
        lam = np.random.beta(self.alpha, self.alpha)

        mixed_imgs: torch.Tensor = lam * imgs + (1 - lam) * imgs[idx]
        mixed_labels: torch.Tensor = lam * labels + (1 - lam) * labels[idx]

        return mixed_imgs, mixed_labels


class SwapMixup:
    """
    onset, wakeupイベントを含むwindowを取得し、それを別の日のイベントのwindowとmixupする
    イベント近傍のパターンが増えることで多様性を出す
    """

    def __init__(self, window_size: int = 100, swap_channels=[0, 1], alpha: float = 0.4):
        self.window_size = window_size
        self.swap_channels = swap_channels
        self.alpha = alpha

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
        if (
            (idx - self.window_size < 0)
            or (swap_idx - self.window_size < 0)
            or (idx + self.window_size >= len(feature))
            or (swap_idx + self.window_size >= len(swap_feature))
            or (idx - self.window_size >= len(feature))
            or (swap_idx - self.window_size >= len(swap_feature))
        ):
            return feature

        lam = np.random.beta(self.alpha, self.alpha)
        feature[idx - self.window_size : idx + self.window_size, self.swap_channels] = (
            lam * feature[idx - self.window_size : idx + self.window_size, self.swap_channels]
            + (1 - lam)
            * swap_feature[
                swap_idx - self.window_size : swap_idx + self.window_size, self.swap_channels
            ]
        )
        return feature
