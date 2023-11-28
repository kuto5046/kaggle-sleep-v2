from typing import Optional

import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.utils.criterions import DiceLoss
from src.utils.criterions import ImbalancedL1Loss


class Spec1D(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
        imbalanced_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.channels_fc = nn.Linear(feature_extractor.out_chans, 1)
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn1 = nn.BCEWithLogitsLoss()
        self.loss_fn2 = ImbalancedL1Loss(imbalanced_loss_weight)
        self.loss_fn3 = DiceLoss()

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        # n_channelとheightを結合
        batch_size = x.shape[0]
        n_timesteps = x.shape[-1]
        x = x.view(batch_size, -1, n_timesteps)  # (batch_size, n_channels*height, n_timesteps
        logits = self.decoder(x)  # (batch_size, n_classes, n_timesteps)

        output = {"logits": logits}
        if labels is not None:
            loss1 = self.loss_fn1(logits, labels)
            loss2 = self.loss_fn2(
                logits[:, :, 0].sigmoid().diff(dim=1), labels[:, :, 0].diff(dim=1)
            )
            loss3 = self.loss_fn3(logits.sigmoid(), labels)
            output["loss"] = loss1 + loss2 + loss3

        return output
