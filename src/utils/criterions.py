import torch
from torch import nn


class InbalancedMSELoss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, logits, targets):
        loss = self.criterion(logits, targets)
        loss[targets != 0] *= self.weight
        return loss.mean()


class InbalancedL1Loss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.criterion = nn.L1Loss(reduction="none")

    def forward(self, logits, targets):
        loss = self.criterion(logits, targets)
        loss[targets != 0] *= self.weight
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = self.criterion(logits, targets)
        probs = logits.sigmoid()
        loss = torch.where(
            targets >= 0.5,
            self.alpha * (1.0 - probs) ** self.gamma * bce_loss,
            probs**self.gamma * bce_loss,
        )
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probs = logits.sigmoid()
        batch_size = targets.size(0)
        m1 = probs.view(batch_size, -1)
        m2 = targets.view(batch_size, -1)
        intersection = (m1 * m2).sum()
        union = m1.sum() + m2.sum()
        return 1 - ((2.0 * intersection + 1) / (union + 1))
