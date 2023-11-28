from typing import Iterable
from typing import List
from typing import Set
from typing import cast

import torch
from torch import einsum
from torch import nn


class ImbalancedMSELoss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, logits, targets):
        loss = self.criterion(logits, targets)
        loss[targets != 0] *= self.weight
        return loss.mean()


class ImbalancedL1Loss(nn.Module):
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


def uniq(a: torch.Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: torch.Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def simplex(t: torch.Tensor, axis=1) -> bool:
    _sum = cast(torch.Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: torch.Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


class BoundaryLoss:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss
