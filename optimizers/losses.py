import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds   = torch.sigmoid(logits)
        inter   = (preds * targets).sum(dim=(2, 3))
        union   = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice    = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice        = DiceLoss()
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice(logits, targets)
                + self.bce_weight  * self.bce(logits,  targets))

    def update_weights(self, dice_weight: float, bce_weight: float):
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight