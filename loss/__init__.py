from typing import Literal

from .dice_coefficient import dice_coeff
from .focal_loss import FocalLoss

LOSSES = {
    "FocalLoss": FocalLoss
}

_loss_t = Literal[
    "FocalLoss"
]

from .init_loss import get_loss