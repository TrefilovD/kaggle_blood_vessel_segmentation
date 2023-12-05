from typing import Literal

from .attention_gated_unet import AttentionGatedUnet

MODELS = {
    "AttentionGatedUnet": AttentionGatedUnet
}

_model_t = Literal[
    "AttentionGatedUnet"
]

from .init_model import get_model