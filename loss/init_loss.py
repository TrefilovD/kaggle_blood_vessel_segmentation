from argparse import Namespace
from typing import Optional

import torch

from clearml import Task

from loss import LOSSES, _loss_t


def get_loss(name: _loss_t, params: Namespace, task: Optional[Task] = None) -> torch.nn.Module:
    builder_loss = LOSSES.get(name)
    if builder_loss:
        if task:
            task.upload_artifact(name='loss.name', artifact_object=name)
            task.upload_artifact(name='loss.params', artifact_object=vars(params))
        return builder_loss(**vars(params))
    raise f"Loss \"{name}\" doesn't supported"