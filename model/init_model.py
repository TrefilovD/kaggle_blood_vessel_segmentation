from argparse import Namespace
from typing import Optional

import torch

from clearml import Task

from model import MODELS, _model_t


def get_model(name: _model_t, params: Namespace, task: Optional[Task] = None) -> torch.nn.Module:
    builder_model = MODELS.get(name)
    if builder_model:
        if task:
            task.upload_artifact(name='model.name', artifact_object=name)
            task.upload_artifact(name='model.params', artifact_object=vars(params))
        return builder_model(**vars(params))
    raise f"Model \"{name}\" doesn't supported"