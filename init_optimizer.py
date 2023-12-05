from argparse import Namespace
from typing import Literal, Iterator, Optional

from clearml import Task
from torch.optim import Adam

_optimizer_t = Literal[
    "Adam"
]

def get_optimizer(name: _optimizer_t, params: Namespace, model_params: Iterator, task: Optional[Task] = None):
    if name == "Adam":
        if task:
            task.upload_artifact(name='optimizer.name', artifact_object=name)
            task.upload_artifact(name='optimizer.params', artifact_object=vars(params))
        return Adam(params=model_params, **vars(params))
    else:
        raise f"Optimizer \"{name}\" doesn't supported"