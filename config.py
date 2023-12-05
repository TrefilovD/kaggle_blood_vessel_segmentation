import yaml

from argparse import Namespace

def parse_config(path: str) -> Namespace:
    assert path.endswith(".yaml") or path.endswith(".yml")

    with open(path, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read config successful")

    for key in cfg.keys():
        if isinstance(cfg[key], dict):
            cfg[key] = Namespace(**cfg[key])

    cfg = Namespace(**cfg)

    return cfg
