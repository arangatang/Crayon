"""
This file should, given a docker image name and a dataset generate a runtool config.yml file
"""
from pydantic import BaseModel
import yaml
import json
from pathlib import Path


class ConfigBase(BaseModel):
    name: str

    def __str__(self):
        return json.dumps(vars(self))


class Algorithm(ConfigBase):
    image: str
    hyperparameters: dict = {}
    instance: str = "local"


class Dataset(ConfigBase):
    meta: dict = {}
    path: dict


def generate_config(
    algorithm: Algorithm,
    dataset: Dataset,
    path: str = None,
    metrics: dict = None,
):
    conf = {algorithm.name: dict(algorithm), dataset.name: dict(dataset)}

    for channel in conf[dataset.name]["path"]:
        if not conf[dataset.name]["path"][channel].startswith("file://"):
            conf[dataset.name]["path"][channel] = Path(
                conf[dataset.name]["path"][channel]
            ).as_uri()

    if metrics:
        conf[algorithm.name]["metrics"] = metrics

    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w+") as fp:
            yaml.dump(conf, fp)

    return conf


if __name__ == "__main__":
    a = Algorithm(
        name="myalgo", image="my_image", hyperparameters={"epochs": 10}
    )
    ds = Dataset(
        name="my_ds", path={"train": "file://smth", "test": "file://smthelse"}
    )
    yml = generate_config(a, ds)
    print(yml)
