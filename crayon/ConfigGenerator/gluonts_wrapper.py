from typing import List
import gluonts
from crayon.ConfigGenerator.generator import Dataset
from crayon.Benchmarker.benchmarker import generate_reference_config


def get_gluonts_dataset(name: str) -> Dataset:
    config = generate_reference_config([name])
    name = name.replace("-", "_")
    ds = Dataset(**config[name])
    return ds
