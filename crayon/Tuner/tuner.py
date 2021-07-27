"""
This file should allow one to tune a docker image using 
different tuning strategies. Initially only grid search
should be implemented but later on maybe other alternatives
can be added such as bayesian search or similar.

One requirement is that each individual configuration 
could be trained a couple of times in order to approximate
its distribution.

One should then be able to choose the best run based on some criteria
i.e max, min
furthermore, one should be able to aggregate the results somehow
i.e.
one may wish to only compare the max values for each run group
or one may wish to average the runs.

possibly this could be done by providing a lambda function.
"""
from typing import Callable
import statistics
from crayon import run_config
from crayon.ConfigGenerator import generate_config, Algorithm, Dataset
from crayon.utils import crayon_dir
from copy import deepcopy
import yaml
from itertools import product
from datetime import datetime
import json
from pydantic import BaseModel
from typing import Union
from pathlib import Path


class GridSearchResults(BaseModel):
    config: str
    value: Union[float, int]
    parameters: dict
    path: str


def grid_search(
    changing_hyperparameters: dict,  # {'epochs':[10, 20, 30], 'freq': ['D', 'H']}
    target_metric: str,  # i.e. abs error
    dataset: Dataset,
    algorithm: Algorithm,
    output_dir: str = crayon_dir().resolve(),
    aggregation_function: Callable = statistics.mean,  # i.e. mean(run1, run2, run3)
    evaluation_function: Callable = lambda new, old: new < old,  # i.e. min
    run_locally: bool = True,
    metrics: dict = None,
    **kwargs,
) -> GridSearchResults:
    # get all combinations of changing hyperparameters
    combos = product(
        *(
            [(key, val) for val in value]
            for key, value in changing_hyperparameters.items()
        )
    )
    combos = [dict(combination) for combination in combos]

    # do grid search
    best_combo = None
    best_value = None
    i = 0
    date = datetime.now().strftime("%B-%d-%Y--%H-%M-%S")
    base_path = Path(output_dir) / "gridsearch" / date
    for num, combo in enumerate(combos):
        print(f"running configuration {num+1} of {len(combos)}")
        # use new hyperparameter configuration
        # and generate new runtool config
        path = base_path / f"config-{i}.yml"

        new = deepcopy(algorithm)
        new.hyperparameters.update(combo)
        generate_config(
            algorithm=new,
            dataset=dataset,
            path=path.resolve(),
            metrics=metrics,
        )

        # execute the runtool
        print(
            f"Training of {algorithm.name} on dataset {dataset.name} commencing.\nParameters under test:\n\n{yaml.dump(combo)}\n"
        )
        try:
            jobs = run_config(
                config=str(path.resolve()),
                combination=f"config['{algorithm.name}'] * config['{dataset.name}']",
                local_output_dir=base_path / "jobs" if run_locally else None,
                **kwargs,
            )
        except Exception as e:
            print(f"Exception occured {e}, failing config: {path.resolve()}")
            continue

        # aggregate runs
        if len(jobs) > 1:
            print("Multiple runs detected, aggregating the results")
            print(jobs.metrics[target_metric])
            result = aggregation_function(jobs.metrics[target_metric])
        else:
            result = jobs[0].metrics[target_metric]

        # Calculate if better than best so far
        print(f"Best metric value: {best_value}\nNew value: {result}")
        if not best_value or evaluation_function(result, best_value):
            print("found new best combination")
            print(f"New best value: {result}")
            print(
                f"Best found hyperparameter configuration\n{yaml.dump(combo, indent=1)}"
            )
            best_value = result
            best_combo = combo
            best_config = path

        i += 1

    if not best_config:
        print("All jobs failed.")

    print(
        "Tuning finished.\n",
        "Best hyperparameter configuration:\n",
        yaml.dump(best_combo),
        "\nBest recorded metric:",
        best_value,
        "\nThe best config can be found in:",
        best_config,
        "\nAll configs can be found in:",
        base_path,
    )

    return GridSearchResults(
        config=str(best_config.resolve()),
        value=best_value,
        parameters=best_combo,
        path=str(base_path.resolve()),
    )
