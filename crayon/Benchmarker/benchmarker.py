"""
This file should given a config, calculate how it performs in comparison to 
previously run algorithms on the same dataset.
"""
import json
from collections import defaultdict
from datetime import datetime
import math
from pathlib import Path
from typing import Callable, Union

import boto3
import numpy as np
from scipy.stats.stats import percentileofscore
import yaml
from crayon.Runner import run_config
from crayon.utils import crayon_dir, crayon_results
from gluonts.dataset.common import CategoricalFeatureInfo
from gluonts.dataset.repository.datasets import (
    get_dataset,
    materialize_dataset,
)
import pandas as pd
from crayon.Runner import Jobs
from uuid import uuid4
import hashlib
from crayon.Verifier.verifier import verify


class Ranking:
    def __init__(self, ranks: dict, rankings: dict):
        self.ranks = ranks
        self.rankings = rankings


def reproducible_hash(*args):
    return str(hashlib.sha1("".join(args).encode("UTF-8")).hexdigest())


def generate_reference_config(
    reference_datasets=["electricity", "m4_daily", "m5", "solar-energy"]
):
    def parse_meta_value(value):
        if value and isinstance(value, list):
            return [
                {"name": v.name, "cardinality": v.cardinality}
                if isinstance(v, CategoricalFeatureInfo)
                else v
                for v in value
            ]
        else:
            return value

    config = {}
    for name in reference_datasets:
        print(
            f"Fetching dataset {name} this may take some time the first time this is run."
        )
        path = materialize_dataset(name)
        meta = dict(get_dataset(name).metadata)

        train_path = path / "train" / "data.json"
        test_path = path / "test" / "data.json"

        assert train_path.exists(), "Unable to find train data"
        assert test_path.exists(), "Unable to find test data"

        name = name.replace("-", "_")
        config[name] = {
            "name": name,
            "path": {
                "train": train_path.resolve().as_uri(),
                "test": test_path.resolve().as_uri(),
            },
            "meta": {k: parse_meta_value(v) for k, v in meta.items()},
        }

    return config


def cdf_scoring_strategy():
    """
    Calculates the "distance" needed on from the optimal value (i.e. 0 for abs_error)
    on the CDF in order to reach 25%, 50%, 75%, 100% coverage of the distribution.

    In the example below, the data for these percentiles are refered to as
    fn.25, fn.50, fn.75 and fn.100

        %
        ^
    100%|                                            --------------
        |                                       ----/
        |                                    --/
        |                                   /
        |                                -/
    50% |                  -------------/
        |     ------------/
        |    /
        |   /
        | -/
    0%  + ---------------------------------------------------------> Error
              ^CDF=0.25     ^CDF=0.5          ^CDF=0.75 ^CDF=1
        <---->
        fn.25
        <----------------->tmp
        fn.50
        <------------------------------------>
        fn.75
        <---------------------------------------------->
        fn.100

    In order to compare two distributions using these values;
    The one with a lower `fn` results in better accuracy.

    Lower `fn.25` for error distribution `A` than error distribution `B` means
    that `A` achieves higher accuracy than `B`
    """
    raise NotImplementedError


def score(data: list, strategy: str = "cdf"):
    def score_avg(data: list, percentage: int):
        # 1. convert from 25% to 0.25
        percentage = percentage / 100

        # 2. convert to index in data 0.25 * len([1,2,3,4]) => 0.25*4=1
        floating_index = percentage * len(data)
        index = int(floating_index // 1)

        # 3. sum all values up to this index
        data = np.sort(data)

        # 4. handle remainder of floating_index
        # TODO

        # Return the average over the range
        if index:
            return np.average(data[0:index])
        else:
            return data[0]

    def score_cdf(data: list, percentage: int):
        # 1. sort
        data = np.sort(data)

        # 1. convert from 25% to 0.25
        percentage = percentage / 100

        # 2. convert to index in data 0.25 * len([1,2,3,4]) => 0.25*4=1
        floating_index = percentage * len(data)
        index = int(floating_index // 1)
        if index == len(data):
            return data[-1]
        return data[index]

    scores = None
    if strategy == "avg":
        scores = [score_avg(data, i) for i in range(1, 101)]
    elif strategy == "cdf":
        scores = [score_cdf(data, i) for i in range(1, 101)]
    return scores


def RMSEforDistributions(data: list):
    if isinstance(data, list):
        data = np.array(data)

    return np.sqrt(np.mean(np.square(data)))


def default_scoring_strategy(data: list):
    return RMSEforDistributions(data)
    # cdf = score(data, "cdf")
    # averages = score(data, "avg")
    # # give distributions with a heavy tail close to optimal value
    # # a lower score (lower is better)
    # return (cdf[25] + averages[25]) * (cdf[89] + averages[89])


def calc_score(jobs: Jobs, target_metric: str):
    try:
        data_to_score = jobs.metrics[target_metric]
        return default_scoring_strategy(data=data_to_score)
    except Exception:
        return np.nan


def generate_ranking(latest_results: list, target_metric: str):
    # remove any previous benchmarks run on different datasets
    # or which did not track the same target metric.
    dataset_names = [run["dataset_name"] for run in latest_results]

    def same_metrics_and_datasets(item):
        if item["dataset_name"] not in dataset_names:
            return False

        for job in item["jobs"]:
            if not target_metric in job["metrics"]:
                return False
        return True

    with crayon_results().open("r") as fp:
        previous_benchmarks = yaml.safe_load(fp)

    if not previous_benchmarks:
        previous_benchmarks = []

    # calculate the score of these benchmarks for the specified target metric
    scored = []
    for run in filter(same_metrics_and_datasets, previous_benchmarks):
        scored.append(
            {
                "score": calc_score(Jobs.from_list(run["jobs"]), target_metric),
                **run,
            }
        )

    for run in latest_results:
        scored.append(
            {
                "score": calc_score(Jobs.from_list(run["jobs"]), target_metric),
                **run,
            }
        )
    # group by dataset used
    groups = {}
    for benchmark in scored:
        ds = benchmark["dataset_name"]
        if ds in groups:
            groups[ds].append(benchmark)
        else:
            groups[ds] = [benchmark]

    # Find the ranking for each dataset as they may differ
    for ds_name in groups:
        max_score = max(i["score"] for i in groups[ds_name]) + 1
        groups[ds_name] = sorted(
            groups[ds_name],
            key=lambda item: max_score if math.isnan(item["score"]) else item["score"],
        )

    # calculate ranking of the latest run based on score
    ranks = {}
    for ds, ranking in groups.items():
        for i in range(len(ranking)):
            if ranking[i]["benchmark_id"] == latest_results[0]["benchmark_id"]:
                ranks[ds] = i
    return Ranking(ranks=ranks, rankings=dict(groups))


def visualize(ranking: Ranking):
    # TODO build web interface
    print("\n\t================ RANKING ================")
    for ds, rank in ranking.ranks.items():
        print("-" * len(ds))
        print(ds)
        print("-" * len(ds))
        if rank is not None:
            print("the rank of the latest run on", ds, "was:", rank + 1, "\n")
        cleaned = {
            "rank": [i for i in range(1, len(ranking.rankings[ds]) + 1)],
            "score": [i["score"] for i in ranking.rankings[ds]],
            "algorithm": [i["algorithm_name"] for i in ranking.rankings[ds]],
            "benchmark_id": [i["benchmark_id"] for i in ranking.rankings[ds]],
        }
        df = pd.DataFrame(cleaned)
        print(df.to_string(index=False, header=True), "\n")


def load_benchmark(benchmark_id: str):
    """
    Loads the data related to a specific benchmark.

    The data returned from this function contains:

    - the config which was used
    - the experiment definitions
    - the Jobs objects related to the benchmark
    - ids for the experiment and the benchmark
    - output paths
    """

    with open(crayon_results()) as fp:
        benchmarks = yaml.safe_load(fp)

    if benchmarks:
        return list(filter(lambda i: i["benchmark_id"] == benchmark_id, benchmarks))
    return False


def plot_benchmark_distribution(benchmark_id: str):
    """
    plots the distribution
    """
    raise NotImplementedError


def benchmark(
    algorithm_config: Union[dict, str],
    algorithm_name: str = "algorithm",
    # dataset_config: Union[dict, str] = None,  # TODO
    # dataset_name: str = None,  # TODO
    # cloud: bool = False,  # TODO
    target_metric: str = "abs_error",
    session: boto3.Session = boto3.Session(),
    save_benchmark: bool = True,
    benchmark_id: str = datetime.now().strftime("%Y/%m/%d/%H-%M-%S"),
    runs=100,
):
    # check if the benchmark id has been used before
    assert not load_benchmark(
        benchmark_id
    ), "Previous benchmark with same id exists, choose another id."

    # 1. generate reference config
    datasets = generate_reference_config()

    # 2. run algorithm on each dataset
    if isinstance(algorithm_config, str):
        with open(algorithm_config, "r") as fp:
            algorithm_config = yaml.safe_load(fp)

    base_dir = crayon_dir() / "benchmark"
    output_dir = base_dir / Path(datetime.now().strftime("%Y/%m/%d/%H-%M-%S"))

    results = []
    for ds_name in datasets:
        if f"{algorithm_name}_{ds_name}" in algorithm_config:
            print(f"Using tuned version {algorithm_name}_{ds_name}")
            algo = f"{algorithm_name}_{ds_name}"
        else:
            algo = algorithm_name

        conf = {
            **algorithm_config,
            ds_name: datasets[ds_name],
        }
        print(
            "starting training of",
            algo,
            "on dataset:",
            ds_name,
        )

        experiment = f"config.{algo} * config.{ds_name}"
        try:
            jobs = run_config(
                config=conf,
                combination=experiment,
                session=session,
                runs=runs,
                local_output_dir=output_dir,
            )
        except Exception as e:
            print(f"An exception occured when running {algo} on {ds_name}.")
            print(e)
            jobs = Jobs([])

        print("Results of", algo, "running on dataset:", ds_name)
        print(jobs.metrics.to_string(index=False, header=True))

        new_results = {
            "experiment_id": reproducible_hash(
                algorithm_name, ds_name
            ),  # TODO look on data instead of names when calculating hash
            "algorithm_name": algorithm_name,
            "algorithm_version": algo,
            "dataset_name": ds_name,
            "config": conf,
            "experiment": experiment,
            "output_path": str(output_dir.resolve()),
            "jobs": jobs.to_dict(),
            "benchmark_id": benchmark_id,
        }
        results.append(new_results)

        # calc score for latest results
        print(
            f"Crayon score for {target_metric} (lower is better):",
            calc_score(jobs, target_metric),
        )

    # compare scores to previous benchmarks
    ranking = generate_ranking(latest_results=results, target_metric=target_metric)
    visualize(ranking)

    # 4. store distributions somewhere (cloud preferably) for now locally
    if save_benchmark:

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        path = crayon_results()
        with path.open("a") as fp:
            yaml.dump(results, fp, Dumper=NoAliasDumper)
            print("stored benchmarking results to:", path)


def get_ranking(target_metric: str, brf: Path = crayon_results(), ids: list = None):
    with brf.open("r") as fp:
        previous_benchmarks = yaml.safe_load(fp)

    if not previous_benchmarks:
        print("no benchmarks found.")
        previous_benchmarks = []

    # calculate the score of these benchmarks for the specified target metric
    scored = []
    for run in previous_benchmarks:
        if ids and run["benchmark_id"] not in ids:
            continue

        scored.append(
            {
                "score": calc_score(Jobs.from_list(run["jobs"]), target_metric),
                **run,
            }
        )

    # group by dataset used
    groups = {}
    for benchmark in scored:
        ds = benchmark["dataset_name"]
        if ds in groups:
            groups[ds].append(benchmark)
        else:
            groups[ds] = [benchmark]

    # Find the ranking for each dataset as they may differ

    for ds_name in groups:
        max_score = max(i["score"] for i in groups[ds_name]) + 1
        groups[ds_name] = sorted(
            groups[ds_name],
            key=lambda item: max_score if math.isnan(item["score"]) else item["score"],
        )

    # calculate ranking of the latest run based on score
    return Ranking(ranks={ds_name: None for ds_name in groups}, rankings=dict(groups))


def display_ranking(target_metric: str, brf: Path = crayon_results()):
    """
    prints the complete ranking table with all datasets and algorithms
    """
    visualize(get_ranking(target_metric, brf))


def verify_if_benchmarks_behave_the_same(
    id_1: str,
    id_2: str,
    target_metric: str,
    id_1_results_file: Path = crayon_results(),
    id_2_results_file: Path = crayon_results(),
):
    """
    Checks whether two seperate runs of a benchmark performs the "same".
    This is for proving that some results are reproducible.
    i.e. if an algorithm still has the same accuracy as before.
    """
    # 1. load the benchmarks with the ids 1 and 2 from the results.yml file
    def load(path, id):
        with path.open("r") as fp:
            benchmarks = yaml.safe_load(fp)

        relevant = filter(lambda i: i["benchmark_id"] == id, benchmarks)
        return {i["dataset_name"]: Jobs.from_list(i["jobs"]) for i in relevant}

    data_1 = load(id_1_results_file, id_1)
    data_2 = load(id_2_results_file, id_2)

    # for all dataset which both benchmarks ran on
    num_passed = num_failed = 0
    for ds in (key for key in data_1 if key in data_2):
        passed = verify(data_1[ds], data_2[ds], target_metric, False)
        if not passed:
            print(
                f"The behaviour of the algorithm could not be verified on dataset: {ds}"
            )
            num_failed += 1
        else:
            num_passed += 1
            print(f"Algorithm verified on dataset {ds}")

    print(f"Passed {num_passed}/{num_passed + num_failed} verifications.")
