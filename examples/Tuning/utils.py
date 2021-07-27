import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml
from crayon import GLUONTS_METRICS, Algorithm, Dataset, grid_search
from crayon.ConfigGenerator.gluonts_wrapper import get_gluonts_dataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset
from numpy import isin, load
import json

defaults = {
    "freq": {"$eval": "__trial__.dataset.meta.freq"},
    "prediction_length": {"$eval": "$trial.dataset.meta.prediction_length"},
    "num_workers": 6,
    "listify_dataset": "yes",
    "epochs": 100,
}
RUNS = 3


def perform_rolling(ds):
    return ds


def extract_cardinality(ds):

    cardinality = [
        json.loads(feat_static_cat["cardinality"])
        for feat_static_cat in ds.meta["feat_static_cat"]
    ]

    if len(cardinality) > 1:
        for i in cardinality:
            if not isinstance(i, int):
                raise NotImplementedError
    if isinstance(cardinality[0], list):
        cardinality = cardinality[0]

    print("extracted cardinality of:", cardinality)
    return cardinality


def tune(
    algorithm,
    path="/home/leonardo/Documents/crayon/tuning/",
    datasets=["m4_daily"],
    gpu=False,
    use_feat_static_cat=True,
    rolling=False,
    slack=True,
    force_int_cardinality=False,
):
    if slack:
        update_slack(
            title=f"Tuning commencing :mega:",
            message="Tuning of {} on {} started".format(algorithm["name"], datasets),
        )

    changing = algorithm.pop("changing_hyperparameters")
    start = time.perf_counter()
    start_time = datetime.now().strftime("%B-%d-%Y--%H-%M-%S")

    algorithm["image"] = (
        "gluonts:gpu_101" if gpu else "arangatang/masterthesis:gluonts_commit_4d1a9a0"
    )
    algorithm["hyperparameters"].update(defaults)

    algo = Algorithm(**algorithm)
    results = []
    try:
        for ds in datasets:
            if slack:
                update_slack(
                    title=f"Grid search started :the_horns:",
                    message="Tuning of {} on {} started".format(algorithm["name"], ds),
                )

            ds_path = Path(path) / algo.name / ds
            last_start = time.perf_counter()
            dss = get_gluonts_dataset(ds)
            # print(dss)

            if rolling:
                dss = perform_rolling(dss)

            if use_feat_static_cat == True:
                cardinality = extract_cardinality(dss)
                if force_int_cardinality:
                    cardinality = cardinality[0]
                algo.hyperparameters["cardinality"] = cardinality
                algo.hyperparameters["use_feat_static_cat"] = True

            print(algo)
            # print(dss)
            best = grid_search(
                algorithm=algo,
                dataset=dss,
                changing_hyperparameters=changing,
                runs=RUNS,
                output_dir=ds_path,
                target_metric="MASE",
            )
            if not best:
                update_slack(
                    title="All jobs failed",
                    message="Tuning of {} on {} failed".format(algorithm["name"], ds),
                )
                continue

            results.append(best)

            output = json.dumps(dict(best), indent=2)
            runtime = f"Time for grid_search was: {time.perf_counter() - last_start}"
            print(output, "\n", runtime)
            with open(ds_path / "results.jsonl", "a+") as fp:
                fp.write(output)
                fp.write("\n")

            if slack:
                update_slack(
                    title="Grid search finished :white_check_mark:",
                    message="Tuning of {} on {} finished, best config:\n```\n{}\n```\n{}".format(
                        algorithm["name"], ds, output, runtime
                    ),
                )

        results = json.dumps(list(map(dict, results)), indent=2)
        message = "Tuning of {} on {} finished\nResults:\n```\n{}\n```\ntotal time: {}".format(
            algorithm["name"],
            datasets,
            results,
            time.perf_counter() - start,
        )
        path = Path(path) / algo.name / f"grid_search_summary_{start_time}.json"
        with open(path, "w+") as fp:
            fp.write(results)

        print(message)
        if slack:
            update_slack(
                title="Tuning ended :partying_face:",
                message=f"{message}\nartifacts stored to: {path}",
            )

    except Exception as e:
        if slack:
            update_slack(
                title="Tuning failed due to crash :boom:",
                message="Tuning of {} on {} crashed with error ```\n{}\n```".format(
                    algorithm["name"], datasets, e
                ),
            )
        raise


def update_slack(title, message, url=None):
    """Send message to slack"""
    if not url:
        return
    slack_data = {
        "username": "NotificationBot",
        "icon_emoji": ":satellite:",
        "channel": "#training_jobs",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{title}*",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{message}",
                },
            },
        ],
    }
    byte_length = str(sys.getsizeof(slack_data))
    headers = {"Content-Type": "application/json", "Content-Length": byte_length}
    response = requests.post(url, data=json.dumps(slack_data), headers=headers)
    if response.status_code != 200:
        print(response)
        pass


def load_multivariate_dataset(dataset_name: str):
    ds = get_dataset(dataset_name)
    grouper_train = MultivariateGrouper()
    grouper_test = MultivariateGrouper()
    return (
        TrainDatasets(
            metadata=ds.metadata,
            train=grouper_train(ds.train),
            test=grouper_test(ds.test),
        ),
        len(ds.train),
        len(ds.test),
    )


def store_multivariate_ds(name: str):
    ds = load_multivariate_dataset(name)


def pprint(data):
    print(yaml.dump(data))


def get_dataset_info(name: str):
    ds = get_dataset(name)
    pprint(ds.metadata)
    print(vars(ds.train))
    print("train_data", len(ds.train("target")))
    print("test_data", len(ds.test("target")))


def get_ds_cardinality(name: str) -> int:
    ds = get_dataset(name)
    print(ds)
    exit()
    return len(ds.train("target"))


if __name__ == "__main__":
    get_dataset_info("electricity")