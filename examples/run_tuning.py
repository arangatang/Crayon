import boto3
from crayon import GLUONTS_METRICS, Algorithm, Dataset, grid_search
from crayon.ConfigGenerator.gluonts_wrapper import get_gluonts_dataset
import time

kwargs = {
    "runs": 2,
    # "output_dir": "/Users/freccero/Documents/crayon",
    "output_dir": "/home/leonardo/Documents/crayon",
    "target_metric": "MASE",
}

algorithms = {
    "simplefeedforward": dict(
        name="simplefeedforward",
        #image="gluonts:gpu_101-cudnn7-ujson",
        image="gluonts:gpu_101",
        hyperparameters={
            "freq": {"$eval": "__trial__.dataset.meta.freq"},
            "prediction_length": {"$eval": "$trial.dataset.meta.prediction_length"},
            "forecaster_name": "gluonts.model.simple_feedforward.SimpleFeedForwardEstimator",
        },
        changing_hyperparameters={
            "learning_rate": [0.0005, 0.001, 0.002],
            "context_length": [
                {"$eval": "$trial.dataset.meta.prediction_length"},
                {"$eval": "2 * $trial.dataset.meta.prediction_length"},
            ],
            "num_hidden_dimensions": [[40, 40], [20, 20], [50, 50]],
        },
    ),
    "deepar": dict(
        name="deepar",
        image="gluonts:gpu_101",
        #image="gluonts:gpu_101-cudnn7-ujson",
        hyperparameters={
            "freq": {"$eval": "__trial__.dataset.meta.freq"},
            "prediction_length": {"$eval": "$trial.dataset.meta.prediction_length"},
            "forecaster_name": "gluonts.model.deepar.DeepAREstimator",
        },
        changing_hyperparameters={
            "learning_rate": [0.0005, 0.001, 0.002],
            "context_length": [
                {"$eval": "$trial.dataset.meta.prediction_length"},
                {"$eval": "2 * $trial.dataset.meta.prediction_length"},
            ],
            "num_layers": [1, 2, 4],
            "num_cells": [20, 40, 60],
        },
    ),
    "deepar_multicore": dict(
        name="deepar",
        # image="gluonts:gpu_101-cudnn7-ujson",
        #image="gluonts-patched-101-cudnn7:latest",
        image="gluonts:gpu_101",
        hyperparameters={
            "freq": {"$eval": "__trial__.dataset.meta.freq"},
            "prediction_length": {"$eval": "$trial.dataset.meta.prediction_length"},
            "forecaster_name": "gluonts.model.deepar.DeepAREstimator",
            "num_workers": 8,
            "listify_dataset": "yes",
        },
        changing_hyperparameters={
            "learning_rate": [0.0005, 0.001, 0.002],
            "context_length": [
                {"$eval": "$trial.dataset.meta.prediction_length"},
                {"$eval": "2 * $trial.dataset.meta.prediction_length"},
            ],
            "num_layers": [1, 2, 4],
            "num_cells": [20, 40, 60],
        },
    ),
}


def tune(name, datasets=["electricity", "traffic", "m4_hourly"]):
    changing = algorithms[name].pop("changing_hyperparameters")
    start = time.perf_counter()

    results = [
        grid_search(
            algorithm=Algorithm(**algorithms[name]),
            dataset=get_gluonts_dataset(ds),
            changing_hyperparameters=changing,
            **kwargs,
        )
        for ds in datasets
    ]
    print(results)
    print(f"total time: {time.perf_counter() - start}")


def tune_local():
    grid_search(
        algorithm=Algorithm(
            name="simplefeedforward",
            image="gluonts:gpu_101-cudnn7-ujson",
            hyperparameters={
                "freq": {"$eval": "__trial__.dataset.meta.freq"},
                "prediction_length": {"$eval": "$trial.dataset.meta.prediction_length"},
                "forecaster_name": "gluonts.model.simple_feedforward.SimpleFeedForwardEstimator",
            },
            changing_hyperparameters={
                "learning_rate": [0.0005, 0.001, 0.002],
                "context_length": [
                    {"$eval": "$trial.dataset.meta.prediction_length"},
                    {"$eval": "2 * $trial.dataset.meta.prediction_length"},
                ],
                "num_hidden_dimensions": [[40, 40], [20, 20], [50, 50]],
            },
        ),
        dataset=get_gluonts_dataset("electricity"),
        **kwargs,
    )

    grid_search(
        algorithm=Algorithm(
            name="deepaar",
            image="gluonts:gpu_101-cudnn7-ujson",
            hyperparameters={
                "freq": {"$eval": "__trial__.dataset.meta.freq"},
                "prediction_length": {"$eval": "$trial.dataset.meta.prediction_length"},
                "forecaster_name": "gluonts.model.deepar.DeepAREstimator",
            },
            changing_hyperparameters={
                "learning_rate": [0.0005, 0.001, 0.002],
                "context_length": [
                    {"$eval": "$trial.dataset.meta.prediction_length"},
                    {"$eval": "2 * $trial.dataset.meta.prediction_length"},
                ],
                "num_layers": [1, 2, 4],
                "num_cells": [20, 40, 60],
            },
        ),
        dataset=get_gluonts_dataset("electricity"),
        **kwargs,
    )


def tune_on_sagemaker():
    grid_search(
        metrics=GLUONTS_METRICS,
        role_arn="arn:aws:iam::817344031547:role/service-role/AmazonSageMaker-ExecutionRole-20200616T115297",
        bucket="freccero",
        algorithm=Algorithm(
            name="simplefeedforward",
            image="817344031547.dkr.ecr.eu-west-1.amazonaws.com/gluonts/cpu:v0.6.4",
            hyperparameters={
                "freq": "D",
                "prediction_length": 7,
                "forecaster_name": "gluonts.model.simple_feedforward.SimpleFeedForwardEstimator",
            },
            instance="ml.m5.xlarge",
        ),
        dataset=Dataset(
            name="electricity",
            path={
                "train": "s3://gluonts-run-tool/gluon_ts_datasets/electricity/train/data.json",
                "test": "s3://gluonts-run-tool/gluon_ts_datasets/electricity/test/data.json",
            },
        ),
        run_locally=False,
        **kwargs,
    )


if __name__ == "__main__":
    # tune("deepar")
    # tune("simplefeedforward", ["traffic", "m4_hourly"])
    tune("deepar_multicore", ["traffic", "m4_hourly"])
    # tune_on_sagemaker()
