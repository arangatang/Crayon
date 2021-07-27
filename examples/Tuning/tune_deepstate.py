from utils import tune

algo = dict(
    name="deepStateEstimator",
    hyperparameters={
        "forecaster_name": "gluonts.model.deepstate.DeepStateEstimator",
        "use_feat_static_cat": False,
        "cardinality": [0],
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "num_layers": [30, 50, 70],
        "num_cells": [2, 3],
        "add_trend": [True, False],
    },
)

# tune(algo)
tune(algo, gpu=False)  # Neither CPU or GPU works
