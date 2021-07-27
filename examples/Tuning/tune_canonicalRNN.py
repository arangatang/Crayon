from utils import tune

algo = dict(
    name="canonicalRNN",
    hyperparameters={
        "forecaster_name": "gluonts.model.canonical.CanonicalRNNEstimator",
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "num_layers": [1, 2, 3],
        "num_cells": [30, 50, 70],
    },
)

# tune(algo)
# tune(algo, datasets=["electricity"])
tune(algo, gpu=False)
