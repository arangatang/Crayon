from utils import tune

algo = dict(
    name="deepFactorEstimator",
    hyperparameters={
        "forecaster_name": "gluonts.model.deep_factor.DeepFactorEstimator",
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "num_hidden_global": [30, 50],
        "num_layers_global": [1, 2],
        "num_factors": [7, 10],
        "num_layers_local": [1, 2],
        "num_hidden_local": [3, 5, 7],
    },
)

# tune(algo)
tune(algo, gpu=False)
