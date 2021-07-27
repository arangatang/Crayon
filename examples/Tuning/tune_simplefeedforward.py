from utils import tune

algo = dict(
    name="simplefeedforward",
    hyperparameters={
        "forecaster_name": "gluonts.model.simple_feedforward.SimpleFeedForwardEstimator",
    },
    changing_hyperparameters={
        # "learning_rate": [0.0001, 0],
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "num_hidden_dimensions": [[40, 40], [20, 20], [60, 60]],
    },
)

# tune(algo)
tune(algo, gpu=False)