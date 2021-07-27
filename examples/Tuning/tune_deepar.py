from utils import tune

deepar = dict(
    name="deepar",
    hyperparameters={
        "forecaster_name": "gluonts.model.deepar.DeepAREstimator",
        # "use_feat_dynamic_real": True,
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "num_layers": [1, 2, 3],
        "num_cells": [20, 40, 60],
    },
)

# tune(deepar)
tune(deepar, gpu=False)