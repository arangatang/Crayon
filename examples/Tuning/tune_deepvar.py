from utils import tune

algo = dict(
    name="deepvar",
    hyperparameters={
        "forecaster_name": "gluonts.model.deepvar.DeepVAREstimator",
        "target_dim": "$trial.dataset.meta.feat_static_cat[0].cardinality",
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "num_layers": [1, 2, 3],
        "num_cells": [20, 40, 60],
        "rank": [3, 5, 7],
        "pick_incomplete": [True, False],
        "use_marginal_transformation": [True, False],
    },
)

# tune(algo)
tune(algo)
