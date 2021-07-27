from utils import tune

algo = dict(
    name="lstnet",
    hyperparameters={
        "forecaster_name": "gluonts.model.lstnet.LSTNetEstimator",
        "num_series": "$trial.dataset.meta.feat_static_cat[0].cardinality",
        "channels": {"$eval": "$trial.algorithm.hyperparameters.context_length"},
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "skip_size": [1, 2],
        "ar_window": [1, 5, 10],
        "rnn_num_layers": [1, 2],
        "rnn_num_cells": [50, 100],
        "skip_rnn_num_layers": [1, 2],
        "skip_rnn_num_cells": [5, 10],
    },
)

tune(algo)
