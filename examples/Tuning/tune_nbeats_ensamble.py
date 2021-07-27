from utils import tune

algo = dict(
    name="NBEATSEnsembleEstimator",
    hyperparameters={
        "forecaster_name": "gluonts.model.n_beats.NBEATSEstimator",
        "loss_function": "MASE",
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "meta_bagging_size": [5, 10],
        "num_stacks": [20, 30],
        "num_blocks": [[1], [2]],
        "widths": [[256], [512]],
    },
)

# tune(algo, continue_from="/home/leonardo/Documents/crayon/tuning/NBEATSEnsembleEstimator/electricity/gridsearch/March-24-2021--15-10-44")
tune(algo, gpu=False)
