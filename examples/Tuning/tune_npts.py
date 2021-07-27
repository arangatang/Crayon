from utils import tune

algo = dict(
    name="npts",
    hyperparameters={
        "forecaster_name": "gluonts.model.npts.NPTSEstimator",
    },
    changing_hyperparameters={
        # "learning_rate": [0.0001, 0.001, 0.01], # no learning rate as it is a predictor
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "use_seasonal_model": [True, False],
        "use_default_time_features": [True, False],
        "feature_scale": [500, 1000, 1500],
    },
)

tune(algo)
