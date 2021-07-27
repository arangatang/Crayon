from utils import tune

forecaster = "gluonts.model.wavenet.WaveNetEstimator"
algo = dict(
    name=forecaster.split(".")[-1],
    hyperparameters={
        "forecaster_name": forecaster,
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "embedding_dimension": [3, 5],
        "num_bins": [512, 1024],
        "hybridize_prediction_net": [True, False],
        "n_residue": [12, 24],
        "n_skip": [16, 32],
        "n_stacks": [1, 2],
    },
)

tune(algo)
