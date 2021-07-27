from utils import tune

algo = dict(
    name="nbeatsEstimator",
    hyperparameters={
        "forecaster_name": "gluonts.model.n_beats.NBEATSEstimator",
        "sharing": [False],
        "expansion_coefficient_lengths": [32],
        "stack_types": ["G"],
        "loss_function": "MASE",
        "block_layers": [4],
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "num_stacks": [20, 30, 40],
        "num_blocks": [1, 2],
        "widths": [256, 512],
    },
)

tune(algo)
