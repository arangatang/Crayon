from utils import tune

forecaster = "gluonts.model.seq2seq.Seq2SeqEstimator"
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
        "mlp_final_dim": [10, 20, 30],
        "mlp_hidden_dimension_seq": [[], [1], [1, 1], [2], [2, 2]],
    },
)

tune(algo)
