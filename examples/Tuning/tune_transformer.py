from utils import tune

algo = dict(
    name="TransformerEstimator",
    hyperparameters={
        "forecaster_name": "gluonts.model.transformer.TransformerEstimator",
    },
    changing_hyperparameters=dict(
        learning_rate=[0.0001, 0.001, 0.01],
        context_length=[
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        embedding_dimension=[10, 20, 30],
        model_dim=[16, 32, 64],
        inner_ff_dim_scale=[2, 4, 6],
    ),
)

# tune(algo)
tune(algo, gpu=False)