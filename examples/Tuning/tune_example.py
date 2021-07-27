from utils import tune

algo = dict(
    name="simplefeedforward",
    hyperparameters={
        "forecaster_name": "gluonts.model.simple_feedforward.SimpleFeedForwardEstimator",
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001],
    },
)

# tune(algo)
tune(algo, gpu=False)