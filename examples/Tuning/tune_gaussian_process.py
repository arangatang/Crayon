from utils import tune, get_ds_cardinality

algo = dict(
    name="gaussianProcessEstimator",
    hyperparameters={
        "forecaster_name": "gluonts.model.gp_forecaster.GaussianProcessEstimator",
        # "cardinality": 321, #electricity
        "cardinality": {
            "$eval": "int($trial.dataset.meta.feat_static_cat[0].cardinality)"
        },  # solar
        "learning_rate": 0.01,
        "context_length": {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        "sample_noise": False,
        "params_scaling": False,
        "max_iter_jitter": 5,
    },
    changing_hyperparameters={
        "learning_rate": [0.0001, 0.001, 0.01],
        "context_length": [
            {"$eval": "$trial.dataset.meta.prediction_length"},
            {"$eval": "2 * $trial.dataset.meta.prediction_length"},
        ],
        "sample_noise": [True, False],
        "params_scaling": [True, False],
        "max_iter_jitter": [5, 10],
    },
)

# algo["hyperparameters"]["cardinality"] = 321  # electricity
# tune(algo, datasets=["electricity"], gpu=False)
# algo["hyperparameters"]["cardinality"] = 137  # solar-energy
# algo["hyperparameters"]["cardinality"] = 4227  # m4_daily
# algo["hyperparameters"]["cardinality"] = None  # m5 has several which is not supported

# tune(algo_electricity, datasets=["electricity"])
tune(
    algo,
    datasets=["m5"],
    gpu=False,
    force_int_cardinality=True,
)
