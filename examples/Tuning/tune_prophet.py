from utils import tune

# for params look here
# https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py
# docker image gluonts:prophet

algo = dict(
    name="ProphetPredictor",
    hyperparameters={
        "forecaster_name": "gluonts.model.prophet.ProphetPredictor",
    },
    changing_hyperparameters=dict(
        prophet_params=[
            dict(growth="linear"),
            # {"growth": "logistic"},
            # {"growth": "linear", "seasonality_mode": "multiplicative"},
            # {"growth": "logistic", "seasonality_mode": "multiplicative"},
        ],
    ),
)

tune(algo, datasets=["electricity"])
