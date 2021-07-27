from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.mx.trainer import Trainer
import mxnet
from utils import load_multivariate_dataset

dataset, train_len, test_len = load_multivariate_dataset("electricity", dim=1)
metadata = dataset.metadata

print("Starting the training")
estimator = DeepVAREstimator(
    prediction_length=dataset.metadata.prediction_length,
    target_dim=train_len,
    freq=metadata.freq,
    trainer=Trainer(epochs=1, ctx=mxnet.cpu()),
)
predictor = estimator.train(training_data=dataset.train)

print("starting backtest")
agg_metrics, _ = backtest_metrics(
    test_dataset=dataset.test,
    predictor=predictor,
)
print(agg_metrics)
print(agg_metrics["MASE"])
