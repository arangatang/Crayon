from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.dataset.common import ListDataset, TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx.trainer import Trainer
import mxnet as mx


def load_dataset(dataset_name: str):
    ds = get_dataset(dataset_name)

    return TrainDatasets(
        metadata=ds.metadata,
        train=ListDataset(ds.train, freq=ds.metadata.freq),
        test=ListDataset(ds.test, freq=ds.metadata.freq),
    )


dataset = load_dataset("electricity")
estimator = MQCNNEstimator(
    prediction_length=dataset.metadata.prediction_length,
    freq=dataset.metadata.freq,
    trainer=Trainer(ctx=mx.gpu()),
)


print("Starting the training")

predictor = estimator.train(training_data=dataset.train)

print("starting backtest")
agg_metrics, _ = backtest_metrics(
    test_dataset=dataset.test,
    predictor=predictor,
)
print("MASE", agg_metrics["MASE"])
