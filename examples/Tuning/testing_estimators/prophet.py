from gluonts.model.prophet import ProphetPredictor
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.dataset.common import ListDataset, TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset


def load_dataset(dataset_name: str):
    ds = get_dataset(dataset_name)

    return TrainDatasets(
        metadata=ds.metadata,
        train=ListDataset(ds.train, freq=ds.metadata.freq),
        test=ListDataset(ds.test, freq=ds.metadata.freq),
    )


dataset = load_dataset("electricity")
estimator = ProphetPredictor(
    prediction_length=dataset.metadata.prediction_length,
    freq=dataset.metadata.freq,
)


print("Starting the training")

print("starting backtest")
agg_metrics, _ = backtest_metrics(
    test_dataset=dataset.test,
    predictor=estimator,
)
print("MASE", agg_metrics["MASE"])
