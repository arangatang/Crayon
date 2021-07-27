from gluonts.model.npts import NPTSEstimator
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.dataset.common import ListDataset, TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset


def load_dataset(dataset_name: str, dim: int = 1):
    ds = get_dataset(dataset_name)

    return TrainDatasets(
        metadata=ds.metadata,
        train=ListDataset(ds.train, freq=ds.metadata.freq),
        test=ListDataset(ds.test, freq=ds.metadata.freq),
    )


dataset = load_dataset("electricity")
estimator = NPTSEstimator(
    prediction_length=dataset.metadata.prediction_length,
    freq=dataset.metadata.freq,
    # context_length =
    kernel_type="exponential"
    # exp_kernel_weights =
    # use_seasonal_model = [True, False],
    # use_default_time_features = [True, False],
    # feature_scale = [500, 1000, 1500]
)


print("Starting the training")

predictor = estimator.train(training_data=dataset.train)

print("starting backtest")
agg_metrics, _ = backtest_metrics(
    test_dataset=dataset.test,
    predictor=predictor,
)
print("MASE", agg_metrics["MASE"])
