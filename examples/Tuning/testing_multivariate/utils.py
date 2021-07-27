from gluonts.dataset.common import ListDataset, TrainDatasets
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.evaluation import MultivariateEvaluator
from gluonts.dataset.repository.datasets import get_dataset


def load_multivariate_dataset(dataset_name: str, dim: int = 1):
    ds = get_dataset(dataset_name)
    grouper = MultivariateGrouper(max_target_dim=dim)
    return (
        TrainDatasets(
            metadata=ds.metadata,
            train=grouper(ListDataset(ds.train, freq=ds.metadata.freq)),
            test=grouper(ListDataset(ds.test, freq=ds.metadata.freq)),
        ),
        dim,
        len(ds.test),
    )