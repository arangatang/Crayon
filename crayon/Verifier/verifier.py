from scipy import stats
from crayon.Runner import Jobs
import numpy as np


def ks_test(
    sample_a: Jobs, sample_b: Jobs, metric_name: str, p_limit: int = 0.05
):
    """
    Returns True if sample_a and sample_b are sampled from the same distribution.
    I.e. if the kolmogorov smirnow test outputs a p value higher than the provided
    p_limit
    """
    _, p = stats.ks_2samp(
        sample_a.metrics[metric_name],
        sample_b.metrics[metric_name],
        alternative="two-sided",
        mode="auto",
    )

    return p > p_limit


def calc_cdf_fit(sample, confidence=0.05):
    """
    Returns how likely it is for the CDF to faile to correctly depict the
    underlying distribution. Rewrite of the Dvoretzky-Kiefer-Wolfowitz inequality.
    """
    return np.sqrt(np.log(2.0 / confidence) / (2 * len(sample)))


def verify(
    sample_a: Jobs, sample_b: Jobs, target_metric: str, ignore_warnings=False
):
    if not ignore_warnings and (
        min(
            len(sample_a.metrics[target_metric]),
            len(sample_b.metrics[target_metric]),
        )
        < 10
        or max(
            len(sample_a.metrics[target_metric]),
            len(sample_b.metrics[target_metric]),
        )
        < 50
    ):
        print(
            "Warning, atleast one of the sample has very few values. it is recommended "
            "that each collection of samples that are to be tested should have minimum 50 samples.\n"
            f"The samples has {len(sample_a)} and {len(sample_b)} items. "
            "Due to this, the accuracy of the kolmogorov smirnov test may be inaccurate."
        )

    return ks_test(sample_a, sample_b, target_metric)
