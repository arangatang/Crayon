from crayon import benchmark

benchmark(
    "tuned_algorithms.yml",
    "TransformerEstimator",
    benchmark_id="TransformerEstimator_100",
    runs=100,
)
