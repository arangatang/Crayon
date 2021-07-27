from crayon import benchmark

benchmark(
    algorithm_config="config.yml",
    algorithm_name="deepar",
    target_metric="MASE",
    benchmark_id="my benchmark",
)
