from crayon import benchmark

benchmark(
    "tuned_simplefeedforward.yml",
    "simplefeedforward",
    benchmark_id="simplefeedforward_100_runs",
)
