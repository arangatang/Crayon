from crayon import benchmark

benchmark(
    "config.yml",
    "prophet",
    benchmark_id="prophet_100_runs_m4_daily",
    runs=100,
    save_benchmark=True,
    datasets_to_run=["m4_daily"],
)
