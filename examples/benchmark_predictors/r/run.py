from crayon import benchmark

benchmark(
    "config.yml",
    "thetaf",
    benchmark_id="theta_m5_2",
    runs=100,
    save_benchmark=True,
    datasets_to_run=["m5"],
    ignore_exceptions=True,
)
