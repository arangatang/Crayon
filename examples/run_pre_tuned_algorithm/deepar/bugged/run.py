from crayon import benchmark

benchmark(
    "deepar.yml",
    "deepar",
    benchmark_id="deepar_bugged_100_solar",
    runs=100,
    datasets_to_run=["solar_energy"],
)
