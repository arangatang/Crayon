from crayon import benchmark

benchmark(
    "deepar.yml",
    "deepar",
    benchmark_id="deepar_pre_bug_100_solar",
    runs=100,
    datasets_to_run=["solar_energy"],
)
