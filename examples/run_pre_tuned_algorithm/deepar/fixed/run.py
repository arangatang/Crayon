from crayon import benchmark

benchmark(
    "deepar.yml",
    "deepar",
    benchmark_id="deepar_fixed_100_m4_daily_final",
    runs=100,
    datasets_to_run=["m4_daily"],
)
