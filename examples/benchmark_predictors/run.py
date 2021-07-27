from crayon import benchmark

benchmark(
    "config.yml",
    "naive2",
    benchmark_id="naive_test_2",
    runs=1,
    save_benchmark=True,
    datasets_to_run=["m4_daily"],
)
# benchmark(
#     "config.yml",
#     "seasonalNaive",
#     benchmark_id="seasonalNaive_100",
#     runs=100,
#     save_benchmark=True,
# )
