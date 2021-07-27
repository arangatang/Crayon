from crayon.Benchmarker.benchmarker import load_benchmark
from crayon.Runner import Jobs
import yaml

benchmark = load_benchmark("2021/01/25/19-50-43")

for partial_benchmark in benchmark:
    print(partial_benchmark["algorithm_name"])
    print(partial_benchmark["dataset_name"])
    print(yaml.dump(partial_benchmark["config"]))
    print(Jobs.from_list(partial_benchmark["jobs"]).metrics.head())
