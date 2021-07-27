from crayon import run_config
import boto3

jobs = run_config(
    config="config.yml",
    combination="config.algorithms * config.datasets",
    runs=2,
    local_output_dir="/Users/freccero/Documents/tmp",
)
print(len(jobs))
print(jobs.metrics.head())
