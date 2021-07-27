from crayon import run_config
import boto3

jobs = run_config(
    config="config.yml",
    combination="config.simplefeedforward * config.constant",
    role_arn="arn:aws:iam::817344031547:role/service-role/AmazonSageMaker-ExecutionRole-20200616T115297",
    bucket="freccero",
    local_output_dir=None,
    session=boto3.Session(),
)
print(jobs[0])
