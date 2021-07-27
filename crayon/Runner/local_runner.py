from typing import Dict, Iterable, List, Union
from sagemaker.estimator import Estimator
from sagemaker.local import LocalSession
from runtool.runtool import generate_sagemaker_json
from runtool.datatypes import Experiment, Experiments
from datetime import datetime


def run_locally(
    client,
    output_path,
    experiment: Union[Experiments, Experiment],
    experiment_name: str = "default experiment name",
    runs: int = 1,
    job_name_expression: str = None,
    tags: dict = {},
) -> Dict[str, str]:
    """
    This method takes an Experiment or a Experiments object and
    executes these on SageMaker.

    Parameters
    ----------
    experiment
        A `runtool.datatypes.Experiment` object
    experiment_name
        The name of the experiment
    runs
        Number of times each job should be repeated
    job_name_expression
        A python expression which will be used to set
        the `TrainingJobName` field in the generated JSON.
    tags
        Any tags that should be set in the training job JSON

    Returns
    -------
    Dict
        Dictionary with the training job name as a key and the AWS ARN of the
        training job as a value.
    """
    json_stream = generate_sagemaker_json(
        experiment,
        runs=runs,
        experiment_name=experiment_name,
        job_name_expression=job_name_expression,
        tags=tags,
        creation_time=datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S"),
        bucket=client.bucket,
        role=client.role,
    )
    return run_training_on_local_machine(list(json_stream), output_path)


def run_training_on_local_machine(
    sm_jsons: Iterable[Dict[str, str]],
    output_path,
):
    if not output_path.startswith("file://"):
        output_path = f"file://{output_path}"

    print("running training jobs in local mode")
    paths = {}
    for run in sm_jsons:
        print(f"Starting next training job ({run['TrainingJobName']})\n")
        path_output = f'{output_path}/{run["TrainingJobName"]}'
        paths[run["TrainingJobName"]] = path_output
        inputs = {
            channel["ChannelName"]: channel["DataSource"]["S3DataSource"]["S3Uri"]
            for channel in run["InputDataConfig"]
        }

        estimator = Estimator(
            image_uri=run["AlgorithmSpecification"]["TrainingImage"],
            role="arn:aws:iam::012345678901:role/service-role/local",  # dummy role to make it work
            instance_count=1,
            instance_type="local_gpu",  # local for cpu local_gpu for gpu support
            output_path=path_output,
            tags=run["Tags"],
            metric_definitions=run["AlgorithmSpecification"]["MetricDefinitions"],
            hyperparameters=run["HyperParameters"],
        )

        estimator.fit(inputs=inputs, job_name=run["TrainingJobName"])
    return paths