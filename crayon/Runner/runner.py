"""
This file should import the runtool
Essentially it should wrap the runtool such that by specifying a 
config file, the system will run the algorithm on some dataset in it.
"""

import runtool
from typing import List, Union, Dict, Iterable
import pandas as pd
from functools import singledispatch
import boto3
from botocore.exceptions import WaiterError
from itertools import chain
import tempfile
import tarfile
import json
import yaml
from pydantic import BaseModel
import pathlib
from crayon.Runner.local_runner import run_locally


class Job(BaseModel):
    """
    Interface for a single training job.
    """

    metrics: dict
    name: str
    source: Union[dict, str]

    def __repr__(self):
        return f"Name: {self.name}\nSource:\n{yaml.dump(self.source)}"

    def to_dict(self):
        return vars(self)

    @classmethod
    def from_dict(cls, dct):
        return Job(**dct)


class Jobs:
    """
    Has convienent functionality for working with training jobs
    """

    def __init__(self, jobs: List[Job]):
        self.data = jobs
        self.metrics = pd.DataFrame(job.metrics for job in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Jobs(self.data[key])
        return self.data[key]

    def to_dict(self):
        return [job.to_dict() for job in self.data]

    @classmethod
    def from_list(cls, job_list: list):
        return Jobs([Job.from_dict(d) for d in job_list])


@singledispatch
def to_jobs(runs) -> Jobs:
    """
    handles when the runs are sagemaker JSON objects
    """

    jobs = [
        Job(
            metrics={
                metric["MetricName"]: metric["Value"]
                for metric in run["FinalMetricDataList"]
            },
            name=run["TrainingJobName"],
            source=run,
        )
        for run in runs
    ]

    return Jobs(jobs)


@to_jobs.register
def to_jobs_local(runs: dict):
    """
    Loads job data from disk
    """
    jobs = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        for name, path in runs.items():
            path = pathlib.Path(path.replace("file://", ""))
            archive_path = path / "model.tar.gz"
            output_path = tmpdir / name

            with tarfile.open(archive_path.resolve()) as tf:
                
                import os
                
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tf, path=output_path.resolve())
                with (output_path / "agg_metrics.json").open("r") as fp:
                    metrics = json.load(fp)

            jobs.append(
                Job(
                    metrics=metrics,
                    name=name,
                    source=str(archive_path.resolve()),
                )
            )
    return Jobs(jobs)


def wait_for_jobs(session: boto3.Session, job_names: List[str]) -> None:
    """
    Waits until each job has either completed, stopped or crashed.
    """
    sm = session.client("sagemaker")
    for name in job_names:
        print("waiting for job to finish:", name)
        try:
            sm.get_waiter("training_job_completed_or_stopped").wait(
                TrainingJobName=name
            )
        except WaiterError:
            print(
                f"The training job: {name} seems to have failed. See link below for further details:\nhttps://console.aws.amazon.com/sagemaker/home?#/jobs/{name}"
            )
        print("finished waiting")


def fetch_job_jsons(session: boto3.Session, job_names: List[str]) -> Iterable:
    """
    Uses sagemaker search to query for training jobs
    """
    streams = []
    max_results = len(job_names)
    while job_names:
        total_length = 0
        names = []
        while job_names and total_length + 1 + len(job_names[-1]) < 1000:
            name = job_names.pop(-1)
            names.append(name)
            total_length += len(name) + 1

        paginator = session.client("sagemaker").get_paginator("search")
        pages = paginator.paginate(
            Resource="TrainingJob",
            PaginationConfig={"PageSize": min(100, max_results)},
            MaxResults=max_results,
            SearchExpression={
                "Filters": [
                    {
                        "Name": "TrainingJobName",
                        "Operator": "In",
                        "Value": ",".join(names),
                    }
                ]
            },
        )

        streams.append(job["TrainingJob"] for page in pages for job in page["Results"])

    return chain.from_iterable(streams)


def run_config(
    config: Union[dict, str],
    combination: str,
    bucket="",
    role_arn: str = "",
    session: boto3.Session = boto3.Session(region_name="eu-west-1"),
    local_output_dir: Union[str, pathlib.Path] = pathlib.Path.home()
    / "Documents"
    / "crayon_output",
    runs: int = 1,
    job_name_expression: str = "'{}-{}-{}'.format(__trial__.algorithm.get('name', 'default-algo'), __trial__.dataset.get('name', 'default-algo'), uid)",
) -> Jobs:
    """
    Uses the runtool to execute an algorithm on a dataset defined in a config file.
    Fetches the results of the jobs from either disk or sagemaker and returns them
    as Jobs objects.
    """
    rt = runtool.Client(role=role_arn, bucket=bucket, session=session)

    if isinstance(config, dict):
        config = runtool.runtool.transform_config(config)
    else:
        config = runtool.load_config(config)

    experiment = eval(combination)

    if local_output_dir:
        print("starting run in local mode")
        if isinstance(local_output_dir, pathlib.Path):
            local_output_dir = local_output_dir.resolve().as_uri()

        if not local_output_dir.startswith("file://"):
            local_output_dir = f"file://{local_output_dir}"

        runs = run_locally(
            rt,
            experiment=experiment,
            runs=runs,
            output_path=local_output_dir,
            job_name_expression=job_name_expression,
        )
        print("Local run artifacts saved at:", runs)
    else:
        print("starting runs on sagemaker")
        names = rt.run(experiment=experiment, runs=runs)
        wait_for_jobs(session=session, job_names=names)
        runs = fetch_job_jsons(session=session, job_names=names)

    return to_jobs(runs)
