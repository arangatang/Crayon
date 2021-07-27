# Crayon

## Prerequisites
The runtool need to be installed from

https://github.com/awslabs/gluon-ts-tools/tree/main/runtool

## Installation

Tested on Python 3.8.5

```bash
pip install -e .
```

Therafter a AWS config file need to be created in `~/.aws/config` on unix based systems and on `C:\Users\USERNAME\.aws\config` if on Windows.  

Inside it the following content need to be placed

```
[default]
region = eu-west-1
```
The region used is ignored when executing locally. However it is a requirement from the boto3 library even though it is ignored when executing locally.


## Examples
Below are commands to run Tuning, Benchmarking, Verification and to present the rankings of all benchmarks.

### Tune
Example scripts for tuning are available in examples/Tuning. After tuning finished, the config file used for the best configuration is presented.
Note that `examples/Tuning` contains a helper script `examples/Tuning/utils.py` which wraps the `grid_search` functionality of Crayon and is intended for GluonTS datasets and algorithms.

```bash
cd examples/Tuning
python tune_example.py
```
For using the tuned configs of multiple datasets together these need to be combined manually. Note that the names of the algorithms in the config should have the format `<algorithm name>_<dataset name>` for the benchmarking system to know which config corresponds to which dataset.

### Benchmark
Example benchmark of deepar or simplefeedforward 
```bash
cd examples/benchmark
python run.py
```

### Verify
When two or more benchmarks has been run, the benchmark ids can be added to the `examples/verify.py` to assert reproducibility.
Thereafter, run:
```
cd examples
python verify.py
```

### Display ranking
After some benchmarks has been run, the rankings can be displayed.

```bash
cd examples
python display_rankings.py
```