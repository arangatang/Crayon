from crayon.ConfigGenerator import Algorithm, Dataset, generate_config
import yaml

# Define your algorithm
algo = Algorithm(
    name="myalgo", image="my_image", hyperparameters={"epochs": 10}
)

# Define dataset related info
ds = Dataset(
    name="my_ds",
    path={
        "train": "/Users/freccero/.mxnet/gluon-ts/datasets/electricity/train/data.json",
        "test": "/Users/freccero/.mxnet/gluon-ts/datasets/electricity/test/data.json",
    },
)

# Convert this to runtool compatible yml and optionally store to file
config = generate_config(algo, ds, "/Users/freccero/Documents/tmp/config.yml")

print(f"Created the config:\n\n{yaml.dump(config)}")
