"""
This file calculates CDF for different functions in the interval [0,10] sampled at each whole number
"""
import numpy as np
import matplotlib.pyplot as plt
import random

# first 25% data sum/len(data which is summed), the close this avg is to the optimal value, the better it is


def resize_list(data: list, length: int) -> list:
    def interpolate(data, new_index, delta):
        # Split floating-point index into whole & fractional parts.
        fp_index = new_index * delta
        index = int(fp_index // 1)
        remainder = fp_index % 1
        next_index = index + 1 if remainder > 0 else index
        return (1 - remainder) * data[index] + remainder * data[next_index]

    def merge(data, new_index, delta):
        # TODO
        raise NotImplementedError

    delta = (len(data) - 1) / (length - 1)

    print(delta)
    if delta <= 1:
        return [interpolate(data, i, delta) for i in range(length)]
    else:
        return [merge(data, i, delta) for i in range(length)]


def score(data: list, strategy: str = "cdf"):
    def score_sum(data: list, percentage: int):
        # 1. convert from 25% to 0.25
        percentage = percentage / 100

        # 2. convert to index in data 0.25 * len([1,2,3,4]) => 0.25*4=1
        floating_index = percentage * len(data)
        index = int(floating_index // 1)

        # 3. sum all values up to this index
        data = np.sort(data)

        # 4. handle remainder of floating_index
        # TODO

        # Return the average over the range
        if index:
            return np.average(data[0:index])
        else:
            return data[0]

    def score_cdf(data: list, percentage: int):
        # 1. sort
        data = np.sort(data)

        # 1. convert from 25% to 0.25
        percentage = percentage / 100

        # 2. convert to index in data 0.25 * len([1,2,3,4]) => 0.25*4=1
        floating_index = percentage * len(data)
        index = int(floating_index // 1)
        if index == len(data):
            return data[-1]
        return data[index]

    scores = None
    if strategy == "sum":
        scores = [score_sum(data, i) for i in range(1, 101)]
    elif strategy == "cdf":
        scores = [score_cdf(data, i) for i in range(1, 101)]
    return scores


def cdf(data):
    # 1. sort
    data = np.sort(data)

    # calculate the proportional values of samples
    p = 1.0 * np.arange(len(data)) / (len(data) - 1)
    return {"data": data, "cdf": p}


functions = {
    "linear": lambda x: x,
    "constant": lambda x: 5.5,
    "threshold": lambda x: 0 if x < 5 else 5 if x == 5 else 10,
    "random": lambda x: random.randint(1, 10),
    "custom": lambda x: [1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5] * 20,
}

result = {}
fig, axs = plt.subplots(len(functions))
plot_index = iter(range(len(functions)))
for function_name, function in functions.items():
    print()
    distribution = [function(x) for x in range(1, 11)]
    if isinstance(distribution[0], list):
        distribution = distribution[0]

    print(function_name, distribution)
    res = score(distribution)
    res_sum = score(distribution, "sum")
    # likely you will see a value less than this,
    # punishes heavy tails far from optimum
    print("score cdf 90%:", res[89])
    # probably you will see a value like this
    # benefits distributions with a peak closer to 0
    print("score average 90%", res_sum[89])

    # if you get lucky you can expect values less than this
    # benefits distributions with some values close to 0
    print("score cdf 10%", res[9])
    # most likely you will see a value such as this if you are lucky
    # benefits consistently low scores
    print("score average 10%", res_sum[9])
    """
    Lower score implies less error. 
    By adding them together distributions which has low values
    all over is benefited.
    the averages are added as it is both of interest to have a low
    "lucky" value if one is interested in the optimal value.
    """
    print("score combined", res_sum[9] + res[9] + res_sum[89] + res[89])
    print(
        "score combined multiplied",
        (res_sum[9] + res[9]) * (res_sum[89] + res[89]),
    )
    p = lambda x: print(f"fn.{x}:", "{:.2f}".format(res[x - 1]))
    li = [5, 25, 50, 75, 95, 100]
    for point in li:
        p(point)

    result[function_name] = cdf(distribution)

    ax = axs[next(plot_index)]
    ax.plot(result[function_name]["data"], result[function_name]["cdf"])
    ax.set_title(function_name)
    ax.set_xlabel("data")
    ax.set_ylabel("cdf")
    ax.set_xlim(0, 10)

fig.tight_layout()
fig.show()
# input()
