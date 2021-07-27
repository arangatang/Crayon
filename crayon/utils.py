from pathlib import Path
from typing import List, Union
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import gc
import seaborn as sns
import pandas
import math


def write_to_file(data, path):
    """
    stores data to path and creates any dirs on the way.
    """
    raise NotImplementedError


def crayon_dir():
    path = Path.home() / "Documents" / "crayon"
    path.mkdir(parents=True, exist_ok=True)
    return path


def crayon_results():
    path = crayon_dir() / "results.yml"
    path.touch(exist_ok=True)
    return path


def crayon_plots_dir():
    path = crayon_dir() / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_benchmark_names(brf: Union[Path, str] = crayon_results()):
    if brf:
        if isinstance(brf, str):
            brf = Path(brf)
    else:
        brf = crayon_results()

    # opening as yaml becomes very slow thus string manipulation is faster
    previous_benchmarks = (
        row.split("  benchmark_id: ")[1].replace("\n", "")
        for row in brf.open("r")
        if "  benchmark_id: " in row
    )

    names = defaultdict(lambda: 0)
    for id in previous_benchmarks:
        names[id] = names[id] + 1

    print("existing benchmarks:", names)


def cm2inch(*tupl):
    tupl = (21 * tupl[0], 29.7 * tupl[1])
    inch = 2.54
    return tuple(i / inch for i in tupl)


def fix_latex_formatting(string: str):
    return string.replace("_", "\_")


def create_violin_plots(
    ids: List[str],
    output: Union[str, Path] = crayon_plots_dir(),
    brf: Union[str, Path] = crayon_results(),
    overwrite=False,
    use_id: bool = False,
    orient="v",
):
    from crayon.Runner import Jobs

    if isinstance(output, str):
        output = Path(output)

    if isinstance(brf, str):
        brf = Path(brf)
    with brf.open("r") as fp:
        previous_benchmarks = yaml.safe_load(fp)

    if not previous_benchmarks:
        print("no benchmarks found.")
        exit(1)

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 28,
            "text.usetex": True,
        }
    )

    path = output / "violin_plots" / "_".join(ids)
    path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    df_data = defaultdict(lambda: defaultdict(lambda: {}))
    for bench in previous_benchmarks:
        if bench["benchmark_id"] in ids:
            jobs = Jobs.from_list(bench["jobs"])
            ds = fix_latex_formatting(bench["dataset_name"])

            id = bench["benchmark_id"] if use_id else bench["algorithm_name"]
            id = fix_latex_formatting(id)
            for metric in jobs.metrics.keys():
                df_data[metric][ds][id] = jobs.metrics[metric]

    for metric, ds_benchmark in df_data.items():
        for ds, data in ds_benchmark.items():
            fig_path = path / ds / f"{metric}.png"
            if not overwrite and fig_path.exists():
                continue
            ax.cla()
            fig_path.parent.mkdir(parents=True, exist_ok=True)

            df = pandas.DataFrame.from_dict(data)

            sns.set(style="whitegrid")
            sns.violinplot(data=df, ax=ax, orient=orient).set(
                title=f"{fix_latex_formatting(metric)} on dataset {ds}",
            )
            if orient == "v":
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
            plt.tight_layout()

            fig.savefig(fig_path)


def create_benchmark_histogram(
    id: str = None,
    output: Union[str, Path] = crayon_plots_dir(),
    brf: Union[str, Path] = crayon_results(),
    overwrite: bool = False,
):
    from crayon.Runner import Jobs

    if isinstance(output, str):
        output = Path(output)

    if isinstance(brf, str):
        brf = Path(brf)

    with brf.open("r") as fp:
        previous_benchmarks = yaml.safe_load(fp)

    if not previous_benchmarks:
        print("no benchmarks found.")
        exit(1)

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 28,
            "text.usetex": True,
        }
    )
    fig, ax = plt.subplots()
    for bench in previous_benchmarks:
        if (not id) or id == bench["benchmark_id"]:

            benchmark_path = output / bench["benchmark_id"] / bench["dataset_name"]
            benchmark_path.mkdir(parents=True, exist_ok=True)
            print(
                "Saving plots for benchmark:",
                bench["benchmark_id"],
                "to",
                benchmark_path,
            )
            jobs = Jobs.from_list(bench["jobs"])
            for name, metrics in jobs.metrics.items():

                if not overwrite and (benchmark_path / f"{name}.png").exists():
                    # print("skipping: ", name)
                    continue
                print("saving", name)
                try:
                    metrics.plot(
                        kind="hist", title=fix_latex_formatting(name), color="b", ax=ax
                    )
                    ax.set_xlabel("Error")
                    ax.set_ylabel("Occurences")
                    mini = min(metrics)
                    maxi = max(metrics)
                    if maxi != mini:
                        ax.set_xlim([mini, maxi])
                    plt.tight_layout()
                    fig.savefig(benchmark_path / f"{name}.png")
                    ax.cla()
                except Exception as e:
                    print("An exeption occured: \n\n")
                    print(e)


def generate_latex_ranking(
    ids: List[str],
    metric: str,
    brf: Union[str, Path] = crayon_results(),
    use_floats: bool = True,
    rename: dict = {},
):

    from crayon.Runner import Jobs
    from crayon.Benchmarker.benchmarker import get_ranking, visualize

    ranking = get_ranking(metric, ids=ids, brf=Path(brf))

    rankings = defaultdict(lambda: {})
    worst_ranks = {}
    for ds, _ in ranking.ranks.items():
        rank = 1
        for i in ranking.rankings[ds]:
            tmp = dict(
                name=fix_latex_formatting(
                    rename[i["benchmark_id"]]
                    if i["benchmark_id"] in rename
                    else i["algorithm_name"]
                ),
                id=fix_latex_formatting(i["benchmark_id"]),
            )
            tmp[ds] = i["score"]
            tmp[f"{ds}_rank"] = rank

            rankings[i["benchmark_id"]].update(tmp)
            if "mean_rank" in rankings[i["benchmark_id"]]:
                rankings[i["benchmark_id"]]["mean_rank"].append(rank)
            else:
                rankings[i["benchmark_id"]]["mean_rank"] = [rank]

            if not math.isnan(i["score"]):
                rank += 1
        worst_ranks[ds] = rank

    for id in rankings:
        rankings[id]["mean_rank"] = np.mean(rankings[id]["mean_rank"])
    rankings = list(rankings.values())

    rankings.sort(key=lambda i: i["mean_rank"])
    # print(yaml.dump(rankings, indent=1))
    # print each row
    def row(
        use_floats,
        name,
        electricity=np.nan,
        electricity_rank=worst_ranks["electricity"],
        solar_energy=np.nan,
        solar_energy_rank=worst_ranks["solar_energy"],
        m4_daily=np.nan,
        m4_daily_rank=worst_ranks["m4_daily"],
        m5=np.nan,
        m5_rank=worst_ranks["m5"],
        mean_rank=np.nan,
        id=None,
    ):
        # in case of missing datasets, the mean rank is not counted properly, do it here instead
        mean_rank = np.mean(
            [electricity_rank, solar_energy_rank, m4_daily_rank, m5_rank]
        )
        if use_floats:
            return f"{name} & {electricity:.2f} ({electricity_rank}) & {solar_energy:.2f} ({solar_energy_rank}) & {m4_daily:.2f} ({m4_daily_rank})  & {m5:.2f} ({m5_rank}) & {mean_rank:.2f} \\\\\\hline"
        return f"{name} & {electricity:.0f} ({electricity_rank}) & {solar_energy:.0f} ({solar_energy_rank}) & {m4_daily:.0f} ({m4_daily_rank})  & {m5:.0f} ({m5_rank}) & {mean_rank:.2f} \\\\\\hline"

    print(
        """
\\begin{table}[htb]
\\centering
\\begin{tabular}{ccccccc}
\\hline
Algorithm & \\rothalf{Electricity} & \\rothalf{Solar Energy} & \\rothalf{M4 Daily} & \\rothalf{M5} & \\rothalf{Mean rank} \\\\
\\hline"""
    )
    for i in rankings:
        print(row(use_floats, **i))
    print(
        """\\end{tabular}
\\caption{Benchmark results}
\\label{tab:benchmark_results_METRIC_USED}
\\end{table}
    """.replace(
            "METRIC_USED", metric
        )
    )

    print("\n\n")
    rankings.sort(key=lambda i: i["name"])
    ranks = {i["name"]: i["mean_rank"] for i in rankings}
    print(ranks)


if __name__ == "__main__":
    brf = "C:/Users/leona/Documents/masterthesis/masterthesis/results.yml"
    ids = [
        "deepar_100",
        "TransformerEstimator_100",
        "NBEATSEnsamble",
        "simplefeedforward_100_runs",
        "MQCNNEstimator",
        "canonicalRNN",
        "GaussianProcessEstimator",
        "MQRNNEstimator",
        "deepFactorEstimator",
        "naive2_100",
        "seasonalNaive_100",
        "thetaf",
        "prophet_100",
    ]
    rename = {
        "deepar_100": "DeepAR",
        "TransformerEstimator_100": "Transformer",
        "NBEATSEnsamble": "NBEATS",
        "simplefeedforward_100_runs": "SimpleFF",
        "MQCNNEstimator": "MQCNN",
        "canonicalRNN": "C-RNN",
        "GaussianProcessEstimator": "GP",
        "MQRNNEstimator": "MQRNN",
        "deepFactorEstimator": "DeepFactor",
        "naive2_100": "Naive2",
        "seasonalNaive_100": "S-Naive",
        "thetaf": "Thetaf",
        "prophet_100": "Prophet",
    }
    generate_latex_ranking(ids, rename=rename, metric="MASE", brf=brf, use_floats=True)
    generate_latex_ranking(ids, rename=rename, metric="MAPE", brf=brf, use_floats=True)
    generate_latex_ranking(
        ids, rename=rename, metric="abs_error", brf=brf, use_floats=False
    )
    generate_latex_ranking(ids, rename=rename, metric="RMSE", brf=brf, use_floats=True)
    generate_latex_ranking(ids, rename=rename, metric="MSIS", brf=brf, use_floats=True)
    # print_benchmark_names(brf=brf)
    # # create_benchmark_histogram(brf=brf, overwrite=False)
    # create_violin_plots(
    #     ["deepar_pre_bug_100", "deepar_bugged_100", "deepar_fixed_100"],
    #     brf=brf,
    #     overwrite=True,
    #     use_id=True,
    # )
    # create_violin_plots(
    #     [
    #         "MQCNNEstimator",
    #         "canonicalRNN",
    #         "GaussianProcessEstimator",
    #     ],
    #     overwrite=True,
    # )

    # create_violin_plots(
    #     [
    #         "MQRNNEstimator",
    #         "deepFactorEstimator",
    #     ],
    #     overwrite=True,
    # )

    # create_violin_plots(["GaussianProcessEstimator"], overwrite=True, orient="h")
