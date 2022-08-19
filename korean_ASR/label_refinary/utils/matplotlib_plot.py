import logging
import argparse

from pathlib import Path
import matplotlib.pyplot as plt

from typing import List


def plot_cdf(
        data,
        save_path,
        bins,
        dataset_type,
        train_type,
):
    save_path = Path(f"{save_path}/filtered_transcript/{train_type}/{dataset_type}/cdf.png")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(data, bins=bins, cumulative=-1)
    fig.savefig(save_path)


def plot_valid(
        data_list: List[str],
        metric_type: str,
        save_path: str,
        ):
    data_dict = {d: float(d.split("/")[-3]) for d in data_list}
    sorted_dict = sorted(data_dict.items(), key=lambda item: item[1])
    sorted_list = [k for (k, v) in sorted_dict]

    filter_weight_list = [float(s.split("/")[-3]) for s in sorted_list]
    score_list = [float(s.split("|")[-3]) for s in sorted_list]

    fig, ax = plt.subplots(1, 1)

    ax.plot(filter_weight_list,score_list)
    plt.ylim([min(score_list)-1, max(score_list)+1])
    plt.title(metric_type)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{metric_type}_rescored.png")

    return sorted_list


def plot_rescored_result(args: argparse.Namespace, logger: logging.RootLogger) -> None:
    train_type = args.train_type
    valid_type = args.valid_type
    save_path = args.save_path

    save_path = Path(f"{save_path}/filtered_score/{train_type}/{valid_type}/")

    result_path = save_path / "RESULTS.md"
    sorted_result_path = save_path / "RESULTS_SORTED.md"

    metric_types = ['WER','CER','SWER']
    with result_path.open("r", encoding='utf-8') as f:
        res = f.readlines()

    st_pos = [idx+4 for idx,r in enumerate(res) if r.startswith("###")]
    line_size = (st_pos[1]-1) - (st_pos[0]+4)

    with sorted_result_path.open("w", encoding='utf-8') as f:
        for idx, metric_type in enumerate(metric_types):
            logger.info("plot generated in {}/{}_rescore.png for {}".format(save_path, metric_type, metric_type))
            sorted_list = plot_valid(res[st_pos[idx]:st_pos[idx]+line_size], metric_type, save_path)

            f.writelines(res[st_pos[idx]-5:st_pos[idx]])
            f.writelines(sorted_list)






