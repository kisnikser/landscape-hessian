import os
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import argparse
from omegaconf import OmegaConf
from collections import defaultdict


myparams = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "Djvu Serif",
    "font.size": 16,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
}
plt.rcParams.update(myparams)


def calculate_mean_std(accuracy_samples):
    """
    Calculate the mean and standard deviation of accuracy across samples.

    Args:
        accuracy_samples (list): List of accuracy lists for each sample.

    Returns:
        tuple: (mean accuracy, std accuracy)
    """
    accuracy_samples = np.array(accuracy_samples)
    mean_accuracy = np.mean(accuracy_samples, axis=0)
    std_accuracy = np.std(accuracy_samples, axis=0)
    return mean_accuracy, std_accuracy


def get_loss_abs_differences(loss_values):
    loss_cumsum_values = np.cumsum(random.sample(loss_values, len(loss_values)))
    loss_mean_values = loss_cumsum_values / np.arange(1, len(loss_cumsum_values) + 1)
    loss_abs_differences = abs(np.diff(loss_mean_values))
    return loss_abs_differences


# calculate the EMA (exponential moving average)
def calculate_ema(data, smoothing_factor=0.99):
    """
    Calculate the Exponential Moving Average (EMA) of a time series.

    Args:
        data (list): The time series data.
        smoothing_factor (float): The smoothing factor (alpha) for the EMA calculation.

    Returns:
        list: The EMA of the input data.
    """
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(smoothing_factor * ema[i - 1] + (1 - smoothing_factor) * data[i])
    return ema


def plot_grouped_results(results, group_by, config):
    """
    Plot results grouped by a specific key (either 'h' or 'L').

    Args:
        results (list): List of result dictionaries.
        group_by (str): Key to group by, either 'h' or 'L'.
        config (omegaconf.DictConfig): Configuration object.
    """
    grouped_results = defaultdict(list)

    # Group results by the specified key
    for res in results:
        key = res[group_by]
        grouped_results[key].append(res)

    # Determine the non-grouped-by key
    non_group_by = "L" if group_by == "h" else "h"

    # Plot each group
    for key, group in grouped_results.items():
        plt.figure()

        # Sort the group by the non-grouped-by value
        group_sorted = sorted(group, key=lambda x: x[non_group_by])

        diff_min = -float(np.inf)
        diff_max = float(np.inf)

        for res in group_sorted:
            h = res["h"]
            L = res["L"]
            samples = res["samples"]
            accuracy_samples = [s["accuracy"] for s in samples]

            means_list = []
            for s in samples:
                loss_values = s["loss_values"]
                vals = np.array([get_loss_abs_differences(loss_values) for _ in range(config.num_samples)])
                means = vals.mean(axis=0)
                means_list.append(means)
            means = np.mean(means_list, axis=0)
            ema = calculate_ema(means)

            # Calculate mean and std of accuracy
            mean_accuracy, std_accuracy = calculate_mean_std(accuracy_samples)

            # Calculate training sample size
            val_batch_size = res["val_batch_size"]
            sample_sizes = [val_batch_size * (idx + 1) for idx in range(len(mean_accuracy))]

            # Truncate data where sample_sizes <= 50,000
            truncate_idx = np.searchsorted(sample_sizes, 50000, side="right")
            sample_sizes_trunc = sample_sizes[:truncate_idx]
            mean_accuracy_trunc = mean_accuracy[:truncate_idx]
            std_accuracy_trunc = std_accuracy[:truncate_idx]

            abs_differences = [ema[idx] for idx in sample_sizes_trunc]
            diff_min = max(diff_min, min(abs_differences))
            diff_max = min(diff_max, max(abs_differences))

            # Plot mean accuracy with error bars for std
            plt.plot(abs_differences, mean_accuracy_trunc, label=f"$h$ = {h}, $L$ = {L}")
            plt.fill_between(
                abs_differences,
                mean_accuracy_trunc - std_accuracy_trunc,
                mean_accuracy_trunc + std_accuracy_trunc,
                alpha=0.3,
            )

        plt.legend()
        plt.title("MNIST")
        plt.xlabel(
            r"$\left| \mathcal{L}_{k+1}(\hat{\boldsymbol{\theta}}) - \mathcal{L}_k(\hat{\boldsymbol{\theta}}) \right|$"
        )
        plt.ylabel("Accuracy on Validation")
        plt.grid(True, alpha=0.3)
        plt.xlim(diff_min, diff_max)
        plt.xscale("log")
        plt.tight_layout()
        save_path = f"{os.path.splitext(config.plot_save_path)[0]}_{group_by}_{key}.pdf"
        plt.savefig(save_path, bbox_inches="tight")


def main(config):
    with open(config.results_path, "r") as f:
        results = json.load(f)

    # Plot results grouped by 'h'
    plot_grouped_results(results, "h", config)

    # Plot results grouped by 'L'
    plot_grouped_results(results, "L", config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting Accuracy vs Training Sample Size")
    parser.add_argument("--config_path", type=str, default="config.yml", help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    main(config)
