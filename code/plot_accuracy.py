import os
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from omegaconf import OmegaConf
from collections import defaultdict


myparams = {
    # "text.usetex": True,
    # "text.latex.preamble": r"\usepackage{amsmath}",
    # "font.family": "Djvu Serif",
    "font.size": 14,
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

        for res in group_sorted:
            h = res["h"]
            L = res["L"]
            samples = res["samples"]
            accuracy_samples = [s["accuracy"] for s in samples]

            # Calculate mean and std of accuracy
            mean_accuracy, std_accuracy = calculate_mean_std(accuracy_samples)

            # Calculate training sample size
            val_batch_size = res["val_batch_size"]
            sample_sizes = [val_batch_size * (idx + 1) for idx in range(len(mean_accuracy))]

            # Truncate data where sample_sizes <= 100,000
            truncate_idx = np.searchsorted(sample_sizes, 100000, side="right")
            sample_sizes_trunc = sample_sizes[:truncate_idx]
            mean_accuracy_trunc = mean_accuracy[:truncate_idx]
            std_accuracy_trunc = std_accuracy[:truncate_idx]

            # Plot mean accuracy with error bars for std
            plt.plot(sample_sizes_trunc, mean_accuracy_trunc, label=f"$h$ = {h}, $L$ = {L}")
            plt.fill_between(
                sample_sizes_trunc,
                mean_accuracy_trunc - std_accuracy_trunc,
                mean_accuracy_trunc + std_accuracy_trunc,
                alpha=0.3,
            )

        plt.legend()
        plt.title(f"Accuracy vs Training Sample Size ({group_by} = {key})")
        plt.xlabel("Sample Size")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        save_path = f"{os.path.splitext(config.plot_save_path)[0]}_{group_by}_{key}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()


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
