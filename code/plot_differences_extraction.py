import matplotlib.pyplot as plt
import numpy as np
import random
import json
import argparse
from omegaconf import OmegaConf

myparams = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'Djvu Serif',
    'font.size': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
}
plt.rcParams.update(myparams)


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
        ema.append(smoothing_factor * ema[i-1] + (1 - smoothing_factor) * data[i])
    return ema


def main(config):
    
    for i in range(len(config.dataset_names)):

        with open(f'results_extraction/{config.output_names[i]}.json', 'r') as f:
            results = json.load(f)
            
        # NUMBER OF LAYERS

        plt.figure()

        for res in results:
            
            h = res['h']
            if h != int(config.hidden_size):
                continue
            
            L = res['L']
            loss_values = res['loss_values']
            
            vals = np.array([get_loss_abs_differences(loss_values) for _ in range(config.num_samples)])
            means = vals.mean(axis=0)
            ema = calculate_ema(means)
            plt.plot(ema, label=f'$h$ = {h}, $L$ = {L}')
            
        plt.legend()
        plt.title(config.dataset_names[i])
        plt.xlabel('Sample size, $k$')
        plt.ylabel(r"$\left| \mathcal{L}_{k+1}(\hat{\boldsymbol{\theta}}) - \mathcal{L}_k(\hat{\boldsymbol{\theta}}) \right|$")
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'figs_extraction/{config.output_names[i]}_num_layers.pdf', bbox_inches='tight')
        
        # HIDDEN SIZE
        
        plt.figure()

        for res in results:
            
            h = res['h']
            L = res['L']
            
            if L != int(config.num_layers):
                continue
            
            loss_values = res['loss_values']
            
            vals = np.array([get_loss_abs_differences(loss_values) for _ in range(config.num_samples)])
            means = vals.mean(axis=0)
            ema = calculate_ema(means)
            plt.plot(ema, label=f'$h$ = {h}, $L$ = {L}')
            
        plt.legend()
        plt.title(config.dataset_names[i])
        plt.xlabel('Sample size, $k$')
        plt.ylabel(r"$\left| \mathcal{L}_{k+1}(\hat{\boldsymbol{\theta}}) - \mathcal{L}_k(\hat{\boldsymbol{\theta}}) \right|$")
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'figs_extraction/{config.output_names[i]}_hidden_size.pdf', bbox_inches='tight')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting graphs for losses differences")
    parser.add_argument("--config_path", type=str, default='configs/configs_plot/config_plot_extraction.yml', help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    main(config)