# Unraveling the Hessian: A Key to Smooth Convergence in Loss Function Landscapes

This repository contains the source code for the paper "Unraveling the Hessian: A Key to Smooth Convergence in Loss Function Landscapes". The code includes data processing, model building, and visualization of results for exploring the role of the Hessian matrix in exploring loss functions.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Files Description](#files-description)

## Installation <a name="installation"></a>
To use this project, you need to have Python and the required packages installed on your computer.

Clone the repo:
```bash
git clone https://github.com/kisnikser/landscape-hessian.git
cd landscape-hessian/code
```

Install dependencies:
```bash
conda create -n landscape-hessian python=3.10
conda activate landscape-hessian
pip install -r requirements.txt
```

## Usage <a name="usage"></a>
To run the project, you can execute the Python scripts directly. For example, to get loss values, you can run:
```bash
python get_loss_values.py --config_path=configs/configs_direct/mnist.yml
```
Similarly, to plot differences, you can run:
```bash
python plot_differences.py --config_path=configs/configs_plot/config_plot.yml
```
Make sure to adjust the configuration files in the `configs` directory according to your needs.

## Files Description <a name="files-description"></a>
- `configs`: This directory contains configuration files for different experiments.
  - `configs_direct`: Configuration files for experiments with Direct Image Classification.
  - `configs_extraction`: Configuration files for experiments with Image features extraction.
  - `configs_plot`: Configuration files for plotting.
- `environment.yml`: This file specifies the required packages for the conda environment.
- `figs` and `figs_extraction`: These directories contain the generated figures for the paper.
- `get_loss_values.py` and `get_loss_values_extraction.py`: These scripts compute loss values for different experiments.
- `plot_differences.py` and `plot_differences_extraction.py`: These scripts plot the differences in loss values.
- `results` and `results_extraction`: These directories contain the results of the experiments in JSON format.