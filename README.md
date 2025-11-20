# Introduction
This is the repository containing all codes used for my Master`s thesis on analyzing the collapse of loss functions in the context of the maximal update parametrization ($\mu P$), which was originally introduced in [Tensor Programs V](https://arxiv.org/abs/2203.03466).

## Installation
To set up the simulation environment, clone this repository and change to its directory.
Afterwards enter `pixi install` to set up the default environment which only uses the CPU device.
Optionally, also enter `pixi install -e gpu` to also set up the environment to run the training process on the GPU device.
Alternatively, the python environment can also be set up using the `requiremnents.txt` by entering `pip install requirements.txt` inside a virtual environment. 

## Quick Start
Enter the following command fro, the repositories root directory to run the training process for the network and training configurations set in the `experiments/example_config.yaml` file. 
```
pixi run python src/run_experiment.py --path experiments/example_config.yaml  # To run the simulation using the pixi default environment
python src/run_experiment.py --path experiments/example_config.yaml  # To run the simulation using a standard python environment
```

To run the training on the GPU device use: 
```
pixi run -e gpu python src/run_experiment.py --path experiments/example_config.yaml
```

The training logs can be analyzed by running:
```
pixi run python visualizations/compare_runs.py --log-dir path/to/directory/log_file  
python visualizations/compare_runs.py --log-dir path/to/directory/log_file   
```

To verify that the ($\mu P$) parametrization is correctly implemented, a script to check the evolution of layer magnitudes is provided in `visualizations` and can be executed using: 
```
pixi run python visualizations/analyze_acts.py --log-dir path/to/directory/log_file  
python visualizations/analyze_acts.py --log-dir path/to/directory/log_file   
```

## Experiments
To run trainings for multiple networks simply provide the directory that contains the config files as input to the --path CLI flag, as for example: 
```
pixi run python src/run_experiment.py --path experiments/mup  # To run the simulation using the pixi default environment
```
The code will automatically search for all config files in the `configs/` subdirectory of the experiment and assemble the `config.yaml` files using the `master_config.yaml` file. All outputs will be saved in the experiments `log/` directory as for instance `experiments/mup/logs`. If this directory is not already present it will be created automatically. Assembled config files using the`master_config.yaml` file and all other files contained in `configs/`, will look structurally like the `example_config.yaml` file that was already referneced in the [Quick Start](#quick-start) section.
The output path of the logs can also be set manually from the CLI flag `--output_dir`.

### Configs 
The config files specify all simulation settings that are used for training the networks. In principle every config file can provide completely independent simulation settings. Optionally, config settings that should be identical between different training processes of networks specified in the `configs/` directory, can be set using the `master_config.yaml` file and referenced in the distinct config files as in the example experiments `standard/` and `mup/`. 
The abc-parametrization used for the network initialization and trainings can be adresses by the `param_scheme` instance in the config files.

## Cluster Usage
The simualtions can alos be executed on a cluster using SLURM as wokrload manager. 
To run the simulations managed by SLURM use the following command that will distribute task using job arrays for the distinct network trainings 
```
sbatch submit.sh experiments/mup
```