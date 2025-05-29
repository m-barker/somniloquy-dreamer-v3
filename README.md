# Somniloquy
This repository contains the latest version of the codebase for the *"Translating Latent State Plans Into Natural Language"*  extended abstract, accepted to RLDM 2025.

Note that this project has evolved since the paper was submitted in Januray 2025. If you want to use the exact codebase used in the RLDM work, please use [this commit](https://github.com/m-barker/somniloquy-dreamer-v3/commit/7a6be7075856da9e705d617822ac68b52cecdcc7).

## Installation
First, setup a virtual environment for `python3.9` using your venv manager of choice, e.g.:

`virtualenv -p python3.9 somniloquy`

Then, install the required packages:

`pip install -r requirements.txt`

Note that this has only been tested on an Ubuntu 22 system, you may have to tweak the Cuda version etc. dependent on your host system.

## Usage
To verify that everything has been installed correctly, you can run a much smaller version of Somniloquy that should fit into most GPUs (needing <8GB VRAM) using the command:

```bash
python3.9 somniloquy.py --configs crafter-language-small --seed 500 --logdir logdir/crafter-langauge-small-seed-500 --wandb False
```

where the `--configs` argument specifies which set of parameters to use (task, etc.), the `--seed` parameter specifies the environment and network seed to use, the `--logdir` parameter specifies where to save all outputs of the run (evaluation details, replay buffer, etc.), and the `--wandb` parameter is used to disable logging of the run to Weights and Biases.

### Deterministic Environment Experiments
To run the set the Crafter experiments for a given seed, for example seed 100, run the following command:

```bash
python3.9 somniloquy.py --configs crafter-language --seed 100 --logdir logdir/crafter-langauge-seed-100
```

Note that by default, the translation gradients do not flow into the RSSM to update the learned latent representation. This can be enabled by adding the `-language_grads True` argument.

If using Weights and Biases (recommended), the translation performance metrics are logged throughout training, for example, under the `mean_imagined_bleu_score` metric, which is the metric used in the paper. 

Additionally, all translation metrics, and actual latent plan translations, can be found under `/logdir/<name>/evaluation/latent_translations` and the corresponding observations, and decoded plan observations, can be found under `/logdir/<name>/evaluation/latent_plan_plots`.

### Stochastic MiniGrid Environment
To run the stochastic MiniGrid experiment, the process is much the same as for deterministic environments, with an extra, post-hoc processing needed to generate the Total Varational Distance (TVD) plot given in the paper.

To run a given seed, the command is as above, with the `--config` and `--logdir` arguments replaced to be, e.g.,:

```bash
python3.9 somniloquy.py --configs minigrid-language --seed 500 --logdir logdir/minigrid-language-no-grad-seed-500
```

Stochastic plan translations can then be found under `/logdir/<name>/evaluation/latent_translations` and the corresponding observations, and decoded plan observations, can be found under `/logdir/<name>/evaluation/latent_plan_plots`.


## Acknowledgements
The base world component is directly taken from [this excellent PyTorch Dreamer V3 implementation](https://github.com/NM512/dreamerv3-torch), which is licensed under the MIT License.
