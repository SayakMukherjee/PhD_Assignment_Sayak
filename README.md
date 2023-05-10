# PhD Assignment: Delft University of Technology

This repository contains the code for the experiments mentioned in the assignment. It is implemented using [PyTorch Lightning](https://lightning.ai/) based on [PyTorch Lightning Example](https://github.com/CVLab-TUDelft/pytorch-lightning-example) developed at the Computer Vision Lab at TU Delft.

## Installation
This code is written in `Python 3.9` and requires the packages listed in `environment-cuda11.3.yaml`.

To run the code, set up a virtual environment using `conda`:

```
cd <path-to-experiments-directory>
conda env create --file environment-cuda11.3.yaml.yml
conda activate mscthesis
```

## Running experiments

To run an experiment create a new configuration file in `scripts` directory. The experiments are run in two phase namely pre-training and fine-tuning. It can be run directly the following commands

```
cd <path-to-experiments-directory>\src

# pre-training
python  pretrain.py --exp_config ..\scripts\<config-file-name>.yaml

# fine-tuning
python  train.py --exp_config ..\scripts\<config-file-name>.yaml
```

or by submitting job using SLURM.


Example of a config file:

```yaml

seed: 12345
optimize:
  optimizer: Adam
  lr: 0.001
  weight_decay: 1e-5
train:
  method: Autoencoder
  model: LeNet_Autoencoder
  epochs: 100
  batch_size: 64
  loss: MSE
  use_reg: True
  save_dir: ../models
dataset:
  name: STL10
  channels: 3
  is_corrupted: True
  root: ../data
  num_workers: 4
  use_collate: False
wandb:
  log: True
  dir: ../wandb
  experiment_name: ''
  entity: ''
  project: ''

```

### Options:

- Method training mode: `Autoencoder` or `Finetune` or `SimCLR`
- Model used for training: `LeNet` or `LeNet_Autoencoder` or `ResNet` or `ResNetBackbone`
- Dataset for training: `STL10`
- Loss function: `MSE` or `PSE` (Used only for autoencoder pre-training)
- To pre-training using corrupted or clean data set `is_corrupted` as `True` or `False`
- `use_collate` can only be used with `SimCLR` method
- `use_reg` is for using the regularization term based on contractive AE