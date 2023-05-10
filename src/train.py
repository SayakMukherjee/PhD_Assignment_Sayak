#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# ---------------------------------------------------------------------------
# Contains the main driver code for fine-tuning
# ---------------------------------------------------------------------------

import os
import click
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from pathlib import Path
from datasets import get_dataset
from models import get_model
from optim import get_methods

@click.command()
@click.option('--exp_config', type=click.Path(exists=True), default=None,
              help='Configuration file for the experiment (default: None).')
def main(exp_config):

    # Load defaults and overwrite by command-line arguments
    config = OmegaConf.load(Path(exp_config))
    # cmd_config = OmegaConf.from_cli()
    # config = OmegaConf.merge(config, cmd_config)
    print(OmegaConf.to_yaml(config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Seed everything. Note that this does not make training entirely
    # deterministic.
    pl.seed_everything(config.seed, workers=True)

    wandb_path = Path(config.wandb.dir)
    if not Path.exists(wandb_path):
        Path.mkdir(wandb_path, exist_ok=True, parents=True)

    # Set cache dir to W&B logging directory
    os.environ["WANDB_CACHE_DIR"] = wandb_path.as_posix()
    wandb_logger = WandbLogger(
        save_dir=wandb_path,
        project=config.wandb.project,
        name=config.wandb.experiment_name + '_finetune',

        # log_model='all' if config.wandb.log else None,
        offline=not config.wandb.log,

        # Keyword args passed to wandb.init()
        entity=config.wandb.entity,
        config=OmegaConf.to_object(config),
    )

    # Create datasets and loaders
    dataset = get_dataset(config.dataset.name)(config=config)

    train_dataset_loader = dataset.get_train_loader()
    test_dataset_loader = dataset.get_test_loader()

    # Create model
    model = get_model(config.train.model)(config)
    model_dict = model.state_dict()

    # Initialize from pre-training
    import_path = Path('../pretrained_models').joinpath(config.wandb.experiment_name + '.ckpt')
    loaded_net_dict = torch.load(import_path, map_location=torch.device(device))
    loaded_net_dict = loaded_net_dict['net_dict']

    new_state_dict = {k: v for k,v in loaded_net_dict.items() if k in model_dict}

    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)

    # Finetuning method
    method = get_methods(config.train.method)(config, model)

    checkpoint_callback = ModelCheckpoint(
        dirpath= Path(config.train.save_dir),
        every_n_epochs=10,
        filename= config.wandb.experiment_name + '-finetune-{epoch:02d}')

    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        logger=wandb_logger,
        # Use DDP training by default, even for CPU training
        strategy="ddp",
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback]
    )

    # Train + validate (if validation dataset is implemented)
    trainer.fit(method, train_dataset_loader)

    # Test (if test dataset is implemented)
    if test_dataset_loader is not None:
        trainer.test(method, test_dataset_loader)

if __name__ == '__main__':
    main()