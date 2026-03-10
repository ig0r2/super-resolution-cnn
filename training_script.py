from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from datasets import get_training_set, get_test_set
from utils.plot import plot_training_history
from utils.trainer import Trainer
from utils.logger import Logger
from models import get_model

if __name__ == "__main__":
    # load config
    with open("config/training.yaml") as f:
        config = yaml.safe_load(f)

    # load dataset
    upscale_factor = config['models'][0]['model']['params']['upscale_factor']
    train_set = get_training_set(upscale_factor=upscale_factor, patch_size=config['patch_size'],
                                 preload=config['train_preload'])
    val_set = get_test_set(name="DIV2K", upscale_factor=upscale_factor, preload=config['val_preload'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.backends.cudnn.is_available(): torch.backends.cudnn.benchmark = True

    # for each model
    for config_m in config['models']:
        config['model'] = config_m['model']
        log_path = f"logs/training/training_{config['model']['checkpoint_name']}.txt"

        with Logger(log_path):
            model = get_model(config['model']).to(device)
            criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_epochs'], gamma=0.5)

            Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                    config=config, device=device, train_set=train_set, val_set=val_set).train()
