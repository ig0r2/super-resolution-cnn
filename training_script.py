import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from utils.trainer import Trainer
from utils.trainer_multiscale import TrainerMultiscale
from utils.logger import Logger
from models import get_model

if __name__ == "__main__":
    CONFIG_FILE = "config/training_multi_jpeg.yaml"
    MULTISCALE = True
    JPEG_DEGRADATION = True

    ######################################################
    # load config
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.backends.cudnn.is_available(): torch.backends.cudnn.benchmark = True

    # create trainer (load datasets)
    if MULTISCALE:
        trainer = TrainerMultiscale(config=config, device=device, jpeg_degradation=JPEG_DEGRADATION)
    else:
        trainer = Trainer(config=config, device=device, jpeg_degradation=JPEG_DEGRADATION)

    # for each model
    for config_m in config['models']:
        config['model'] = config_m['model']
        log_path = f"logs/training/training_{config['model']['checkpoint_name']}.txt"

        with Logger(log_path):
            model = get_model(config['model']).to(device)
            criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_epochs'], gamma=0.5)

            trainer.set_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                              config=config).train()
