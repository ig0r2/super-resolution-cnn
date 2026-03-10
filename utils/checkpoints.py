import torch
from models import get_model


def load_model_from_checkpoint(path, device):
    """
    Ucitava checkpoint i instancira model
    :param path: putanja
    :param device: torch.device
    :return: (model, model_config iz checkpointa)
    """
    checkpoint = torch.load(path, map_location=device)
    model = get_model(checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint['model_config']
