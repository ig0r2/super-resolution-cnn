# Dictionary of [class name]: Class reference
MODEL_REGISTRY = {}


def register_model(cls):
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


def get_model(model_config):
    """
    Instancira odgovarajuci model za model_config
    """
    return MODEL_REGISTRY[model_config['name']](**model_config['params'])


from .rfdn.model import SR_RFDN, SR_RFDN_Multi
from .imdn.model import SR_IMDN, SR_IMDN_Multi
from .edsr.model import SR_EDSR, SR_EDSR_Multi
from .edsr.fast_edsr import SR_FastEDSR, SR_FastEDSR_Multi
from .regular_models import RegularModel
