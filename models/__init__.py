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


from .rfdn.model import SR_RFDN
from .imdn.model import SR_IMDN
from .edsr.model import SR_EDSR
from .edsr.fast_edsr import SR_FastEDSR
from .custom.ern import SR_CustomERN
from .regular_models import RegularModel
