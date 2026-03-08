from copy import deepcopy
from os import path as osp
import importlib

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import LOSS_REGISTRY
# from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, PerceptualLoss, WeightedTVLoss, g_path_regularize,
#                      gradient_penalty_loss, r1_penalty)

# __all__ = [
#     'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss', 'GANLoss', 'gradient_penalty_loss',
#     'r1_penalty', 'g_path_regularize'
# ]


from .losses import (LPIPSLoss)

__all__ = [
    'LPIPSLoss'
]

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_loss.py')]
# import all the model modules
# _model_modules = [importlib.import_module(f'basicsr.losses.{file_name}') for file_name in model_filenames]
_model_modules = [importlib.import_module(f'realesrgan.losses.{file_name}') for file_name in model_filenames]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must constain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
