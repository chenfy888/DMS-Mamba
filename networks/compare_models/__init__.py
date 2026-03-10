from .munet_multi_concat import build_model as MUNET_CONCAT
from .munet_multi_sum import build_model as MUNET_SUM
from .mamba_unet import build_model as MAMBA_UNET
from .mamba_unet_concat import build_model as MAMBA_UNET_CONCAT
from .mmunet import build_model as MMUNET

from .compare.swinir import build_model as SwinIR
from .mambaIR import build_model as MambaIR
from .panmanba import build_model as PanMamba
from .panmanba_baseline import build_model as PanMambaBaseline
from .MMR import build_model as MMR
from .dms_mamba import build_model as dmsmamba
from .MMR_knee import build_model as MMRKnee
from .panmanba_knee_ASFFsum import build_model as PanMambaKneeASFFsum

# ======================

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'plans_handler'))

model_factory = {
    'munet_concat': MUNET_CONCAT,
    'munet_sum': MUNET_SUM,  
    'mamba_unet': MAMBA_UNET,
    'mamba_unet_concat': MAMBA_UNET_CONCAT,
    'swinir': SwinIR,
    'mmunet': MMUNET,
    'mambaIR': MambaIR,
    'panmamba': PanMamba,
    'panmamba_baseline': PanMambaBaseline,
    'mmr_mamba': MMR,
    'mmr_mamba_knee': MMRKnee,
    'panmamba_knee_ASFFsum': PanMambaKneeASFFsum,
    'dmsmamba': dmsmamba,
}


def build_model_from_name(args):
    if args.model_name not in model_factory.keys():
        print(f"Error: Model '{args.model_name}' not found.")
        print(f"Available models: {list(model_factory.keys())}")
    
    assert args.model_name in model_factory.keys(), 'unknown model name'

    return model_factory[args.model_name](args)