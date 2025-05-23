import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from dlib.stdcl import STDClassifier

from dlib.unet import Unet
from dlib.unet import UnetFCAM
from dlib.unetplusplus import UnetPlusPlus
from dlib.manet import MAnet
from dlib.linknet import Linknet
from dlib.fpn import FPN
from dlib.pspnet import PSPNet
from dlib.deeplabv3 import DeepLabV3, DeepLabV3Plus
from dlib.pan import PAN

from dlib.poolings import core
from dlib.poolings import wildcat


from dlib import encoders
from dlib import utils
from dlib import losses

from dlib.configure import constants
from dlib import div_classifiers

from typing import Optional
import torch


def create_model(
    task: str,
    arch: str,
    method: str,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    **kwargs,
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parametes

    """
    spec_arch = [constants.ACOLARCH, constants.ADLARCH, constants.SPGARCH]
    if arch in spec_arch:
        assert encoder_name in constants.BACKBONES
        assert task == constants.STD_CL

        return div_classifiers.models[method][encoder_name](
            encoder_weights=encoder_weights, in_channels=in_channels, **kwargs)

    elif arch == constants.TSCAMCLASSIFIER:
        assert encoder_name in constants.TSCAM_BACKBONES
        assert task == constants.STD_CL
        assert method == constants.METHOD_TSCAM
        return div_classifiers.models[method][encoder_name](
            encoder_weights, **kwargs)
    
    elif arch == constants.SATCLASSIFIER:
        assert encoder_name in constants.SAT_BACKBONES
        assert task == constants.STD_CL
        assert method == constants.METHOD_SAT
        return div_classifiers.models[method][encoder_name](
            encoder_weights, **kwargs)
        
    elif arch == constants.SCMCLASSIFIER:
        assert encoder_name in constants.SCM_BACKBONES
        assert task == constants.STD_CL
        assert method == constants.METHOD_SCM
        return div_classifiers.models[method][encoder_name](
            encoder_weights, **kwargs)
        
    elif arch == constants.NLCCAMCLASSIFIER:
        assert encoder_name == constants.NLCCAM_VGG16
        assert task == constants.STD_CL
        assert method == constants.METHOD_NLCCAM
        return div_classifiers.models[method][encoder_name](
            encoder_weights, **kwargs)
        

    archs = [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3,
             DeepLabV3Plus, PAN, STDClassifier, UnetFCAM]
    if task == constants.STD_CL:
        archs = [STDClassifier]
    elif task == constants.F_CL:
        archs = [UnetFCAM]
    elif task == constants.SEG:
        archs = [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3,
                 DeepLabV3Plus, PAN]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError(
            "Wrong architecture type `{}`. "
            "Avalibale options are: {}".format(
                arch, list(archs_dict.keys()),))
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        **kwargs,
    )
