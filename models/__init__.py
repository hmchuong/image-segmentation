
from .hrnet import HRNet, LARGE_CONFIG, SMALL_CONFIG_V2
from .deeplabv3plus import Deeplab_v3plus


def create_model_cityscapes(model_name):
  if model_name == "DeeplabV3+":
    return Deeplab_v3plus(19)
  elif model_name == "HRNet-W18":
    return HRNet(19, 3, SMALL_CONFIG_V2)
  elif model_name == "HRNet-W48":
    return HRNet(19, 3, LARGE_CONFIG)
