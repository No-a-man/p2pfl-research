from binarySVM_lightning import BinarySVMAligned
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel


def model_build_fn(*args, **kwargs):
    compression = kwargs.pop("compression", None)
    # Default params can be overriden via kwargs
    input_size = kwargs.pop("input_size", 28 * 28)
    lr = kwargs.pop("lr", 1e-3)
    C = kwargs.pop("C", 1.0)
    # BinarySVMAligned expects `lr_rate` as the learning-rate arg name
    return LightningModel(BinarySVMAligned(input_size=input_size, lr_rate=lr, C=C), compression=compression)
