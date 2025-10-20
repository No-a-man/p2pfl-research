from simpleTextSVM import SimpleTextSVM
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel

def model_build_fn(*args, **kwargs):
    compression = kwargs.pop("compression", None)
    return LightningModel(SimpleTextSVM(*args, **kwargs), compression=compression)
