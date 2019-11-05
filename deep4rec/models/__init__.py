import tensorflow as tf

tf.enable_eager_execution()

from deep4rec.models.fm import FM
from deep4rec.models.nfm import NeuralFM
from deep4rec.models.nmf import NeuralMF
from deep4rec.models.widedeep import WideDeep
from deep4rec.models.spectralcf import SpectralCF


models = [FM, NeuralFM, NeuralMF, WideDeep, SpectralCF]


def options():
    """Returns a list containing the available models."""
    return [str(model.__name__) for model in models]
