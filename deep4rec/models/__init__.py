import tensorflow as tf

tf.enable_eager_execution()

from deep4rec.models.fm import FM
from deep4rec.models.nfm import NeuralFM
from deep4rec.models.nmf import NeuralMF
from deep4rec.models.widedeep import WideDeep
