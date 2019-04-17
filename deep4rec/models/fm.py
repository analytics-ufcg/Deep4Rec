'''
Implementation of Factorization Machines.

Paper: Factorization Machines
link: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
Author: Steffen Rendle, Osaka University
'''

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

class FM(tf.keras.Model):

  def __init__(self, ds, num_units=128):
    super(FM, self).__init__()
    self._num_weights = ds.num_features_one_hot
    self._num_units = num_units
    self._num_features = ds.num_features

    self.w = tf.keras.layers.Embedding(self._num_weights, num_units,
                                       input_length=self._num_features,
                                       embeddings_regularizer=tf.keras.regularizers.l2())
    self.w0 = tf.keras.layers.Embedding(self._num_features, 1,
                                        input_length=self._num_features,
                                        embeddings_initializer='zeros')
    self.bias = tfe.Variable(tf.constant(0.0))


  def call(self, one_hot_features, features=None):
    '''
    Args:
      one_hot_features: A dense tensor of shape [batch_size, self._num_features]
        that indicates which features are present in this input.
      features: A dense tensor of shape [batch_size, self._num_features] that indicates
        the value of each feature.
    
    Returns:
      A tuple containing the loss.
    '''
    # TODO: add support to other features.
    non_zero_weights = self.w(one_hot_features)  # [batch_size, num_features, num_units]
    
    sum_nzw = tf.reduce_sum(non_zero_weights, 1)  # [batch_size, num_units]
    squared_sum_nzw = tf.square(sum_nzw)  # [batch_size, num_units]
    
    squared_nzw = tf.square(non_zero_weights)  # [batch_size, num_features, num_units]
    sum_squared_nzw = tf.reduce_sum(squared_nzw, 1)  # [batch_size, num_units]
    
    fm = 0.5 * (squared_sum_nzw - sum_squared_nzw)
    bilinear = tf.reduce_sum(fm, 1, keep_dims=True) # [batch_size, 1]
    weight_bias = tf.reduce_sum(self.w0(one_hot_features), 1) # [batch_size, 1]

    pred_rating = tf.add_n([bilinear, weight_bias]) + self.bias
    
    return pred_rating
