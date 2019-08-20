import tensorflow as tf
import numpy as np

def conv2d(input, in_dim, out_dim, conv_table, name, 
           reuse=False, stride=1, activation='elu', padding='VALID'):

    shape = np.asarray([10, 1, in_dim, out_dim])
    with tf.variable_scope(name, reuse= reuse):
        weight = tf.get_variable('weight', 
                     shape= shape, 
                     initializer = tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', 
                     shape= shape[-1], 
                     initializer= tf.constant_initializer(0))
        # width = 20 * 4**subdivision
        # input = (n_batch, 1, width, n_channel), conv_table (kernelsize, width)
        # Turns x into an array of shape (n_batch, 1, kernel=10, width, n_channel)
        x = tf.gather(input, conv_table, axis=2)
        # squeeze the array into shape (n_batch, kernel=10, width, n_channel)
        x = tf.squeeze(x, axis=1)
        output = tf.nn.conv2d(x, weight, [1, stride, stride, 1], padding) + bias
        if activation == 'elu':
            output = tf.nn.elu(output)
        
        return output

def channel_conv(input, in_dim, out_dim, name, reuse=False, 
                 stride=1, activation='elu', padding='VALID'):

    shape = np.asarray([1, 1, in_dim, out_dim])
    with tf.variable_scope(name, reuse= reuse):
        weight = tf.get_variable('weight', 
                     shape= shape, 
                     initializer = tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', 
                     shape= shape[-1], 
                     initializer= tf.constant_initializer(0))
        # input = (n_batch, 1, width, n_channel)
        output = tf.nn.conv2d(input, weight, [1, stride, stride, 1], padding) + bias

        if activation == 'elu':
            output = tf.nn.elu(output)
        return output

def maxpool(x, adj_table, pooling_table):

    # input = (n_batch, 1, width, n_channel)
    # Turns x into an array of shape (n_batch, 1, kernel=4, width, n_channel)
    x = tf.gather(x, adj_table, axis=2)
    # squeeze the array into shape (n_batch, kernel=4, width, n_channel)
    x = tf.squeeze(x, axis=1)
    # pool_table make the subdivision-1, with size: (width/4)
    # Picks out correct pool indexes (n_batch, kernel=4, width/4, n_channel)
    x = tf.gather(x, pooling_table, axis=2)
    # Max from pool (n_batch, 1, width/4, n_channel)
    x = tf.reduce_max(x, axis=1, keepdims=True)
    return x

def upsample(x, upsample_table):
    # input = (n_batch, 1, width, n_channel)
    # Turns x into an array of shape (n_batch, 1, width*4, n_channel)
    x = tf.keras.layers.UpSampling2D((1, 4))(x)
    # unpooling shape (n_batch, 1, width*4, n_channel)
    x = tf.gather(x, upsample_table, axis=2)
    return x

def avgpool(x):
    x = tf.reduce_mean(x, axis=[1, 2])
    return x
