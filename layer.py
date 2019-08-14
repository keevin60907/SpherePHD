import tensorflow as tf
import numpy as np

def conv2d(input, in_dim, out_dim, conv_table, name, 
           reuse, stride=1, activation='relu', padding='VALID'):

    shape = np.asarray([10, 1, in_dim, out_dim])
    with tf.variable_scope(name, reuse= reuse):
        weight = tf.get_variable('weight', 
                     shape= shape, 
                     initializer= tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', 
                     shape= shape[-1], 
                     initializer= tf.constant_initializer(0))
        # width = 20 * 4**subdivision
        # input = (n_batch, 1, width, n_channel), conv_table (kernelsize, width)
        # Turns x into an array of shape (n_batch, 1, kernel=10, width, n_channel)
        x = tf.gather(input, conv_table, axis=2)
        # squeeze the array into shape (n_batch, kernel=10, width, n_channel)
        x = tf.squeeze(x, axis=1)
        output = tf.nn.conv2d(input, weight, [1, strides, strides, 1], padding) + bias
        
        if activation == 'relu':
            output = tf.nn.relu(output)
        
        return output

def maxpool(x, callTable, poolTable):
    # input = (n_batch, 1, width, n_channel)
    # Turns x into an array of shape (n_batch, 1, kernel=4, width, n_channel)
    x = tf.gather(x, callTable, axis=2)
    # squeeze the array into shape (n_batch, kernel=4, width, n_channel)
    x = tf.squeeze(x, axis=1)
    # pool_table make the subdivision-1, with size: (width/4)
    # Picks out correct pool indexes (n_batch, kernel=4, width/4, n_channel)
    x = tf.gather(x, poolTable, axis=2)
    # Max from pool (n_batch, 1, inWidth/4, n_channel)
    x = tf.reduce_max(x, axis=1, keepdims=True)
    return x

def avgpool(x):
    x = tf.reduce_mean(x, axis=[1, 2])
    return x


def conv_net(x, conv_tables, adj_tables, pooling_tables, div):
    conv1 = conv2d(x, in_dim=3, out_dim=256, conv_table[div].T)
    conv1 = conv2d(conv1, in_dim=256, out_dim=256, conv_table[div].T)
    conv1 = conv2d(conv1, in_dim=256, out_dim=256, conv_table[div].T)
    conv1 = maxpool(conv1, adj_tables[div].T, pooling_tables[div - 1].T)

    conv2 = conv2d(conv1, in_dim=256, out_dim=256, conv_table[div].T)
    conv2 = conv2d(conv2, in_dim=256, out_dim=256, conv_table[div].T)
    conv2 = conv2d(conv2, in_dim=256, out_dim=256, conv_table[div].T)
    conv2 = maxpool(conv2, adj_tables[div].T, pooling_tables[div - 1].T)
    out = avgpool(conv3)
    return out
