import numpy as np
import pandas as pd
import random

import tensorflow as tf  ### version 1.01
import tflearn
from tflearn.layers.conv import conv_1d, max_pool_1d, avg_pool_1d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization, local_response_normalization


# ### create convolutional neural network
# def get_cnn(df):
#     x_input_size = df.shape[0]
#     net = input_data(shape=[None, x_input_size, 1], name='input')
#     # net = avg_pool_1d(net, 2) #avg
#     net = conv_1d(net, 4, 6, activation='relu')  # 3d-tensor, num of conv filters, size of filters
#     # net = avg_pool_1d(net, 2) #avg
#     net = batch_normalization(net)
#     net = conv_1d(net, 3, 3, activation='relu')  # 3d-tensor, num of conv filters, size of filters
#     # net = avg_pool_1d(net, 2) #avg
#     # net = batch_normalization(net)
#     net = fully_connected(net, 40, activation='relu')
#     net = dropout(net, 0.8)
#     net = fully_connected(net, 2, activation='softmax')
#     return net



### create convolutional neural network
def get_cnn(df):
    x_input_size = df.shape[0]
    net = input_data(shape=[None, x_input_size, 1], name='input')
    net = batch_normalization(net)
    net = avg_pool_1d(net, 3) ### mean window 3
    net = conv_1d(net, 3, 6)  ### num of conv filters, size of filters
    net = fully_connected(net, 10, activation='tanh')
    net = dropout(net, 0.8)
    net = fully_connected(net, 2, activation='softmax')
    return net


# ### create convolutional neural network
def get_cnn_linear(df, seed):
    # random.seed(seed)
    x_input_size = df.shape[0]
    net = input_data(shape=[None, x_input_size, 1], name='input')
    net = batch_normalization(net)
    net = avg_pool_1d(net, 3) ### mean window 3
    net = conv_1d(net, 3, 6)  ### num of conv filters, size of filters
    net = fully_connected(net, 10, activation='tanh')
    net = dropout(net, 0.8)
    net = fully_connected(net, 1, activation='linear')
    return net


# # ### create convolutional neural network
# def get_cnn_linear(df, seed=123):
#     random.seed(seed)
#     l1_nodes = random.randrange(1,10)
#     l2_nodes = random.randrange(1,10)
#     l2_size = random.randrange(2,20)
#     l3_nodes = random.randrange(2,20)
#
#     x_input_size = df.shape[0]
#     net = input_data(shape=[None, x_input_size, 1], name='input')
#     net = batch_normalization(net)
#     net = avg_pool_1d(net, l1_nodes) ### mean window 3
#     net = conv_1d(net, l2_nodes, l2_size)  ### num of conv filters, size of filters
#     net = fully_connected(net, l3_nodes, activation='tanh')
#     net = dropout(net, 0.8)
#     net = fully_connected(net, 1, activation='linear')
#
#     print('# NETWORK #')
#     print('l1_nodes: {}'.format(l1_nodes) )
#     print('l2_nodes: {}'.format(l2_nodes) )
#     print('l2_size: {}'.format(l2_size) )
#     print('l3_nodes: {}'.format(l3_nodes) )
#
#     return net







#
# # ### create convolutional neural network
# def get_cnn_linear(df):
#     x_input_size = df.shape[0]
#     net = input_data(shape=[None, x_input_size, 1], name='input')
#     net = batch_normalization(net)
#     net = avg_pool_1d(net, 3) ### mean window 3
#     net = conv_1d(net, 3, 6)  ### num of conv filters, size of filters
#     net = fully_connected(net, 10, activation='relu')
#     net = dropout(net, 0.8)
#     net = fully_connected(net, 1, activation='linear')
#     return net







def normalize_input(df):
    out_df = np.log2(df)
    # out_df = out_df -  np.mean(out_df, axis=0)
    return out_df






