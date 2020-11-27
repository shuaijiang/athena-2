# coding=utf-8
# Copyright (C) 2020 ATHENA AUTHORS; Ne Luo
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
""" an implementation of resnet block """

import tensorflow as tf
from tensorflow.keras.regularizers import l2

class ResnetBasicBlock(tf.keras.layers.Layer):
    """ Basic block of resnet
        Reference to paper "Deep residual learning for image recognition"
    """
    def __init__(self, num_filter, stride=1):
        super().__init__()
        layers = tf.keras.layers
        self.conv1 = layers.Conv2D(filters=num_filter,
                                   kernel_size=(3, 3),
                                   strides=stride,
                                   padding="same")
        self.conv2 = layers.Conv2D(filters=num_filter,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.add = layers.add
        self.relu = tf.nn.relu
        self.downsample_layer = self.make_downsample_layer(
            num_filter=num_filter, stride=stride
        )

    def call(self, inputs):
        """ call model """
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        residual = self.downsample_layer(inputs)
        output = self.add([residual, output])
        output = self.relu(output)
        return output

    def make_downsample_layer(self, num_filter, stride):
        """ perform downsampling using conv layer with stride != 1 """
        if stride != 1:
            downsample = tf.keras.Sequential()
            downsample.add(tf.keras.layers.Conv2D(filters=num_filter,
                                                  kernel_size=(1, 1),
                                                  strides=stride))
            downsample.add(tf.keras.layers.BatchNormalization())
        else:
            downsample = lambda x: x
        return downsample


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    Returns:
        Output tensor for the residual block.
    """
    bn_axis = 3
    weight_decay = 1e-4
    layers = tf.keras.layers

    if conv_shortcut:
        #shortcut = layers.Conv2D(
        #        4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.Conv2D(
            filters, 1, strides=stride,
            use_bias=False,
            kernel_initializer='orthogonal',
            kernel_regularizer=l2(weight_decay),
            name=name + '_0_conv'
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
        filters, kernel_size,
        strides=stride, padding='same',
        use_bias=False,
        kernel_initializer='orthogonal',
        kernel_regularizer=l2(weight_decay),
        name=name + '_1_conv'
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size,
        strides=1, padding='same',
        use_bias=False,
        kernel_initializer='orthogonal',
        kernel_regularizer=l2(weight_decay),
        name=name + '_2_conv'
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    #x = layers.Activation('relu', name=name + '_2_relu')(x)

    #x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    #x = layers.BatchNormalization(
    #    axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x

def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
        Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    Returns:
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x

def stack_fn(x):
    x = stack1(x, 16, 3, stride1=1, name='conv2')
    x = stack1(x, 32, 4, name='conv3')
    x = stack1(x, 64, 6, name='conv4')
    x = stack1(x, 128, 3, name='conv5')
    return x

def stack16_fn(x):
    x = stack1(x, 16, 2, stride1=1, name='conv2')
    x = stack1(x, 32, 2, name='conv3')
    x = stack1(x, 64, 2, name='conv4')
    x = stack1(x, 128, 2, name='conv5')
    return x