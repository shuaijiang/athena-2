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
""" an implementation of resnet model that can be used as a sample
    for speaker recognition """

import tensorflow as tf
from .base import BaseModel
from ..loss import SoftmaxLoss, MSELoss, AAMSoftmaxLoss, ProtoLoss, AngleProtoLoss, GE2ELoss
from ..metrics import MeanAbsoluteError
from ..layers.resnet_block import ResnetBasicBlock, stack_fn
from ..utils.hparam import register_and_parse_hparams
from tensorflow.keras.regularizers import l2

SUPPORTED_LOSS = {
    "softmax": SoftmaxLoss,
    "mseloss": MSELoss,
    "aamsoftmax": AAMSoftmaxLoss,
    "prototypical": ProtoLoss,
    "angular_prototypical": AngleProtoLoss,
    "ge2e": GE2ELoss
}

class AgeResnetRelu(BaseModel):
    """ A sample implementation of resnet 34 for speaker recognition
        Reference to paper "Deep residual learning for image recognition"
        The implementation is the same as the standard resnet with 34 weighted layers,
        excepts using only 1/4 amount of filters to reduce computation.
    """
    default_config = {
        "num_speakers": None,
        "hidden_size": 128,
        "num_filters": [16, 32, 64, 128],
        "num_layers": [3, 4, 6, 3],
        "loss": "mseloss",
        "max_age": 100,
        "scale": 15
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        # maximum value of speakers' ages
        self.max_age = self.hparams.max_age

        self.loss_function = self.init_loss(self.hparams.loss)
        self.metric = MeanAbsoluteError(max_val=100, is_norm=False, name="MAE")

        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"],
                                      dtype=data_descriptions.sample_type["input"])

        bn_axis = 3
        weight_decay = 1e-4

        x = input_features
        x = layers.Conv2D(
            16, 7,
            strides=2, use_bias=False,
            kernel_initializer='orthogonal',
            kernel_regularizer=l2(weight_decay),
            padding='same',
            name='conv1_conv',
        )(x)

        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

        x = layers.MaxPooling2D(3, strides=2, name='pool1_pool', padding='same')(x)

        x = stack_fn(x)

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

        x = layers.Dense(self.hparams.hidden_size,
                           kernel_initializer='orthogonal',
                           use_bias=True, trainable=True,
                           kernel_regularizer=l2(weight_decay),
                           bias_regularizer=l2(weight_decay),
                           name='projection')(x)

        x = layers.Dense(1,
                            activation=tf.keras.activations.relu,
                            kernel_initializer='orthogonal',
                            use_bias=False, trainable=True,
                            kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay),
                            name='prediction')(x)
        x = tf.keras.activations.relu(x, max_value=self.max_age)
        # Create model
        self.age_resnet = tf.keras.Model(inputs=input_features, outputs=x, name="resnet-34")
        print(self.age_resnet.summary())

    def make_resnet_block_layer(self, num_filter, num_blocks, stride=1):
        """ returns sequential layer composed of resnet block """
        resnet_block = tf.keras.Sequential()
        resnet_block.add(ResnetBasicBlock(num_filter, stride))
        for _ in range(1, num_blocks):
            resnet_block.add(ResnetBasicBlock(num_filter, 1))
        return resnet_block

    def make_final_layer(self, embedding_size, num_class, loss):
        layers = tf.keras.layers
        if loss in ("softmax", "mseloss"):
            final_layer = layers.Dense(
                num_class,
                kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                input_shape=(embedding_size,),
            )
        elif loss in ("amsoftmax", "aamsoftmax"):
            # calculate cosine
            embedding = layers.Input(shape=tf.TensorShape([None, embedding_size]), dtype=tf.float32)
            initializer = tf.initializers.GlorotNormal()
            weight = tf.Variable(initializer(
                                shape=[embedding_size, num_class], dtype=tf.float32))
            embedding_norm = tf.math.l2_normalize(embedding, axis=1)
            weight_norm = tf.math.l2_normalize(weight, axis=0)
            cosine = tf.matmul(embedding_norm, weight_norm)
            final_layer = tf.keras.Model(inputs=embedding,
                                         outputs=cosine, name="final_layer")
        else:
            # return embedding directly
            final_layer = lambda x: x
        return final_layer

    def call(self, samples, training=None):
        """ call model """
        input_features = samples["input"]
        output = self.age_resnet(input_features, training=training)
        return output

    def init_loss(self, loss):
        """ initialize loss function """
        if loss == "softmax":
            loss_function = SUPPORTED_LOSS[loss](
                                num_classes=self.num_class
                            )
        elif loss == "mseloss":
            loss_function = SUPPORTED_LOSS[loss](
                                max_scale=self.max_age,
                                is_norm=False
                            )
        else:
            raise NotImplementedError
        return loss_function

    def pooling_layer(self, output, pooling_type):
        """aggregate frame-level feature vectors to obtain
        utterance-level embedding

        Args:
            output: frame-level feature vectors
            pooling_type (str): "average_pooling" or "statistic_pooling"

        Returns:
            utterance-level embedding, shape::

            "average_pooling": [batch_size, dim]
            "statistic_pooling": [batch_size, dim * 2]
        """
        output = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[1], -1])
        if pooling_type == "average_pooling":
            output, _ = tf.nn.moments(output, 1)
        elif pooling_type == "statistic_pooling":
            mean, variance = tf.nn.moments(output, 1)
            output = tf.concat(mean, tf.sqrt(variance + 1e-6), 1)
        else:
            raise NotImplementedError
        return output

    def get_loss(self, outputs, samples, training=None):
        loss = self.loss_function(outputs, samples)
        self.metric.update_state(outputs, samples)
        metrics = {self.metric.name: self.metric.result()}
        return loss, metrics


    def decode(self, samples, hparams=None, decoder=None):
        outputs = self.call(samples, training=False)
        return outputs

