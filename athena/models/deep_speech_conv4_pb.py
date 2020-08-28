# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
""" a implementation of deep speech 2 model can be used as a sample for ctc model """

import tensorflow as tf
from absl import logging
from ..utils.hparam import register_and_parse_hparams
from .base import BaseModel
from ..loss import CTCLoss
from ..metrics import CTCAccuracy
from ..layers.commons import SUPPORTED_RNNS

class DeepSpeechModelConv4PB(BaseModel):
    """ a sample implementation of CTC model """
    default_config = {
        "conv_filters": 256,
        "rnn_hidden_size": 1024,
        "num_rnn_layers": 6,
        "rnn_type": "gru"
    }
    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.num_classes = data_descriptions.num_class + 1
        self.loss_function = CTCLoss(blank_index=-1)
        self.metric = CTCAccuracy()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        layers = tf.keras.layers
        input_feature = layers.Input(
            shape=data_descriptions.sample_shape["input"],
            dtype=tf.float32,
            name="input_feature"
        )
        inner = layers.Conv2D(
            filters=self.hparams.conv_filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(input_feature)#[:,1:-1,:,:]
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu6(inner)
        #inner = tf.identity(inner, name="cnn_output")
        #self.enc_1 = tf.keras.Model(inputs=input_feature,
        #                          outputs=inner)
        inner = layers.Conv2D(
            filters=self.hparams.conv_filters,
            kernel_size=(4, 4),
            strides=(2, 1),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(inner)[:,1:-1,:,:]
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu6(inner)
        _, _, dim, channels = inner.get_shape().as_list()
        output_dim = dim * channels
        inner = layers.Reshape((-1, output_dim))(inner) #[:,:-1,:]
        inner = tf.identity(inner, name="cnn_output")
        self.enc = tf.keras.Model(inputs=input_feature,
                                  outputs=inner)
        logging.info(self.enc.summary())

        rnn_type = self.hparams.rnn_type
        rnn_hidden_size = self.hparams.rnn_hidden_size
        self.inputs = []
        self.inputs.append(input_feature)
        self.outputs = []
        for i in range(self.hparams.num_rnn_layers):
            initial = tf.keras.layers.Input(
                shape=(512, ),
                dtype=tf.float32,
                name="initial_"+str(i))
            self.inputs.append(initial)

            inner, last_state = tf.keras.layers.RNN(
                cell=[SUPPORTED_RNNS[rnn_type](rnn_hidden_size)],
                return_sequences=True,
                return_state=True
            )(inputs=inner, initial_state=[initial])
            state = tf.identity(last_state[0], name="state_"+str(i))
            self.outputs.append(state)
            inner = layers.BatchNormalization()(inner)
        inner = layers.Dense(rnn_hidden_size, activation=tf.nn.relu6)(inner)
        inner = layers.Dense(self.num_classes)(inner)
        inner = tf.nn.log_softmax(inner)
        logits = tf.identity(inner, name="logits")
        self.outputs.append(logits)
        self.net = tf.keras.Model(inputs=self.inputs,
                                  outputs=self.outputs)
        logging.info(self.net.summary())


    def call(self, samples, training=None):
        """ call function """
        #return self.net(samples["input"], training=training)
        return self.enc(samples, training=training)

    def compute_logit_length(self, samples):
        """ used for get logit length """
        input_length = tf.cast(samples["input_length"], tf.float32)
        logit_length = tf.math.ceil(input_length / 2)
        logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length
