# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei
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
# Only support tensorflow 2.0
# pylint: disable=invalid-name, no-member, wildcard-import, unused-wildcard-import, redefined-outer-name
""" a sample implementation of LAS for HKUST """
import os
import sys
import json
from absl import logging
from athena import *
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants


from athena.main import (
    parse_config,
    build_model_from_jsonfile,
    SUPPORTED_DATASET_BUILDER
)

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    json_file = sys.argv[1]
    config = None
    with open(json_file) as f:
        config = json.load(f)
    p = parse_config(config)
    BaseSolver.initialize_devices(p.solver_gpu)
    sess = K.get_session()
    _, model, _, checkpointer = build_model_from_jsonfile(json_file, pre_run=False)

    ckpt_index_list = [1, 2, 3] # TODO: load top k models
    ckpt_v_list = []
    #restore v from ckpts
    for idx in ckpt_index_list:
        ckpt_path = os.path.join(p.ckpt, 'ckpt-' + str(idx))
        checkpointer.restore(ckpt_path) #current variables will be updated
        var_list = []
        for i in model.trainable_variables:
            v = tf.constant(i.value())
            var_list.append(v)
        ckpt_v_list.append(var_list)
    #compute average, and assign to current variables
    for i in range(len(model.trainable_variables)):
        v = [tf.expand_dims(ckpt_v_list[j][i],[0]) for j in range(len(ckpt_v_list))]
        v = tf.reduce_mean(tf.concat(v,axis=0),axis=0)
        model.trainable_variables[i].assign(v)

    #checkpointer.restore_from_best()
    model.save_weights('./model.h5')

    print("Done!")
