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
import sys
import json
from absl import logging
from athena import *
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants, remove_training_nodes

from athena.main import (
    parse_config,
    build_model_from_jsonfile, 
    SUPPORTED_DATASET_BUILDER
)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    for op in graph.get_operations():
        print(op.name + ':\t' + str(op.values()))
    with graph.as_default():
        #freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        #output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names)
        
        return frozen_graph

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
    tf.compat.v1.disable_v2_behavior()
    _, model, _, checkpointer = build_model_from_jsonfile(json_file, pre_run=False)

    model.load_weights('./model.h5')
    frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.net.outputs])
    # For Transformer
    #frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.encoder_pb.outputs]) 
    #frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.decoder_pb.outputs])
    with tf.compat.v1.gfile.GFile("./model.pb", "wb") as in_f:
        in_f.write(frozen_graph.SerializeToString())
    print("Done!")
