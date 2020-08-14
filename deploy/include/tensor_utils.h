/* Copyright (C) 2020 ATHENA AUTHORS;
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================== */

#include "tensorflow/core/public/session.h"


int initSession(tensorflow::Session *&session,
                const std::string &graph_path,
                tensorflow::Status &status);

tensorflow::Tensor getMatrixFromFile(const std::string &filename);

Eigen::Tensor<float, 1> computeLogSoftmax(tensorflow::Tensor score);

void createInputStructureEncoder(std::vector
                                 <std::pair<std::string, tensorflow::Tensor>> &inputs);

void createOutputNameStructureEncoder(std::vector<std::string> &output_names);

void createInputStructureDecoder(std::vector
                                 <std::pair<std::string, tensorflow::Tensor>> &inputs);

void createOutputNameStructureDecoder(std::vector<std::string> &output_names);
