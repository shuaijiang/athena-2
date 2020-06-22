# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Shuaijiang Zhao; Ne Luo
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
# pylint: disable=no-member, invalid-name
""" audio dataset """

import os
import sys
from absl import logging
import tensorflow as tf
import kaldiio
from athena.transform import AudioFeaturizer
from ...utils.hparam import register_and_parse_hparams
from ..text_featurizer import TextFeaturizer
from ..feature_normalizer import FeatureNormalizer
from .base import BaseDatasetBuilder


class SpeechRecognitionDatasetKaldiIOBuilder(BaseDatasetBuilder):
    """ SpeechRecognitionDatasetKaldiIOBuilder

    Args:
        for __init__(self, config=None)

    Config::
        audio_config: the config file for feature extractor, default={'type':'Fbank'}
        vocab_file: the vocab file, default='data/utils/ch-en.vocab'

    Interfaces::
        __len__(self): return the number of data samples
        num_class(self): return the max_index of the vocabulary + 1
        @property:
          sample_shape:
            {"input": tf.TensorShape([None, self.audio_featurizer.dim,
                                  self.audio_featurizer.num_channels]),
             "input_length": tf.TensorShape([1]),
             "output_length": tf.TensorShape([1]),
             "output": tf.TensorShape([None])}
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "text_config": {"type":"vocab", "model":"athena/utils/vocabs/ch-en.vocab"},
        "cmvn_file": None,
        "remove_unk": True,
        "input_length_range": [20, 50000],
        "output_length_range": [1, 10000],
        "speed_permutation": [1.0],
        "data_scps_dir": None,
        "merge_label": False
    }

    def __init__(self, config=None):
        super().__init__()
        # hparams
        self.hparams = register_and_parse_hparams(
            self.default_config, config, cls=self.__class__)
        logging.info("hparams: {}".format(self.hparams))

        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        self.feature_normalizer = FeatureNormalizer(self.hparams.cmvn_file)
        self.text_featurizer = TextFeaturizer(self.hparams.text_config)

        if self.hparams.data_scps_dir is not None:
            self.load_scps(self.hparams.data_scps_dir)

    def reload_config(self, config):
        """ reload the config """
        if config is not None:
            self.hparams.override_from_dict(config)

    def preprocess_data(self, file_dir, apply_sort_filter=True):
        """ Generate a list of tuples (feat_key, speaker). """
        logging.info("Loading kaldi-format feats.scp, labels.scp and utt2spk (optional) from {}".format(file_dir))
        self.kaldi_io_feats = kaldiio.load_scp(os.path.join(file_dir, "feats.scp"))
        self.kaldi_io_labels = kaldiio.load_scp(os.path.join(file_dir, "labels.scp"))

        # data checking
        if self.kaldi_io_feats.keys() != self.kaldi_io_labels.keys():
            logging.info("Error: feats.scp and labels.scp does not contain same keys, please check your data.")
            sys.exit()

        # initialize all speakers with 'global' unless 'utterance_key speaker' is specified in "utt2spk"
        self.speakers = dict.fromkeys(self.kaldi_io_feats.keys(), 'global')
        if os.path.exists(os.path.join(file_dir, "utt2spk")):
            with open(os.path.join(file_dir, "utt2spk"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    key, spk = line.strip().split(" ", 1)
                    self.speakers[key] = spk

        self.entries = []
        for key in self.kaldi_io_feats.keys():
            self.entries.append(tuple([key, self.speakers[key]]))
        
        if apply_sort_filter:
            logging.info("Sorting and filtering data, this is very slow, please be patient ...")
            self.entries.sort(key=lambda item: self.kaldi_io_feats[item[0]].shape[0])
            self.filter_sample_by_unk()
            self.filter_sample_by_input_length()
            self.filter_sample_by_output_length()
        return self

    def load_scps(self, file_dir):
        """ load kaldi-format feats.scp, labels.scp and utt2spk (optional) """
        return self.preprocess_data(file_dir)

    def __getitem__(self, index):
        key, speaker = self.entries[index]
        feat = self.kaldi_io_feats[key]
        feat = feat.reshape(feat.shape[0], feat.shape[1], 1)
        feat = tf.convert_to_tensor(feat)
        feat = self.feature_normalizer(feat, speaker)
        label = list(self.kaldi_io_labels[key])
        if self.hparams.merge_label:
            label = self.merge_label(label)

        feat_length = feat.shape[0]
        label_length = len(label)
        return {
            "input": feat,
            "input_length": feat_length,
            "output_length": label_length,
            "output": label,
        }

    def __len__(self):
        """ return the number of data samples """
        return len(self.entries)

    @property
    def num_class(self):
        """ return the max_index of the vocabulary + 1"""
        return len(self.text_featurizer)

    @property
    def speaker_list(self):
        """ return the speaker list """
        return self.speakers

    @property
    def audio_featurizer_func(self):
        """ return the audio_featurizer function """
        return self.audio_featurizer

    @property
    def sample_type(self):
        return {
            "input": tf.float32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.int32,
        }

    @property
    def sample_shape(self):
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None, dim, nc]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None]),
        }

    @property
    def sample_signature(self):
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return (
            {
                "input": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            },
        )

    def filter_sample_by_unk(self):
        """filter samples which contain unk
        """
        if self.hparams.remove_unk is False:
            return self
        filter_entries = []
        unk = self.text_featurizer.unk_index
        if unk == -1:
            return self
        for items in self.entries:
            if unk not in self.kaldi_io_labels[items[0]]:
                filter_entries.append(items)
        self.entries = filter_entries
        return self

    def filter_sample_by_input_length(self):
        """filter samples by input length

        The length of filterd samples will be in [min_length, max_length)

        Args:
            self.hparams.input_length_range = [min_len, max_len]
            min_len: the minimal length (ms for csv-format data, and frame amount for scp-format data)
            max_len: the maximal length (ms for csv-format data, and frame amount for scp-format data)
        returns:
            entries: a filtered list of tuples
            (wav_filename, wav_len, transcripts, speed, speaker)
        """
        min_len = self.hparams.input_length_range[0]
        max_len = self.hparams.input_length_range[1]
        filter_entries = []
        for items in self.entries:
            if self.kaldi_io_feats[items[0]].shape[0] in range(min_len, max_len):
                filter_entries.append(items)
        self.entries = filter_entries
        return self

    def filter_sample_by_output_length(self):
        """filter samples by output length

        The length of filterd samples will be in [min_length, max_length)

        Args:
            self.hparams.output_length_range = [min_len, max_len]
            min_len: the minimal length
            max_len: the maximal length
        returns:
            entries: a filtered list of tuples
            (wav_filename, wav_len, transcripts, speed, speaker)
        """
        min_len = self.hparams.output_length_range[0]
        max_len = self.hparams.output_length_range[1]
        filter_entries = []
        for items in self.entries:
            if self.kaldi_io_labels[items[0]].shape[0] in range(min_len, max_len):
                filter_entries.append(items)
        self.entries = filter_entries
        return self

    def merge_label(self, label):
        """ merge the label which is aligned at frame level
            can be used for computing ctc loss
        Args:
            label: the label  aligned at frame level of one utterance
        returns
            merged_label: the merged label as an unique label sequence
        """
        merged_label_list = []
        last_label = label[0]
        merged_label_list.append(last_label)
        for cur_label in label:
            if cur_label != last_label:
                merged_label_list.append(cur_label)
                last_label = cur_label
        return merged_label_list

    def compute_cmvn_if_necessary(self, is_necessary=True):
        """ compute cmvn file
        """
        if not is_necessary:
            return self
        if os.path.exists(self.hparams.cmvn_file):
            return self
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        with tf.device("/cpu:0"):
            self.feature_normalizer.compute_cmvn_kaldiio(
                self.entries, self.speakers, self.kaldi_io_feats, feature_dim
            )
        self.feature_normalizer.save_cmvn()
        return self
