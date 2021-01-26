import tensorflow as tf
from models.attention.attention_model import BeamSearchModel
from models.attention.attention_model_utils import build_attention_decoder, \
    build_beam_search_placeholders
from models.attention.attention_model_utils import set_mode
from models.e2e_model import E2EModel
from collections import namedtuple
from models.encoder import build_encoder
import pdb


AttentionPlaceholders = namedtuple(
    'AttentionPlaceholders',
    ['group_input_len', 'group_input_seq', 'group_output_mask', 'group_output_seq']
)

TransformerPlaceholders = namedtuple(
    'AttentionPlaceholders',
    ['group_input_len', 'group_input_seq', 'group_input_mask', 'group_output_mask', 'group_output_seq']
)

CTCPlaceholders = namedtuple(
    'CTCPlaceholders',
    ['group_feat', 'group_seq_lens','group_feat_mask' , 'group_label']
)

def build_models(cfg, num_gpus, train_data, mtl_weight):
    # bulid end to end model(support CTC/Attention)
    train_model = build_e2e_model(cfg, gpu_nums=num_gpus,
                                            num_input_dims=train_data.num_input_dims,
                                            num_output_syms=train_data.num_output_syms,
                                            reuse_vars=False, mode='train', mtl_weight=mtl_weight)
    dev_model = build_e2e_model(cfg, gpu_nums=num_gpus,
                                            num_input_dims=train_data.num_input_dims,
                                            num_output_syms=train_data.num_output_syms,
                                            reuse_vars=True, mode='dev', mtl_weight=mtl_weight)
    if mtl_weight == 0.0:
        argmax_model, beam_search_model = None, None
    else:
        argmax_model = build_e2e_model(cfg, gpu_nums=num_gpus,
                                                num_input_dims=train_data.num_input_dims,
                                                num_output_syms=train_data.num_output_syms,
                                                reuse_vars=True, mode='test',
                                                data_loader=train_data, mtl_weight=mtl_weight)
        beam_search_model = build_beam_search_model(
                        train_data.num_input_dims, train_data.num_output_syms, cfg,
                        data_loader=train_data, reuse_vars=True, beam_size=cfg.beam_size)
    return train_model, dev_model, argmax_model, beam_search_model


def build_e2e_model(cfg, gpu_nums=1, num_input_dims=10, num_output_syms=0, reuse_vars=False, mode='train',
                    data_loader=None, mtl_weight=1.0):
    """ build a e2e model
    Args:
        cfg: It comes from the config.py, including params to build graph
             and control process of training, validating or evaluating
        gpu_nums: number of choosed gpu
        mode: this param affects the option 'is_train' for construction of
              graph. Since the process of evaluating and training use different
              statistics in batch_normalization, please set mode='eval' when you
              are doing a evaluating job
    Returns:
        a object of class mtl_model
    """
    set_mode(mode)
    weight_init = tf.initializers.variance_scaling(distribution="uniform")
    argmax_decode = (mode == 'test')
    with tf.variable_scope('e2e_model', reuse=reuse_vars, initializer=weight_init):
        attention_holder = build_transformer_placeholders(input_dim=cfg.FEAT_DIM, gpu_nums=gpu_nums)
        #attention_holder = build_attention_placeholders(input_dim=cfg.FEAT_DIM, gpu_nums=gpu_nums)
        ctc_holder = build_ctc_placeholders(input_dim=cfg.FEAT_DIM, gpu_nums=gpu_nums)
        with tf.device('/gpu:0'):
            # build Multi_gpu_ctc_model graph
            is_train = bool(mode == 'train')

            model = E2EModel(cfg, gpu_nums=gpu_nums,
                             attention_placeholder=attention_holder,
                             ctc_placeholder=ctc_holder,
                             num_input_dims=num_input_dims,
                             num_output_syms=num_output_syms,
                             weight_init=weight_init,
                             argmax_decode=argmax_decode,
                             data_loader=data_loader,
                             is_train=is_train, mtl_weight=mtl_weight)
            return model

def build_beam_search_model(input_dim, num_output_syms, cfg, mode='test',
                            input_normalizer=None,
                            data_loader=None, scope=None, reuse_vars=True,
                            beam_size=None, word_insert_score=0.0):
    set_mode(mode)
    weight_init = tf.uniform_unit_scaling_initializer()
    # we need this variable scope to be able to reuse/restore parameters
    with tf.variable_scope(scope or 'e2e_model/Model', reuse=tf.AUTO_REUSE, initializer=weight_init):
        # for beam search decode, we're not using multi gpu, so we use a new placeholder configuration
        # TODO: do beam search decode on multi gpu?
        phs = build_beam_search_placeholders(input_dim)

        if input_normalizer is not None:
            encoder_input = input_normalizer.normalize(phs.input_seq)
        else:
            encoder_input = phs.input_seq
        encoder_input = tf.transpose(encoder_input, perm=[1,0,2])
        # Encoder
        with tf.variable_scope("e2e_encoder", reuse=True) as scope:
            input_mask = phs.input_mask
            input_mask = tf.transpose(input_mask, perm=[1,0])
            encoder = build_encoder(cfg, encoder_input, phs.input_len, input_mask, input_dim)

        # Decoder
        with tf.variable_scope("attn_decoder", reuse=True) as scope:
            decoder = build_attention_decoder(
                encoder, phs, cfg, num_output_syms,
                weight_init,
                argmax_decode=False, beam_search=True,
                data_loader=data_loader)

    if beam_size is None:
        beam_size = cfg.model.beam_search.beam_size

    model = BeamSearchModel(cfg, decoder, beam_size, max_seq_len=cfg.max_seq_len)
    return model


def build_ctc_placeholders(input_dim=40, gpu_nums=0):
    group_input_seq_len = []
    batch_size = tf.Dimension(None)
    input_dim = tf.Dimension(input_dim)
    #returns
    group_input_seq = []
    group_input_len = []
    group_input_mask = []
    group_sparse_labels = []
    for i in range(gpu_nums):
        group_input_seq_len.append(tf.Dimension(None))
        group_input_seq.append(tf.placeholder(tf.float32, shape=[group_input_seq_len[i], batch_size, input_dim],
                                              name='input_seq' + str(i)))
        group_input_mask.append(tf.placeholder(tf.int32, shape=[group_input_seq_len[i], batch_size],
                                               name='input_mask' + str(i)))
        group_input_len.append(tf.placeholder(tf.int32, [batch_size], name='input_len' + str(i)))
        group_sparse_labels.append(tf.sparse_placeholder(tf.int32, name="label_batch_%d" % i))

    return CTCPlaceholders(group_input_seq, group_input_len, group_input_mask, group_sparse_labels)

def build_transformer_placeholders(input_dim=40, gpu_nums=0):
    group_input_seq_len = []
    group_output_seq_len = []
    batch_size = tf.Dimension(None)
    input_dim = tf.Dimension(input_dim)
    #returns
    group_input_seq = []
    group_input_len = []
    group_input_mask = []
    group_output_seq = []
    group_output_mask = []
    for i in range(gpu_nums):
        group_input_seq_len.append(tf.Dimension(None))
        group_input_seq.append(tf.placeholder(tf.float32, shape=[group_input_seq_len[i], batch_size, input_dim],
                                              name='input_seq' + str(i)))
        group_input_len.append(tf.placeholder(tf.int32, [batch_size], name='input_len' + str(i)))
        group_input_mask.append(tf.placeholder(tf.int32, shape=[group_input_seq_len[i], batch_size],
                                               name='input_mask' + str(i)))
        group_output_seq_len.append(tf.Dimension(None))
        group_output_seq.append(tf.placeholder(tf.int32, shape=[group_output_seq_len[i], batch_size],
                                               name='output_seq' + str(i)))
        group_output_mask.append(tf.placeholder(tf.int32, shape=[group_output_seq_len[i], batch_size],
                                                name='output_mask' + str(i)))
    return TransformerPlaceholders(group_input_len, group_input_seq, group_input_mask, group_output_mask, group_output_seq)

def build_attention_placeholders(input_dim=40, gpu_nums=0):
    group_input_seq_len = []
    group_output_seq_len = []
    batch_size = tf.Dimension(None)
    input_dim = tf.Dimension(input_dim)
    #returns
    group_input_seq = []
    group_input_len = []
    group_output_seq = []
    group_output_mask = []
    for i in range(gpu_nums):
        group_input_seq_len.append(tf.Dimension(None))
        group_input_seq.append(tf.placeholder(tf.float32, shape=[group_input_seq_len[i], batch_size, input_dim],
                                              name='input_seq' + str(i)))
        group_input_len.append(tf.placeholder(tf.int32, [batch_size], name='input_len' + str(i)))
        group_output_seq_len.append(tf.Dimension(None))
        group_output_seq.append(tf.placeholder(tf.int32, shape=[group_output_seq_len[i], batch_size],
                                               name='output_seq' + str(i)))
        group_output_mask.append(tf.placeholder(tf.int32, shape=[group_output_seq_len[i], batch_size],
                                                name='output_mask' + str(i)))

    return AttentionPlaceholders(group_input_len, group_input_seq, group_output_mask, group_output_seq)
