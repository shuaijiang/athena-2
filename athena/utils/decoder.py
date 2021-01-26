from abc import ABC, abstractmethod

import tensorflow as tf

from models.attention import attention_model_utils

import pdb


class AttentionDecoder(ABC):
    """
    Decodes the output by (optionally) attending over encoder hidden states.
    The RNN can be made to avoid using attention (in which case, it is
    equivalent to a simple seq2seq model).

    Also, it can be used to do argmax decoding, in which at each timestep, the
    symbol corresponding to the max logit from previous time-step is used to
    decode the next symbol.

    """
    def __init__(self, encoder, output_projection, attn_context, output_seq,
                 scheduled_sampling_rate=0.,
                 argmax_decode=False, argmax_extra_rollout=100,
                 beam_search=False, input_seq_ph=None, input_len_ph=None, input_mask_ph=None,
                 data_loader=None, dtype=tf.float32, cfg=None):
        if argmax_decode and data_loader is None:
            raise ValueError("`data_loader` must be provided when doing "
                             "argmax decoding")

        self.cfg = cfg

        self.logits = None
        self.alphas = None
        self.decoder_inputs = None
        self.decoder_masks = None

        self.encoder = encoder
        self.output_projection = output_projection
        self.attn_context = attn_context
        self.output_seq = output_seq
        self.scheduled_sampling_rate = scheduled_sampling_rate
        self.argmax_decode = argmax_decode
        self.argmax_extra_rollout = argmax_extra_rollout
        self.data_loader = data_loader
        self.dtype = dtype

        self.num_time_steps = tf.shape(self.output_seq)[0]
        self.batch_size = tf.unstack(tf.shape(
            self.encoder.collapsed_output_seq))[1]
        self.num_output_syms = self.output_projection.num_output_symbols()
        self.beam_search = beam_search
        self.input_seq_ph = input_seq_ph
        self.input_len_ph = input_len_ph
        self.input_mask_ph = input_mask_ph

        # decoding related
        self.MAX_STEPS = 200
        #self.MAX_STEPS = tf.floordiv(
        #    tf.reduce_max(input_len_ph, name='decoder_max_steps'), 10)
        self.stop_symbol = self.num_output_syms - 1
        self.start_symbol = self.num_output_syms - 2

    @abstractmethod
    def step(self, decoder_input, curr_state, curr_alpha, batch_index=0):
        """Do a single decoder step.

        return new_logits, new_state, new_alpha
        """
        pass

    def build_loop_step(self):

        def loop_step(ts, eos_emitted, curr_state, logits,
                      alphas, decoder_inputs, decoder_masks):

            decoder_input, decoder_masks, eos_emitted = self.select_input(
                ts, logits, decoder_masks, eos_emitted)

            if self.attn_context is not None:
                curr_alpha = alphas.read(ts)
            else:
                curr_alpha = None

            new_logits, new_state, new_alpha = self.step(decoder_input, curr_state, curr_alpha, 0)

            if self.attn_context is not None:
                alphas = alphas.write(ts + 1, new_alpha)
            logits = logits.write(ts + 1, new_logits)
            decoder_inputs = decoder_inputs.write(ts, decoder_input)

            return (ts + 1, eos_emitted, new_state, logits, alphas,
                    decoder_inputs, decoder_masks)

        return loop_step

    def build_arrays(self):
        if self.argmax_decode:
            dynamic_size = True
            size = 1
        else:
            dynamic_size = False
            size = self.num_time_steps

        logits = tf.TensorArray(
            tf.float32, size=size + 1, dynamic_size=dynamic_size,
            infer_shape=False, clear_after_read=False,
            name='logits')
        alphas = tf.TensorArray(
            tf.float32, size=size + 1, dynamic_size=dynamic_size,
            infer_shape=False, clear_after_read=False,
            name='alphas')
        decoder_inputs = tf.TensorArray(
            tf.int32, size=size, dynamic_size=dynamic_size,
            infer_shape=True, clear_after_read=False,
            name='decoder_inputs')
        decoder_masks = tf.TensorArray(
            tf.bool, size=size, dynamic_size=dynamic_size,
            infer_shape=True, clear_after_read=False,
            name='decoder_masks')


        batch_size = self.batch_size
        logits = logits.write(
            0, self.output_projection.initial_logits(batch_size))
        alphas = alphas.write(0, self.attn_context.initial_alpha(batch_size))
        eos_emitted = tf.fill([batch_size], False)

        return logits, alphas, decoder_inputs, decoder_masks, eos_emitted

    def select_input(self, ts, logits, decoder_masks, eos_emitted):
        def select_argmax_input():
            current_logits = logits.read(ts)
            # TensorArray seems to make TF forget shape info.
            current_logits = tf.reshape(
                current_logits, [self.batch_size, self.num_output_syms])
            decoder_input = tf.to_int32(tf.argmax(current_logits, 1))
            decoder_input = tf.stop_gradient(decoder_input)
            return decoder_input

        def select_sample_input():
            current_logits = logits.read(ts)
            decoder_input = tf.to_int32(tf.multinomial(current_logits, 1))
            decoder_input = tf.stop_gradient(decoder_input)
            return tf.squeeze(decoder_input, [1])

        def select_real_input():
            decoder_input = tf.slice(self.output_seq, [ts, 0], [1, -1])
            decoder_input = tf.squeeze(decoder_input, [0])
            return decoder_input

        if self.argmax_decode:

            # Only select real input for ts=0
            decoder_input = tf.cond(ts > 0,
                                    select_argmax_input,
                                    select_real_input)
        else:
            if attention_model_utils.get_mode() == 'train':
                random_num = tf.random_uniform([], 0., 1.)
                decoder_input = tf.cond(
                    random_num < self.scheduled_sampling_rate,
                    select_sample_input, select_real_input)
            else:
                # Don't use scheduled sampling if in dev mode (when not
                # using argmax decoding)
                decoder_input = select_real_input()

        if self.data_loader is not None:
            stop_symbol = self.data_loader.stop_symbol
            eos_emitted = tf.logical_or(
                eos_emitted, tf.equal(decoder_input, stop_symbol))
            decoder_mask = tf.logical_not(eos_emitted)
            decoder_masks = decoder_masks.write(ts, decoder_mask)

        return decoder_input, decoder_masks, eos_emitted

    @property
    @abstractmethod
    def initial_state(self):
        pass

    def build(self):
        if self.beam_search:
            print('build beam search loop')
            self.build_beam_search_loop(beam_size=self.cfg.beam_size, batch_size=self.cfg.BATCH)
            #if self.cfg is not None and hasattr(self.cfg, 'is_frozen') and self.cfg.is_frozen:
            #    self.build_beam_search_loop()
            #else:
            #    self.build_beam_search()
        else:
            print('build loop')
            self.build_loop()

    def build_loop(self):
        num_time_steps = tf.shape(self.output_seq)[0]
        batch_size = tf.unstack(tf.shape(self.encoder.collapsed_output_seq))[1]
        initial_state = self.initial_state

        logits, alphas, decoder_inputs, decoder_masks, eos_emitted =\
            self.build_arrays()

        def continue_while(ts, eos_emitted, *_):
            if self.argmax_decode:
                max_time_steps = tf.to_int32(
                    self.argmax_extra_rollout + tf.to_float(num_time_steps)
                )
                all_eos_emitted = tf.reduce_all(eos_emitted)
                return tf.logical_and(ts < max_time_steps,
                                      tf.logical_not(all_eos_emitted))
            else:
                return ts < num_time_steps

        loop_step = self.build_loop_step()

        ts = tf.constant(0)
        _, _, _, logits, alphas, decoder_inputs, decoder_masks = tf.while_loop(
            continue_while, loop_step,
            [ts, eos_emitted, initial_state, logits, alphas,
             decoder_inputs, decoder_masks],
            parallel_iterations=1, back_prop=True, swap_memory=True)

        self.logits = logits.stack()
        self.alphas = alphas.stack()
        self.decoder_inputs = decoder_inputs.stack()

        # Prepend row of zeros to mask and convert to float32 from bool
        if self.data_loader is not None:
            self.decoder_masks = tf.to_float(tf.concat(axis=0, values=[
                tf.fill([1, batch_size], False), decoder_masks.stack()]))

    def build_beam_search(self):
        """Setup beam seach ops

        Args:
            input_seq (tf.Placeholder): placeholder used to compute
                encoder output
            input_len (tf.Placeholder): placeholder used to compute
                encoder output
        """

        if self.input_seq_ph is None or self.input_len_ph is None:
            raise ValueError("Must provide input placeholders to do "
                             "beam search")

        self.decoder_input_ph, self.curr_state_ph, self.curr_alpha_ph = \
            self.beam_search_placeholders()

        new_logits, new_state, new_alpha = self.step(
            self.decoder_input_ph, self.curr_state_ph, self.curr_alpha_ph)

        Z = tf.reduce_logsumexp(new_logits,
                                reduction_indices=(1,), keep_dims=True)
        new_logprobs = new_logits - Z

        self.new_logprobs_op = new_logprobs
        self.new_state_op = new_state
        self.new_alpha_op = new_alpha
        self.init_state_op = self.initial_state
        self.init_alpha_op = self.attn_context.initial_alpha(self.batch_size)
        self.encoder_states = self.encoder.collapsed_output_seq

    @abstractmethod
    def beam_search_placeholders(self):
        """Return placeholders used in beam search.

        return decoder_input_ph, curr_state_ph, curr_alpha_ph
        """
        pass

    def initialize_beam_search(self, input_seq, input_len, input_mask, sess):
        """
        Initialize beam search for a single sequence.

        Args:
            input_seq (ndarray): (time_steps, 1, num_feats)
            input_len (ndarray): (1,)

        Caches:
            encoder_state_vals
            attn_feat_vals

        Returns:
            init_state_vals (ndarray): (1, dec_dim)
            init_alpha_vals (ndarray): (enc_steps, 1)

        """
        feed_dict = {
            self.input_seq_ph: input_seq,
            self.input_len_ph: input_len,
            self.input_mask_ph: input_mask,
        }

        fetches = [self.encoder_states, self.attn_context.attn_features,
                   self.init_state_op, self.init_alpha_op]

        # Save encoder_states and attn_features and return others
        # TODO: Cache encoder_states and attn_features in graph
        # cf: http://stackoverflow.com/a/33662680/665254
        (self.encoder_state_vals, self.attn_feat_vals,
         init_state_vals, init_alpha_vals) = sess.run(fetches, feed_dict)

        return init_state_vals, init_alpha_vals

    def beam_search_step(self, decoder_input, curr_state, curr_alpha, sess):
        """Do a single decoder step for use in beam search

        Args:
            decoder_input (ndarray): (batch_size,) symbol indices (ints)
            curr_state (ndarray): (batch_size, state_size)
            curr_alpha (ndarray): (batch_size, enc_steps)

        Returns:
            new_logprobs (ndarray): (batch_size, num_symbols)
            new_state (ndarray): (batch_size, state_size)
            new_alphas (ndarray): (batch_size, enc_steps)
        """
        feed_dict = {
            self.encoder_states: self.encoder_state_vals,
            self.attn_context.attn_features: self.attn_feat_vals,
            self.decoder_input_ph: decoder_input,
            self.curr_state_ph: curr_state,
            self.curr_alpha_ph: curr_alpha,
        }

        fetches = [self.new_logprobs_op, self.new_state_op, self.new_alpha_op]
        return sess.run(fetches, feed_dict)

    def get_logits(self):
        """
        Returns:
        Logits for all time-steps. 3d tensor of shape:
            (time_steps + 1, batch_size, num_output_symbols)
        The first time step is all zeros and should be ignored.
        """
        return self.logits

    def get_alphas(self):
        """
        Returns:
        Alpha values for all time-steps. 3d tensor of shape:
            (time_steps + 1, encoder_time_steps, batch_size)
        The first time step is all zeros and should be ignored.
        """
        return self.alphas

    def get_decoder_inputs(self):
        """
        Returns:
        decoder inputs for all time-steps. 2d tensor of shape:
            (time_steps, batch_size)
        """
        return self.decoder_inputs

    def get_decoder_masks(self):
        """
        Returns:
        decoder masks for all time-steps. 2d tensor of shape:
            (time_steps, batch_size)
        """
        return self.decoder_masks

    def build_beam_search_loop(self, beam_size=4, batch_size=8):
        """Build Beam Search decoding graph

        NOTE: 1. ONLY compatable with LAS for now!!
        NOTE: 2. ONLY batch size == 1
        """
        print("Build Beam Search Decoding Graph")
        print("  beam size: {}".format(beam_size))
        print("  batch size: {}".format(batch_size))

        DEBUG = False
        MAX_STEPS = self.MAX_STEPS
        num_symbols = self.num_output_syms
        stop_symbol = self.stop_symbol
        start_symbol = self.start_symbol
        num_unstop_symbols = num_symbols - 1

        def condition(batch_index, ts, stop, *_):
            return tf.logical_and(tf.less(ts, MAX_STEPS), tf.logical_not(stop))

        def loop_step(batch_index, ts, stop_decoder, states, alphas, cand_seqs,
            cand_scores, completed_scores, completed_scores_scaled,
            completed_seqs, completed_lens):
            """
            Args:
              batch_index: batch index
              ts (int): time step
              stop_decoder (bool): stop decoding
              ys (?): [beam_size]
              states (float): [beam_size, state_size]
              alphas (float): [beam_size, alpha_size]
              cand_scores: [beam_size], sequence score
              cand_seqs: [beam_size, ts], ts increases over time

            Returns:
              logits shape: [beam_size, output_dim]
              state: [beam_size, state_size]
              alpha: [beam_size, alpha_size]

            """
            # 1. get score from one step decoder
            # logits = tf.one_hot(ts, depth=num_symbols, off_value=0.0, dtype=tf.float32)
            if DEBUG: ts = tf.Print(ts, [ts], message='ts: ')
            ys = cand_seqs[:, ts]
            if DEBUG: ys = tf.Print(ys, [ys], message='Y(t-1): ')
            logits, states, alphas = self.step(ys, states, alphas, batch_index)
            if DEBUG: logits = tf.Print(logits, [logits], message='logits: ')
            Z = tf.reduce_logsumexp(logits, 1, keep_dims=True)
            if DEBUG: Z = tf.Print(Z, [Z], message='Z: ')
            logprobs = tf.subtract(logits, Z)  # [beam_size, num_symbols]
            new_scores = tf.add(logprobs, tf.expand_dims(cand_scores, 1)) # [beam_size, num_symbols]
            if DEBUG: new_scores = tf.Print(new_scores, [new_scores], message='new_scores: ')

            num_unstop_symbols = tf.shape(new_scores)[1] - 1
            new_uncompleted_scores, new_completed_scores = tf.split(
                new_scores, [num_unstop_symbols, 1], 1)
            if DEBUG: new_uncompleted_scores = tf.Print(
                new_uncompleted_scores, [new_uncompleted_scores],
                message='new_uncompleted_scores: ')

            # 2. Update completed seqs  --------------------------------------
            # 2.1 update scores
            new_completed_scores = tf.squeeze(new_completed_scores, -1) # [beam_size]
            all_completed_scores = tf.concat([completed_scores,
                new_completed_scores], 0)  # [2*beam_size]

            # 2.2 choose top K from scaled_scores
            new_completed_scores_scaled = tf.div(
                new_completed_scores, tf.to_float(ts+1))
            all_scores_scaled = tf.concat(
                [completed_scores_scaled, new_completed_scores_scaled], 0)
            completed_scores_scaled, indices = tf.nn.top_k(all_scores_scaled,
                k=beam_size, sorted=False)
            if DEBUG: indices = tf.Print(
                indices, [indices], message='top K completed indices: ')

            # 2.2 update len
            new_completed_lens = tf.fill([beam_size], tf.add(ts, 1)) # [beam_size]
            all_lens = tf.concat(
                [completed_lens, new_completed_lens], 0)  # [2*beam_size]
            completed_lens = tf.gather(all_lens, indices,
                validate_indices=True, axis=0)  # [beam_size]
            if DEBUG: completed_lens = tf.Print(completed_lens, [completed_lens],
                message='completed lens', summarize=5)

            # 2.3 update seqs
            all_completed = tf.concat([completed_seqs, cand_seqs], 0)
            completed_seqs = tf.gather(all_completed, indices,
                validate_indices=True, axis=0)  # [beam_size, ts]
            if DEBUG: completed_seqs = tf.Print(
                completed_seqs, [completed_seqs], message='completed seqs: ',
                summarize=MAX_STEPS+2)

            # 2.4 stop decoding loop
            max_uncompleted = tf.reduce_max(new_uncompleted_scores)
            completed_scores = tf.gather(all_completed_scores, indices,
                validate_indices=True, axis=0)
            min_completed = tf.reduce_min(completed_scores)
            stop_decoder = tf.greater(min_completed, max_uncompleted)
            # 2. Update completed seqs  --------------------------------------

            # 3. Update uncompleted sequences --------------------------------
            # new_uncompleted_scores: [beam_size, num_symbols-1]
            # top_k: [beam_size]. indices of top k scores
            def f0(): return new_uncompleted_scores[0,:]
            def f1(): return new_uncompleted_scores
            un_scores = tf.cond(tf.equal(ts,0), f0, f1)
            new_flat = tf.squeeze(tf.reshape(un_scores, [-1,1]))  # [beam_size*num_unstop_symbols]

            # get top K symbols
            cand_scores, flat_indices = tf.nn.top_k(new_flat, k=beam_size, sorted=False)
            cand_parents = tf.div(flat_indices, num_unstop_symbols)
            _ys = tf.mod(flat_indices, num_unstop_symbols)  # [beam_size], y(t) for next step
            A = tf.gather(cand_seqs[:,0:ts+1], cand_parents)  #[beam_size, ts+1]
            B = tf.expand_dims(_ys, -1)  # [beam_size, 1]
            C = tf.fill([beam_size, MAX_STEPS+2 - ts - 2], stop_symbol)
            cand_seqs = tf.concat([A, B, C], 1)  # [beam_size, MAX_STEPS]
            if DEBUG: cand_seqs = tf.Print(cand_seqs, [cand_seqs],
                message='cand seqs: ', summarize=MAX_STEPS+2)
            cand_seqs = tf.reshape(cand_seqs, [beam_size, MAX_STEPS+2])
            cand_scores.set_shape([beam_size])
            completed_seqs = tf.reshape(completed_seqs, [beam_size, MAX_STEPS+2])


            s1_shape = [beam_size, self.attention_cell.state_size]
            s2_shape = [beam_size, self.decoder_cell.state_size]
            s3_shape = [beam_size, self.attn_context.context_size]

            # prepare data for next step
            # states = tf.gather(states, cand_parents, axis=0)
            # states = self.select_states(states, cand_parents)
            states = tuple(tf.gather(el, cand_parents) for el in states)
            states[0].set_shape(s1_shape)
            states[1].set_shape(s2_shape)
            states[2].set_shape(s3_shape)
            alphas = tf.gather(alphas, cand_parents, axis=1)
            alphas_shape = [self.attn_context.num_encoder_states, beam_size]
            alphas = tf.reshape(alphas, alphas_shape)
            # alphas.set_shape(alphas_shape)
            # 3. Update uncompleted sequences --------------------------------

            ts = tf.add(ts, 1)
            return batch_index, ts, stop_decoder, states, alphas, cand_seqs, \
                cand_scores, completed_scores, completed_scores_scaled, \
                completed_seqs, completed_lens

        def condition_bacth(batch_index, stop, *_):
            return tf.logical_and(tf.less(batch_index, batch_size), tf.logical_not(stop))

        def loop_batch(batch_index, stop_batch,
                       batch_completed_seqs, batch_completed_scores, batch_completed_scores_scaled):

            # states = self.initial_state
            # TODO: only LAS
            init_attn_state = self.attention_cell.zero_state(beam_size, self.dtype)
            init_decoder_state = self.decoder_cell.zero_state(beam_size, self.dtype)
            init_context = self.attn_context.initial_context(beam_size)
            states = (init_attn_state, init_decoder_state, init_context)

            alphas = self.attn_context.initial_alpha(beam_size)  # batch?
            alphas_shape = [self.attn_context.num_encoder_states, beam_size]

            aa = tf.fill([beam_size, 1], start_symbol)
            bb = tf.fill([beam_size, MAX_STEPS+1], stop_symbol)
            cand_seqs = tf.concat([aa, bb], 1)
            if DEBUG: cand_seqs = tf.Print(cand_seqs, [cand_seqs],
                message='Initial cand seqs', summarize=beam_size*(MAX_STEPS+2))
            cand_scores = tf.fill([beam_size], 0.0)

            completed_seqs = tf.fill([beam_size, MAX_STEPS+2], 0) #stop_symbol)
            completed_lens = tf.fill([beam_size], 0)
            completed_scores = tf.fill([beam_size], -99.0)  # float.min?
            completed_scores_scaled = tf.fill([beam_size], -99.0) # float.min?


            ts = tf.constant(0)
            stop_decoder = tf.constant(False)
            batch_index, ts, stop_decoder, states, alphas, cand_seqs, \
                cand_scores, completed_scores, completed_scores_scaled,  \
                completed_seqs, completed_lens = tf.while_loop(
                condition, loop_step,
                [batch_index, ts, stop_decoder, states, alphas, cand_seqs,
                 cand_scores, completed_scores, completed_scores_scaled,
                 completed_seqs, completed_lens],
                parallel_iterations=1, back_prop=False, swap_memory=True)

            #batch_completed_seqs = \
            #tf.Print(batch_completed_seqs, [tf.shape(batch_completed_seqs)], message='batch_completed_seqs shape1')
            batch_completed_seqs = tf.concat([batch_completed_seqs, completed_seqs], axis=0)
            batch_completed_seqs = batch_completed_seqs[beam_size:,:]
            batch_completed_seqs = tf.reshape(batch_completed_seqs, [batch_size * beam_size,  MAX_STEPS+2])
            #batch_completed_seqs = \
            #tf.Print(batch_completed_seqs, [tf.shape(batch_completed_seqs)], message='batch_completed_seqs shape2')

            batch_completed_scores = tf.concat([batch_completed_scores, completed_scores], axis=0)
            batch_completed_scores = batch_completed_scores[beam_size:]
            batch_completed_scores = tf.reshape(batch_completed_scores, [batch_size * beam_size])

            batch_completed_scores_scaled = tf.concat([batch_completed_scores_scaled, completed_scores_scaled], axis=0)
            batch_completed_scores_scaled = batch_completed_scores_scaled[beam_size:]
            batch_completed_scores_scaled = \
                tf.reshape(batch_completed_scores_scaled, [batch_size * beam_size])

            # LM rescore 1-best
            #best = tf.argmax(completed_scores_scaled, output_type=tf.int32)
            #length = completed_lens[best] + 1
            #self.decoding_op = tf.identity(completed_seqs[best, :length],
            #    name='decoding_output')
            batch_index = tf.add(batch_index, 1)
            return batch_index, stop_batch, batch_completed_seqs, \
                   batch_completed_scores, batch_completed_scores_scaled

        batch_index = tf.constant(0)
        stop_batch= tf.constant(False)

        batch_completed_seqs = tf.fill([batch_size * beam_size, self.MAX_STEPS + 2], 0)
        batch_completed_scores = tf.fill([batch_size * beam_size], -99.0)
        batch_completed_scores_scaled = tf.fill([batch_size * beam_size], -99.0)
        batch_index, stop_batch, batch_completed_seqs, \
        batch_completed_scores, batch_completed_scores_scaled = \
                    tf.while_loop(condition_bacth, loop_batch,
                                  [batch_index, stop_batch, batch_completed_seqs,
                                   batch_completed_scores, batch_completed_scores_scaled],
                                  parallel_iterations=1, back_prop=False, swap_memory=True)
        # LM rescore n-best
        batch_completed_seqs = \
            tf.reshape(batch_completed_seqs, [batch_size * beam_size,  MAX_STEPS+2])
        batch_completed_scores = tf.reshape(batch_completed_scores, [batch_size * beam_size])
        batch_completed_scores_scaled = \
            tf.reshape(batch_completed_scores_scaled, [batch_size * beam_size])

        self.decoding_op = tf.identity(batch_completed_seqs, name='decoding_output')
        self.decoding_op2 = tf.identity(batch_completed_scores, name='decoding_scores')
        self.decoding_op3 = tf.identity(batch_completed_scores_scaled, name='decoding_scores_scaled')


class SimpleRNNAttentionDecoder(AttentionDecoder):
    """
    This is a basic Attention Decoder as described in Bahdanau et al.

    For each time-step of the RNN:
        1. The decoder calls the attn_context object to compute a context
        vector (as a linear combination of attn_features).
        2. Using the context vector, previous RNN state and previous symbol,
        the RNN computes the next state.
        3. The decoder then calls the output_projection object on this output
        to compute logits over the next symbol.
    """
    def __init__(self, decoder_cell, encoder, output_projection, attn_context,
                 output_seq, scheduled_sampling_rate=0.,
                 argmax_decode=False, argmax_extra_rollout=100,
                 beam_search=False, input_seq_ph=None, input_len_ph=None,
                 data_loader=None, dtype=tf.float32,
                 scope=None, cfg=None):
        super().__init__(
            encoder, output_projection, attn_context, output_seq,
            scheduled_sampling_rate, argmax_decode, argmax_extra_rollout,
            beam_search, input_seq_ph, input_len_ph,
            data_loader, dtype, cfg)
        self.decoder_cell = decoder_cell

        with tf.variable_scope(scope or type(self).__name__):
            self.build()

    @property
    def initial_state(self):
        return self.decoder_cell.zero_state(self.batch_size, self.dtype)

    def step(self, decoder_input, curr_state, curr_alpha):

        num_output_syms = self.output_projection.num_output_symbols()
        decoder_input_one_hot = tf.one_hot(decoder_input, num_output_syms)

        if self.attn_context is not None:
            context, new_alpha = self.attn_context.compute_context(
                curr_state, curr_alpha)
            full_input = tf.concat(axis=1, values=[
                decoder_input_one_hot, context])
        else:
            full_input = decoder_input_one_hot

        output, new_state = self.decoder_cell(full_input, curr_state)
        new_logits = self.output_projection.compute_logits(output)

        return new_logits, new_state, new_alpha

    def beam_search_placeholders(self):
        enc_steps = tf.Dimension(None)
        batch_size = tf.Dimension(None)
        dec_state_size = self.decoder_cell.state_size

        decoder_input = tf.placeholder(tf.int32, (batch_size,),
                                       name='decoder_input')
        curr_state = tf.placeholder(tf.float32,
                                    (batch_size, dec_state_size),
                                    name='curr_state')
        curr_alpha = tf.placeholder(tf.float32,
                                    (enc_steps, batch_size),
                                    name='curr_alpha')

        return decoder_input, curr_state, curr_alpha


class LASDecoder(AttentionDecoder):
    """
    This implements the Two Level Decoder as described in the "Listen, Attend
    and Spell" paper.

    It uses two RNNs - an Attention RNN and a Decoder RNN. The Attention RNN
    runs before the attention context vector is computed, which means that the
    symbol that was chosen in the previous step is used to compute the context
    vector.

    For each time-step of the RNN:
        1. The decoder runs one step of Attention RNN as a function of
        previous state and current input.
        2. Using the output of the Attention RNN, the decoder then calls the
        attn_context object to compute a context vector (as a linear
        combination of attn_features).
        3. Using the context vector and output of Attention RNN, the RNN then
        runs one step of Decoder RNN.
        3. The decoder then calls the output_projection object on this output
        of Decoder RNN to compute logits over the next symbol.
    """
    def __init__(self, decoder_cell, attention_cell, encoder,
                 output_projection, attn_context, output_seq,
                 scheduled_sampling_rate=0.,
                 argmax_decode=False, argmax_extra_rollout=100,
                 beam_search=False, input_seq_ph=None, input_len_ph=None, input_mask_ph=None,
                 data_loader=None, dtype=tf.float32,
                 scope=None, cfg=None):
        super().__init__(
            encoder, output_projection, attn_context, output_seq,
            scheduled_sampling_rate, argmax_decode, argmax_extra_rollout,
            beam_search, input_seq_ph, input_len_ph, input_mask_ph,
            data_loader, dtype, cfg)
        self.decoder_cell = decoder_cell
        self.attention_cell = attention_cell

        with tf.variable_scope(scope or type(self).__name__):
            self.build()

    @property
    def initial_state(self):
        init_attn_state = self.attention_cell.zero_state(
            self.batch_size, self.dtype)
        init_decoder_state = self.decoder_cell.zero_state(
            self.batch_size, self.dtype)
        if self.attn_context is not None:
            init_context = self.attn_context.initial_context(self.batch_size)
        else:
            init_context = tf.constant(0)
        init_state = (init_attn_state, init_decoder_state, init_context)
        return init_state

    def step(self, decoder_input, curr_state, curr_alpha, batch_index):

        curr_attn_state, curr_decoder_state, curr_context = curr_state

        num_output_syms = self.output_projection.num_output_symbols()
        decoder_input_one_hot = tf.one_hot(decoder_input, num_output_syms)

        # Run one step of the Attention Layer RNN
        full_input = tf.concat(
            axis=1, values=[decoder_input_one_hot, curr_context])
        #full_input = tf.Print(full_input, [full_input], message='Full input')
        #curr_attn_state = tf.Print(curr_attn_state, [curr_attn_state], message='curr_attn_state')
        attn_output, new_attn_state = self.attention_cell(
            full_input, curr_attn_state, scope='attention_cell')
        #attn_output = tf.Print(attn_output, [attn_output], message='attn_output')
        #attn_output = tf.Print(attn_output, [tf.shape(attn_output)], message='attn_output shape')
        #new_attn_state = tf.Print(new_attn_state, [new_attn_state], message='new_attn_state')
        #new_attn_state = tf.Print(new_attn_state, [tf.shape(new_attn_state)], message='new_attn_state shape')

        # Compute new attention context
        if self.attn_context is not None:
            new_context, new_alpha = self.attn_context.compute_context_single_batch(
                new_attn_state, curr_alpha, batch_index)
            #new_context = tf.Print(new_context, [new_context], message='new_context')
            #new_alpha = tf.Print(new_alpha, [new_alpha], message='new_alpha')
        else:
            #curr_context = tf.Print(curr_context, [curr_context], message='curr_context')
            new_context = curr_context
            new_alpha = None

        # Run one step of the Decoder Layer RNN
        decoder_layer_input = tf.concat(
            axis=1, values=[attn_output, new_context])
        output, new_decoder_state = self.decoder_cell(
            decoder_layer_input, curr_decoder_state, scope='decoder_cell')
        #output = tf.Print(output, [output], message='output')

        # Compute logits
        new_logits = self.output_projection.compute_logits(output)

        new_state = (new_attn_state, new_decoder_state, new_context)

        return new_logits, new_state, new_alpha

    def beam_search_placeholders(self):
        enc_steps = tf.Dimension(None)
        batch_size = tf.Dimension(None)
        attn_state_size = self.attention_cell.state_size
        dec_state_size = self.decoder_cell.state_size
        context_size = self.attn_context.context_size

        decoder_input = tf.placeholder(tf.int32, (batch_size,),
                                       name='decoder_input')
        curr_attn_state = tf.placeholder(tf.float32,
                                         (batch_size, attn_state_size),
                                         name='curr_attn_state')
        curr_decoder_state = tf.placeholder(tf.float32,
                                            (batch_size, dec_state_size),
                                            name='curr_decoder_state')
        curr_context = tf.placeholder(tf.float32,
                                      (batch_size, context_size),
                                      name='curr_context')
        curr_alpha = tf.placeholder(tf.float32,
                                    (enc_steps, batch_size),
                                    name='curr_alpha')

        curr_state = (curr_attn_state, curr_decoder_state, curr_context)

        return decoder_input, curr_state, curr_alpha
