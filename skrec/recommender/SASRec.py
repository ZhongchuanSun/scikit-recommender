"""
Paper: Self-Attentive Sequential Recommendation
Author: Wang-Cheng Kang, and Julian McAuley
Reference: https://github.com/kang205/SASRec
"""
__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["SASRec"]

import numpy as np
from typing import Dict
import tensorflow as tf
from .base import AbstractRecommender
from ..utils.py import EarlyStopping
from ..utils.tf1x import inner_product
from ..utils.py import pad_sequences, batch_randint_choice, BatchIterator
from ..utils.py import Config
from ..run_config import RunConfig


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0.0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs


class SASRecConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 l2_emb=0.0,
                 hidden_units=64,
                 dropout_rate=0.5,
                 max_len=50,
                 num_blocks=2,
                 num_heads=1,
                 batch_size=128,
                 epochs=1000,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.l2_emb: float = l2_emb
        self.hidden_units: int = hidden_units
        self.dropout_rate: float = dropout_rate
        self.max_len: int = max_len
        self.num_blocks: int = num_blocks
        self.num_heads: int = num_heads
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.l2_emb, float) and self.l2_emb >= 0
        assert isinstance(self.hidden_units, int) and self.hidden_units > 0
        assert isinstance(self.dropout_rate, float) and 1 > self.dropout_rate >= 0
        assert isinstance(self.max_len, int) and self.max_len > 0
        assert isinstance(self.num_blocks, int) and self.num_blocks > 0
        assert isinstance(self.num_heads, int) and self.num_heads > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class SASRec(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = SASRecConfig(**model_config)
        super().__init__(run_config, self.config)

        self.users_num, self.items_num = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.train_data.to_user_dict_by_time()
        self.all_users = list(self.user_pos_train.keys())

        self._build_model()
        tf_config = tf.ConfigProto()  # allow_soft_placement=False, log_device_placement=True
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        self.test_item_seqs = self._process_test()

    def _process_test(self):
        item_seqs = [self.user_pos_train[user][-self.config.max_len:] if user in self.user_pos_train else [self.items_num]
                     for user in range(self.users_num)]

        test_item_seqs = pad_sequences(item_seqs, value=self.items_num, max_len=self.config.max_len,
                                       padding='pre', truncating='pre', dtype=np.int32)
        return test_item_seqs

    def _generate_train_data(self):
        item_seq_list, item_pos_list = [], []
        all_users = BatchIterator(self.all_users, batch_size=1024, shuffle=False, drop_last=False)
        for bat_users in all_users:
            bat_seq = [self.user_pos_train[u][:-1] for u in bat_users]
            bat_pos = [self.user_pos_train[u][1:] for u in bat_users]

            # padding
            bat_seq = pad_sequences(bat_seq, value=self.items_num, max_len=self.config.max_len,
                                    padding='pre', truncating='pre')
            bat_pos = pad_sequences(bat_pos, value=self.items_num, max_len=self.config.max_len,
                                    padding='pre', truncating='pre')

            item_seq_list.extend(bat_seq)
            item_pos_list.extend(bat_pos)
        return item_seq_list, item_pos_list

    def _sample_negative(self):
        item_neg_list = []
        all_users = BatchIterator(self.all_users, batch_size=1024, shuffle=False, drop_last=False)
        for bat_users in all_users:
            n_neg_items = [len(self.user_pos_train[u][1:]) for u in bat_users]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.items_num, n_neg_items, replace=True, exclusion=exclusion, thread_num=4)
            # padding
            bat_neg = pad_sequences(bat_neg, value=self.items_num, max_len=self.config.max_len,
                                    padding='pre', truncating='pre')

            item_neg_list.extend(bat_neg)
        return item_neg_list

    def _create_variable(self):
        # self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        batch_size = None
        self.item_seq_ph = tf.placeholder(tf.int32, [batch_size, self.config.max_len], name="item_seq")
        self.item_pos_ph = tf.placeholder(tf.int32, [batch_size, self.config.max_len], name="item_pos")
        self.item_neg_ph = tf.placeholder(tf.int32, [batch_size, self.config.max_len], name="item_neg")
        self.is_training = tf.placeholder(tf.bool, name="training_flag")

        l2_regularizer = tf.contrib.layers.l2_regularizer(self.config.l2_emb)
        item_embeddings = tf.get_variable('item_embeddings', dtype=tf.float32,
                                          shape=[self.items_num, self.config.hidden_units],
                                          regularizer=l2_regularizer)

        zero_pad = tf.zeros([1, self.config.hidden_units], name="padding")
        item_embeddings = tf.concat([item_embeddings, zero_pad], axis=0)
        self.item_embeddings = item_embeddings * (self.config.hidden_units ** 0.5)

        self.position_embeddings = tf.get_variable('position_embeddings', dtype=tf.float32,
                                                   shape=[self.config.max_len, self.config.hidden_units],
                                                   regularizer=l2_regularizer)

    def _build_model(self):
        self._create_variable()
        reuse = None
        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq = tf.nn.embedding_lookup(self.item_embeddings, self.item_seq_ph)
            item_emb_table = self.item_embeddings

            # Positional Encoding
            position = tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq_ph)[1]), 0),
                               [tf.shape(self.item_seq_ph)[0], 1])
            t = tf.nn.embedding_lookup(self.position_embeddings, position)
            pos_emb_table = self.position_embeddings

            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=self.config.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq_ph, self.items_num)), -1)
            self.seq *= mask

            # Build blocks

            for i in range(self.config.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.config.hidden_units,
                                                   num_heads=self.config.num_heads,
                                                   dropout_rate=self.config.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq),
                                           num_units=[self.config.hidden_units, self.config.hidden_units],
                                           dropout_rate=self.config.dropout_rate,
                                           is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)  # (b, l, d)
            last_emb = self.seq[:, -1, :]  # (b, d), the embedding of last item of each session

        pos = tf.reshape(self.item_pos_ph, [tf.shape(self.item_seq_ph)[0] * self.config.max_len])  # (b*l,)
        neg = tf.reshape(self.item_neg_ph, [tf.shape(self.item_seq_ph)[0] * self.config.max_len])  # (b*l,)
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)  # (b*l, d)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)  # (b*l, d)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.item_seq_ph)[0] * self.config.max_len, self.config.hidden_units])  # (b*l, d)

        # prediction layer
        self.pos_logits = inner_product(pos_emb, seq_emb)  # (b*l,)
        self.neg_logits = inner_product(neg_emb, seq_emb)  # (b*l,)

        # ignore padding items (self.items_num)
        is_target = tf.reshape(tf.to_float(tf.not_equal(pos, self.items_num)),
                               [tf.shape(self.item_seq_ph)[0] * self.config.max_len])

        pos_loss = -tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * is_target
        neg_loss = -tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * is_target
        self.loss = tf.reduce_sum(pos_loss + neg_loss) / tf.reduce_sum(is_target)

        try:
            reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = self.loss + reg_losses
        except:
            pass

        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta2=0.98).minimize(self.loss)

        # for predication/test
        items_embeddings = item_emb_table[:-1]  # remove the padding item
        self.all_logits = tf.matmul(last_emb, items_embeddings, transpose_b=True)

    def fit(self):
        item_seq_list, item_pos_list = self._generate_train_data()
        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            item_neg_list = self._sample_negative()
            data = BatchIterator(item_seq_list, item_pos_list, item_neg_list,
                                 batch_size=self.config.batch_size, shuffle=True, drop_last=False)
            for bat_item_seq, bat_item_pos, bat_item_neg in data:
                feed = {self.item_seq_ph: bat_item_seq,
                        self.item_pos_ph: bat_item_pos,
                        self.item_neg_ph: bat_item_neg,
                        self.is_training: True}

                self.sess.run(self.train_opt, feed_dict=feed)

            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def evaluate(self, test_users=None):
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        bat_seq = [self.test_item_seqs[u] for u in users]
        feed = {self.item_seq_ph: bat_seq,
                self.is_training: False}
        bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
        return bat_ratings
