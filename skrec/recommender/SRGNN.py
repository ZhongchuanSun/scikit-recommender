"""
Paper: Session-Based Recommendation with Graph Neural Networks
Author: Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan
Reference: https://github.com/CRIPAC-DIG/SR-GNN
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["SRGNN"]

import math
import numpy as np
import tensorflow as tf
from typing import Dict
from .base import AbstractRecommender
from ..utils.py import Config
from ..utils.py import EarlyStopping
from ..utils.py import pad_sequences
from ..utils.py import BatchIterator
from ..run_config import RunConfig


class SRGNNConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 l2_reg=1e-5,
                 hidden_size=64,
                 lr_dc=0.1,
                 lr_dc_step=3,
                 step=1,
                 nonhybrid=False,
                 max_seq_len=200,
                 batch_size=256,
                 epochs=500,
                 early_stop=50,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.l2_reg: float = l2_reg
        self.hidden_size: int = hidden_size
        self.lr_dc: float = lr_dc
        self.lr_dc_step: int = lr_dc_step
        self.step: int = step
        self.nonhybrid: bool = nonhybrid
        # max_seq_len is used to save gpu memory by limiting the max length of item sequence
        self.max_seq_len: int = max_seq_len
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.l2_reg, float) and self.l2_reg >= 0
        assert isinstance(self.hidden_size, int) and self.hidden_size > 0
        assert isinstance(self.lr_dc, float) and self.lr_dc >= 0
        assert isinstance(self.lr_dc_step, int) and self.lr_dc_step >= 0
        assert isinstance(self.step, int) and self.step > 0
        assert isinstance(self.nonhybrid, bool)
        assert isinstance(self.max_seq_len, int) and self.max_seq_len > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class SRGNN(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = SRGNNConfig(**model_config)
        super().__init__(run_config, self.config)

        self.num_users, self.num_item = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.train_data.to_user_dict_by_time()

        self.train_seq = []
        self.train_tar = []
        for user, seqs in self.user_pos_train.items():
            for i in range(1, len(seqs)):
                self.train_seq.append(seqs[-i-self.config.max_seq_len:-i])
                self.train_tar.append(seqs[-i])

        self._build_model()
        tf_config = tf.ConfigProto()  # allow_soft_placement=False, log_device_placement=True
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())

    def _create_variable(self):
        self.mask_ph = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, None])
        self.alias_ph = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size, None])
        self.item_ph = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size, None])
        self.target_ph = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size])

        self.adj_in_ph = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, None, None])
        self.adj_out_ph = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, None, None])

        stdv = 1.0 / math.sqrt(self.config.hidden_size)
        w_init = tf.random_uniform_initializer(-stdv, stdv)
        self.nasr_w1 = tf.get_variable('nasr_w1', [self.config.hidden_size, self.config.hidden_size],
                                       dtype=tf.float32, initializer=w_init)
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.config.hidden_size, self.config.hidden_size],
                                       dtype=tf.float32, initializer=w_init)
        self.nasr_v = tf.get_variable('nasrv', [1, self.config.hidden_size], dtype=tf.float32, initializer=w_init)
        self.nasr_b = tf.get_variable('nasr_b', [self.config.hidden_size], dtype=tf.float32,
                                      initializer=tf.zeros_initializer())

        embedding = tf.get_variable(shape=[self.num_item, self.config.hidden_size], name='embedding',
                                    dtype=tf.float32, initializer=w_init)
        zero_pad = tf.zeros([1, self.config.hidden_size], name="padding")
        self.embedding = tf.concat([embedding, zero_pad], axis=0)

        self.W_in = tf.get_variable('W_in', shape=[self.config.hidden_size, self.config.hidden_size],
                                    dtype=tf.float32, initializer=w_init)
        self.b_in = tf.get_variable('b_in', [self.config.hidden_size], dtype=tf.float32, initializer=w_init)
        self.W_out = tf.get_variable('W_out', [self.config.hidden_size, self.config.hidden_size],
                                     dtype=tf.float32, initializer=w_init)
        self.b_out = tf.get_variable('b_out', [self.config.hidden_size], dtype=tf.float32, initializer=w_init)

        self.B = tf.get_variable('B', [2 * self.config.hidden_size, self.config.hidden_size], initializer=w_init)

    def ggnn(self):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item_ph)  # (b,l,d)
        cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        with tf.variable_scope('gru'):
            for i in range(self.config.step):
                fin_state = tf.reshape(fin_state, [self.config.batch_size, -1, self.config.hidden_size])  # (b,l,d)
                fin_state_tmp = tf.reshape(fin_state, [-1, self.config.hidden_size])  # (b*l,d)

                fin_state_in = tf.reshape(tf.matmul(fin_state_tmp, self.W_in) + self.b_in,
                                          [self.config.batch_size, -1, self.config.hidden_size])  # (b,l,d)

                # fin_state_tmp = tf.reshape(fin_state, [-1, self.hidden_size])  # (b*l,d)
                fin_state_out = tf.reshape(tf.matmul(fin_state_tmp, self.W_out) + self.b_out,
                                           [self.config.batch_size, -1, self.config.hidden_size])  # (b,l,d)

                av_in = tf.matmul(self.adj_in_ph, fin_state_in)  # (b,l,d)
                av_out = tf.matmul(self.adj_out_ph, fin_state_out)  # (b,l,d)
                av = tf.concat([av_in, av_out], axis=-1)  # (b,l,2d)

                av = tf.expand_dims(tf.reshape(av, [-1, 2 * self.config.hidden_size]), axis=1)  # (b*l,1,2d)
                # fin_state_tmp = tf.reshape(fin_state, [-1, self.hidden_size])  # (b*l,d)

                state_output, fin_state = tf.nn.dynamic_rnn(cell, av, initial_state=fin_state_tmp)
        return tf.reshape(fin_state, [self.config.batch_size, -1, self.config.hidden_size])  # (b,l,d)

    def _session_embedding(self, re_embedding):
        # re_embedding  (b,l,d)
        rm = tf.reduce_sum(self.mask_ph, 1)  # (b,), length of each session
        last_idx = tf.stack([tf.range(self.config.batch_size), tf.to_int32(rm) - 1], axis=1)  # (b, 2) index of last item
        last_id = tf.gather_nd(self.alias_ph, last_idx)  # (b,) alias id of last item
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.config.batch_size), last_id], axis=1))  # (b,d) embedding of last item

        seq_h = [tf.nn.embedding_lookup(re_embedding[i], self.alias_ph[i]) for i in range(self.config.batch_size)]
        seq_h = tf.stack(seq_h, axis=0)  # batch_size*T*d
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.config.hidden_size]), self.nasr_w2)
        last = tf.reshape(last, [self.config.batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.config.batch_size, -1, self.config.hidden_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.config.hidden_size]), self.nasr_v,
                         transpose_b=True) * tf.reshape(self.mask_ph, [-1, 1])
        if not self.config.nonhybrid:
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.config.batch_size, -1, 1]) * seq_h, 1),
                            tf.reshape(last, [-1, self.config.hidden_size])], -1)
            sess_embedding = tf.matmul(ma, self.B)
        else:
            sess_embedding = tf.reduce_sum(tf.reshape(coef, [self.config.batch_size, -1, 1]) * seq_h, 1)

        return sess_embedding

    def _build_model(self):
        self._create_variable()
        with tf.variable_scope('ggnn_model', reuse=None):
            node_embedding = self.ggnn()
            sess_embedding = self._session_embedding(node_embedding)

        item_embedding = self.embedding[:-1]
        self.all_logits = tf.matmul(sess_embedding, item_embedding, transpose_b=True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_ph, logits=self.all_logits)
        loss = tf.reduce_mean(loss)

        var_list = tf.trainable_variables()
        l2_loss = [tf.nn.l2_loss(v) for v in var_list if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]
        loss_train = loss + self.config.l2_reg * tf.add_n(l2_loss)

        global_step = tf.Variable(0)
        decay = self.config.lr_dc_step * len(self.train_seq) / self.config.batch_size
        learning_rate = tf.train.exponential_decay(self.config.lr, global_step=global_step, decay_steps=decay,
                                                   decay_rate=self.config.lr_dc, staircase=True)
        self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss_train, global_step=global_step)

    def fit(self):
        train_seq_len = [(idx, len(seq)) for idx, seq in enumerate(self.train_seq)]
        train_seq_len = sorted(train_seq_len, key=lambda x: x[1], reverse=True)
        train_seq_index, _ = list(zip(*train_seq_len))

        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            for bat_index in self._shuffle_index(train_seq_index):
                item_seqs = [self.train_seq[idx] for idx in bat_index]
                bat_tars = [self.train_tar[idx] for idx in bat_index]
                bat_adj_in, bat_adj_out, bat_alias, bat_items, bat_mask = self._build_session_graph(item_seqs)
                feed = {self.target_ph: bat_tars,
                        self.item_ph: bat_items,
                        self.adj_in_ph: bat_adj_in,
                        self.adj_out_ph: bat_adj_out,
                        self.alias_ph: bat_alias,
                        self.mask_ph: bat_mask}

                self.sess.run(self.train_opt, feed_dict=feed)

            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def _shuffle_index(self, seq_index):
        """NOTE: two-step shuffle for saving memory"""
        index_chunks = BatchIterator(seq_index, batch_size=self.config.batch_size*32,
                                     shuffle=False, drop_last=False)  # chunking
        index_chunks = list(index_chunks)
        index_chunks_iter = BatchIterator(index_chunks, batch_size=1,
                                          shuffle=True, drop_last=False)  # shuffle index chunk
        for indexes in index_chunks_iter:  # outer-layer shuffle
            indexes_iter = BatchIterator(indexes[0], batch_size=self.config.batch_size,
                                         shuffle=True, drop_last=True)  # inner-layer shuffle
            for bat_index in indexes_iter:
                yield bat_index

    def _build_session_graph(self, bat_items):
        A_in, A_out, alias_inputs = [], [], []
        all_mask = [[1] * len(items) for items in bat_items]
        bat_items = pad_sequences(bat_items, value=self.num_item)

        unique_nodes = [np.unique(items).tolist() for items in bat_items]
        max_n_node = np.max([len(nodes) for nodes in unique_nodes])
        for u_seq, u_node, mask in zip(bat_items, unique_nodes, all_mask):
            adj_mat = np.zeros((max_n_node, max_n_node))
            id_map = {node: idx for idx, node in enumerate(u_node)}
            if len(u_seq) > 1:
                alias_previous = [id_map[i] for i in u_seq[:len(mask) - 1]]
                alias_next = [id_map[i] for i in u_seq[1:len(mask)]]
                adj_mat[alias_previous, alias_next] = 1

            u_sum_in = np.sum(adj_mat, axis=0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(adj_mat, u_sum_in)

            u_sum_out = np.sum(adj_mat, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(adj_mat.transpose(), u_sum_out)

            A_in.append(u_A_in)
            A_out.append(u_A_out)
            alias_inputs.append([id_map[i] for i in u_seq])

        items = pad_sequences(unique_nodes, value=self.num_item)
        all_mask = pad_sequences(all_mask, value=0)
        return A_in, A_out, alias_inputs, items, all_mask

    def evaluate(self, test_users=None):
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        bat_user = users
        cur_batch_size = len(bat_user)
        bat_items = [self.user_pos_train[user][-self.config.max_seq_len:] for user in bat_user]
        bat_adj_in, bat_adj_out, bat_alias, bat_items, bat_mask = self._build_session_graph(bat_items)
        if cur_batch_size < self.config.batch_size:  # padding
            pad_size = self.config.batch_size - cur_batch_size
            bat_adj_in = np.concatenate([bat_adj_in, [bat_adj_in[-1]] * pad_size], axis=0)
            bat_adj_out = np.concatenate([bat_adj_out, [bat_adj_out[-1]] * pad_size], axis=0)
            bat_alias = np.concatenate([bat_alias, [bat_alias[-1]] * pad_size], axis=0)
            bat_items = np.concatenate([bat_items, [bat_items[-1]] * pad_size], axis=0)
            bat_mask = np.concatenate([bat_mask, [bat_mask[-1]] * pad_size], axis=0)

        feed = {self.item_ph: bat_items,
                self.adj_in_ph: bat_adj_in,
                self.adj_out_ph: bat_adj_out,
                self.alias_ph: bat_alias,
                self.mask_ph: bat_mask}
        bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
        ratings = bat_ratings[:cur_batch_size]
        return ratings
