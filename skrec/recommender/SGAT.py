"""
Paper: Sequential Graph Collaborative Filtering
Author: Zhongchuan Sun, Bin Wu, Youwei Wang, and Yangdong Ye
Reference: https://github.com/ZhongchuanSun/SGAT
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["SGAT"]

import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from typing import Dict
from collections import defaultdict
from .base import AbstractRecommender
from ..utils.py import Config
from ..utils.py import EarlyStopping
from ..utils.py import pad_sequences
from ..utils.tf1x import bpr_loss, l2_loss, l2_distance
from ..io import SequentialPairwiseIterator
from ..utils.common import normalize_adj_matrix
from ..run_config import RunConfig


class SGATConfig(Config):
    def __init__(self,
                 lr=0.001,
                 reg=1e-4,
                 n_layers=5,
                 n_seqs=5,
                 n_next=3,
                 embed_size=64,
                 batch_size=1024,
                 epochs=500,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.n_layers: int = n_layers
        self.n_seqs: int = n_seqs
        self.n_next: int = n_next
        self.embed_size: int = embed_size
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, float) and self.reg >= 0
        assert isinstance(self.n_layers, int) and self.n_layers >= 0
        assert isinstance(self.n_seqs, int) and self.n_seqs > 0
        assert isinstance(self.n_next, int) and self.n_next > 0
        assert isinstance(self.embed_size, int) and self.embed_size > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


def mexp(x, tau=1.0):
    # normalize att_logit to avoid negative value
    x_max = tf.reduce_max(x)
    x_min = tf.reduce_min(x)
    norm_x = (x-x_min) / (x_max-x_min)

    # calculate attention for each pair of items
    # used for calculating softmax
    exp_x = tf.exp(norm_x/tau)
    return exp_x


class SGAT(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = SGATConfig(**model_config)
        super().__init__(run_config, self.config)

        self.users_num, self.items_num = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.train_data.to_user_dict_by_time()
        self._process_test()

        self._build_model()
        tf_config = tf.ConfigProto()  # allow_soft_placement=False, log_device_placement=True
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())

    def _process_test(self):
        item_seqs = [self.user_pos_train[user][-self.config.n_seqs:] if user in self.user_pos_train else [self.items_num]
                     for user in range(self.users_num)]
        self.test_item_seqs = pad_sequences(item_seqs, value=self.items_num, max_len=self.config.n_seqs,
                                            padding='pre', truncating='pre', dtype=np.int32)

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.head_ph = tf.placeholder(tf.int32, [None, self.config.n_seqs], name="head_item")  # the previous item
        self.pos_tail_ph = tf.placeholder(tf.int32, [None, self.config.n_next], name="pos_tail_item")  # the next item
        self.neg_tail_ph = tf.placeholder(tf.int32, [None, self.config.n_next], name="neg_tail_item")  # the negative item

    def _construct_graph(self):
        th_rs_dict = defaultdict(list)
        for user, pos_items in self.user_pos_train.items():
            for h, t in zip(pos_items[:-1], pos_items[1:]):
                th_rs_dict[(t, h)].append(user)

        th_rs_list = sorted(th_rs_dict.items(), key=lambda x: x[0])

        user_list, head_list, tail_list = [], [], []
        for (t, h), r in th_rs_list:
            user_list.extend(r)
            head_list.extend([h] * len(r))
            tail_list.extend([t] * len(r))

        # attention mechanism

        # the auxiliary constant to calculate softmax
        row_idx, nnz = np.unique(tail_list, return_counts=True)
        count = {r: n for r, n in zip(row_idx, nnz)}
        nnz = [count[i] if i in count else 0 for i in range(self.items_num)]
        nnz = np.concatenate([[0], nnz])
        rows_idx = np.cumsum(nnz)

        # the auxiliary constant to calculate the weight between two node
        edge_num = np.array([len(r) for (t, h), r in th_rs_list], dtype=np.int32)
        edge_num = np.concatenate([[0], edge_num])
        edge_idx = np.cumsum(edge_num)

        sp_idx = [[t, h] for (t, h), r in th_rs_list]

        adj_mean_norm = self._get_mean_norm(edge_num[1:], sp_idx)

        return head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_mean_norm

    def _load_graph(self):
        dir_name = os.path.dirname(self.dataset.data_dir)
        dir_name = os.path.join(dir_name, "_sgat_data")
        elem_file = os.path.join(dir_name, "graph_elem.npy")
        index_file = os.path.join(dir_name, "index.npy")
        adj_file = os.path.join(dir_name, "adj_mean_norm.npz")

        if os.path.exists(elem_file) and \
                os.path.exists(index_file) and \
                os.path.exists(adj_file):
            head_list, tail_list, user_list = np.load(elem_file, allow_pickle=True)
            rows_idx, edge_idx, sp_idx = np.load(index_file, allow_pickle=True)
            adj_mean_norm = sp.load_npz(adj_file)
        else:
            head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_mean_norm = self._construct_graph()
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            np.save(elem_file, [head_list, tail_list, user_list])
            np.save(index_file, [rows_idx, edge_idx, sp_idx])
            sp.save_npz(adj_file, adj_mean_norm)

        return head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_mean_norm

    def _init_constant(self):
        head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_norm = self._load_graph()

        # attention mechanism
        self.att_head_idx = tf.constant(head_list, dtype=tf.int32, shape=None, name="att_head_idx")
        self.att_tail_idx = tf.constant(tail_list, dtype=tf.int32, shape=None, name="att_tail_idx")
        self.att_user_idx = tf.constant(user_list, dtype=tf.int32, shape=None, name="att_user_idx")

        # the auxiliary constant to calculate softmax
        self.rows_end_idx = tf.constant(rows_idx[1:], dtype=tf.int32, shape=None, name="rows_end_idx")
        self.row_begin_idx = tf.constant(rows_idx[:-1], dtype=tf.int32, shape=None, name="row_begin_idx")

        # the auxiliary constant to calculate the weight between two node
        self.edge_end_idx = tf.constant(edge_idx[1:], dtype=tf.int32, shape=None, name="edge_end_idx")
        self.edge_begin_idx = tf.constant(edge_idx[:-1], dtype=tf.int32, shape=None, name="edge_begin_idx")

        # the index of sparse matrix
        self.sp_tensor_idx = tf.constant(sp_idx, dtype=tf.int64)

    def _get_mean_norm(self, edge_num, sp_idx):
        adj_num = np.array(edge_num, dtype=np.float32)
        rows, cols = list(zip(*sp_idx))
        adj_mat = sp.csr_matrix((adj_num, (rows, cols)), shape=(self.items_num, self.items_num))

        return normalize_adj_matrix(adj_mat, "left").astype(np.float32)

    def _init_variable(self):
        # embedding parameters
        init = tf.random.truncated_normal([self.users_num, self.config.embed_size], mean=0.0, stddev=0.01)
        self.user_embeddings = tf.Variable(init, dtype=tf.float32, name="user_embeddings")

        init = tf.random.truncated_normal([self.items_num, self.config.embed_size], mean=0.0, stddev=0.01)
        self.item_embeddings = tf.Variable(init, dtype=tf.float32, name="item_embeddings")

        self.item_biases = tf.Variable(tf.zeros([self.items_num]), dtype=tf.float32, name="item_biases")

    def _get_attention(self, item_embeddings):
        h_embed = tf.nn.embedding_lookup(item_embeddings, self.att_head_idx)
        r_embed = tf.nn.embedding_lookup(self.user_embeddings, self.att_user_idx)
        t_embed = tf.nn.embedding_lookup(item_embeddings, self.att_tail_idx)

        att_logit = l2_distance(h_embed+r_embed, t_embed)
        exp_logit = mexp(-att_logit)

        exp_logit = tf.concat([[0], exp_logit], axis=0)
        sum_exp_logit = tf.cumsum(exp_logit)

        pre_sum = tf.gather(sum_exp_logit, self.edge_begin_idx)
        next_sum = tf.gather(sum_exp_logit, self.edge_end_idx)
        sum_exp_logit_per_edge = next_sum - pre_sum

        # convert to spares tensor
        exp_logit = tf.SparseTensor(indices=self.sp_tensor_idx, values=sum_exp_logit_per_edge,
                                    dense_shape=[self.items_num, self.items_num])

        # normalize attention score to a probability vector
        next_sum = tf.gather(sum_exp_logit, self.rows_end_idx)
        pre_sum = tf.gather(sum_exp_logit, self.row_begin_idx)
        sum_exp_logit_per_row = next_sum - pre_sum + 1e-6
        sum_exp_logit_per_row = tf.reshape(sum_exp_logit_per_row, shape=[self.items_num, 1])

        att_score = exp_logit / sum_exp_logit_per_row
        return att_score

    def _aggregate(self, item_embedding, neighbor_embedding):
        item_embeddings = item_embedding + neighbor_embedding
        return item_embeddings

    def _mean_history(self, item_embeddings):
        self.logger.info("mean_history")
        # fuse to get short-term embeddings
        pad_value = tf.zeros([1, self.config.embed_size], dtype=tf.float32)
        item_embeddings = tf.concat([item_embeddings, pad_value], axis=0)

        item_seq_embs = tf.nn.embedding_lookup(item_embeddings, self.head_ph)  # (b,l,d)
        mask = tf.cast(tf.not_equal(self.head_ph, self.items_num), dtype=tf.float32)  # (b,l)
        his_emb = tf.reduce_sum(item_seq_embs, axis=1) / tf.reduce_sum(mask, axis=1, keepdims=True)  # (b,d)/(b,1)
        return his_emb

    def _fusion_layer(self, embed_g, embed_s=None):
        head_emb = embed_g
        if embed_s is not None:
            head_emb += embed_s
        return head_emb

    def _forward_head_emb(self, user_emb, item_embeddings):
        # embed item sequence
        his_embs = self._mean_history(item_embeddings)

        # fusion representation
        head_emb_g = tf.nn.embedding_lookup(item_embeddings, self.head_ph[:, -1])  # b*d
        head_emb = self._fusion_layer(head_emb_g, his_embs)

        return head_emb

    def _build_model(self):
        self._create_placeholder()
        self._init_constant()
        self._init_variable()
        item_embeddings = self.item_embeddings
        # Graph Convolution
        for k in range(self.config.n_layers):
            att_scores = self._get_attention(item_embeddings)
            neighbor_embeddings = tf.sparse_tensor_dense_matmul(att_scores, item_embeddings)
            item_embeddings = self._aggregate(item_embeddings, neighbor_embeddings)

        # Translation-based Recommendation
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)  # b*d
        head_emb = self._forward_head_emb(user_emb, item_embeddings)

        pos_tail_emb = tf.nn.embedding_lookup(item_embeddings, self.pos_tail_ph)  # b*t*d
        neg_tail_emb = tf.nn.embedding_lookup(item_embeddings, self.neg_tail_ph)  # b*t*d

        pos_tail_bias = tf.gather(self.item_biases, self.pos_tail_ph)  # b*t
        neg_tail_bias = tf.gather(self.item_biases, self.neg_tail_ph)  # b*t

        pre_emb = head_emb + user_emb  # b*d
        pre_emb = tf.expand_dims(pre_emb, axis=1)  # b*1*d
        pos_rating = -l2_distance(pre_emb, pos_tail_emb) + pos_tail_bias  # b*t
        neg_rating = -l2_distance(pre_emb, neg_tail_emb) + neg_tail_bias  # b*t

        pairwise_loss = tf.reduce_sum(bpr_loss(pos_rating, neg_rating))

        # reg loss
        emb_reg = l2_loss(user_emb, head_emb, pos_tail_emb, neg_tail_emb, pos_tail_bias, neg_tail_bias)

        # objective loss and optimizer
        obj_loss = pairwise_loss + self.config.reg*emb_reg
        self.update_opt = tf.train.AdamOptimizer(self.config.lr).minimize(obj_loss)

        # for prediction
        self.item_embeddings_final = tf.Variable(tf.zeros([self.items_num, self.config.embed_size]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.assign_opt = tf.assign(self.item_embeddings_final, item_embeddings)

        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)  # b*d
        head_emb = self._forward_head_emb(user_emb, self.item_embeddings_final)
        pre_emb = head_emb + user_emb  # b*d

        pre_emb = tf.expand_dims(pre_emb, axis=1)  # b*1*d
        j_emb = tf.expand_dims(self.item_embeddings_final, axis=0)  # 1*n*d

        self.prediction = -l2_distance(pre_emb, j_emb) + self.item_biases  # b*n

    def fit(self):
        data_iter = SequentialPairwiseIterator(self.dataset.train_data,
                                               num_previous=self.config.n_seqs, num_next=self.config.n_next,
                                               pad=self.items_num, batch_size=self.config.batch_size,
                                               shuffle=True, drop_last=False)

        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            for bat_users, bat_head, bat_pos_tail, bat_neg_tail in data_iter:
                feed = {self.user_ph: bat_users,
                        self.head_ph: bat_head.reshape([-1, self.config.n_seqs]),
                        self.pos_tail_ph: bat_pos_tail.reshape([-1, self.config.n_next]),
                        self.neg_tail_ph: bat_neg_tail.reshape([-1, self.config.n_next])
                        }
                self.sess.run(self.update_opt, feed_dict=feed)

            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def evaluate(self, test_users=None):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        last_items = [self.test_item_seqs[u] for u in users]
        feed = {self.user_ph: users, self.head_ph: last_items}
        bat_ratings = self.sess.run(self.prediction, feed_dict=feed)
        return bat_ratings
