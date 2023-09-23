"""
Paper: Recurrent Neural Networks with Top-k Gains for Session-based Recommendations
Author: Balázs Hidasi, and Alexandros Karatzoglou
Reference: https://github.com/hidasib/GRU4Rec
           https://github.com/Songweiping/GRU4Rec_TensorFlow
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["GRU4RecPlus"]


import numpy as np
import tensorflow as tf
from typing import List, Dict
from .base import AbstractRecommender
from ..utils.py import EarlyStopping
from ..utils.py import Config
from ..utils.tf1x import l2_loss
from ..run_config import RunConfig


class GRU4RecPlusConfig(Config):
    def __init__(self,
                 lr=0.001,
                 reg=0.0,
                 bpr_reg=1.0,
                 layers=[64],
                 batch_size=128,
                 loss="bpr_max",
                 hidden_act="tanh",
                 final_act="linear",
                 n_sample=2048,
                 sample_alpha=0.75,
                 epochs=500,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.bpr_reg: float = bpr_reg
        self.layers: List[int] = layers
        self.batch_size: int = batch_size
        self.loss: str = loss  # loss = top1_max, bpr_max
        self.hidden_act: str = hidden_act  # hidden_act = relu, tanh
        self.final_act: str = final_act  # final_act = linear, relu, leaky_relu
        self.n_sample: int = n_sample
        self.sample_alpha: float = sample_alpha  # 0 < sample_alpha <= 1
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, float) and self.reg >= 0
        assert isinstance(self.bpr_reg, float) and self.bpr_reg >= 0
        assert isinstance(self.layers, list)
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.loss, str) and self.loss in {"top1_max", "bpr_max"}
        assert isinstance(self.hidden_act, str) and self.hidden_act in {"relu", "tanh"}
        assert isinstance(self.final_act, str) and self.final_act in {"linear", "relu", "leaky_relu"}
        assert isinstance(self.n_sample, int) and self.n_sample >= 0
        assert isinstance(self.sample_alpha, float) and 0 < self.sample_alpha <= 1
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class GRU4RecPlus(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = GRU4RecPlusConfig(**model_config)
        super().__init__(run_config, self.config)
        config: GRU4RecPlusConfig = self.config

        if config.hidden_act == "relu":
            self.hidden_act = tf.nn.relu
        elif config.hidden_act == "tanh":
            self.hidden_act = tf.nn.tanh
        else:
            raise ValueError("There is not hidden_act named '%s'." % config.hidden_act)

        # final_act = leaky-relu
        if config.final_act == "relu":
            self.final_act = tf.nn.relu
        elif config.final_act == "linear":
            self.final_act = tf.identity
        elif config.final_act == "leaky_relu":
            self.final_act = tf.nn.leaky_relu
        else:
            raise ValueError("There is not final_act named '%s'." % config.final_act)

        if config.loss == "bpr_max":
            self.loss_fun = self._bpr_max_loss
        elif config.loss == "top1_max":
            self.loss_fun = self._top1_max_loss
        else:
            raise ValueError("There is not loss named '%s'." % config.loss)

        self.users_num, self.items_num = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.train_data.to_user_dict_by_time()
        self.data_ui, self.offset_idx = self._init_data()

        # for sampling negative items
        _, pop = np.unique(self.data_ui[:, 1], return_counts=True)
        pop = np.power(pop, self.config.sample_alpha)
        pop_cumsum = np.cumsum(pop)
        self.pop_cumsum = pop_cumsum / pop_cumsum[-1]

        self._build_model()
        tf_config = tf.ConfigProto()  # allow_soft_placement=False, log_device_placement=True
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_data(self):
        data_ui = self.dataset.train_data.to_user_item_pairs_by_time()

        _, idx = np.unique(data_ui[:, 0], return_index=True)
        offset_idx = np.zeros(len(idx)+1, dtype=np.int32)
        offset_idx[:-1] = idx
        offset_idx[-1] = len(data_ui)

        return data_ui, offset_idx

    def _create_variable(self):
        self.X_ph = tf.placeholder(tf.int32, [self.config.batch_size], name='input')
        self.Y_ph = tf.placeholder(tf.int32, [self.config.batch_size+self.config.n_sample], name='output')
        self.state_ph = [tf.placeholder(tf.float32, [self.config.batch_size, n_unit], name='layer_%d_state' % idx)
                         for idx, n_unit in enumerate(self.config.layers)]

        init = tf.random.truncated_normal([self.items_num, self.config.layers[0]], mean=0.0, stddev=0.01)
        self.input_embeddings = tf.Variable(init, dtype=tf.float32, name="input_embeddings")

        init = tf.random.truncated_normal([self.items_num, self.config.layers[-1]], mean=0.0, stddev=0.01)
        self.item_embeddings = tf.Variable(init, dtype=tf.float32, name="item_embeddings")
        self.item_biases = tf.Variable(tf.zeros([self.items_num]), dtype=tf.float32, name="item_biases")

    def _softmax_neg(self, logits):
        # logits: (b, size_y)
        hm = 1.0 - tf.eye(tf.shape(logits)[0], tf.shape(logits)[1])
        logits = logits * hm
        logits = logits - tf.reduce_max(logits, axis=1, keep_dims=True)
        e_x = tf.exp(logits) * hm  # (b, size_y)
        e_x = e_x / tf.reduce_sum(e_x, axis=1, keep_dims=True)
        return e_x  # (b, size_y)

    def _bpr_max_loss(self, logits):
        # logits: (b, size_y)
        softmax_scores = self._softmax_neg(logits)  # (b, size_y)
        pos_logits = tf.matrix_diag_part(logits)  # (b,)
        pos_logits = tf.reshape(pos_logits, shape=[-1, 1])  # (b, 1)
        prob = tf.sigmoid((pos_logits - logits))  # (b, size_y)
        prob = tf.reduce_sum(tf.multiply(prob, softmax_scores), axis=1)  # (b,)
        loss = -tf.log(prob + 1e-24)
        reg_loss = tf.reduce_sum(tf.multiply(tf.pow(logits, 2), softmax_scores), axis=1)  # (b,)

        return tf.reduce_mean(loss + self.config.bpr_reg*reg_loss)

    def _top1_max_loss(self, logits):
        softmax_scores = self._softmax_neg(logits)  # (b, size_y)

        pos_logits = tf.matrix_diag_part(logits)  # (b,)
        pos_logits = tf.reshape(pos_logits, shape=[-1, 1])  # (b, 1)
        prob = tf.sigmoid(-pos_logits + logits) + tf.sigmoid(tf.pow(logits, 2))
        loss = tf.reduce_sum(tf.multiply(prob, softmax_scores), axis=1)

        return tf.reduce_mean(loss)

    def _build_model(self):
        self._create_variable()
        # get embedding and bias
        # b: batch size
        # l1: the dim of the first layer
        # ln: the dim of the last layer
        # size_y: the length of Y_ph, i.e., n_sample+batch_size

        cells = [tf.nn.rnn_cell.GRUCell(size, activation=self.hidden_act) for size in self.config.layers]
        drop_cell = [tf.nn.rnn_cell.DropoutWrapper(cell) for cell in cells]
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(drop_cell)
        inputs = tf.nn.embedding_lookup(self.input_embeddings, self.X_ph)  # (b, l1)
        outputs, state = stacked_cell(inputs, state=self.state_ph)
        self.u_emb = outputs  # outputs: (b, ln)
        self.final_state = state  # [(b, l1), (b, l2), ..., (b, ln)]

        # for training
        items_embed = tf.nn.embedding_lookup(self.item_embeddings, self.Y_ph)  # (size_y, ln)
        items_bias = tf.gather(self.item_biases, self.Y_ph)  # (size_y,)

        logits = tf.matmul(outputs, items_embed, transpose_b=True) + items_bias  # (b, size_y)
        logits = self.final_act(logits)

        loss = self.loss_fun(logits)

        # reg loss
        reg_loss = l2_loss(inputs, items_embed, items_bias)
        final_loss = loss + self.config.reg*reg_loss
        self.update_opt = tf.train.AdamOptimizer(self.config.lr).minimize(final_loss)

    def _sample_neg_items(self, size):
        samples = np.searchsorted(self.pop_cumsum, np.random.rand(size))
        return samples

    def fit(self):
        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")

        data_ui, offset_idx = self.data_ui, self.offset_idx
        data_items = data_ui[:, 1]
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            state = [np.zeros([self.config.batch_size, n_unit], dtype=np.float32) for n_unit in self.config.layers]
            user_idx = np.random.permutation(len(offset_idx) - 1)
            iters = np.arange(self.config.batch_size, dtype=np.int32)
            maxiter = iters.max()
            start = offset_idx[user_idx[iters]]
            end = offset_idx[user_idx[iters]+1]
            finished = False
            while not finished:
                min_len = (end - start).min()
                out_idx = data_items[start]
                for i in range(min_len-1):
                    in_idx = out_idx
                    out_idx = data_items[start+i+1]
                    out_items = out_idx
                    if self.config.n_sample:
                        neg_items = self._sample_neg_items(self.config.n_sample)
                        out_items = np.hstack([out_items, neg_items])

                    feed = {self.X_ph: in_idx, self.Y_ph: out_items}
                    for l in range(len(self.config.layers)):
                        feed[self.state_ph[l]] = state[l]

                    _, state = self.sess.run([self.update_opt, self.final_state], feed_dict=feed)

                start = start+min_len-1
                mask = np.arange(len(iters))[(end - start) <= 1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_idx)-1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_idx[user_idx[maxiter]]
                    end[idx] = offset_idx[user_idx[maxiter]+1]
                if len(mask):
                    for i in range(len(self.config.layers)):
                        state[i][mask] = 0

            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def _get_user_embeddings(self):
        users = np.array(list(self.user_pos_train.keys()), dtype=np.int32)
        u_nnz = np.array([len(self.user_pos_train[u]) for u in users], dtype=np.int32)
        users = users[np.argsort(-u_nnz)]
        user_embeddings = np.zeros([self.users_num, self.config.layers[-1]], dtype=np.float32)  # saving user embedding

        data_ui, offset_idx = self.data_ui, self.offset_idx
        data_items = data_ui[:, 1]

        state = [np.zeros([self.config.batch_size, n_unit], dtype=np.float32) for n_unit in self.config.layers]
        batch_iter = np.arange(self.config.batch_size, dtype=np.int32)
        next_iter = batch_iter.max() + 1

        start = offset_idx[users[batch_iter]]
        end = offset_idx[users[batch_iter] + 1]  # the start index of next user

        batch_mask = np.ones([self.config.batch_size], dtype=np.int32)
        while np.sum(batch_mask) > 0:
            min_len = (end - start).min()

            for i in range(min_len):
                cur_items = data_items[start + i]
                feed = {self.X_ph: cur_items}
                for l in range(len(self.config.layers)):
                    feed[self.state_ph[l]] = state[l]

                u_emb, state = self.sess.run([self.u_emb, self.final_state], feed_dict=feed)

            start = start + min_len
            mask = np.arange(self.config.batch_size)[(end - start) == 0]
            for idx in mask:
                u = users[batch_iter[idx]]
                user_embeddings[u] = u_emb[idx]  # saving user embedding
                if next_iter < self.users_num:
                    batch_iter[idx] = next_iter
                    start[idx] = offset_idx[users[next_iter]]
                    end[idx] = offset_idx[users[next_iter] + 1]
                    next_iter += 1
                else:
                    batch_mask[idx] = 0
                    start[idx] = 0
                    end[idx] = offset_idx[-1]

            for i, _ in enumerate(self.config.layers):
                state[i][mask] = 0

        return user_embeddings

    def evaluate(self, test_users=None):
        self.cur_user_embeddings = self._get_user_embeddings()
        self.cur_item_embeddings, self.cur_item_biases = self.sess.run([self.item_embeddings, self.item_biases])
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        user_embeddings = self.cur_user_embeddings[users]
        all_ratings = np.matmul(user_embeddings, self.cur_item_embeddings.T) + self.cur_item_biases

        # final_act = leaky-relu
        if self.final_act == tf.nn.relu:
            all_ratings = np.maximum(all_ratings, 0)
        elif self.final_act == tf.identity:
            all_ratings = all_ratings
        elif self.final_act == tf.nn.leaky_relu:
            all_ratings = np.maximum(all_ratings, all_ratings*0.2)
        else:
            pass

        all_ratings = np.array(all_ratings, dtype=np.float32)
        return all_ratings
