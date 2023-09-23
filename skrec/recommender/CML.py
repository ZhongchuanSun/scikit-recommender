"""
Paper: Collaborative Metric Learning
Author: Cheng-Kang Hsieh, Longqi Yang, Yin Cui, Tsung-Yi Lin, Serge Belongie and Deborah Estrin
Reference: https://github.com/changun/CollMetric
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["CML"]

import tensorflow as tf
from tensorflow import keras
from typing import Dict
from .base import AbstractRecommender
from ..utils.py import Config
from ..utils.py import EarlyStopping
from ..utils.py import BatchIterator
from ..utils.tf1x import euclidean_distance, hinge_loss
from ..io.data_iterator import _generate_positive_items, _sampling_negative_items
from ..run_config import RunConfig


class CMLConfig(Config):
    def __init__(self,
                 lr=0.05,
                 reg=10.0,
                 embed_size=64,
                 margin=0.5,
                 clip_norm=1.0,
                 dns=10,
                 batch_size=256,
                 epochs=500,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.embed_size: int = embed_size
        self.margin: float = margin
        self.clip_norm: float = clip_norm
        self.dns: int = dns  # dns > 1
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, (float, int)) and self.reg >= 0
        assert isinstance(self.embed_size, int) and self.embed_size > 0
        assert isinstance(self.margin, float) and self.margin >= 0
        assert isinstance(self.clip_norm, float) and self.clip_norm >= 0
        assert isinstance(self.dns, int) and self.dns > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class CML(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = CMLConfig(**model_config)
        super().__init__(run_config, self.config)

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.train_data.to_user_dict()

        self._build_model()
        tf_config = tf.ConfigProto()  # allow_soft_placement=False, log_device_placement=True
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())

    def _create_variable(self):
        # B: batch size
        # L: number of negative items
        # D: embedding size
        self.user_h = tf.placeholder(tf.int32, [None], name="user")  # (B,)
        self.pos_item_h = tf.placeholder(tf.int32, [None], name="pos_item")  # (B,)
        self.neg_item_h = tf.placeholder(tf.int32, [None, self.config.dns], name="neg_item")  # (B,L)
        user_emb_init = tf.keras.initializers.normal(stddev=1 / (self.config.embed_size ** 0.5))
        item_emb_init = tf.keras.initializers.normal(stddev=1 / (self.config.embed_size ** 0.5))

        self.user_elayer = keras.layers.Embedding(self.num_users, self.config.embed_size,
                                                  embeddings_initializer=user_emb_init)
        self.item_elayer = keras.layers.Embedding(self.num_items, self.config.embed_size,
                                                  embeddings_initializer=item_emb_init)

    def cov_loss(self, matrix):
        n_rows = tf.cast(tf.shape(matrix)[0], tf.float32)
        matrix = matrix - tf.reduce_mean(matrix, axis=0)
        cov = tf.matmul(matrix, matrix, transpose_a=True) / n_rows
        cov = tf.matrix_set_diag(cov, tf.zeros(self.config.embed_size, dtype=tf.float32))

        f2_norm = tf.sqrt(tf.reduce_sum(tf.square(cov)))
        return f2_norm

    def _build_model(self):
        self._create_variable()
        user_embedding = self.user_elayer(self.user_h)  # (B,D)
        pos_item_embedding = self.item_elayer(self.pos_item_h)  # (B,D)
        neg_item_embedding = self.item_elayer(self.neg_item_h)  # (B,L,D)

        # positive item to user distance
        # d_ui = tf.squared_difference(user_embedding, pos_item_embedding)
        d_ui = euclidean_distance(user_embedding, pos_item_embedding)  # (B,)

        # negative items to user distance
        user_embedding_t = tf.expand_dims(user_embedding, axis=1)  # (B,1,D)
        d_ujs = euclidean_distance(user_embedding_t, neg_item_embedding)  # (B,L)
        # d_ujs = tf.squeeze(tf.squared_difference(user_embedding, neg_item_embedding))
        d_uj = tf.reduce_min(d_ujs, axis=1)  # (B,)

        loss = hinge_loss(d_uj-d_ui, margin=self.config.margin)

        # calculate w_ui
        impostors = tf.greater(tf.expand_dims(d_ui, axis=-1)-d_ujs+self.config.margin, 0)  # (B,L)
        rank = tf.reduce_mean(tf.cast(impostors, tf.float32), axis=1) * self.num_items  # (B,)
        w_ui = tf.log(rank + 1)

        # embedding loss
        loss = tf.reduce_sum(w_ui*loss)

        # covariance regularization
        j_idx = tf.expand_dims(tf.argmin(d_ujs, axis=1, output_type=tf.int32), -1)  # (B,1)
        j_idx = tf.squeeze(tf.batch_gather(self.neg_item_h, j_idx))
        neg_item_embedding = self.item_elayer(j_idx)  # (B,D)
        item_embedding = tf.concat([pos_item_embedding, neg_item_embedding], axis=0)
        f2_norm = self.cov_loss(user_embedding) + self.cov_loss(item_embedding)

        # total loss and update operation
        total_loss = loss + self.config.reg * f2_norm
        self.update = tf.train.AdagradOptimizer(learning_rate=self.config.lr).minimize(total_loss)

        with tf.control_dependencies([self.update]):
            # training items id
            train_item_idx = tf.concat([self.pos_item_h, tf.squeeze(j_idx)], axis=-1)
            # updated embedding of users and items
            user_emb = self.user_elayer(self.user_h)
            item_emb = self.item_elayer(train_item_idx)

            # constrain embedding
            user_normed = tf.clip_by_norm(user_emb, self.config.clip_norm, axes=-1)
            item_normed = tf.clip_by_norm(item_emb, self.config.clip_norm, axes=-1)

            # assign
            self.update = [tf.scatter_update(self.user_elayer.weights[0], self.user_h, user_normed),
                           tf.scatter_update(self.item_elayer.weights[0], train_item_idx, item_normed)]

        # for test
        user_embedding = tf.reshape(self.user_elayer(self.user_h), [-1, 1, self.config.embed_size])
        item_embedding = tf.reshape(self.item_elayer.weights[0], [1, self.num_items, self.config.embed_size])
        self.pre_logits = -euclidean_distance(user_embedding, item_embedding)

    def fit(self):
        user_n_pos, all_users, pos_items = _generate_positive_items(self.user_pos_train)
        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            neg_items = _sampling_negative_items(user_n_pos, self.config.dns, self.num_items,
                                                 self.user_pos_train).squeeze()
            data_iter = BatchIterator(all_users, pos_items, neg_items,
                                      batch_size=self.config.batch_size,
                                      shuffle=True, drop_last=False)

            for user, pos_item, neg_item in data_iter:
                feed = {self.user_h: user, self.pos_item_h: pos_item, self.neg_item_h: neg_item}
                self.sess.run(self.update, feed_dict=feed)

            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def evaluate(self, test_users=None):
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        ratings = self.sess.run(self.pre_logits, feed_dict={self.user_h: users})
        return ratings
