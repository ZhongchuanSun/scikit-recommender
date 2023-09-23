"""
Paper: Improving Pairwise Learning for Item Recommendation from Implicit Feedback
Author: Steffen Rendle, and Christoph Freudenthaler
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["AOBPR"]

import numpy as np
from typing import Dict
from ..base import AbstractRecommender
from ...io import PairwiseIterator
from ...utils.py import randint_choice
from ...utils.py import Config
from ...utils.py import EarlyStopping
from ...run_config import RunConfig
from .pyx_aobpr_func import aobpr_update
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AOBPRConfig(Config):
    def __init__(self,
                 lr=1e-2,
                 reg=5e-2,
                 embed_size=64,
                 alpha=6682,
                 epochs=500,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.embed_size: int = embed_size
        self.alpha: int = alpha
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, float) and self.reg >= 0
        assert isinstance(self.embed_size, int) and self.embed_size > 0
        assert isinstance(self.alpha, int) and self.alpha > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class AOBPR(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = AOBPRConfig(**model_config)
        super().__init__(run_config, self.config)
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        low, high = 0.0, 1.0
        self.user_embeds = np.random.uniform(low=low, high=high, size=[self.num_users, self.config.embed_size]).astype(np.float32)
        self.item_embeds = np.random.uniform(low=low, high=high, size=[self.num_items, self.config.embed_size]).astype(np.float32)

        rank = np.arange(1, self.num_items+1)
        rank_prob = np.exp(-rank/self.config.alpha)
        self.rank_prob = rank_prob/np.sum(rank_prob)

    def fit(self):
        data_iter = PairwiseIterator(self.dataset.train_data,
                                     batch_size=len(self.dataset.train_data),
                                     shuffle=False, drop_last=False)

        user1d, pos_item1d, _ = list(data_iter)[0]
        len_data = len(user1d)
        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        shuffle_idx = np.arange(len_data)
        for epoch in range(self.config.epochs):
            rank_idx = randint_choice(self.num_items, size=len_data,
                                      replace=True, p=self.rank_prob)
            np.random.shuffle(shuffle_idx)
            aobpr_update(user1d[shuffle_idx], pos_item1d[shuffle_idx],
                         rank_idx, self.config.lr, self.config.reg,
                         self.user_embeds, self.item_embeds)

            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def evaluate(self, test_users=None):
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        user_embedding = self.user_embeds[users]
        all_ratings = np.matmul(user_embedding, self.item_embeds.T)
        return all_ratings
