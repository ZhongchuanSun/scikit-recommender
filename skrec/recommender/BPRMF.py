"""
Paper: BPR: Bayesian Personalized Ranking from Implicit Feedback
Author: Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["BPRMF"]

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from .base import AbstractRecommender
from ..run_config import RunConfig
from ..utils.py import Config
from ..utils.py import EarlyStopping
from ..utils.torch import inner_product, bpr_loss, l2_loss, get_initializer
from ..io import PairwiseIterator


class BPRMFConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=1e-3,
                 n_dim=64,
                 batch_size=1024,
                 epochs=1000,
                 early_stop=200,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.n_dim: int = n_dim
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, float) and self.reg >= 0
        assert isinstance(self.n_dim, int) and self.n_dim > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class _MF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(_MF, self).__init__()

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)

        self.item_biases = nn.Embedding(num_items, 1)

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        init = get_initializer("normal")
        zero_init = get_initializer("zeros")

        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)
        zero_init(self.item_biases.weight)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        item_bias = self.item_biases(item_ids).squeeze()
        ratings = inner_product(user_embeds, item_embeds) + item_bias
        return ratings

    def predict(self, user_ids):
        user_embeds = self.user_embeddings(user_ids)
        ratings = torch.matmul(user_embeds, self.item_embeddings.weight.T)
        ratings += self.item_biases.weight.squeeze()
        return ratings


class BPRMF(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = BPRMFConfig(**model_config)
        super().__init__(run_config, self.config)
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.mf = _MF(self.num_users, self.num_items, self.config.n_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.mf.parameters(), lr=self.config.lr)

    def fit(self):
        data_iter = PairwiseIterator(self.dataset.train_data,
                                     batch_size=self.config.batch_size,
                                     shuffle=True, drop_last=False)

        self.logger.info("metrics:".ljust(12)+f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.mf.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                yui = self.mf(bat_users, bat_pos_items)
                yuj = self.mf(bat_users, bat_neg_items)

                loss = bpr_loss(yui, yuj).sum()
                reg_loss = l2_loss(self.mf.user_embeddings(bat_users),
                                   self.mf.item_embeddings(bat_pos_items),
                                   self.mf.item_embeddings(bat_neg_items),
                                   self.mf.item_biases(bat_pos_items),
                                   self.mf.item_biases(bat_neg_items)
                                   )
                loss += self.config.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12)+f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")
        # group_results = self.evaluate_group()
        # result = "\n".join([f"{label}:".ljust(12) + result.values_str for label, result in group_results])
        # self.logger.info("\n"+result)

    def evaluate(self, test_users=None):
        self.mf.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users) -> np.ndarray:
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.mf.predict(users).cpu().detach().numpy()
