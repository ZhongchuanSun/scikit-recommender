"""
Paper: Factorizing Personalized Markov Chains for Next-Basket Recommendation
Author: Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["FPMC"]


import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from .base import AbstractRecommender
from ..utils.py import Config
from ..utils.py import EarlyStopping
from ..utils.torch import get_initializer
from ..utils.torch import bpr_loss, l2_loss, inner_product
from ..io import SequentialPairwiseIterator
from ..run_config import RunConfig


class FPMCConfig(Config):
    def __init__(self,
                 lr=0.001,
                 reg=0.001,
                 embed_size=64,
                 batch_size=1024,
                 epochs=500,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.embed_size: int = embed_size
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, float) and self.reg >= 0
        assert isinstance(self.embed_size, int) and self.embed_size > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class _FPMC(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(_FPMC, self).__init__()

        # user and item embeddings
        self.UI_embeddings = nn.Embedding(num_users, embed_dim)
        self.IU_embeddings = nn.Embedding(num_items, embed_dim)
        self.IL_embeddings = nn.Embedding(num_items, embed_dim)
        self.LI_embeddings = nn.Embedding(num_items, embed_dim)

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        init = get_initializer("normal")
        init(self.UI_embeddings.weight)
        init(self.IU_embeddings.weight)
        init(self.IL_embeddings.weight)
        init(self.LI_embeddings.weight)

    def forward(self, user_ids, last_items, pre_items):
        ui_emb = self.UI_embeddings(user_ids)  # b*d
        pre_iu_emb = self.IU_embeddings(pre_items)  # b*d
        pre_il_emb = self.IL_embeddings(pre_items)  # b*d
        last_emb = self.LI_embeddings(last_items)  # b*d

        hat_y = inner_product(ui_emb, pre_iu_emb) + inner_product(last_emb, pre_il_emb)

        return hat_y

    def predict(self, user_ids, last_items):
        ui_emb = self.UI_embeddings(user_ids)  # b*d
        last_emb = self.LI_embeddings(last_items)  # b*d
        ratings = torch.matmul(ui_emb, self.IU_embeddings.weight.T) + \
                  torch.matmul(last_emb, self.IL_embeddings.weight.T)

        return ratings


class FPMC(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = FPMCConfig(**model_config)
        super().__init__(run_config, self.config)

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_dict = self.dataset.train_data.to_user_dict_by_time()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.fpmc = _FPMC(self.num_users, self.num_items, self.config.embed_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.fpmc.parameters(), lr=self.config.lr)

    def fit(self):
        data_iter = SequentialPairwiseIterator(self.dataset.train_data,
                                               num_previous=1, num_next=1,
                                               batch_size=self.config.batch_size,
                                               shuffle=True, drop_last=False)

        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.fpmc.train()
            for bat_users, bat_last_items, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_last_items = torch.from_numpy(bat_last_items).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                yui = self.fpmc(bat_users, bat_last_items, bat_pos_items)
                yuj = self.fpmc(bat_users, bat_last_items, bat_neg_items)

                loss = bpr_loss(yui, yuj).sum()
                reg_loss = l2_loss(self.fpmc.UI_embeddings(bat_users),
                                   self.fpmc.LI_embeddings(bat_last_items),
                                   self.fpmc.IU_embeddings(bat_pos_items),
                                   self.fpmc.IU_embeddings(bat_neg_items),
                                   self.fpmc.IL_embeddings(bat_pos_items),
                                   self.fpmc.IL_embeddings(bat_neg_items)
                                   )
                loss += self.config.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def evaluate(self, test_users=None):
        self.fpmc.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        last_items = [self.user_pos_dict[u][-1] for u in users]
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        last_items = torch.from_numpy(np.asarray(last_items)).long().to(self.device)
        return self.fpmc.predict(users, last_items).cpu().detach().numpy()
