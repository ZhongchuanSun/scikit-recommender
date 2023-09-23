"""
Paper: Translation-based Recommendation
Author: Ruining He, Wang-Cheng Kang, and Julian McAuley
Reference: https://drive.google.com/file/d/0B9Ck8jw-TZUEVmdROWZKTy1fcEE/view?usp=sharing
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["TransRec"]

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Dict
from .base import AbstractRecommender
from ..utils.py import Config
from ..utils.py import EarlyStopping
from ..utils.torch import bpr_loss, l2_loss, l2_distance
from ..utils.torch import get_initializer
from ..io import SequentialPairwiseIterator
from ..run_config import RunConfig


class TransRecConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=0.0,
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


class _TransRec(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(_TransRec, self).__init__()

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self.global_transition = Parameter(torch.Tensor(1, embed_dim))

        self.item_biases = nn.Embedding(num_items, 1)

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        init = get_initializer("normal")
        zero_init = get_initializer("zeros")

        zero_init(self.user_embeddings.weight)
        init(self.global_transition)
        init(self.item_embeddings.weight)
        zero_init(self.item_biases.weight)

    def forward(self, user_ids, last_items, pre_items):
        user_embs = self.user_embeddings(user_ids)
        last_item_embs = self.item_embeddings(last_items)
        pre_item_embs = self.item_embeddings(pre_items)
        pre_item_bias = self.item_biases(pre_items)

        transed_emb = user_embs + self.global_transition + last_item_embs
        hat_y = -l2_distance(transed_emb, pre_item_embs) + torch.squeeze(pre_item_bias)

        return hat_y

    def predict(self, user_ids, last_items):
        user_embs = self.user_embeddings(user_ids)
        last_item_embs = self.item_embeddings(last_items)
        transed_emb = user_embs + self.global_transition + last_item_embs
        ratings = -l2_distance(transed_emb.unsqueeze(dim=1), self.item_embeddings.weight)

        ratings += torch.squeeze(self.item_biases.weight)
        return ratings


class TransRec(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = TransRecConfig(**model_config)
        super().__init__(run_config, self.config)

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_dict = self.dataset.train_data.to_user_dict_by_time()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.transrec = _TransRec(self.num_users, self.num_items, self.config.embed_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.transrec.parameters(), lr=self.config.lr)

    def fit(self):
        data_iter = SequentialPairwiseIterator(self.dataset.train_data,
                                               num_previous=1, num_next=1,
                                               batch_size=self.config.batch_size,
                                               shuffle=True, drop_last=False)

        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.transrec.train()
            for bat_users, bat_last_items, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_last_items = torch.from_numpy(bat_last_items).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                yui = self.transrec(bat_users, bat_last_items, bat_pos_items)
                yuj = self.transrec(bat_users, bat_last_items, bat_neg_items)

                loss = bpr_loss(yui, yuj).sum()
                reg_loss = l2_loss(self.transrec.user_embeddings(bat_users),
                                   self.transrec.global_transition,
                                   self.transrec.item_embeddings(bat_last_items),
                                   self.transrec.item_embeddings(bat_pos_items),
                                   self.transrec.item_embeddings(bat_neg_items),
                                   self.transrec.item_biases(bat_pos_items),
                                   self.transrec.item_biases(bat_neg_items)
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
        self.transrec.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        last_items = [self.user_pos_dict[u][-1] for u in users]
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        last_items = torch.from_numpy(np.asarray(last_items)).long().to(self.device)
        return self.transrec.predict(users, last_items).cpu().detach().numpy()
