"""
Paper: Collaborative Denoising Auto-Encoder for Top-N Recommender Systems
Author: Yao Wu, Christopher DuBois, Alice X. Zheng, and Martin Ester
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["CDAE"]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as torch_sp
from typing import Dict
from .base import AbstractRecommender
from ..utils.py import Config
from ..utils.py import EarlyStopping
from ..utils.py import BatchIterator
from ..utils.torch import l2_loss, get_initializer, inner_product
from ..utils.torch import square_loss, sigmoid_cross_entropy
from ..utils.torch import sp_mat_to_sp_tensor, dropout_sparse
from ..utils.py import randint_choice
from ..run_config import RunConfig


class CDAEConfig(Config):
    def __init__(self,
                 lr=0.001,
                 reg=0.001,
                 hidden_dim=64,
                 dropout=0.5,
                 num_neg=5,
                 hidden_act="sigmoid",
                 loss_func="sigmoid_cross_entropy",
                 batch_size=256,
                 epochs=1000,
                 early_stop=200,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.hidden_dim: int = hidden_dim
        self.dropout: float = dropout
        self.num_neg: int = num_neg
        self.hidden_act: str = hidden_act  # hidden_act = identity, sigmoid
        self.loss_func: str = loss_func  # loss_func = sigmoid_cross_entropy, square
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, float) and self.reg >= 0
        assert isinstance(self.hidden_dim, int) and self.hidden_dim > 0
        assert isinstance(self.dropout, float) and self.dropout < 1.0
        assert isinstance(self.num_neg, int) and self.num_neg >= 0
        assert isinstance(self.hidden_act, str) and self.hidden_act in {"identity", "sigmoid"}
        assert isinstance(self.loss_func, str) and self.loss_func in {"sigmoid_cross_entropy", "square"}
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class _CDAE(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, dropout, hidden_act):
        super(_CDAE, self).__init__()

        # user and item embeddings
        self.en_embeddings = nn.Embedding(num_items, embed_dim)
        self.en_offset = nn.Parameter(torch.Tensor(embed_dim))
        self.de_embeddings = nn.Embedding(num_items, embed_dim)
        self.de_bias = nn.Embedding(num_items, 1)
        self.user_embeddings = nn.Embedding(num_users, embed_dim)

        self.dropout = dropout
        self.hidden_act = hidden_act

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        init = get_initializer("normal")
        zero_init = get_initializer("zeros")

        init(self.en_embeddings.weight)
        zero_init(self.en_offset)

        init(self.de_embeddings.weight)
        zero_init(self.de_bias.weight)

        init(self.user_embeddings.weight)

    def forward(self, user_ids, bat_idx, sp_item_mat, bat_items):
        hidden = self._encoding(user_ids, sp_item_mat)  # (b,d)

        # decoding
        de_item_embs = self.de_embeddings(bat_items)  # (l,d)
        de_bias = self.de_bias(bat_items).squeeze()
        hidden = F.embedding(bat_idx, hidden)  # (l,d)

        ratings = inner_product(hidden, de_item_embs) + de_bias

        # reg loss
        bat_items = torch.unique(bat_items, sorted=False)
        reg_loss = l2_loss(self.en_embeddings(bat_items), self.en_offset,
                           self.user_embeddings(user_ids),
                           self.de_embeddings(bat_items), self.de_bias(bat_items))

        return ratings, reg_loss

    def _encoding(self, user_ids, sp_item_mat):

        corruption = dropout_sparse(sp_item_mat, 1-self.dropout, self.training)

        en_item_embs = self.en_embeddings.weight  # (n,d)
        hidden = torch_sp.mm(corruption, en_item_embs)  # (b,n)x(n,d)->(b,d)

        user_embs = self.user_embeddings(user_ids)  # (b,d)
        hidden += user_embs  # add user vector
        hidden += self.en_offset.view([1, -1])  # add bias
        hidden = self.hidden_act(hidden)  # hidden activate, z_u
        return hidden  # (b,d)

    def predict(self, user_ids, sp_item_mat):
        user_emb = self._encoding(user_ids, sp_item_mat)  # (b,d)
        ratings = user_emb.matmul(self.de_embeddings.weight.T)  # (b,d)x(d,n)->(b,n)
        ratings += self.de_bias.weight.view([1, -1])
        return ratings


class CDAE(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = CDAEConfig(**model_config)
        super().__init__(run_config, self.config)

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.train_csr_mat = self.dataset.train_data.to_csr_matrix()
        self.train_csr_mat.data[:] = 1.0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.config.hidden_act == "identity":
            hidden_act = nn.Identity()
        elif self.config.hidden_act == "sigmoid":
            hidden_act = nn.Sigmoid()
        else:
            raise ValueError(f"hidden activate function '{self.config.hidden_act}' is invalid.")

        if self.config.loss_func == "sigmoid_cross_entropy":
            self.loss_func = sigmoid_cross_entropy
        elif self.config.loss_func == "square_loss":
            self.loss_func = square_loss
        else:
            raise ValueError(f"loss function '{self.config.loss_func}' is invalid.")

        self.cdae = _CDAE(self.num_users, self.num_items, self.config.hidden_dim,
                          self.config.dropout, hidden_act).to(self.device)
        self.optimizer = torch.optim.Adam(self.cdae.parameters(), lr=self.config.lr)

    def fit(self):
        train_users = [user for user in range(self.num_users) if self.train_csr_mat[user].nnz]
        user_iter = BatchIterator(train_users, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.cdae.train()
            for bat_users in user_iter:
                bat_sp_mat = self.train_csr_mat[bat_users]
                bat_items = []
                bat_labels = []
                bat_idx = []  # used to decoder
                for idx, _ in enumerate(bat_users):
                    pos_items = bat_sp_mat[idx].indices
                    neg_items = randint_choice(self.num_items, size=bat_sp_mat[idx].nnz*self.config.num_neg,
                                               replace=True, exclusion=pos_items)
                    neg_items = np.unique(neg_items)
                    bat_sp_mat[idx, neg_items] = 1

                    bat_items.append(pos_items)
                    bat_labels.append(np.ones_like(pos_items, dtype=np.float32))
                    bat_items.append(neg_items)
                    bat_labels.append(np.zeros_like(neg_items, dtype=np.float32))
                    bat_idx.append(np.full(len(pos_items)+len(neg_items), idx, dtype=np.int32))

                bat_items = np.concatenate(bat_items)
                bat_labels = np.concatenate(bat_labels)
                bat_idx = np.concatenate(bat_idx)
                bat_users = np.asarray(bat_users)

                bat_sp_mat = sp_mat_to_sp_tensor(bat_sp_mat).to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)

                bat_idx = torch.from_numpy(bat_idx).long().to(self.device)
                bat_users = torch.from_numpy(bat_users).long().to(self.device)

                hat_y, reg_loss = self.cdae(bat_users, bat_idx, bat_sp_mat, bat_items)

                loss = self.loss_func(hat_y, bat_labels)
                loss = loss.sum()

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
        self.cdae.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        user_ids = torch.from_numpy(np.asarray(users)).long().to(self.device)
        sp_item_mat = sp_mat_to_sp_tensor(self.train_csr_mat[users]).to(self.device)
        ratings = self.cdae.predict(user_ids, sp_item_mat).cpu().detach().numpy()
        return ratings
