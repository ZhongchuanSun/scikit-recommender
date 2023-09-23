"""
Paper: Hierarchical Gating Networks for Sequential Recommendation
Author: Chen Ma, Peng Kang, and Xue Liu
Reference: https://github.com/allenjack/HGN
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["HGN"]

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Dict
from .base import AbstractRecommender
from ..utils.py import EarlyStopping
from ..utils.py import Config
from ..utils.torch import bpr_loss, get_initializer
from ..io import SequentialPairwiseIterator
from ..run_config import RunConfig


class HGNConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=1e-3,
                 seq_L=5,
                 seq_T=3,
                 embed_size=64,
                 batch_size=1024,
                 epochs=1000,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.seq_L: int = seq_L
        self.seq_T: int = seq_T
        self.embed_size: int = embed_size
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, float) and self.reg >= 0
        assert isinstance(self.seq_L, int) and self.seq_L > 0
        assert isinstance(self.seq_T, int) and self.seq_T > 0
        assert isinstance(self.embed_size, int) and self.embed_size > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class _HGN(nn.Module):
    def __init__(self, num_users, num_items, dims, seq_L, item_pad_idx=None):
        super(_HGN, self).__init__()
        self._item_pad_idx = item_pad_idx
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims, padding_idx=item_pad_idx)

        self.feature_gate_item = nn.Linear(dims, dims)
        self.feature_gate_user = nn.Linear(dims, dims)

        self.instance_gate_item = Parameter(torch.Tensor(dims, 1))
        self.instance_gate_user = Parameter(torch.Tensor(dims, seq_L))

        self.W2 = nn.Embedding(num_items, dims, padding_idx=item_pad_idx)
        self.b2 = nn.Embedding(num_items, 1, padding_idx=item_pad_idx)

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        # weight initialization
        init = get_initializer("normal")
        he_init = get_initializer("he_uniform")
        xavier_init = get_initializer("xavier_uniform")
        zero_init = get_initializer("zeros")

        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)

        he_init(self.feature_gate_user.weight)
        he_init(self.feature_gate_item.weight)
        zero_init(self.feature_gate_user.bias)
        zero_init(self.feature_gate_item.bias)

        xavier_init(self.instance_gate_item)
        xavier_init(self.instance_gate_user)

        init(self.W2.weight)
        zero_init(self.b2.weight)

        if self._item_pad_idx is not None:
            zero_init(self.item_embeddings.weight[self._item_pad_idx])
            zero_init(self.W2.weight[self._item_pad_idx])

    def _forward_user(self, user_emb, item_embs):
        # feature gating
        gate = torch.sigmoid(self.feature_gate_item(item_embs) +
                             self.feature_gate_user(user_emb).unsqueeze(1))
        gated_item = item_embs * gate

        # instance gating
        term1 = torch.matmul(gated_item, self.instance_gate_item.unsqueeze(0))  # (b,l,d)x(1,d,1)->(b,l,1)
        term2 = user_emb.mm(self.instance_gate_user)  # (b,d)x(d,l)->(b,l)

        instance_score = torch.sigmoid(term1.squeeze() + term2)  # (b,l)
        union_out = gated_item * instance_score.unsqueeze(2)
        union_out = torch.sum(union_out, dim=1)
        union_out = union_out / torch.sum(instance_score, dim=1).unsqueeze(1)
        return union_out

    def forward(self, user_ids, item_seq_ids, target_item_ids):
        item_embs = self.item_embeddings(item_seq_ids)
        user_emb = self.user_embeddings(user_ids)
        union_out = self._forward_user(user_emb, item_embs)

        w2 = self.W2(target_item_ids)
        b2 = self.b2(target_item_ids)

        # MF
        # res = w2.matmul(user_emb.unsqueeze(dim=2)).squeeze()+b2.squeeze()
        torch.matmul(w2, user_emb.unsqueeze(dim=2))
        res = torch.baddbmm(b2, w2, user_emb.unsqueeze(2)).squeeze()

        # union-level
        # res += union_out.unsqueeze(1).matmul(w2.permute(0, 2, 1)).squeeze()
        res += torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

        # item-item product
        # res += item_embs.matmul(w2.permute(0, 2, 1)).squeeze().sum(dim=1)
        rel_score = item_embs.bmm(w2.permute(0, 2, 1))
        rel_score = torch.sum(rel_score, dim=1)
        res += rel_score

        return res

    def clip_parameters_norm(self):
        # to avoid NAN
        self.feature_gate_user.weight.data.renorm_(p=2, dim=1, maxnorm=1)
        self.feature_gate_item.weight.data.renorm_(p=2, dim=1, maxnorm=1)

    def predict(self, user_ids, item_seq_ids):
        item_embs = self.item_embeddings(item_seq_ids)
        user_emb = self.user_embeddings(user_ids)
        union_out = self._forward_user(user_emb, item_embs)
        w2 = self.W2.weight
        b2 = self.b2.weight.squeeze()

        # MF
        res = user_emb.mm(w2.T) + b2  # (b,d)x(d,n)->(b,n)

        # union-level
        res += union_out.mm(w2.T)  # (b,d)x(d,n)->(b,n)

        # item-item product
        res += torch.matmul(item_embs, w2.T.unsqueeze(dim=0)).sum(dim=1)  # (b,l,d)x(1,d,n)->(b,l,n)->(b,n)

        return res


class HGN(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = HGNConfig(**model_config)
        super().__init__(run_config, self.config)
        config = self.config

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.pad_idx = self.num_items
        self.num_items += 1

        self.user_truncated_seq = self.dataset.train_data.to_truncated_seq_dict(config.seq_L,
                                                                                pad_value=self.pad_idx,
                                                                                padding='pre', truncating='pre')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.hgn = _HGN(self.num_users, self.num_items, config.embed_size, config.seq_L, self.pad_idx).to(self.device)
        self.optimizer = torch.optim.Adam(self.hgn.parameters(), weight_decay=config.reg, lr=config.lr)

    def fit(self):
        data_iter = SequentialPairwiseIterator(self.dataset.train_data,
                                               num_previous=self.config.seq_L, num_next=self.config.seq_T,
                                               pad=self.pad_idx, batch_size=self.config.batch_size,
                                               shuffle=True, drop_last=False)

        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.hgn.train()
            for bat_users, bat_item_seqs, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_item_seqs = torch.from_numpy(bat_item_seqs).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                bat_items = torch.cat([bat_pos_items, bat_neg_items], dim=1)
                bat_ratings = self.hgn(bat_users, bat_item_seqs, bat_items)

                yui, yuj = torch.split(bat_ratings, [self.config.seq_T, self.config.seq_T], dim=1)
                loss = bpr_loss(yui, yuj).sum()

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
        self.hgn.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        bat_seq = [self.user_truncated_seq[u] for u in users]
        bat_seq = torch.from_numpy(np.asarray(bat_seq)).long().to(self.device)
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        all_ratings = self.hgn.predict(users, bat_seq)
        return all_ratings.cpu().detach().numpy()
