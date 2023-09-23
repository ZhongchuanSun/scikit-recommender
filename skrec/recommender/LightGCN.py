"""
Paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
Author: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang
Reference: https://github.com/hexiangnan/LightGCN
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["LightGCN"]

import os
import torch
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np
import scipy.sparse as sp
from .base import AbstractRecommender
from ..utils.torch import inner_product, bpr_loss, l2_loss, get_initializer
from ..utils.py import EarlyStopping
from ..io import PairwiseIterator
from ..utils.common import normalize_adj_matrix
from ..utils.torch import sp_mat_to_sp_tensor
from ..utils.py import Config
from ..run_config import RunConfig


class LightGCNConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=1e-3,
                 embed_size=64,
                 n_layers=3,
                 adj_type="pre",
                 batch_size=1024,
                 epochs=1000,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.embed_size: int = embed_size
        self.n_layers: int = n_layers
        self.adj_type: str = adj_type  # plain, norm, gcmc, pre
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, float) and self.reg >= 0
        assert isinstance(self.embed_size, int) and self.embed_size > 0
        assert isinstance(self.n_layers, int) and self.n_layers> 0
        assert self.adj_type in {"plain", "norm", "gcmc", "pre"}
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        init = get_initializer("xavier_uniform")
        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)

    def forward(self, users, items):
        user_embeddings, item_embeddings = self._forward_gcn()
        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        ratings = inner_product(user_embs, item_embs)
        return ratings

    def _forward_gcn(self):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            ego_embeddings = torch_sp.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        ratings = torch.matmul(user_embs, self._item_embeddings_final.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn()


class LightGCN(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = LightGCNConfig(**model_config)
        super().__init__(run_config, self.config)
        config = self.config

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        adj_matrix = self._load_adj_mat(config.adj_type)
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, config.embed_size,
                                  adj_matrix, config.n_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=config.lr)

    def _load_adj_mat(self, adj_type):
        output_dir = self.dataset.data_dir
        output_dir = os.path.join(output_dir, f"_{self.__class__.__name__}_data")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        adj_mat_file = os.path.join(output_dir, f"{adj_type}_adj.npz")
        if os.path.exists(adj_mat_file):
            adj_matrix = sp.load_npz(adj_mat_file)
        else:
            adj_matrix = self._create_adj_mat(adj_type)
            sp.save_npz(adj_mat_file, adj_matrix)
        return adj_matrix

    def _create_adj_mat(self, adj_type):
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        ratings = np.ones_like(users_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        up_left_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = up_left_adj + up_left_adj.T

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalize_adj_matrix(adj_mat + sp.eye(adj_mat.shape[0]), norm_method="left")
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalize_adj_matrix(adj_mat, norm_method="left")
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            adj_matrix = normalize_adj_matrix(adj_mat, norm_method="symmetric")
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalize_adj_matrix(adj_mat, norm_method="left")
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix

    def fit(self):
        data_iter = PairwiseIterator(self.dataset.train_data,
                                     batch_size=self.config.batch_size,
                                     shuffle=True, drop_last=False)

        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.lightgcn.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)

                _bat_users = torch.cat([bat_users, bat_users], dim=0)
                _bat_items = torch.cat([bat_pos_items, bat_neg_items], dim=0)

                hat_y = self.lightgcn(_bat_users, _bat_items)
                yui, yuj = torch.split(hat_y, [len(bat_pos_items), len(bat_neg_items)], dim=0)

                loss = bpr_loss(yui, yuj).mean()
                reg_loss = l2_loss(self.lightgcn.user_embeddings(bat_users),
                                   self.lightgcn.item_embeddings(bat_pos_items),
                                   self.lightgcn.item_embeddings(bat_neg_items)
                                   )
                loss += self.config.reg * reg_loss / self.config.batch_size
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
        self.lightgcn.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
