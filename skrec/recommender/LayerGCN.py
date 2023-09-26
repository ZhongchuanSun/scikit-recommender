"""
Paper: Layer-refined Graph Convolutional Networks for Recommendation
Author: Xin Zhou, Donghui Lin, Yong Liu, and Chunyan Miao
ICDE 2023
References: https://github.com/enoche/ImRec
            https://github.com/enoche/MMRec/blob/master/src/models/layergcn.py
"""

import numpy as np
import scipy.sparse as sp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrec.recommender.base import AbstractRecommender
from skrec.run_config import RunConfig
from skrec.utils.py import Config
from skrec.io import RSDataset
from typing import Dict
from skrec.utils.py import EarlyStopping
from skrec.io import PairwiseIterator


class LayerGCNConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=1e-2,
                 embed_dim=64,
                 n_layers=4,
                 dropout=0.0,
                 batch_size=2048,
                 epochs=1000,
                 early_stop=200,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg  # [1e-02, 1e-03, 1e-04, 1e-05]
        self.embed_dim: int = embed_dim
        self.n_layers: int = n_layers
        self.dropout: float = dropout  # [0.0, 0.1, 0.2]

        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss


class BPRLoss(nn.Module):

    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class _LayerGCN(nn.Module):
    def __init__(self, config: LayerGCNConfig, dataset: RSDataset, device):
        super(_LayerGCN, self).__init__()
        self.device = device
        # load dataset info
        self.interaction_matrix = dataset.train_data.to_coo_matrix().astype(np.float32)

        # load parameters info
        self.latent_dim = config.embed_dim  # int type:the embedding size of lightGCN
        self.n_layers = config.n_layers  # int type:the layer num of lightGCN
        self.reg_weight = config.reg  # float32 type: the weight decay for l2 normalizaton
        self.dropout = config.dropout

        self.n_users = dataset.num_users
        self.n_items = dataset.num_items

        self.n_nodes = self.n_users + self.n_items

        # define layers and loss
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.latent_dim)))

        # normalized adj matrix
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = None
        self.forward_adj = None
        self.pruning_random = False

        # edge prune
        self.edge_indices, self.edge_values = self.get_edge_info()

        self.mf_loss = BPRLoss()
        self.reg_loss = L2Loss()

    # def post_epoch_processing(self):
    #     with torch.no_grad():
    #         return '=== Layer weights: {}'.format(F.softmax(self.layer_weights.exp(), dim=0))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj_matrix
            return
        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        if self.pruning_random:
            # pruning randomly
            keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len)).to(self.device)
        else:
            # pruning edges by pro
            keep_idx = torch.multinomial(self.edge_values, keep_len)         # prune high-degree nodes
        self.pruning_random = True ^ self.pruning_random
        keep_indices = self.edge_indices[:, keep_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj_matrix.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat([self.user_embeddings, self.item_embeddings], 0)
        return ego_embeddings

    def forward(self):
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = ego_embeddings
        embeddings_layers = []

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.forward_adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)
        user_all_embeddings, item_all_embeddings = torch.split(ui_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def bpr_loss(self, u_embeddings, i_embeddings, user, pos_item, neg_item):
        u_embeddings = u_embeddings[user]
        posi_embeddings = i_embeddings[pos_item]
        negi_embeddings = i_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        m = torch.nn.LogSigmoid()
        bpr_loss = torch.sum(-m(pos_scores - neg_scores))
        #mf_loss = self.mf_loss(pos_scores, neg_scores)

        return bpr_loss

    def emb_loss(self, user, pos_item, neg_item):
        # calculate BPR Loss
        u_ego_embeddings = self.user_embeddings[user]
        posi_ego_embeddings = self.item_embeddings[pos_item]
        negi_ego_embeddings = self.item_embeddings[neg_item]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        return reg_loss

    def calculate_loss(self, user, pos_item, neg_item):
        self.forward_adj = self.masked_adj
        user_all_embeddings, item_all_embeddings = self.forward()

        mf_loss = self.bpr_loss(user_all_embeddings, item_all_embeddings, user, pos_item, neg_item)
        reg_loss = self.emb_loss(user, pos_item, neg_item)

        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, users):
        self.forward_adj = self.norm_adj_matrix
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[users]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


class LayerGCN(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = LayerGCNConfig(**model_config)
        super().__init__(run_config, self.config)
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model: _LayerGCN = _LayerGCN(self.config, self.dataset, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        lr_scheduler = [1.0, 50]
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

    def fit(self):
        data_iter = PairwiseIterator(self.dataset.train_data,
                                     batch_size=self.config.batch_size,
                                     shuffle=True, drop_last=False)
        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")

        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.model.pre_epoch_processing()
            self.model.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                loss = self.model.calculate_loss(bat_users, bat_pos_items, bat_neg_items)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()
            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def evaluate(self, test_users=None):
        self.model.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.model.full_sort_predict(users).cpu().detach().numpy()
