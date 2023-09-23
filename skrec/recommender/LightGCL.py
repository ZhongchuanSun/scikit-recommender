"""
Paper: LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation
Author: Xuheng Cai, Chao Huang, Lianghao Xia, and Xubin Ren
ICLR 2023
Reference: https://github.com/HKUDS/LightGCL
"""

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from typing import Dict
from collections import defaultdict
import numpy as np
from .base import AbstractRecommender
from ..io import PairwiseIterator
from ..utils.torch import sp_mat_to_sp_tensor
from ..utils.py import EarlyStopping
from ..utils.py import Config
from ..run_config import RunConfig


class LightGCLConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 lambda1=0.2,
                 d=64,
                 gnn_layer=2,
                 batch_size=2048,
                 svd_q=5,
                 dropout=0.0,
                 temp=0.2,
                 lambda2=1e-7,
                 epochs=500,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.lambda1: float = lambda1  # weight of cl loss
        self.d: int = d  # embedding size
        self.gnn_layer: int = gnn_layer
        self.batch_size: int = batch_size
        self.svd_q: int = svd_q  # rank
        self.dropout: float = dropout
        self.temp: float = temp  # temperature in cl loss
        self.lambda2: float = lambda2  # l2 reg weight
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.lambda1, float) and self.lambda1 >= 0
        assert isinstance(self.d, int) and self.d > 0
        assert isinstance(self.gnn_layer, int) and self.gnn_layer > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.svd_q, int) and self.svd_q > 0
        assert isinstance(self.dropout, float) and self.dropout >= 0
        assert isinstance(self.temp, float) and self.temp > 0
        assert isinstance(self.lambda2, float) and self.lambda2 >= 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)


class _LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, device):
        super(_LightGCL,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            # mask = self.train_csr[uids.cpu().numpy()].toarray()
            # mask = torch.Tensor(mask).cuda(torch.device(self.device))
            # preds = preds * (1-mask) - 1e8 * mask
            # predictions = preds.argsort(descending=True)
            return preds  # predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            loss_s = 0.0
            if self.lambda_1 > 0:
                G_u_norm = self.G_u
                E_u_norm = self.E_u
                G_i_norm = self.G_i
                E_i_norm = self.E_i
                neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
                neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
                pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
                loss_s = -pos_score + neg_score
                loss_s = self.lambda_1 * loss_s

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            # loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()
            loss_r = -F.logsigmoid(pos_scores - neg_scores).mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + loss_s + loss_reg
            return loss


class LightGCL(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = LightGCLConfig(**model_config)
        super().__init__(run_config, self.config)

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.user_group = defaultdict(list)
        user_pos_train = self.dataset.train_data.to_user_dict()
        for user, item_seq in user_pos_train.items():
            self.user_group[len(item_seq)].append(user)

        train = self.dataset.train_data.to_csr_matrix().tocoo()
        train.data[:] = 1.0
        train_csr = (train != 0).astype(np.float32)

        # normalizing the adj matrix
        rowD = np.array(train.sum(1)).squeeze()
        colD = np.array(train.sum(0)).squeeze()
        for i in range(len(train.data)):
            train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)

        adj_norm = sp_mat_to_sp_tensor(train)
        adj_norm = adj_norm.coalesce().cuda(self.device)
        self.logger.info('Adj matrix normalized.')

        # perform svd reconstruction
        adj = sp_mat_to_sp_tensor(train).coalesce().cuda(self.device)
        self.logger.info('Performing SVD...')
        svd_u, s, svd_v = torch.svd_lowrank(adj, q=self.config.svd_q)
        u_mul_s = svd_u @ (torch.diag(s))
        v_mul_s = svd_v @ (torch.diag(s))
        del s
        self.logger.info('SVD done.')

        self.model = _LightGCL(self.num_users, self.num_items, self.config.d, u_mul_s, v_mul_s, svd_u.T, svd_v.T,
                               train_csr, adj_norm, self.config.gnn_layer, self.config.temp,
                               self.config.lambda1, self.config.lambda2,
                               self.config.dropout, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=0, lr=self.config.lr)

    def fit(self):
        data_iter = PairwiseIterator(self.dataset.train_data,
                                     batch_size=self.config.batch_size,
                                     shuffle=True, drop_last=False)
        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.model.train()
            for uids, pos, neg in data_iter:
                uids = torch.from_numpy(uids).long().to(self.device)
                pos = torch.from_numpy(pos).long().to(self.device)
                neg = torch.from_numpy(neg).long().to(self.device)
                iids = torch.cat([pos, neg], dim=0)

                # feed
                self.optimizer.zero_grad()
                loss = self.model(uids, iids, pos, neg)
                loss.backward()
                self.optimizer.step()
                # torch.cuda.empty_cache()
            result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{result.values_str}")
            if early_stopping(result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def evaluate(self, test_users=None):
        self.model.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        predictions = self.model(users, None, None, None, test=True)
        return predictions.cpu().detach().numpy()
