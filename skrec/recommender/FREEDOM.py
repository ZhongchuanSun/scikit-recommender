"""
Paper: A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation
Author: Xin Zhou, Zhiqi Shen
ACM MM 2023
References: https://github.com/enoche/FREEDOM
"""

import os
from typing import Dict
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrec.recommender.base import AbstractRecommender
from skrec.run_config import RunConfig
from skrec.utils.py import Config
from skrec.io import RSDataset
from skrec.io import PairwiseIterator
from skrec.utils.py import EarlyStopping


class FREEDOMConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=0.0,
                 embed_dim=64,
                 feat_dim=64,
                 lambda_coeff=0.9,
                 n_mm_layers=1,
                 n_ui_layers=2,
                 knn_k=10,
                 mm_image_weight=0.1,
                 dropout=0.8,
                 batch_size=2048,
                 epochs=1000,
                 early_stop=200,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg  # [0.0, 1e-05, 1e-04, 1e-03]
        self.embed_dim: int = embed_dim
        self.feat_dim: int = feat_dim
        self.lambda_coeff: float = lambda_coeff
        self.n_mm_layers: int = n_mm_layers
        self.n_ui_layers: int = n_ui_layers
        self.knn_k: int = knn_k
        self.mm_image_weight: float = mm_image_weight
        self.dropout: float = dropout  # [0.8, 0.9]

        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()


class _FREEDOM(nn.Module):
    def __init__(self, config: FREEDOMConfig, dataset: RSDataset, device):
        super(_FREEDOM, self).__init__()
        self.device = device
        self.embedding_dim = config.embed_dim
        self.feat_embed_dim = config.feat_dim
        self.knn_k = config.knn_k
        self.lambda_coeff = config.lambda_coeff
        self.n_layers = config.n_mm_layers
        self.n_ui_layers = config.n_ui_layers
        self.reg_weight = config.reg
        self.build_item_graph = True
        self.mm_image_weight = config.mm_image_weight
        self.dropout = config.dropout

        self.n_users = dataset.num_users
        self.n_items = dataset.num_items
        self.n_nodes = self.n_users + self.n_items

        self.v_feat = dataset.img_features
        self.t_feat = dataset.txt_features

        # load dataset info
        self.interaction_matrix = dataset.train_data.to_coo_matrix().astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if not os.path.exists(dataset.cache_dir):
            os.makedirs(dataset.cache_dir)
        mm_adj_file = os.path.join(dataset.cache_dir, '_cache_mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            v_feat_t = torch.from_numpy(self.v_feat).type(torch.FloatTensor)
            self.image_embedding = nn.Embedding.from_pretrained(v_feat_t, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            t_feat_t = torch.from_numpy(self.t_feat).type(torch.FloatTensor)
            self.text_embedding = nn.Embedding.from_pretrained(t_feat_t, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind).to(self.device)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

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
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

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

    def forward(self, adj):
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, users, pos_items, neg_items):
        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
        return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)

    def full_sort_predict(self, users):
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[users]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


class FREEDOM(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = FREEDOMConfig(**model_config)
        super().__init__(run_config, self.config)
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model: _FREEDOM = _FREEDOM(self.config, self.dataset, self.device).to(self.device)
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
