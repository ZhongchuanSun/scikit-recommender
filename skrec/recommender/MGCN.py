"""
Paper: Multi-View Graph Convolutional Network for Multimedia Recommendation
Author: Penghang Yu, Zhiyi Tan, Guanming Lu, and Bing-Kun Bao
ACM MM 2023
References: https://github.com/demonph10/MGCN
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrec.recommender.base import AbstractRecommender
from skrec.run_config import RunConfig
from skrec.utils.py import Config
from skrec.io import RSDataset
from typing import Dict, List
from skrec.io import PairwiseIterator
from skrec.utils.py import EarlyStopping


class MGCNConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=1e-4,
                 embed_dim=64,
                 n_ui_layers=2,
                 n_layers=1,
                 lambda_coeff=0.9,
                 knn_k=10,
                 cl_loss=0.001,
                 lr_scheduler=[0.96, 50],
                 batch_size=2048,
                 epochs=1000,
                 early_stop=200,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg  # [0.1, 0.01]
        self.embed_dim: int = embed_dim
        self.n_ui_layers: int = n_ui_layers
        self.n_layers: int = n_layers
        self.lambda_coeff: float = lambda_coeff
        self.knn_k: int = knn_k
        self.cl_loss: float = cl_loss  # [0.001, 0.01, 0.1]
        self.lr_scheduler: List[int] = lr_scheduler

        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm


def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


class _MGCN(nn.Module):
    def __init__(self, config: MGCNConfig, dataset: RSDataset, device):
        super(_MGCN, self).__init__()
        self.sparse = True
        self.device = device
        self.cl_loss = config.cl_loss
        self.n_ui_layers = config.n_ui_layers
        self.embedding_dim = config.embed_dim
        self.knn_k = config.knn_k
        self.n_layers = config.n_layers
        self.reg_weight = config.reg

        self.n_users = dataset.num_users
        self.n_items = dataset.num_items

        self.v_feat = dataset.img_features
        self.t_feat = dataset.txt_features

        # load dataset info
        self.interaction_matrix = dataset.train_data.to_coo_matrix().astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if not os.path.exists(dataset.cache_dir):
            os.makedirs(dataset.cache_dir)

        image_adj_file = os.path.join(dataset.cache_dir, 'image_mgcn_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset.cache_dir, 'text_mgcn_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            v_feat_t = torch.from_numpy(self.v_feat).type(torch.FloatTensor)
            self.image_embedding = nn.Embedding.from_pretrained(v_feat_t, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            t_feat_t = torch.from_numpy(self.t_feat).type(torch.FloatTensor)
            self.text_embedding = nn.Embedding.from_pretrained(t_feat_t, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        # Behavior-Guided Purifier
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))

        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Behavior-Aware Fuser
        att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / users.shape[0]

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, users, pos_items, neg_items):
        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

    def full_sort_predict(self, users):
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[users]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


class MGCN(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = MGCNConfig(**model_config)
        super().__init__(run_config, self.config)
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model: _MGCN = _MGCN(self.config, self.dataset, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        lr_scheduler = self.config.lr_scheduler
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
