"""
Paper: Bootstrap Latent Representations for Multi-modal Recommendation
Author: Xin Zhou, Hongyu Zhou, Yong Liu, Zhiwei Zeng, Chunyan Miao, Pengwei Wang, Yuan You, and Feijun Jiang
WWW 2023
References: https://github.com/enoche/BM3
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from skrec.recommender.base import AbstractRecommender
from skrec.run_config import RunConfig
from skrec.utils.py import Config
from skrec.io import RSDataset
from typing import Dict
from skrec.utils.py import EarlyStopping
from skrec.io import InteractionIterator


class BM3Config(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=0.1,
                 embed_dim=64,
                 feat_dim=64,
                 n_layers=1,
                 dropout=0.3,
                 cl_weight=2.0,
                 batch_size=2048,
                 epochs=1000,
                 early_stop=200,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg  # [0.1, 0.01]
        self.embed_dim: int = embed_dim
        self.feat_dim: int = feat_dim
        self.n_layers: int = n_layers
        self.cl_weight: float = cl_weight  # 2.0
        self.dropout: float = dropout  # [0.3, 0.5]

        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class _BM3(nn.Module):
    def __init__(self, config: BM3Config, dataset: RSDataset, device):
        super(_BM3, self).__init__()
        self.device = device
        self.embedding_dim = config.embed_dim
        self.feat_embed_dim = config.embed_dim
        self.n_layers = config.n_layers
        self.reg_weight = config.reg
        self.cl_weight = config.cl_weight
        self.dropout = config.dropout

        self.n_users = dataset.num_users
        self.n_items = dataset.num_items
        self.n_nodes = self.n_users + self.n_items

        self.v_feat = dataset.img_features
        self.t_feat = dataset.txt_features

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.train_data.to_coo_matrix().astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        nn.init.xavier_normal_(self.predictor.weight)

        if self.v_feat is not None:
            v_feat_t = torch.from_numpy(self.v_feat).type(torch.FloatTensor)
            self.image_embedding = nn.Embedding.from_pretrained(v_feat_t, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            t_feat_t = torch.from_numpy(self.t_feat).type(torch.FloatTensor)
            self.text_embedding = nn.Embedding.from_pretrained(t_feat_t, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
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

    def forward(self):
        h = self.item_id_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, users, items):
        # online network
        u_online_ori, i_online_ori = self.forward()
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
               self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

    def full_sort_predict(self, users):
        u_online, i_online = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[users], i_online.transpose(0, 1))
        return score_mat_ui


class BM3(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = BM3Config(**model_config)
        super().__init__(run_config, self.config)
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model: _BM3 = _BM3(self.config, self.dataset, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        lr_scheduler = [1.0, 50]
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

    def fit(self):
        data_iter = InteractionIterator(self.dataset.train_data,
                                        batch_size=self.config.batch_size,
                                        shuffle=True, drop_last=False)
        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")

        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.model.train()
            for bat_users, bat_pos_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                loss = self.model.calculate_loss(bat_users, bat_pos_items)

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
