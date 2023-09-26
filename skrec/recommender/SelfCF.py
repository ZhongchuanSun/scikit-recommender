"""
Paper: SelfCF: A Simple Framework for Self-supervised Collaborative Filtering
Author: Xin Zhou, Aixin Sun, Yong Liu, Jie Zhang, and Chunyan Miao
TORS 2023
References: https://github.com/enoche/SelfCF
SELFCF_{ed}: embedding dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skrec.recommender.base import AbstractRecommender
from skrec.run_config import RunConfig
from skrec.utils.py import Config
from skrec.io import RSDataset
from typing import Dict
from skrec.utils.py import EarlyStopping
from skrec.io import InteractionIterator
import scipy.sparse as sp


class SelfCFConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=0.0,
                 embed_dim=64,
                 n_layers=2,
                 dropout=0.5,
                 batch_size=2048,
                 epochs=1000,
                 early_stop=200,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        self.embed_dim: int = embed_dim
        self.n_layers: int = n_layers
        self.dropout: float = dropout  # [0.3, 0.5]

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


class LightGCN_Encoder(nn.Module):
    def __init__(self, config: SelfCFConfig, dataset: RSDataset, device):
        super(LightGCN_Encoder, self).__init__()
        # load dataset info
        self.device = device
        self.interaction_matrix = dataset.train_data.to_coo_matrix().astype(np.float32)

        self.user_count = dataset.num_users
        self.item_count = dataset.num_items
        self.n_users = self.user_count
        self.n_items = self.item_count
        self.latent_size = config.embed_dim
        self.n_layers = 3 if config.n_layers is None else config.n_layers
        self.layers = [self.latent_size] * self.n_layers

        self.drop_ratio = 1.0
        self.drop_flag = True

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self.get_norm_adj_mat().to(self.device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.latent_size)))
        })

        return embedding_dict

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
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
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(self.device)
        return out * (1. / (1 - rate))

    def forward(self, users, pos_items):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        items = pos_items
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        return user_all_embeddings, item_all_embeddings


class SELFCFED_LGN(nn.Module):
    def __init__(self, config: SelfCFConfig, dataset: RSDataset, device):
        super(SELFCFED_LGN, self).__init__()
        self.device = device
        self.latent_size = config.embed_dim
        self.dropout = config.dropout
        self.reg_weight = config.reg

        self.n_users = dataset.num_users
        self.n_items = dataset.num_items

        self.online_encoder = LightGCN_Encoder(config, dataset, device)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.reg_loss = L2Loss()

    def forward(self, users, pos_items):
        u_online, i_online = self.online_encoder(users, pos_items)
        with torch.no_grad():
            u_target, i_target = u_online.clone(), i_online.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

        return u_online, u_target, i_online, i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def loss_fn(self, p, z):  # negative cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def calculate_loss(self, users, pos_items):
        u_online, u_target, i_online, i_target = self.forward(users, pos_items)
        reg_loss = self.reg_loss(u_online, i_online)

        u_online, i_online = self.predictor(u_online), self.predictor(i_online)

        loss_ui = self.loss_fn(u_online, i_target)/2
        loss_iu = self.loss_fn(i_online, u_target)/2

        return loss_ui + loss_iu + self.reg_weight * reg_loss

    def full_sort_predict(self, users):
        u_online, u_target, i_online, i_target = self.get_embedding()
        score_mat_ui = torch.matmul(u_online[users], i_target.transpose(0, 1))
        score_mat_iu = torch.matmul(u_target[users], i_online.transpose(0, 1))
        scores = score_mat_ui + score_mat_iu

        return scores


class SelfCF(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = SelfCFConfig(**model_config)
        super().__init__(run_config, self.config)
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model: SELFCFED_LGN = SELFCFED_LGN(self.config, self.dataset, self.device).to(self.device)
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
