"""
Paper: Disentangled Negative Sampling for Collaborative Filtering
Authors: Riwei Lai, Li Chen, Yuhan Zhao, Rui Chen, and Qilong Han
WSDM 2023
https://dl.acm.org/doi/10.1145/3539597.3570419
Reference: https://github.com/Riwei-HEU/DENS
"""

import torch
import torch.nn as nn
import torch.utils
import numpy as np
from typing import Dict
import scipy.sparse as sp
from collections import defaultdict
from .base import AbstractRecommender
from ..io import PairwiseIterator
from ..utils.py import EarlyStopping
from ..utils.py import Config
from ..run_config import RunConfig


class DENSConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 l2=1e-4,
                 gamma=0.3,
                 dim=64,
                 batch_size=2048,
                 context_hops=3,
                 K=1,
                 n_negs=6,
                 ns="dens",
                 pool="mean",
                 warmup=100,
                 mess_dropout=False,
                 mess_dropout_rate=0.1,
                 edge_dropout=False,
                 edge_dropout_rate=0.1,
                 alpha=1.0,
                 epochs=1000,
                 early_stop=100,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.l2: float = l2
        self.gamma: float = gamma
        self.dim: int = dim
        self.batch_size: int = batch_size
        self.context_hops: int = context_hops
        self.K: int = K
        self.n_negs: int = n_negs
        self.ns: str = ns
        self.pool: str = pool
        self.warmup: int = warmup
        self.mess_dropout: bool = mess_dropout
        self.mess_dropout_rate: float = mess_dropout_rate
        self.edge_dropout: bool = edge_dropout
        self.edge_dropout_rate: float = edge_dropout_rate
        self.alpha: float = alpha
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.l2, float) and self.l2 >= 0
        assert isinstance(self.gamma, float) and self.gamma >= 0
        assert isinstance(self.dim, int) and self.dim > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.context_hops, int) and self.context_hops >= 0
        assert isinstance(self.K, int) and self.K > 0
        assert isinstance(self.n_negs, int) and self.n_negs > 0
        assert isinstance(self.ns, str) and self.ns in {"rns", "dns", "dens"}
        assert isinstance(self.warmup, int) and self.warmup >= 0
        assert isinstance(self.mess_dropout, bool)
        assert isinstance(self.mess_dropout_rate, float) and self.mess_dropout_rate >= 0
        assert isinstance(self.edge_dropout, bool)
        assert isinstance(self.edge_dropout_rate, float) and self.edge_dropout_rate >= 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]


class _DENS(nn.Module):
    def __init__(self, n_users, n_items, args_config, adj_mat, device):
        super(_DENS, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K
        self.alpha = args_config.alpha
        self.warmup = args_config.warmup
        # gating
        self.gamma = args_config.gamma
        self.device = device

        self.user_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.item_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)

        self.pos_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.neg_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, cur_epoch, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)

        if self.ns == 'rns':  # n_negs = 1
            neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]
        elif self.ns == 'dns':
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.dynamic_negative_sampling(user_gcn_emb, item_gcn_emb,
                                                                   user,
                                                                   neg_item[:, k * self.n_negs: (k + 1) * self.n_negs]))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)
        elif self.ns == 'dens':
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.dise_negative_sampling(cur_epoch, user_gcn_emb, item_gcn_emb,
                                                                user,
                                                                neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                                pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)
        else:
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.mix_negative_sampling(user_gcn_emb, item_gcn_emb,
                                                               user,
                                                               neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                               pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(cur_epoch, user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)

    def dise_negative_sampling(self, cur_epoch, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]

        gate_p = torch.sigmoid(self.item_gate(p_e) + self.user_gate(s_e))
        gated_p_e = p_e * gate_p  # [batch_size, n_hops+1, channel]

        gate_n = torch.sigmoid(self.neg_gate(n_e) + self.pos_gate(gated_p_e).unsqueeze(1))
        gated_n_e = n_e * gate_n  # [batch_size, n_negs, n_hops+1, channel]

        n_e_sel = (1 - min(1, cur_epoch / self.warmup)) * n_e - gated_n_e  # [batch_size, n_negs, n_hops+1, channel]
        # n_e_sel = (1 - max(0, 1 - (cur_epoch / self.warmup))) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]
        # n_e_sel = (1 - self.alpha) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]

        """dynamic negative sampling"""
        scores = (s_e.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
               range(neg_items_emb_.shape[1]), indices, :]

    def dynamic_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates):
        s_e = user_gcn_emb[user]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]

        if self.pool == 'mean':
            s_e = s_e.mean(dim=1)  # [batch_size, channel]
            n_e = n_e.mean(dim=2)  # [batch_size, n_negs, channel]

        """dynamic negative sampling"""
        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs]
        indices = torch.max(scores, dim=1)[1].detach()  # [batch_size]
        neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze()

        return item_gcn_emb[neg_item]

    def mix_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
               range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, cur_epoch, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size,
                                                                                                       self.K, -1)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]

        mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        if self.ns == 'dens' and self.gamma > 0.:
            gate_pos = torch.sigmoid(self.item_gate(pos_gcn_embs) + self.user_gate(user_gcn_emb))
            gated_pos_e_r = pos_gcn_embs * gate_pos
            gated_pos_e_ir = pos_gcn_embs - gated_pos_e_r

            gate_neg = torch.sigmoid(self.neg_gate(neg_gcn_embs) + self.pos_gate(gated_pos_e_r).unsqueeze(1))
            gated_neg_e_r = neg_gcn_embs * gate_neg
            gated_neg_e_ir = neg_gcn_embs - gated_neg_e_r

            gated_pos_e_r = self.pooling(gated_pos_e_r)
            gated_neg_e_r = self.pooling(gated_neg_e_r.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(
                batch_size, self.K, -1)

            gated_pos_e_ir = self.pooling(gated_pos_e_ir)
            gated_neg_e_ir = self.pooling(gated_neg_e_ir.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(
                batch_size, self.K, -1)

            gated_pos_scores_r = torch.sum(torch.mul(u_e, gated_pos_e_r), axis=1)
            gated_neg_scores_r = torch.sum(torch.mul(u_e.unsqueeze(dim=1), gated_neg_e_r), axis=-1)  # [batch_size, K]

            gated_pos_scores_ir = torch.sum(torch.mul(u_e, gated_pos_e_ir), axis=1)
            gated_neg_scores_ir = torch.sum(torch.mul(u_e.unsqueeze(dim=1), gated_neg_e_ir), axis=-1)  # [batch_size, K]

            # BPR
            # mf_loss += self.gamma * torch.mean(torch.log(1 + torch.exp(gated_neg_scores_r - gated_neg_scores_ir).sum(dim=1)))
            # mf_loss += self.gamma * (torch.mean(torch.log(1 + torch.exp(gated_neg_scores_r - gated_neg_scores_ir).sum(dim=1))) + torch.mean(torch.log(1 + torch.exp(gated_pos_scores_ir.unsqueeze(dim=1) - gated_neg_scores_ir).sum(dim=1)))) / 2
            mf_loss += self.gamma * (
                        torch.mean(torch.log(1 + torch.exp(gated_pos_scores_ir - gated_pos_scores_r))) + torch.mean(
                    torch.log(1 + torch.exp(gated_neg_scores_r - gated_neg_scores_ir).sum(dim=1))) + torch.mean(
                    torch.log(1 + torch.exp(gated_neg_scores_r - gated_pos_scores_r.unsqueeze(dim=1)).sum(
                        dim=1))) + torch.mean(torch.log(
                    1 + torch.exp(gated_pos_scores_ir.unsqueeze(dim=1) - gated_neg_scores_ir).sum(dim=1)))) / 4

        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                      + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                      + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss


def build_sparse_graph(data_cf, n_users, n_items):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat)


class DENS(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = DENSConfig(**model_config)
        super().__init__(run_config, self.config)

        self.user_gcn_emb, self.item_gcn_emb = None, None

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.user_group = defaultdict(list)
        user_pos_train = self.dataset.train_data.to_user_dict()
        for user, item_seq in user_pos_train.items():
            self.user_group[len(item_seq)].append(user)

        train_cf = self.dataset.train_data.to_user_item_pairs()
        norm_mat = build_sparse_graph(train_cf, self.num_users, self.num_items)

        self.model = _DENS(self.num_users, self.num_items, self.config, norm_mat, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=0, lr=self.config.lr)

    def fit(self):
        data_iter = PairwiseIterator(self.dataset.train_data,
                                     num_neg=self.config.K*self.config.n_negs,
                                     batch_size=self.config.batch_size,
                                     shuffle=True, drop_last=False)
        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        for epoch in range(self.config.epochs):
            self.model.train()
            for uids, pos, neg in data_iter:
                batch = dict()
                batch['users'] = torch.from_numpy(uids).long().to(self.device)
                batch['pos_items'] = torch.from_numpy(pos).long().to(self.device)
                batch['neg_items'] = torch.from_numpy(neg).long().to(self.device)

                # feed
                self.optimizer.zero_grad()
                batch_loss, _, _ = self.model(epoch, batch)
                batch_loss.backward()
                self.optimizer.step()

            result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{result.values_str}")
            if early_stopping(result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def evaluate(self, test_users=None):
        self.model.eval()
        self.user_gcn_emb, self.item_gcn_emb = self.model.generate()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        predictions = self.model.rating(self.user_gcn_emb[users], self.item_gcn_emb).detach().cpu()
        return predictions.cpu().detach().numpy()
