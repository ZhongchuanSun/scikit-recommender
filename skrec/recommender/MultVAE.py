"""
Paper: Variational Autoencoders for Collaborative Filtering
Author: Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara
Reference: https://github.com/dawenl/vae_cf
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["MultVAE"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from .base import AbstractRecommender
from ..utils.py import Config
from ..utils.py import EarlyStopping
from ..utils.py import BatchIterator
from ..utils.torch import l2_loss, get_initializer
from ..run_config import RunConfig


class MultVAEConfig(Config):
    def __init__(self,
                 lr=1e-3,
                 reg=0.0,
                 p_dims=[64],
                 q_dims=None,
                 keep_prob=0.5,
                 anneal_steps=200000,
                 anneal_cap=0.2,
                 batch_size=256,
                 epochs=1000,
                 early_stop=200,
                 **kwargs):
        super().__init__()
        self.lr: float = lr
        self.reg: float = reg
        # p_dims is decoder's dimensions and q_dims is encoder's dimensions
        # if q_dims is None, it will be symmetrical with p_dims
        self.p_dims: List[int] = p_dims
        self.q_dims: List[int] = q_dims
        self.keep_prob: float = keep_prob
        self.anneal_steps: int = anneal_steps
        self.anneal_cap: float = anneal_cap
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self._validate()

    def _validate(self):
        assert isinstance(self.lr, float) and self.lr > 0
        assert isinstance(self.reg, float) and self.reg >= 0
        assert isinstance(self.p_dims, list)
        assert self.q_dims is None or isinstance(self.q_dims, list)
        assert isinstance(self.keep_prob, float) and self.keep_prob >= 0
        assert isinstance(self.anneal_steps, int) and self.anneal_steps >= 0
        assert isinstance(self.anneal_cap, float) and self.anneal_cap >= 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.epochs, int) and self.epochs >= 0
        assert isinstance(self.early_stop, int)


class _MultVAE(nn.Module):
    def __init__(self, q_dims, p_dims, keep_prob):
        super(_MultVAE, self).__init__()

        # user and item embeddings
        self.layers_q = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(q_dims[:-1], q_dims[1:])):
            if i == len(q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance, respectively
                d_out *= 2
            self.layers_q.append(nn.Linear(d_in, d_out, bias=True))

        self.layers_p = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(p_dims[:-1], p_dims[1:])):
            self.layers_p.append(nn.Linear(d_in, d_out, bias=True))

        self.dropout = nn.Dropout(1-keep_prob)

        # weight initialization

        self.reg_params = [layer.weight for layer in self.layers_q] + \
                          [layer.weight for layer in self.layers_p]
        self.reset_parameters()

    def reset_parameters(self):
        init = get_initializer("normal")

        for layer in self.layers_q:
            init(layer.weight)
            init(layer.bias)

        for layer in self.layers_p:
            init(layer.weight)
            init(layer.bias)

    def q_graph(self, input_x):
        mu_q, std_q, kl_dist = None, None, None
        h = F.normalize(input_x, p=2, dim=1)
        h = self.dropout(h)

        for i, layer in enumerate(self.layers_q):
            h = layer(h)
            if i != len(self.layers_q) - 1:
                h = F.tanh(h)
            else:
                size = int(h.shape[1] / 2)
                mu_q, logvar_q = torch.split(h, size, dim=1)
                std_q = torch.exp(0.5 * logvar_q)
                kl_dist = torch.sum(0.5*(-logvar_q + logvar_q.exp() + mu_q.pow(2) - 1), dim=1).mean()

        return mu_q, std_q, kl_dist

    def p_graph(self, z):
        h = z

        for i, layer in enumerate(self.layers_p):
            h = layer(h)
            if i != len(self.layers_p) - 1:
                h = F.tanh(h)

        return h

    def forward(self, input_x):
        # q-network
        mu_q, std_q, kl_dist = self.q_graph(input_x)
        epsilon = std_q.new_empty(std_q.shape)
        epsilon.normal_()
        sampled_z = mu_q + float(self.training)*epsilon*std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return logits, kl_dist

    def predict(self, input_x):
        ratings, _ = self.forward(input_x)

        return ratings


class MultVAE(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = MultVAEConfig(**model_config)
        super().__init__(run_config, self.config)
        config = self.config

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.train_csr_mat = self.dataset.train_data.to_csr_matrix()
        self.train_csr_mat.data[:] = 1.0

        p_dims = config.p_dims
        self.p_dims = p_dims + [self.num_items]
        if config.q_dims is None:
            self.q_dims = self.p_dims[::-1]
        else:
            q_dims = config.q_dims
            q_dims = [self.num_items] + q_dims
            assert q_dims[0] == self.p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == self.p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.multvae = _MultVAE(self.q_dims, self.p_dims, self.config.keep_prob).to(self.device)
        self.optimizer = torch.optim.Adam(self.multvae.parameters(), lr=self.config.lr)

    def fit(self):
        train_users = [user for user in range(self.num_users) if self.train_csr_mat[user].nnz]
        user_iter = BatchIterator(train_users, batch_size=self.config.batch_size, shuffle=True, drop_last=False)

        self.logger.info("metrics:".ljust(12) + f"\t{self.evaluator.metrics_str}")
        early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)
        update_count = 0.0
        for epoch in range(self.config.epochs):
            self.multvae.train()
            for bat_users in user_iter:
                bat_input = self.train_csr_mat[bat_users].toarray()
                if self.config.anneal_steps > 0:
                    anneal = min(self.config.anneal_cap, 1.*update_count/self.config.anneal_steps)
                else:
                    anneal = self.config.anneal_cap

                bat_input = torch.from_numpy(bat_input).float().to(self.device)

                logits, kl_dist = self.multvae(bat_input)
                log_softmax_var = F.log_softmax(logits, dim=-1)
                neg_ll = -torch.mul(log_softmax_var, bat_input).sum(dim=-1).mean()

                # apply regularization to weights
                reg_var = l2_loss(*self.multvae.reg_params)
                reg_var *= self.config.reg

                # l2 regularization multiply 0.5 to the l2 norm
                # multiply 2 so that it is back in the same scale
                neg_elbo = neg_ll + anneal*kl_dist + 2*reg_var

                self.optimizer.zero_grad()
                neg_elbo.backward()
                self.optimizer.step()
                update_count += 1
            cur_result = self.evaluate()
            self.logger.info(f"epoch {epoch}:".ljust(12) + f"\t{cur_result.values_str}")
            if early_stopping(cur_result):
                self.logger.info("early stop")
                break

        self.logger.info("best:".ljust(12) + f"\t{early_stopping.best_result.values_str}")

    def evaluate(self, test_users=None):
        self.multvae.eval()
        return self.evaluator.evaluate(self, test_users)

    def predict(self, users):
        bat_input = self.train_csr_mat[users].toarray()
        bat_input = torch.from_numpy(bat_input).float().to(self.device)
        ratings = self.multvae.predict(bat_input).cpu().detach().numpy()
        return ratings
