"""
Paper: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer
Author: Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang
Reference: https://github.com/FeiSun/BERT4Rec
"""
__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["BERT4Rec"]

import os
import numpy as np
import time
import pickle
from typing import Dict
from skrec.utils.py import Config
from skrec.recommender.base import AbstractRecommender
from . import modeling
import tensorflow as tf
from .bert4rec_utils import model_fn_builder, input_fn_builder, EvalHooks
from ...run_config import RunConfig


class BERT4RecConfig(Config):
    def __init__(self,
                 # prepare data
                 max_seq_len=5,
                 masked_lm_prob=0.4,
                 sliding_step=1,
                 dupe_factor=10,
                 # bert model
                 att_drop=0.2,
                 h_drop=0.5,
                 h_size=64,
                 att_heads=2,
                 init_range=0.02,
                 h_act="gelu",
                 n_layers=2,
                 # train
                 lr=1e-4,
                 batch_size=256,
                 save_ckpt_epoch=10,
                 init_ckpt=None,
                 epochs=3000,
                 early_stop=80,
                 verbose=10,
                 pool_size=10,
                 **kwargs):
        super().__init__()
        # prepare data
        self.max_seq_len: int = max_seq_len
        self.masked_lm_prob: float = masked_lm_prob
        self.sliding_step: int = sliding_step
        self.dupe_factor: int = dupe_factor
        # bert model
        self.att_drop: float = att_drop
        self.h_drop: float = h_drop
        self.h_size: int = h_size
        self.att_heads: int = att_heads
        # do not tune
        self.init_range: float = init_range
        self.h_act: str = h_act
        self.n_layers: int = n_layers
        # train
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.save_ckpt_epoch: int = save_ckpt_epoch
        self.init_ckpt: str = init_ckpt
        self.epochs: int = epochs
        self.early_stop: int = early_stop
        self.verbose: int = verbose
        self.pool_size: int = pool_size


class BERT4Rec(AbstractRecommender):
    def __init__(self, run_config: RunConfig, model_config: Dict):
        self.config = BERT4RecConfig(**model_config)
        super().__init__(run_config, self.config)
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        self.logger.info("prepare data...")
        output_dir = self.dataset.data_dir
        tf_record_name = [f"max_seq_len={self.config.max_seq_len}",
                          f"masked_lm_prob={self.config.masked_lm_prob}",
                          f"sliding_step={self.config.sliding_step}",
                          f"dupe_factor={self.config.dupe_factor}"]
        tf_record_name = "_".join(tf_record_name)

        output_dir = os.path.join(output_dir, f"_{self.__class__.__name__}_data")
        self.train_input_file = os.path.join(output_dir, tf_record_name+'.train.tfrecord')
        self.test_input_file = os.path.join(output_dir, tf_record_name+'.test.tfrecord')
        vocab_filename = os.path.join(output_dir, tf_record_name+'.vocab')
        self.user_history_filename = os.path.join(output_dir, tf_record_name+'.his')
        num_trains_file = os.path.join(output_dir, tf_record_name+'.train.num.npy')

        if not os.path.exists(self.train_input_file) or \
                not os.path.exists(self.test_input_file) or \
                not os.path.exists(vocab_filename) or \
                not os.path.exists(self.user_history_filename) or \
                not os.path.exists(num_trains_file):
            from . import bert4rec_gen_data
            bert4rec_gen_data.main(self.config, self.dataset, output_dir, tf_record_name)

        with open(vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
        self.item_size = len(vocab.token_to_ids)
        self.num_instances = np.load(num_trains_file)

    def _build_model(self):
        self.logger.info("build model...")
        config = self.config
        bert_config = modeling.BertConfig(self.item_size,
                                          hidden_size=config.h_size,
                                          num_hidden_layers=config.n_layers,
                                          num_attention_heads=config.att_heads,
                                          intermediate_size=config.h_size*4,
                                          hidden_act=config.h_act,
                                          hidden_dropout_prob=config.h_drop,
                                          attention_probs_dropout_prob=config.att_drop,
                                          max_position_embeddings=config.max_seq_len,
                                          type_vocab_size=2,
                                          initializer_range=config.init_range)
        timestamp = time.time()
        checkpoint_dir = os.path.join(self.dataset.data_dir, f"_{self.__class__.__name__}_ckpt_dir")
        checkpoint_dir = os.path.join(checkpoint_dir, f"{timestamp}")
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        num_instances_per_epoch = int(self.num_instances/(config.dupe_factor+1))
        self.steps_per_epoch = int(num_instances_per_epoch / config.batch_size)
        num_train_steps = self.steps_per_epoch * config.epochs
        max_pre_seq = int(round(config.max_seq_len*config.masked_lm_prob))

        save_ckpt_steps = self.steps_per_epoch*config.save_ckpt_epoch
        tf_config = tf.ConfigProto()  # allow_soft_placement=False, log_device_placement=True
        tf_config.gpu_options.allow_growth = True

        run_config = tf.estimator.RunConfig(
            session_config=tf_config,
            model_dir=checkpoint_dir,
            save_checkpoints_steps=save_ckpt_steps,
            keep_checkpoint_max=1,
            save_summary_steps=0)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=config.init_ckpt,
            learning_rate=config.lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=100,
            use_tpu=False,
            use_one_hot_embeddings=False,
            item_size=self.item_size)

        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={
                "batch_size": config.batch_size
            })

        self.train_input_fn = input_fn_builder(
            input_files=[self.train_input_file],
            max_seq_length=config.max_seq_len,
            max_predictions_per_seq=max_pre_seq,
            is_training=True)

        self.eval_input_fn = input_fn_builder(
            input_files=[self.test_input_file],
            max_seq_length=config.max_seq_len,
            max_predictions_per_seq=max_pre_seq,
            is_training=False)

    def fit(self):
        config = self.config
        eval_hook = EvalHooks(config, self.user_history_filename, self.evaluator, self.logger)
        cur_steps = 0
        for epoch in range(0, config.epochs, config.verbose):
            self.estimator.train(input_fn=self.train_input_fn,
                                 max_steps=cur_steps+self.steps_per_epoch*config.verbose)
            cur_steps += self.steps_per_epoch*config.verbose
            self.estimator.evaluate(input_fn=self.eval_input_fn, steps=None, hooks=[eval_hook])
        self.logger.info("best:".ljust(12) + f"\t{eval_hook.early_stopping.best_result.values_str}")
