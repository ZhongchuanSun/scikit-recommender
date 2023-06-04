import os
import logging
from typing import Union, Tuple, List
from skrec import RankingEvaluator
from skrec import Dataset
from skrec import Config
from skrec.recommender.base import AbstractRecommender
from skrec import merge_config_with_cmd_args
from skrec import ModelRegistry
old_handlers = logging.root.handlers[:]


class RunConfig(Config):
    def __init__(self,
                 recommender="TransRec",
                 data_dir="dataset/Beauty_loo_u5_i5",  # the directory of training and test sets
                 file_column="UIRT",  # UI, UIR, UIT, UIRT
                 column_sep='\t',
                 gpu_id=0,
                 metric=("Precision", "Recall", "MAP", "NDCG", "MRR"),  # ("Precision", "Recall", "MAP", "NDCG", "MRR")
                 top_k=(10, 20, 30, 40, 50, 100),
                 # large test_batch_size might cause GPU memory-consuming, especially dataset is large
                 test_batch_size=64,
                 test_thread=4,
                 seed=2021,
                 **kwargs):
        super(RunConfig, self).__init__()
        self.recommender: str = recommender
        self.data_dir: str = data_dir
        self.file_column: str = file_column
        self.sep: str = column_sep
        self.gpu_id = gpu_id
        self.metric: Union[None, str, Tuple[str], List[str]] = metric
        self.top_k: Union[int, List[int], Tuple[int]] = top_k
        self.test_batch_size: int = test_batch_size
        self.test_thread: int = test_thread
        self.seed = seed

    def _validate(self):
        assert isinstance(self.recommender, str)
        assert isinstance(self.data_dir, str)
        assert isinstance(self.file_column, str)
        assert isinstance(self.sep, str)
        assert isinstance(self.test_batch_size, int) and self.test_batch_size > 0
        assert isinstance(self.test_thread, int) and self.test_thread > 0
        assert isinstance(self.seed, int) and self.seed >= 0


def _set_random_seed(seed=2020):
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

    try:
        import tensorflow as tf
        tf.set_random_seed(seed)
        print("set tensorflow seed")
    except:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print("set pytorch seed")
    except:
        pass


def main():
    # read config
    run_config = {"recommender": "BPRMF",
                  "data_dir": "dataset/ml-100k_ratio_by_time_u5_i5",
                  "file_column": "UIRT",
                  "sep": ',',
                  "gpu_id": 0,
                  "metric": ("Recall", "NDCG"),
                  "top_k": (10, 20, 30, 40, 50),
                  "test_thread": 4,
                  "test_batch_size": 64,
                  "seed": 2021
                  }
    run_config = merge_config_with_cmd_args(run_config)
    run_config = RunConfig(**run_config)
    model_name = run_config.recommender

    registry = ModelRegistry()
    registry.load_skrec_model(model_name)
    # registry.register_model(model_name, model_class)

    model_class = registry.get_model(model_name)
    if not model_class:
        print(f"Recommender '{model_name}' is not found.")

    model_config = {"lr": 1e-3, "early_stop": 100, "lambda1": 0.0}
    model_config = merge_config_with_cmd_args(model_config)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(run_config.gpu_id)
    _set_random_seed(run_config.seed)

    dataset = Dataset(run_config.data_dir, run_config.sep, run_config.file_column)
    evaluator = RankingEvaluator(dataset.train_data.to_user_dict(),
                                 dataset.test_data.to_user_dict(),
                                 metric=run_config.metric, top_k=run_config.top_k,
                                 batch_size=run_config.test_batch_size,
                                 num_thread=run_config.test_thread)

    model: AbstractRecommender = model_class(dataset, model_config, evaluator)
    logging.root.handlers = old_handlers
    model.fit()


if __name__ == "__main__":
    main()
