import os
from typing import Union, Tuple, List
from importlib.util import find_spec
from importlib import import_module
from skrec import RankingEvaluator
from skrec import Dataset
from skrec import Config
from skrec.recommender.base import AbstractRecommender


class RunConfig(Config):
    def __init__(self,
                 recommender="BPRMF",
                 data_dir="",
                 file_column="UIRT", column_sep='\t',
                 gpu_id=0,
                 metric=("Precision", "Recall", "MAP", "NDCG", "MRR"),
                 top_k=(10, 20, 30, 40, 50, 100),
                 test_batch_size=64,
                 test_thread=4,
                 seed=2021,
                 **kwargs):
        super(RunConfig, self).__init__(**kwargs)
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


def import_model_and_config(model_name: str):
    spec_path = f"skrec.recommender.{model_name}"
    if find_spec(spec_path):
        module = import_module(spec_path)
    else:
        raise ModuleNotFoundError(f"Module '{spec_path}' is not found.")

    if hasattr(module, model_name):
        Model = getattr(module, model_name)
    else:
        raise ImportError(f"Import {model_name} failed from {module.__file__}!")

    if hasattr(module, f"{model_name}Config"):
        ModelConfig = getattr(module, f"{model_name}Config")
    else:
        raise ImportError(f"Import {model_name}Config failed from {module.__file__}!")
    return Model, ModelConfig


def main():
    cfg_file = "./skrec.ini"
    # read config
    run_config = RunConfig()
    run_config.parse_args_from_ini(cfg_file, "skrec")  # parse args from file and overwrite the default values
    run_config.parse_args_from_cmd()  # parse args from cmd and overwrite the previous values
    model_name = run_config.recommender

    Model, ModelConfig = import_model_and_config(model_name)

    model_config: Config = ModelConfig()
    # parse args from file and overwrite the default values
    model_config.parse_args_from_ini(cfg_file, model_name)
    model_config.parse_args_from_cmd()  # parse args from cmd and overwrite the previous values

    os.environ['CUDA_VISIBLE_DEVICES'] = str(run_config.gpu_id)
    _set_random_seed(run_config.seed)

    dataset = Dataset(run_config.data_dir, run_config.sep, run_config.file_column)
    evaluator = RankingEvaluator(dataset.train_data.to_user_dict(),
                                 dataset.test_data.to_user_dict(),
                                 metric=run_config.metric, top_k=run_config.top_k,
                                 batch_size=run_config.test_batch_size,
                                 num_thread=run_config.test_thread)

    model: AbstractRecommender = Model(dataset, model_config, evaluator)
    model.fit()


if __name__ == "__main__":
    main()
