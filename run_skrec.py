import os
import logging
from skrec.recommender.base import AbstractRecommender
from skrec import merge_config_with_cmd_args
from skrec import ModelRegistry
from skrec import RunConfig
old_handlers = logging.root.handlers[:]


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
    run_dict = {"recommender": "KGAT",
                "data_dir": "dataset/yelp2018",
                "file_column": "UIR",
                "column_sep": ',',
                "gpu_id": 0,
                "metric": ("Recall", "NDCG"),
                "top_k": (10, 20, 30, 40, 50),
                "test_thread": 4,
                "test_batch_size": 64,
                "seed": 2021
                }
    run_dict = merge_config_with_cmd_args(run_dict)

    run_config = RunConfig(**run_dict)
    model_name = run_config.recommender

    registry = ModelRegistry()
    registry.load_skrec_model(model_name)
    # registry.register_model(model_name, model_class)

    model_class = registry.get_model(model_name)
    if not model_class:
        print(f"Recommender '{model_name}' is not found.")

    model_params = {"lr": 1e-3, "epochs": 2}
    model_params = merge_config_with_cmd_args(model_params)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(run_config.gpu_id)
    _set_random_seed(run_config.seed)

    model: AbstractRecommender = model_class(run_config, model_params)
    logging.root.handlers = old_handlers
    model.fit()


if __name__ == "__main__":
    main()
