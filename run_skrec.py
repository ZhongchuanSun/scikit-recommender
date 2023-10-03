import os
from skrec import merge_config_with_cmd_args
from skrec import ModelRegistry
from skrec import RunConfig
from skrec.utils.hyperopt import HyperOpt


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


def list_gpus():
    import torch
    gpu_str = list()
    gpu_str.append(torch.cuda.is_available())  # 查看是否有可用GPU
    gpu_str.append(torch.cuda.device_count())  # 查看GPU数量
    for i in range(torch.cuda.device_count()):
        gpu_str.append(torch.cuda.get_device_capability(i))  # 查看指定GPU容量
        gpu_str.append(torch.cuda.get_device_name(i))  # 查看指定GPU名称
        gpu_str.append(torch.cuda.get_device_properties(i))  # i为第几张卡，显示该卡的详细信息
    return "\n".join([str(s) for s in gpu_str])


def main():
    # read config
    run_dict = {"recommender": "MGCN",
                "data_dir": "dataset_mm/elec",
                "file_column": "UIRT",
                "sep": '\t',
                "hyperopt": False,
                "gpu_id": 0,
                "metric": ("Precision", "Recall", "MAP", "NDCG"),
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
    if os.path.exists("unarchived_models"):
        registry.load_skrec_model(model_name, "unarchived_models")

    model_class, config_class = registry.get_model(model_name)
    if not model_class:
        print(f"Recommender '{model_name}' is not found.")

    model_params = {"lr": 1e-3, "epochs": 500}
    # model_params = dict()
    model_params = merge_config_with_cmd_args(model_params)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(run_config.gpu_id)
    # os.environ['ROCR_VISIBLE_DEVICES'] = str(run_config.gpu_id)
    _set_random_seed(run_config.seed)

    hyperopt = HyperOpt(run_config, model_class, config_class, model_params)
    # print(list_gpus())
    hyperopt.run()


if __name__ == "__main__":
    main()
