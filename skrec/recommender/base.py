__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["AbstractRecommender"]


import os
import time
from typing import Union, List, Dict, Callable
import numpy as np
from ..io import Logger, CFDataset
from ..utils.py import Config, slugify
from ..utils.py import MetricReport
from ..run_config import RunConfig
from ..utils.py import RankingEvaluator


class AbstractRecommender(object):
    def __init__(self, run_config: Dict, model_config: Dict):
        run_config = RunConfig(**run_config)
        self.config = self.config_class(**model_config)
        self.dataset = CFDataset(run_config.data_dir, run_config.sep, run_config.file_column)
        self.evaluator = RankingEvaluator(self.dataset.train_data.to_user_dict(),
                                          self.dataset.test_data.to_user_dict(),
                                          metric=run_config.metric, top_k=run_config.top_k,
                                          batch_size=run_config.test_batch_size,
                                          num_thread=run_config.test_thread)
        self.logger: Logger = self._create_logger(self.dataset, self.config)

    @property
    def config_class(self) -> Callable:
        raise NotImplementedError

    def _create_logger(self, dataset: CFDataset, config: Config) -> Logger:
        timestamp = time.time()
        model_name = self.__class__.__name__
        data_name = dataset.data_name

        param_str = f"{data_name}_{model_name}_{config.to_string('_')}"
        param_str = slugify(param_str, max_length=255 - 100)
        # run_id: data_name, model_name, hyper-parameters, timestamp
        run_id = f"{param_str}_{timestamp:.8f}"

        log_dir = os.path.join("log", data_name, self.__class__.__name__)
        logger_name = os.path.join(log_dir, run_id + ".log")
        logger = Logger(logger_name)

        # show basic information
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Model: {self.__class__.__module__}")

        logger.info(f"\n{dataset.statistic_info}")
        cfg_str = config.to_string('\n')
        logger.info(f"\nHyper-parameters:\n{cfg_str}\n")

        return logger

    def fit(self):
        raise NotImplementedError

    def evaluate(self) -> MetricReport:
        raise NotImplementedError

    def predict(self, users: Union[List[int], np.ndarray]) -> np.ndarray:
        raise NotImplementedError
