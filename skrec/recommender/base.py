__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["AbstractRecommender"]


import os
import time
import platform
from typing import Union, List, Tuple
import numpy as np
from ..io import Logger, RSDataset, group_users_by_interactions
from ..utils.py import Config, slugify
from ..utils.py import MetricReport
from ..run_config import RunConfig
from ..utils.py import RankingEvaluator


class AbstractRecommender(object):
    def __init__(self, run_config: RunConfig, model_config: Config):
        self.dataset = RSDataset(run_config.data_dir, run_config.sep, run_config.file_column)
        self.logger: Logger = self._create_logger(self.dataset, model_config)
        self.dataset.set_logger(self.logger)
        self.evaluator = RankingEvaluator(self.dataset.train_data.to_user_dict(),
                                          self.dataset.test_data.to_user_dict(),
                                          metric=run_config.metric, top_k=run_config.top_k,
                                          batch_size=run_config.test_batch_size,
                                          num_thread=run_config.test_thread)

        self._user_groups = group_users_by_interactions(self.dataset)

    def _create_logger(self, dataset: RSDataset, config: Config) -> Logger:
        timestamp = time.time()
        model_name = self.__class__.__name__
        data_name = dataset.data_name

        param_str = f"{data_name}_{model_name}_{config.to_string('_')}"
        param_str = slugify(param_str, max_length=255 - 100)
        # run_id: data_name, model_name, hyper-parameters, timestamp
        run_id = f"{param_str}_{timestamp:.8f}"

        log_dir = os.path.join("log", dataset.data_dir, self.__class__.__name__)
        logger_name = os.path.join(log_dir, run_id + ".log")
        logger = Logger(logger_name)

        # show basic information
        logger.info(f"Server:\t{platform.node()}")
        logger.info(f"Workspace:\t{os.getcwd()}")
        logger.info(f"PID:\t{os.getpid()}")
        logger.info(f"Model:\t{self.__class__.__module__}")

        logger.info(f"\n{dataset.statistic_info}")
        cfg_str = config.to_string('\n')
        logger.info(f"\nHyper-parameters:\n{cfg_str}\n")

        return logger

    def fit(self):
        raise NotImplementedError

    def evaluate(self, test_users=None) -> MetricReport:
        raise NotImplementedError

    def evaluate_group(self) -> List[Tuple[str, MetricReport]]:
        group_results = []
        for user_group in self._user_groups:
            results = self.evaluate(user_group.users)
            group_results.append((user_group.label, results))
        return group_results

    def predict(self, users: Union[List[int], np.ndarray]) -> np.ndarray:
        raise NotImplementedError
