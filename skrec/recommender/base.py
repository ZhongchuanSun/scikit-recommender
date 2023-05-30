__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["AbstractRecommender"]


import os
import time
from typing import Union, List
import numpy as np
from ..io import Logger, Dataset
from ..utils.py import Config, slugify
from ..utils.py import MetricReport


class AbstractRecommender(object):
    def __init__(self, dataset: Dataset, config: Config):
        self.logger: Logger = self._create_logger(dataset, config)

    def _create_logger(self, dataset: Dataset, config: Config) -> Logger:
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
        # TODO how to early stop fitting
        raise NotImplementedError

    def evaluate(self) -> MetricReport:
        raise NotImplementedError

    def predict(self, users: Union[List[int], np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def is_early_stop(self, new_result: MetricReport, cmp_metric: str="NDCG@10", stop_epochs: int=50) -> bool:
        if not hasattr(self, "xx_best_result_"):
            self.xx_best_result_: MetricReport = new_result
            self.xx_stop_counter_ = 0

        self.xx_stop_counter_ += 1
        if 0 < stop_epochs < self.xx_stop_counter_:
            self.logger.info("early stop")
            return True

        if new_result[cmp_metric] > self.xx_best_result_[cmp_metric]:
            self.xx_best_result_ = new_result
            self.xx_stop_counter_ = 0

        return False

    @property
    def best_result(self) -> MetricReport:
        if hasattr(self, "xx_best_result_"):
            return self.xx_best_result_
        else:
            return MetricReport(["None"], [0])
