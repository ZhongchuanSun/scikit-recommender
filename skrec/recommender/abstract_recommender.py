__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["AbstractRecommender"]

from reckit import Logger
from reckit import Configurator
from reckit import Evaluator
from reckit import typeassert
from data import Dataset
import abc
import time
import os


@typeassert(config=Configurator, data_name=str)
def _create_logger(config, data_name):
    # create a logger
    timestamp = time.time()
    model_name = config.recommender
    param_str = f"{data_name}_{model_name}_{config.summarize()}"
    run_id = f"{param_str[:150]}_{timestamp:.8f}"

    log_dir = os.path.join("log", data_name, model_name)
    logger_name = os.path.join(log_dir, run_id + ".log")
    logger = Logger(logger_name)

    return logger


class AbstractRecommender(object):
    def __init__(self, config):
        self.dataset = Dataset(config.data_dir, config.sep, config.file_column)

        user_train_dict = self.dataset.train_data.to_user_dict()
        user_test_dict = self.dataset.test_data.to_user_dict()
        self.evaluator = Evaluator(user_train_dict, user_test_dict,
                                   metric=config.metric, top_k=config.top_k,
                                   batch_size=config.test_batch_size,
                                   num_thread=config.test_thread)

    def fit(self):
        # TODO 如何判停?
        raise NotImplementedError

    def predict(self, users):
        raise NotImplementedError
