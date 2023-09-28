from typing import Dict
from copy import deepcopy
import os
import time
import logging
import platform
from skrec.recommender.base import AbstractRecommender
from skrec.utils.py import MetricReport, EarlyStopping
from skrec import RunConfig
from skrec.utils.py import slugify
from hyperopt import fmin, tpe, hp, Trials, space_eval
import json
from skrec.io import Logger, RSDataset
old_handlers = logging.root.handlers[:]

__all__ = ["HyperOpt"]


class HyperOpt(object):
    def __init__(self, run_config: RunConfig, model_class, config_class, fixed_params):
        run_config.hyperopt = run_config.hyperopt and bool(config_class.param_space())
        self._run_config = run_config
        self._model_class = model_class
        self._config_class = config_class
        self._fixed_params = fixed_params
        self._current_model: AbstractRecommender = None
        self._best_trial = None
        if run_config.hyperopt is False:
            return

        self._param_space = {key: hp.choice(key, value) for key, value in config_class.param_space().items()}
        self._num_combos = config_class.num_combos()
        self._patience = max(int(self._num_combos / 2), 10)
        self._early_stopping = EarlyStopping(metric="NDCG@10", patience=self._patience)

        self._dataset = RSDataset(run_config.data_dir, run_config.sep, run_config.file_column)
        self.logger = self._create_logger()

    def _create_logger(self):
        timestamp = time.time()
        param_str = f"{self._dataset.data_name}_{self._model_class.__name__}"
        param_str = slugify(param_str, max_length=255 - 100)
        # run_id: data_name, model_name, timestamp
        run_id = f"hyperopt_{param_str}_{timestamp:.8f}"

        log_dir = os.path.join("log", self._dataset.data_dir, self._model_class.__name__)
        logger_name = os.path.join(log_dir, run_id + ".log")
        logger = Logger(logger_name)

        # show basic information
        logger.info(f"Task: Tune Hyper-Parameters")
        logger.info(f"Server:\t{platform.node()}")
        logger.info(f"Workspace:\t{os.getcwd()}")
        logger.info(f"PID:\t{os.getpid()}")
        logger.info(f"Model:\t{self._model_class.__module__}")
        logger.info(f"Dataset:\t{os.path.abspath(self._dataset.data_dir)}")
        logger.info("Hyper-Parameters Info:\t" + json.dumps(self._config_class.param_space()))
        logger.info("")

        return logger

    @property
    def fixed_params(self) -> Dict:
        return deepcopy(self._fixed_params)

    def run(self):
        if self._run_config.hyperopt is True:
            trials = Trials()
            self.logger.info(f"Early stopping patience:\t{self._patience}")
            self.logger.info(f"fmin max evals count:\t{self._num_combos}")
            best = fmin(fn=self.objective, space=self._param_space, algo=tpe.suggest,
                        max_evals=self._num_combos, trials=trials,
                        early_stop_fn=self.early_stop_fn, verbose=False)
            self.logger.info("Best params:\t" + json.dumps(space_eval(self._param_space, best), default=str))
            self.logger.info("\n\nBest results:")
            self.logger.info(self.trial2value(self._best_trial))
            self.logger.info("\nDetailed results:\n" + json.dumps(self._early_stopping.best_result.results, default=str))
        else:
            model: AbstractRecommender = self._model_class(self._run_config, self.fixed_params)
            logging.root.handlers = old_handlers
            model.fit()

    def objective(self, hp_params) -> float:
        model_params = self.fixed_params
        model_params.update(hp_params)
        self._current_model: AbstractRecommender = self._model_class(self._run_config, model_params)
        logging.root.handlers = old_handlers
        result: MetricReport = self._current_model.fit()
        loss = -result[self._early_stopping.key_metric]
        if self._early_stopping(result):
            return -10.0+loss  # the value of metrics in IR cannot greater than 1.0.
        else:
            return loss

    def early_stop_fn(self, trials: Trials):
        latest = trials.trials[-1]
        if len(trials.trials) == 1:
            self.logger.info(self.trial2title(latest))
        self.logger.info(self.trial2value(latest))
        stopped = latest["result"]["loss"] < -1.01
        if not stopped:
            self._best_trial = trials.best_trial
        return stopped, []  # the value of metrics in IR cannot greater than 1.0.

    def trial2title(self, trial: Dict) -> str:
        titles = ["tid"]
        param_dict = self.get_real_params(trial)
        titles.extend(param_dict.keys())
        titles.extend(["loss", "book_time", "refresh_time", "log_file"])
        return '\t'.join([f"{v}".ljust(20) for v in titles])

    def trial2value(self, trial: Dict) -> str:
        values = [trial["tid"]]
        param_dict = self.get_real_params(trial)
        values.extend(param_dict.values())
        values.append(trial["result"]["loss"])
        values.append(trial["book_time"])
        values.append(trial["refresh_time"])
        values.append(os.path.basename(self._current_model.logger.logger_name))

        return '\t'.join([f"{v}".ljust(20) for v in values])

    def get_real_params(self, trial: Dict) -> Dict:
        vals = trial["misc"]["vals"]
        # unpack the one-element lists to values
        # and skip over the 0-element lists
        rval = {}
        for k, v in list(vals.items()):
            if v:
                rval[k] = v[0]
        return space_eval(self._param_space, rval)
