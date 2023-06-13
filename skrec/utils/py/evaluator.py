__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["MetricReport", "RankingEvaluator", "EarlyStopping"]

from typing import Sequence, Dict, Union, Optional, Tuple, List, Iterable
from collections import OrderedDict
import numpy as np
import itertools
from colorama import Fore, Style
from .batch_iterator import BatchIterator
from .cython import eval_score_matrix

_text_colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]


class MetricReport(object):
    def __init__(self, metrics: Sequence[str], values: Sequence[float]):
        assert len(metrics) == len(values), f"The lengths of metrics and values " \
                                            f"are not equal ({len(metrics)}!={len(values)})."
        self._results = OrderedDict(zip(metrics, values))

    def metrics(self):
        return self._results.keys()

    @property
    def metrics_str(self) -> str:
        _colors = itertools.cycle(_text_colors)
        return '\t'.join([c+f"{m}".ljust(12)+Style.RESET_ALL
                          for c, m in zip(_colors, self.metrics())])

    def values(self):
        return self._results.values()

    @property
    def values_str(self) -> str:
        _colors = itertools.cycle(_text_colors)
        return '\t'.join([c+f"{v:.8f}".ljust(12)+Style.RESET_ALL
                          for c, v in zip(_colors, self.values())])

    def items(self):
        return self._results.items()

    def __getitem__(self, item):
        if item not in self._results:
            raise KeyError(item)
        return self._results[item]

    def __str__(self):
        return self._results.__str__()


_metric2id = {"Precision": 1, "Recall": 2, "MAP": 3, "NDCG": 4, "MRR": 5}
_id2metric = {value: key for key, value in _metric2id.items()}


class RankingEvaluator(object):
    """Evaluator for item ranking task.

    Evaluation metrics of `Evaluator` are configurable and can
    automatically fit both leave-one-out and fold-out data splitting
    without specific indication:

    * **First**, evaluation metrics of this class are configurable via the
      argument `metric`. Now there are five configurable metrics: `Precision`,
      `Recall`, `MAP`, `NDCG` and `MRR`.

    * **Second**, this class and its evaluation metrics can automatically fit
      both leave-one-out and fold-out data splitting without specific indication.

      In **leave-one-out** evaluation:
        1) `Recall` is equal to `HitRatio`;
        2) The implementation of `NDCG` is compatible with fold-out;
        3) `MAP` and `MRR` have same numeric values;
        4) `Precision` is meaningless.
    """

    def __init__(self, user_train_dict: Optional[Dict[int, np.ndarray]],
                 user_test_dict: Dict[int, np.ndarray],
                 metric: Union[None, str, Tuple[str], List[str]]=None,
                 top_k: Union[int, List[int], Tuple[int]]=50,
                 batch_size: int=256, num_thread: int=8):
        """Initializes a new `Evaluator` instance.

        Args:
            user_train_dict (dict, None): Each key is user ID and the corresponding
                value is the list of **training items**.
            user_test_dict (dict): Each key is user ID and the corresponding
                value is the list of **test items**.
            metric (None or list of str): If `metric == None`, metric will
                be set to `["Precision", "Recall", "MAP", "NDCG", "MRR"]`.
                Otherwise, `metric` must be one or a sublist of metrics
                mentioned above. Defaults to `None`.
            top_k (int or list of int): `top_k` controls the Top-K item ranking
                performance. If `top_k` is an integer, K ranges from `1` to
                `top_k`; If `top_k` is a list of integers, K are only assigned
                these values. Defaults to `50`.
            batch_size (int): An integer to control the test batch size.
                Defaults to `1024`.
            num_thread (int): An integer to control the test thread number.
                Defaults to `8`.

        Raises:
             ValueError: If `metric` or one of its element is invalid.
        """
        super(RankingEvaluator, self).__init__()
        if metric is None:
            metric = ["Precision", "Recall", "MAP", "NDCG", "MRR"]
        elif isinstance(metric, str):
            metric = [metric]
        elif isinstance(metric, (tuple, list)):
            metric = list(metric)
        else:
            raise TypeError("The type of 'metric' (%s) is invalid!" % metric.__class__.__name__)

        for m in metric:
            assert m in _metric2id, f"'{metric}' is not in ('Precision', 'Recall', 'MAP', 'NDCG', 'MRR')."

        self.user_pos_train = dict()
        self.user_pos_test = dict()
        self.set_train_data(user_train_dict)
        self.set_test_data(user_test_dict)

        self.metrics_num = len(metric)
        self.metrics = [_metric2id[m] for m in metric]
        self.num_thread = num_thread
        self.batch_size = batch_size

        if isinstance(top_k, int):
            self.max_top = top_k
            self.top_show = np.arange(top_k) + 1
        else:
            self.max_top = max(top_k)
            self.top_show = np.sort(top_k)

    def set_train_data(self, user_train_dict: Optional[Dict[int, np.ndarray]]=None):
        self.user_pos_train = user_train_dict if user_train_dict is not None else dict()

    def set_test_data(self, user_test_dict: Dict[int, np.ndarray]):
        assert len(user_test_dict) > 0, "'user_test_dict' can be empty."
        self.user_pos_test = user_test_dict

    @property
    def metrics_list(self) -> List[str]:
        return [f"{_id2metric[mid]}@{str(k)}" for mid in self.metrics for k in self.top_show]

    @property
    def metrics_str(self) -> str:
        """Get all metrics information.

        Returns:
            str: A string consist of all metrics information, such as
                `"Precision@10    Precision@20    NDCG@10    NDCG@20"`.
        """
        _colors = itertools.cycle(_text_colors)
        return '\t'.join([c + f"{m}".ljust(12) + Style.RESET_ALL
                          for c, m in zip(_colors, self.metrics_list)])

    def evaluate(self, model, test_users: Optional[Iterable[int]]=None) -> MetricReport:
        """Evaluate `model`.

        Args:
            model: The model need to be evaluated. This model must have
                a method `predict(self, users)`, where the argument
                `users` is a list of users and the return is a 2-D array that
                contains `users` rating/ranking scores on all items.
            test_users: The users will be used to test.
                Default is None and means test all users in user_pos_test.

        Returns:
            str: A single-line string consist of all results, such as
                `"0.18663847    0.11239596    0.35824192    0.21479650"`.
        """
        # B: batch size
        # N: the number of items
        assert hasattr(model, "predict"), "the model must have attribute 'predict'."
        if test_users is not None:
            test_users = [u for u in test_users if u in self.user_pos_test]
        else:
            test_users = list(self.user_pos_test.keys())

        assert isinstance(test_users, Iterable), "'test_user' must be iterable."

        test_users = BatchIterator(test_users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        bat_results = []
        for batch_users in test_users:
            test_items = [self.user_pos_test[u] for u in batch_users]
            ranking_score = model.predict(batch_users)  # (B,N)
            assert isinstance(ranking_score, np.ndarray), "'ranking_score' must be an np.ndarray"

            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            for idx, user in enumerate(batch_users):
                if user in self.user_pos_train and len(self.user_pos_train[user]) > 0:
                    train_items = self.user_pos_train[user]
                    ranking_score[idx][train_items] = -np.inf

            result = eval_score_matrix(ranking_score, test_items, self.metrics,
                                       top_k=self.max_top, thread_num=self.num_thread)  # (B,k*metric_num)
            bat_results.append(result)

        # concatenate the batch results to a matrix
        all_results = np.concatenate(bat_results, axis=0)  # (num_users, metrics_num*max_top)
        final_results = np.mean(all_results, axis=0)  # (1, metrics_num*max_top)

        final_results = np.reshape(final_results, newshape=[self.metrics_num, self.max_top])  # (metrics_num, max_top)
        final_results = final_results[:, self.top_show - 1]

        final_results = np.reshape(final_results, newshape=[-1])
        return MetricReport(self.metrics_list, final_results)


class EarlyStopping:
    def __init__(self, metric: str="NDCG@10", patience: int=100):
        self._metric: str = metric
        self._patience: int = patience  # Number of epochs to wait for improvement
        self._best_score: MetricReport = None  # Current best score of the monitored metric
        self._counter: int = 0  # Counter for the number of epochs without improvement

    def __call__(self, val_result: MetricReport):
        if self._best_score is None:
            self._best_score = val_result
        elif val_result[self._metric] <= self._best_score[self._metric]:
            self._counter += 1
            if self._counter >= self._patience > 0:
                return True  # Trigger the stop condition for training
        else:
            self._best_score = val_result
            self._counter = 0

        return False  # Continue training

    @property
    def best_result(self) -> MetricReport:
        if self._best_score is not None:
            return self._best_score
        else:
            return MetricReport(["None"], [0])
