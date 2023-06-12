__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["ImplicitFeedback", "KnowledgeGraph", "CFDataset", "KGDataset"]


import os
import pickle
import warnings
from typing import Dict, Callable, List
from copy import deepcopy
import functools
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy.sparse as sp
import atexit
from ..utils.py import pad_sequences
from ..utils.common import PostInitMeta

_USER = "user"
_ITEM = "item"
_RATING = "rating"
_TIME = "time"
_DColumns = {"UI": [_USER, _ITEM],
             "UIR": [_USER, _ITEM, _RATING],
             "UIT": [_USER, _ITEM, _TIME],
             "UIRT": [_USER, _ITEM, _RATING, _TIME]
             }

_HEAD = "head"
_TAIL = "tail"
_RELATION = "relation"


class DataCacheABC(object):
    def __init__(self):
        self._cache_buffer = dict()
        self._cache_modified = False

    def _is_cached(self, key) -> bool:
        return key in self._cache_buffer

    def _get_from_cache(self, key):
        return deepcopy(self._cache_buffer[key])

    def _set_to_cache(self, key, value):
        self._cache_buffer[key] = value
        self._cache_modified = True

    def clear_cache(self):
        self._cache_buffer.clear()
        self._cache_modified = True

    def loads_cached_data(self, cached_data: Dict):
        self._cache_buffer = deepcopy(cached_data)

    def dumps_cached_data(self) -> Dict:
        return deepcopy(self._cache_buffer)

    def is_cache_modified(self) -> bool:
        return self._cache_modified


def data_cache(func):
    # read from buffer
    @functools.wraps(func)
    def wrapper(self: DataCacheABC, *args, **kwargs):
        # Generate a cache key based on the function name and arguments
        cache_key = pickle.dumps((func.__name__, args, kwargs))

        if self._is_cached(cache_key):
            # If the cached result exists, retrieve and return it
            return self._get_from_cache(cache_key)
        else:
            # Otherwise, process the data and store the result in the cache
            result = func(self, *args, **kwargs)
            self._set_to_cache(cache_key, result)
            return result

    return wrapper


class ImplicitFeedback(DataCacheABC):
    def __init__(self, data: pd.DataFrame=None, num_users: int=None, num_items: int=None):
        super().__init__()
        assert data is None or isinstance(data, pd.DataFrame)

        if data is None or data.empty:
            self._data = pd.DataFrame()
            self.num_users = 0
            self.num_items = 0
            self.num_ratings = 0
        else:
            self._data = data
            self.num_users = num_users if num_users is not None else max(data[_USER]) + 1
            self.num_items = num_items if num_items is not None else max(data[_ITEM]) + 1
            self.num_ratings = len(data)

    def is_empty(self) -> bool:
        return self._data is None or self._data.empty

    @data_cache
    def to_user_item_pairs(self) -> np.ndarray:
        ui_pairs = self._data[[_USER, _ITEM]].to_numpy(copy=True, dtype=np.int32)
        return ui_pairs

    @data_cache
    def to_user_item_pairs_by_time(self) -> np.ndarray:
        if _TIME not in self._data:
            raise ValueError("This dataset do not contain timestamp.")
        data_uit = self._data[[_USER, _ITEM, _TIME]]
        data_uit = data_uit.sort_values(by=["user", "time"], inplace=False)
        data_ui = data_uit[[_USER, _ITEM]].to_numpy(copy=True, dtype=np.int32)
        return data_ui

    @data_cache
    def to_csr_matrix(self) -> sp.csr_matrix:
        users, items = self._data[_USER].to_numpy(), self._data[_ITEM].to_numpy()
        ratings = np.ones(len(users), dtype=np.float32)
        csr_mat = sp.csr_matrix((ratings, (users, items)), shape=(self.num_users, self.num_items), copy=True)
        return csr_mat

    @data_cache
    def to_csc_matrix(self) -> sp.csc_matrix:
        return self.to_csr_matrix().tocsc()

    @data_cache
    def to_dok_matrix(self) -> sp.dok_matrix:
        return self.to_csr_matrix().todok()

    @data_cache
    def to_coo_matrix(self) -> sp.coo_matrix:
        return self.to_csr_matrix().tocoo()

    @data_cache
    def to_user_dict(self) -> Dict[int, np.ndarray]:
        user_dict = OrderedDict()
        user_grouped = self._data.groupby(_USER)
        for user, user_data in user_grouped:
            user_dict[user] = user_data[_ITEM].to_numpy(dtype=np.int32)
        return user_dict

    @data_cache
    def to_user_dict_by_time(self) -> Dict[int, np.ndarray]:
        # in chronological
        if _TIME not in self._data:
            raise ValueError("This dataset do not contain timestamp.")

        user_dict = OrderedDict()
        user_grouped = self._data.groupby(_USER)
        for user, user_data in user_grouped:
            user_data = user_data.sort_values(by=[_TIME])
            user_dict[user] = user_data[_ITEM].to_numpy(dtype=np.int32)

        return user_dict

    @data_cache
    def to_item_dict(self) -> Dict[int, np.ndarray]:
        item_dict = OrderedDict()
        item_grouped = self._data.groupby(_ITEM)
        for item, item_data in item_grouped:
            item_dict[item] = item_data[_USER].to_numpy(dtype=np.int32)

        return item_dict

    @data_cache
    def to_truncated_seq_dict(self, max_len: int, pad_value: int=0,
                              padding='pre', truncating='pre') -> Dict[int, np.ndarray]:
        user_seq_dict = self.to_user_dict_by_time()
        if max_len is None:
            max_len = max([len(seqs) for seqs in user_seq_dict.values()])
        item_seq_list = [item_seq[-max_len:] for item_seq in user_seq_dict.values()]
        item_seq_arr = pad_sequences(item_seq_list, value=pad_value, max_len=max_len,
                                     padding=padding, truncating=truncating, dtype=np.int32)

        seq_dict = OrderedDict([(user, item_seq) for user, item_seq in
                                zip(user_seq_dict.keys(), item_seq_arr)])
        return seq_dict

    def __len__(self):
        return len(self._data)


class KnowledgeGraph(DataCacheABC):
    def __init__(self, data: pd.DataFrame=None, num_entities: int=None, num_relations: int=None):
        super().__init__()
        assert data is None or isinstance(data, pd.DataFrame)

        if data is None or data.empty:
            self._data = pd.DataFrame()
            self.num_entities = 0
            self.num_relations = 0
            self.num_triplets = 0
        else:
            self._data = data
            self.num_entities = num_entities if num_entities is not None else max(max(data[_HEAD]), max(data[_TAIL])) + 1
            self.num_relations = num_relations if num_relations is not None else max(data[_RELATION]) + 1
            self.num_triplets = len(data)

    def is_empty(self) -> bool:
        return self._data is None or self._data.empty

    @data_cache
    def to_triplets(self) -> np.ndarray:
        triplets = self._data[[_HEAD, _RELATION, _TAIL]].to_numpy(copy=True, dtype=np.int32)
        return triplets

    @data_cache
    def to_head_dict(self) -> Dict[int, Dict[str, np.ndarray]]:
        head_dict = OrderedDict()
        head_grouped = self._data.groupby(_HEAD)
        for head, head_data in head_grouped:
            head_dict[head] = {_RELATION: head_data[_RELATION].to_numpy(dtype=np.int32),
                               _TAIL: head_data[_TAIL].to_numpy(dtype=np.int32)}
        return head_dict

    @data_cache
    def to_tail_dict(self) -> Dict[int, Dict[str, np.ndarray]]:
        tail_dict = OrderedDict()
        tail_grouped = self._data.groupby(_TAIL)
        for tail, tail_data in tail_grouped:
            tail_dict[tail] = {_RELATION: tail_data[_RELATION].to_numpy(dtype=np.int32),
                               _HEAD: tail_data[_HEAD].to_numpy(dtype=np.int32)}
        return tail_dict

    @data_cache
    def to_relation_dict(self) -> Dict[int, Dict[str, np.ndarray]]:
        rel_dict = OrderedDict()
        rel_grouped = self._data.groupby(_RELATION)
        for rel, rel_data in rel_grouped:
            rel_dict[rel] = {_HEAD: rel_data[_HEAD].to_numpy(dtype=np.int32),
                             _TAIL: rel_data[_TAIL].to_numpy(dtype=np.int32)}
        return rel_dict

    @data_cache
    def to_csr_matrix_dict(self) -> Dict[int, sp.csr_matrix]:
        rel_csr_dict = OrderedDict()
        rel_dict = self.to_relation_dict()
        for rel, data in rel_dict.items():
            heads, tails = data[_HEAD], data[_TAIL]
            pass
            # users, items = self._data[_USER].to_numpy(), self._data[_ITEM].to_numpy()
            ones = np.ones(len(heads), dtype=np.float32)
            csr_mat = sp.csr_matrix((ones, (heads, tails)), shape=(self.num_entities, self.num_entities), copy=True)
            rel_csr_dict[rel] = csr_mat
        return rel_csr_dict

    @data_cache
    def to_coo_matrix_dict(self) -> Dict[int, sp.coo_matrix]:
        rel_coo_dict = OrderedDict()
        rel_csr_dict = self.to_csr_matrix_dict()
        for rel, data in rel_csr_dict.items():
            rel_coo_dict[rel] = data.tocsc()
        return rel_coo_dict


class SocialNetwork(DataCacheABC):
    # TODO
    pass


class CFDataset(metaclass=PostInitMeta):
    def __init__(self, data_dir, sep, columns):
        """CFDataset

        Notes:
            The prefix name of data files is same as the data_dir, and the
            suffix/extension names are 'train', 'test', 'user2id', 'item2id'.
            Directory structure:
                data_dir
                    ├── data_dir.train      // training data
                    ├── data_dir.valid      // validation data, optional
                    ├── data_dir.test       // test data
                    ├── data_dir.user2id    // user to id, optional
                    ├── data_dir.item2id    // item to id, optional

        Args:
            data_dir: The directory of dataset.
            sep: The separator/delimiter of file columns.
            columns: The format of columns, must be one of 'UI',
                'UIR', 'UIT' and 'UIRT'
        """

        self._data_dir = data_dir
        self._cache_file = os.path.join(self.data_dir, "_cache_" + self.data_name + ".bin")
        self._load_cf_data(sep, columns)

    def __post_init__(self):
        self._restore_cached_data()
        atexit.register(self._save_cached_data)

    @property
    def data_name(self):
        return os.path.split(self.data_dir)[-1]

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def _file_prefix(self):
        return os.path.join(self.data_dir, self.data_name)

    @staticmethod
    def _read_csv(csv_file, sep, header, names, handle: Callable=lambda x: x):
        if os.path.isfile(csv_file):
            csv_data = pd.read_csv(csv_file, sep=sep, header=header, names=names)
        else:
            handle(f"'{csv_file}' does not exist.")
            csv_data = pd.DataFrame()
        return csv_data

    @staticmethod
    def _read_map_file(map_file, sep):
        if os.path.isfile(map_file):
            maps = pd.read_csv(map_file, sep=sep, header=None).to_numpy()
            maps = OrderedDict(maps)
            reverses = OrderedDict([(second, first) for first, second in maps.items()])
        else:
            maps = None
            reverses = None
            warnings.warn(f"'{map_file}' does not exist.")
        return maps, reverses

    def _load_cf_data(self, sep, columns):
        # Load collaborative filtering model data
        if columns not in _DColumns:
            key_str = ", ".join(_DColumns.keys())
            raise ValueError("'columns' must be one of '%s'." % key_str)

        columns = _DColumns[columns]

        # load data
        def raise_error(err: str): raise FileNotFoundError(err)
        _train_data = self._read_csv(self._file_prefix + ".train", sep=sep, names=columns,
                                     header=None, handle=raise_error)
        _valid_data = self._read_csv(self._file_prefix + ".valid", sep=sep, names=columns,
                                     header=None, handle=warnings.warn)
        _test_data = self._read_csv(self._file_prefix + ".test", sep=sep, names=columns,
                                    header=None, handle=raise_error)

        if _train_data.isnull().values.any():
            warnings.warn(f"'Training data has None value, please check the file or the separator.")
        if _valid_data.isnull().values.any():
            warnings.warn(f"'Validation data has None value, please check the file or the separator.")
        if _test_data.isnull().values.any():
            warnings.warn(f"'Test data has None value, please check the file or the separator.")

        self.user2id, self.id2user = self._read_map_file(self._file_prefix + ".user2id", sep)
        self.item2id, self.id2item = self._read_map_file(self._file_prefix + ".item2id", sep)

        # statistical information
        data_info = [(max(data[_USER]), max(data[_ITEM]), len(data))
                     for data in [_train_data, _valid_data, _test_data] if not data.empty]
        self.num_users = max([d[0] for d in data_info]) + 1
        self.num_items = max([d[1] for d in data_info]) + 1
        self.num_ratings = sum([d[2] for d in data_info])

        # convert to to the object of ImplicitFeedback
        self.train_data = ImplicitFeedback(_train_data, num_users=self.num_users, num_items=self.num_items)
        self.valid_data = ImplicitFeedback(_valid_data, num_users=self.num_users, num_items=self.num_items)
        self.test_data = ImplicitFeedback(_test_data, num_users=self.num_users, num_items=self.num_items)

    @property
    def statistic_info(self):
        """The statistic of dataset.

        Returns:
            str: The summary of statistic information
        """
        if 0 in {self.num_users, self.num_items, self.num_ratings}:
            return ""
        else:
            num_users, num_items = self.num_users, self.num_items
            num_ratings = self.num_ratings
            sparsity = 1 - 1.0 * num_ratings / (num_users * num_items)

            statistic = ["Dataset statistic information:",
                         "Name: %s" % self.data_name,
                         f"The number of users: {num_users}",
                         f"The number of items: {num_items}",
                         f"The number of ratings: {num_ratings}",
                         f"Average actions of users: {(1.0 * num_ratings / num_users):.2f}",
                         f"Average actions of items: {(1.0 * num_ratings / num_items):.2f}",
                         f"The sparsity of the dataset: {(sparsity * 100):.6f}%%",
                         "",
                         f"The number of training: {len(self.train_data)}",
                         f"The number of validation: {len(self.valid_data)}",
                         f"The number of testing: {len(self.test_data)}"
                         ]

            statistic = "\n".join(statistic)
            return statistic

    def _read_from_cache_file(self):
        cache_data = dict()
        try:
            with open(self._cache_file, 'rb') as fin:
                cache_data = pickle.load(fin)
        except Exception as e:
            warnings.warn(f"_read_cache_file error: {e}")
        return cache_data

    def _write_to_cache_file(self, cache_data):
        try:
            with open(self._cache_file, 'wb') as fout:
                pickle.dump(cache_data, fout)
        except Exception as e:
            warnings.warn(f"_write_to_cache_file error: {e}")

    def _update_cache_file(self, new_caches):
        cache_data = dict()
        if os.path.exists(self._cache_file):
            cache_data = self._read_from_cache_file()

        cache_data.update(new_caches)
        self._write_to_cache_file(cache_data)

    def _save_cached_data(self):
        if not self._is_cache_modified():
            return

        _t_data = self._dumps_cached_data()
        # save cached data
        self._update_cache_file(_t_data)

    def _restore_cached_data(self):
        if self._is_data_updated():
            return
        # restore cached data
        _t_data = self._read_from_cache_file()
        try:
            self._loads_cached_data(_t_data)
        except Exception as e:
            warnings.warn(f"_restore_cached_data error: {e}")

    def _is_cache_modified(self) -> bool:
        return self.train_data.is_cache_modified() or \
               self.valid_data.is_cache_modified() or \
               self.test_data.is_cache_modified()

    def _is_data_updated(self) -> bool:
        if not os.path.exists(self._cache_file):
            return True
        cached_time = os.path.getmtime(self._cache_file)

        for file_suffix in [".train", ".test", ".valid"]:
            filename = self._file_prefix + file_suffix
            if os.path.exists(filename) and os.path.getmtime(filename) > cached_time:
                return True
        return False

    def _dumps_cached_data(self):
        _t_data = dict()
        _t_data["train_data"] = self.train_data.dumps_cached_data()
        _t_data["test_data"] = self.test_data.dumps_cached_data()
        _t_data["valid_data"] = self.valid_data.dumps_cached_data()
        return _t_data

    def _loads_cached_data(self, _t_data):
        # load cached data
        try:
            self.train_data.loads_cached_data(_t_data["train_data"])
            self.test_data.loads_cached_data(_t_data["test_data"])
            self.valid_data.loads_cached_data(_t_data["valid_data"])
        except Exception as e:
            warnings.warn(f"_loads_cached_data error: {e}")


class KGDataset(CFDataset):
    def __init__(self, data_dir, sep, columns):
        super().__init__(data_dir, sep, columns)
        self._load_kg_data(sep)

    def _load_kg_data(self, sep):
        # Load knowledge graph data
        def raise_error(err: str): raise FileNotFoundError(err)
        _kg_data = self._read_csv(self._file_prefix + ".kg", sep=sep, names=[_HEAD, _RELATION, _TAIL],
                                  header=None, handle=raise_error)
        if _kg_data.isnull().values.any():
            warnings.warn(f"'Knowledge graph data has None value, please check the file or the separator.")
        _kg_data = _kg_data.drop_duplicates()

        self.kg_data = KnowledgeGraph(_kg_data)
        self.num_entities = self.kg_data.num_entities
        self.num_relations = self.kg_data.num_relations
        self.num_triplets = self.kg_data.num_triplets

    @property
    def statistic_info(self):
        cf_info = super().statistic_info
        statistic = ["",
                     f"The number of entities: {self.num_entities}",
                     f"The number of relations: {self.num_relations}",
                     f"The number of triplets: {self.num_triplets}"
                     ]
        kg_info = "\n".join(statistic)

        return cf_info + kg_info

    def _is_cache_modified(self) -> bool:
        return super()._is_cache_modified() or self.kg_data.is_cache_modified()

    def _is_data_updated(self) -> bool:
        if super()._is_data_updated():
            return True
        cached_time = os.path.getmtime(self._cache_file)
        kg_time = os.path.getmtime(self._file_prefix + ".kg")
        return kg_time > cached_time

    def _dumps_cached_data(self):
        _t_data = super()._dumps_cached_data()
        _t_data["kg_data"] = self.kg_data.dumps_cached_data()
        return _t_data

    def _loads_cached_data(self, _t_data):
        # load cached data
        super()._loads_cached_data(_t_data)
        try:
            self.kg_data.loads_cached_data(_t_data["kg_data"])
        except Exception as e:
            warnings.warn(f"_loads_cached_data error: {e}")


class SocialDataset(CFDataset):
    # TODO
    pass
