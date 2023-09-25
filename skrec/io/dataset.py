__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["ImplicitFeedback", "KnowledgeGraph", "RSDataset",
           "UserGroup", "group_users_by_interactions"]


import os
import pickle
import warnings
from typing import Dict, Callable, List, Set
from copy import deepcopy
import functools
from collections import OrderedDict, defaultdict
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


def _read_csv(csv_file, sep, header, names, handle: Callable = lambda x: x):
    if os.path.isfile(csv_file):
        csv_data = pd.read_csv(csv_file, sep=sep, header=header, names=names)
    else:
        handle(f"'{csv_file}' does not exist.")
        csv_data = pd.DataFrame()
    return csv_data


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
    def to_set_of_users(self) -> Set[int]:
        return set(self._data[_USER].unique())

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


class DataMeta(object):
    def __init__(self, data_dir, sep, columns):
        self._data_dir = data_dir
        self.sep = sep
        self.columns = columns

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def data_name(self) -> str:
        return os.path.split(self.data_dir)[-1]

    @property
    def file_prefix(self):
        return os.path.join(self.data_dir, self.data_name)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.data_dir, "_data_cache")


class CacheOpt(metaclass=PostInitMeta):
    def __init__(self, cache_file=None):
        self._cache_file: str = cache_file

    def __post_init__(self):
        self._restore_cached_data()  # restore the cached data after initializing the object
        atexit.register(self._save_cached_data)  # dump the cached data before destroying the object

    def _read_from_cache_file(self) -> Dict:
        cache_data = dict()
        try:
            with open(self._cache_file, 'rb') as fin:
                cache_data = pickle.load(fin)
        except Exception as e:
            warnings.warn(f"_read_cache_file error: {e}")
        return cache_data

    def _write_to_cache_file(self, cache_data):
        try:
            _cache_dir = os.path.dirname(self._cache_file)
            if not os.path.exists(_cache_dir):
                os.makedirs(_cache_dir)
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
        raise NotImplementedError

    def _is_data_updated(self) -> bool:
        raise NotImplementedError

    def _dumps_cached_data(self) -> Dict:
        raise NotImplementedError

    def _loads_cached_data(self, _t_data):
        raise NotImplementedError


class CFData(CacheOpt):
    def __init__(self, d_m: DataMeta):
        super().__init__()
        self._d_m = d_m
        self._cache_file = os.path.join(d_m.cache_dir, d_m.data_name + "_cf" + ".bin")
        self._load_cf_data(d_m.sep, d_m.columns)

    def __post_init__(self):
        self._restore_cached_data()  # restore the cached data after initializing the object
        atexit.register(self._save_cached_data)  # dump the cached data before destroying the object

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
            raise ValueError(f"'columns' must be one of '{key_str}'.")

        columns = _DColumns[columns]

        # load data
        def raise_error(err: str): raise FileNotFoundError(err)
        _train_data = _read_csv(self._d_m.file_prefix + ".train", sep=sep, names=columns,
                                header=None, handle=raise_error)
        _valid_data = _read_csv(self._d_m.file_prefix + ".valid", sep=sep, names=columns,
                                header=None, handle=warnings.warn)
        _test_data = _read_csv(self._d_m.file_prefix + ".test", sep=sep, names=columns,
                               header=None, handle=raise_error)

        if _train_data.isnull().values.any():
            warnings.warn(f"'Training data has None value, please check the file or the separator.")
        if _valid_data.isnull().values.any():
            warnings.warn(f"'Validation data has None value, please check the file or the separator.")
        if _test_data.isnull().values.any():
            warnings.warn(f"'Test data has None value, please check the file or the separator.")

        self.user2id, self.id2user = self._read_map_file(self._d_m.file_prefix + ".user2id", sep)
        self.item2id, self.id2item = self._read_map_file(self._d_m.file_prefix + ".item2id", sep)

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
    def statistic_info(self) -> str:
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
                         f"Name: {self._d_m.data_name}",
                         f"Name: {os.path.abspath(self._d_m.data_dir)}",
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

    def _is_cache_modified(self) -> bool:
        return self.train_data.is_cache_modified() or \
               self.valid_data.is_cache_modified() or \
               self.test_data.is_cache_modified()

    def _is_data_updated(self) -> bool:
        if not os.path.exists(self._cache_file):
            return True
        cached_time = os.path.getmtime(self._cache_file)

        for file_suffix in [".train", ".test", ".valid"]:
            filename = self._d_m.file_prefix + file_suffix
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


class KGData(CacheOpt):
    def __init__(self, d_m: DataMeta):
        super().__init__()
        self._d_m = d_m
        self._cache_file = os.path.join(d_m.cache_dir, d_m.data_name + "_kg" + ".bin")
        self._load_kg_data(d_m.sep)

    def _load_kg_data(self, sep):
        # Load knowledge graph data
        def raise_error(err: str): raise FileNotFoundError(err)
        _kg_data = _read_csv(self._d_m.file_prefix + ".kg", sep=sep, names=[_HEAD, _RELATION, _TAIL],
                             header=None, handle=raise_error)
        if _kg_data.isnull().values.any():
            warnings.warn(f"'Knowledge graph data has None value, please check the file or the separator.")
        _kg_data = _kg_data.drop_duplicates()

        self.kg_data = KnowledgeGraph(_kg_data)

    @property
    def statistic_info(self) -> str:
        statistic = ["",
                     f"The number of entities: {self.kg_data.num_entities}",
                     f"The number of relations: {self.kg_data.num_relations}",
                     f"The number of triplets: {self.kg_data.num_triplets}"
                     ]
        kg_info = "\n".join(statistic)

        return kg_info

    def _is_cache_modified(self) -> bool:
        return self.kg_data.is_cache_modified()

    def _is_data_updated(self) -> bool:
        if not os.path.exists(self._cache_file):
            return True
        cached_time = os.path.getmtime(self._cache_file)
        kg_time = os.path.getmtime(self._d_m.file_prefix + ".kg")
        return kg_time > cached_time

    def _dumps_cached_data(self):
        _t_data = dict()
        _t_data["kg_data"] = self.kg_data.dumps_cached_data()
        return _t_data

    def _loads_cached_data(self, _t_data):
        # load cached data
        try:
            self.kg_data.loads_cached_data(_t_data["kg_data"])
        except Exception as e:
            warnings.warn(f"_loads_cached_data error: {e}")


class MMData(object):
    def __init__(self, d_m: DataMeta):
        self._d_m = d_m
        self._load_mm_data()

    @staticmethod
    def _load_npz_features(file_path):
        if os.path.exists(file_path):
            feat_obj = np.load(file_path, allow_pickle=True)
            features = feat_obj[feat_obj.files[0]]
            return features, features.shape[-1]
        else:
            return None, None

    def _load_mm_data(self):
        self.img_features, self.img_dim = self._load_npz_features(self._d_m.file_prefix + ".img.npz")
        self.txt_features, self.txt_dim = self._load_npz_features(self._d_m.file_prefix + ".txt.npz")
        self.audio_features, self.audio_dim = self._load_npz_features(self._d_m.file_prefix + ".audio.npz")

    @property
    def statistic_info(self) -> str:
        statistic = [""]
        if self.img_features is not None:
            statistic.append(f"The shape of image features: {self.img_features.shape}")
        if self.txt_features is not None:
            statistic.append(f"The shape of txt features: {self.txt_features.shape}")
        if self.audio_features is not None:
            statistic.append(f"The shape of audio features: {self.audio_features.shape}")

        mm_info = "\n".join(statistic)

        return mm_info


class SocialData(CacheOpt):
    pass


class RSDataset(DataMeta):
    def __init__(self, data_dir, sep, columns):
        super().__init__(data_dir, sep, columns)
        self._log_print = print

    def set_logger(self, logger):
        self._log_print = logger.info

    @property
    def cf_data(self) -> CFData:
        if not hasattr(self, "_cf_data"):
            _cf_data = CFData(self)
            self._cf_data = _cf_data
            # logging data statistic info
            self._log_print(_cf_data.statistic_info)
        return self._cf_data

    @property
    def train_data(self) -> ImplicitFeedback:
        return self.cf_data.train_data

    @property
    def valid_data(self) -> ImplicitFeedback:
        return self.cf_data.valid_data

    @property
    def test_data(self) -> ImplicitFeedback:
        return self.cf_data.test_data

    @property
    def num_users(self) -> int:
        return self.cf_data.num_users

    @property
    def num_items(self) -> int:
        return self.cf_data.num_items

    @property
    def num_ratings(self) -> int:
        return self.cf_data.num_ratings

    @property
    def kg_data(self) -> KnowledgeGraph:
        if not hasattr(self, "_kg_data"):
            _kg_data = KGData(self)
            self._kg_data = _kg_data
            self._log_print(_kg_data.statistic_info)
        return self._kg_data.kg_data

    @property
    def num_entities(self) -> int:
        return self.kg_data.num_entities

    @property
    def num_relations(self) -> int:
        return self.kg_data.num_relations

    @num_relations.setter
    def num_relations(self, num: int):
        self.kg_data.num_relations = num

    @property
    def num_triplets(self) -> int:
        return self.kg_data.num_triplets

    @property
    def mm_data(self) -> MMData:
        if not hasattr(self, "_mm_data"):
            _mm_data = MMData(self)
            self._mm_data = _mm_data
            self._log_print(_mm_data.statistic_info)
        return self._mm_data

    @property
    def img_features(self) -> np.ndarray:
        return self.mm_data.img_features

    @property
    def img_dim(self) -> int:
        return self.mm_data.img_dim

    @property
    def txt_features(self) -> np.ndarray:
        return self.mm_data.txt_features

    @property
    def txt_dim(self) -> int:
        return self.mm_data.txt_dim

    @property
    def audio_features(self) -> np.ndarray:
        return self.mm_data.audio_features

    @property
    def audio_dim(self) -> int:
        return self.mm_data.audio_dim

    @property
    def social_data(self) -> SocialData:
        # if not hasattr(self, "_social_data"):
        #     _social_data = SocialData()
        #     self._social_data = _social_data
        #     self._log_print(_social_data.statistic_info)
        # return self._social_data
        raise NotImplementedError

    @property
    def statistic_info(self) -> str:
        info_str = []
        for attr in ["_cf_data", "_kg_data", "_social_data", "_mm_data"]:
            if hasattr(self, attr):
                info_str.append(getattr(self, attr).statistic_info)
        info_str = "\n\n".join(info_str)
        return info_str


class UserGroup(object):
    def __init__(self, users, num_interactions, activities, label):
        self.label = label
        self.num_users = len(users)
        self.num_interactions = num_interactions
        self.users = users
        self.activities = activities


def group_users_by_interactions(dataset: RSDataset, num_groups=4) -> List[UserGroup]:
    user_groups = defaultdict(list)
    user_pos_train = dataset.train_data.to_user_dict()
    for user, item_seq in user_pos_train.items():
        user_groups[len(item_seq)].append(user)

    activities_list, num_users_list = [], []
    for activity, users in user_groups.items():
        activities_list.append(activity)
        num_users_list.append(len(users))

    activities_list = np.array(activities_list)
    num_users_list = np.array(num_users_list)

    # sort activities
    sorted_indices = np.argsort(activities_list)
    activities_list = activities_list[sorted_indices]
    num_users_list = num_users_list[sorted_indices]

    interactions_list = activities_list*num_users_list
    group_index = [0]
    rest_interactions_list = interactions_list

    for g_idx in range(num_groups - 1):
        num_interactions = np.sum(rest_interactions_list)
        num_per = num_interactions / (num_groups - g_idx)
        cum_actions = np.cumsum(rest_interactions_list)
        _idx = max(np.searchsorted(cum_actions, num_per), 1)
        split_idx = _idx - 1 if num_per - cum_actions[_idx - 1] < cum_actions[_idx] - num_per else _idx
        split_idx += 1
        group_index.append(group_index[-1] + split_idx)
        rest_interactions_list = rest_interactions_list[split_idx:]

    group_index = group_index[1:]
    # range labels
    split_len = activities_list[group_index]
    labels = [f"< {split_len[0]}"]
    for mi, ma in zip(split_len[:-1], split_len[1:]):
        labels.append(f"[{mi}, {ma})")
    labels.append(f"≥ {split_len[-1]}")

    # the list of numbers of users
    num_users = [np.sum(n_user) for n_user in np.split(num_users_list, group_index, axis=0)]

    # the list of numbers of interactions
    num_interactions = [np.sum(n_user) for n_user in np.split(interactions_list, group_index, axis=0)]

    activity_groups = np.split(activities_list, group_index, axis=0)

    # group information
    grouped_users = []
    for label, n_users, n_interactions, active_group in zip(labels, num_users, num_interactions, activity_groups):
        users = []
        for active in active_group:
            users.extend(user_groups[active])

        user_group = UserGroup(np.array(users), num_interactions, active_group, label)
        grouped_users.append(user_group)
    return grouped_users
