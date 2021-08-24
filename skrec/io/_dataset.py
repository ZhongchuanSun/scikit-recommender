__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["ImplicitFeedback", "Dataset"]


import os
import pickle
import warnings
from typing import Dict
from copy import deepcopy
from functools import wraps
from collections import OrderedDict
import pandas as pd
import numpy as np
import scipy.sparse as sp
from skrec.common.py_utils import pad_sequences, md5sum

_USER = "user"
_ITEM = "item"
_RATING = "rating"
_TIME = "time"
_DColumns = {"UI": [_USER, _ITEM],
             "UIR": [_USER, _ITEM, _RATING],
             "UIT": [_USER, _ITEM, _TIME],
             "UIRT": [_USER, _ITEM, _RATING, _TIME]
             }


class Interaction(object):
    def __init__(self):
        self._buffer = dict()
        self._buffer_modified_flag = False

    def is_empty(self) -> bool:
        raise NotImplementedError

    def _exist_in_buffer(self, name):
        return name in self._buffer

    def _write_to_buffer(self, name, value):
        self._buffer[name] = value
        self._buffer_modified_flag = True

    def _read_from_buffer(self, name):
        return deepcopy(self._buffer[name])

    def _clean_buffer(self):
        self._buffer.clear()
        self._buffer_modified_flag = True

    def is_buffer_modified(self):
        return self._buffer_modified_flag


def fetch_data(data_generator):
    @wraps(data_generator)
    def wrapper(self: Interaction, *args, **kwargs):
        _data_name = data_generator.__name__
        if self.is_empty():
            raise ValueError("data is empty!")

        if self._exist_in_buffer(_data_name) is False:
            _data = data_generator(self, *args, **kwargs)
            self._write_to_buffer(_data_name, _data)

        return self._read_from_buffer(_data_name)

    return wrapper


class ImplicitFeedback(Interaction):
    def __init__(self, data: pd.DataFrame=None, num_users: int=None, num_items: int=None):
        super(ImplicitFeedback, self).__init__()
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

    @fetch_data
    def to_user_item_pairs(self) -> np.ndarray:
        ui_pairs = self._data[[_USER, _ITEM]].to_numpy(copy=True, dtype=np.int32)
        return ui_pairs

    @fetch_data
    def to_csr_matrix(self) -> sp.csr_matrix:
        users, items = self._data[_USER].to_numpy(), self._data[_ITEM].to_numpy()
        ratings = np.ones(len(users), dtype=np.float32)
        csr_mat = sp.csr_matrix((ratings, (users, items)), shape=(self.num_users, self.num_items), copy=True)
        return csr_mat

    @fetch_data
    def to_csc_matrix(self) -> sp.csc_matrix:
        return self.to_csr_matrix().tocsc()

    @fetch_data
    def to_dok_matrix(self) -> sp.dok_matrix:
        return self.to_csr_matrix().todok()

    @fetch_data
    def to_coo_matrix(self) -> sp.coo_matrix:
        return self.to_csr_matrix().tocoo()

    @fetch_data
    def to_user_dict(self) -> Dict[int, np.ndarray]:
        user_dict = OrderedDict()
        user_grouped = self._data.groupby(_USER)
        for user, user_data in user_grouped:
            user_dict[user] = user_data[_ITEM].to_numpy(dtype=np.int32)
        return user_dict

    @fetch_data
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

    @fetch_data
    def to_item_dict(self) -> Dict[int, np.ndarray]:
        item_dict = OrderedDict()
        item_grouped = self._data.groupby(_ITEM)
        for item, item_data in item_grouped:
            item_dict[item] = item_data[_USER].to_numpy(dtype=np.int32)

        return item_dict

    @fetch_data
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


class Dataset(object):
    def __init__(self, data_dir, sep, columns):
        """Dataset

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
        self.data_name = os.path.split(data_dir)[-1]
        self._file_prefix = os.path.join(self.data_dir, self.data_name)

        # metadata
        self.train_data = ImplicitFeedback()
        self.valid_data = ImplicitFeedback()
        self.test_data = ImplicitFeedback()
        self.user2id = None
        self.item2id = None
        self.id2user = None
        self.id2item = None

        # statistic
        self.num_users = 0
        self.num_items = 0
        self.num_ratings = 0
        self._md5_summary = ""
        self._load_data(sep, columns)

    @property
    def data_dir(self):
        return self._data_dir

    def _load_data(self, sep, columns):
        pkl_file = self._file_prefix + ".pkl"
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as fin:
                _t_data: Dataset = pickle.load(fin)
            if _t_data._md5_summary == self._raw_summary():
                self.__dict__ = _t_data.__dict__
                return

        self._load_from_raw(sep, columns)
        self._md5_summary = self._raw_summary()
        with open(pkl_file, 'wb') as fout:
            pickle.dump(self, fout)

    def __del__(self):
        # if self.train_data.is_buffer_modified() or \
        #         self.valid_data.is_buffer_modified() or \
        #         self.test_data.is_buffer_modified():
        #     pkl_file = self._file_prefix + ".pkl"
        #     pickle.dump(self, open(pkl_file, 'wb'))
        # TODO 如何在析构函数中对模型保存?
        pass

    def _raw_summary(self):
        md5summary = md5sum(self._file_prefix+".train", self._file_prefix+".valid", self._file_prefix+".test")
        md5summary = "_".join([md5 for md5 in md5summary if md5 is not None])
        return md5summary

    def _load_from_raw(self, sep, columns):
        if columns not in _DColumns:
            key_str = ", ".join(_DColumns.keys())
            raise ValueError("'columns' must be one of '%s'." % key_str)

        columns = _DColumns[columns]

        # load data
        train_file = self._file_prefix+".train"
        if os.path.isfile(train_file):
            _train_data = pd.read_csv(train_file, sep=sep, header=None, names=columns)
        else:
            raise FileNotFoundError("%s does not exist." % train_file)

        valid_file = self._file_prefix + ".valid"
        if os.path.isfile(valid_file):
            _valid_data = pd.read_csv(valid_file, sep=sep, header=None, names=columns)
        else:
            _valid_data = pd.DataFrame()
            warnings.warn("%s does not exist." % valid_file)

        test_file = self._file_prefix + ".test"
        if os.path.isfile(test_file):
            _test_data = pd.read_csv(test_file, sep=sep, header=None, names=columns)
        else:
            raise FileNotFoundError("%s does not exist." % test_file)

        user2id_file = self._file_prefix + ".user2id"
        if os.path.isfile(user2id_file):
            _user2id = pd.read_csv(user2id_file, sep=sep, header=None).to_numpy()
            self.user2id = OrderedDict(_user2id)
            self.id2user = OrderedDict([(idx, user) for user, idx in self.user2id.items()])
        else:
            warnings.warn("%s does not exist." % user2id_file)

        item2id_file = self._file_prefix + ".item2id"
        if os.path.isfile(item2id_file):
            _item2id = pd.read_csv(item2id_file, sep=sep, header=None).to_numpy()
            self.item2id = OrderedDict(_item2id)
            self.id2item = OrderedDict([(idx, item) for item, idx in self.item2id.items()])
        else:
            warnings.warn("%s does not exist." % item2id_file)

        # statistical information
        data_list = [data for data in [_train_data, _valid_data, _test_data] if not data.empty]
        all_data = pd.concat(data_list)
        self.num_users = max(all_data[_USER]) + 1
        self.num_items = max(all_data[_ITEM]) + 1
        self.num_ratings = len(all_data)

        # convert to to the object of Interaction
        self.train_data = ImplicitFeedback(_train_data, num_users=self.num_users, num_items=self.num_items)
        self.valid_data = ImplicitFeedback(_valid_data, num_users=self.num_users, num_items=self.num_items)
        self.test_data = ImplicitFeedback(_test_data, num_users=self.num_users, num_items=self.num_items)

    def summary(self):
        """The statistic of dataset.

        Returns:
            str: The summary of statistic
        """
        if 0 in {self.num_users, self.num_items, self.num_ratings}:
            return ""
        else:
            num_users, num_items = self.num_users, self.num_items
            num_ratings = self.num_ratings
            sparsity = 1 - 1.0 * num_ratings / (num_users * num_items)

            statistic = ["Dataset statistics:",
                         "Name: %s" % self.data_name,
                         "The number of users: %d" % num_users,
                         "The number of items: %d" % num_items,
                         "The number of ratings: %d" % num_ratings,
                         "Average actions of users: %.2f" % (1.0 * num_ratings / num_users),
                         "Average actions of items: %.2f" % (1.0 * num_ratings / num_items),
                         "The sparsity of the dataset: %.6f%%" % (sparsity * 100),
                         "",
                         "The number of training: %d" % len(self.train_data),
                         "The number of validation: %d" % len(self.valid_data),
                         "The number of testing: %d" % len(self.test_data)
                         ]
            statistic = "\n".join(statistic)
            return statistic
