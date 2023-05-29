__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Preprocessor"]

import os
import math
import pandas as pd
from ..utils.py import typeassert
from .logger import Logger
from collections import OrderedDict


class Preprocessor(object):
    _USER = "user"
    _ITEM = "item"
    _RATING = "rating"
    _TIME = "time"

    def __init__(self):
        """A class for data preprocessing
        """

        self._column_dict = {"UI": [self._USER, self._ITEM],
                             "UIR": [self._USER, self._ITEM, self._RATING],
                             "UIT": [self._USER, self._ITEM, self._TIME],
                             "UIRT": [self._USER, self._ITEM, self._RATING, self._TIME]}
        self._column_name = None
        self._config = OrderedDict()
        self.all_data = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.user2id = None
        self.item2id = None
        self._dir_path = None
        self._data_name = ""
        self._split_manner = ""
        self._user_min = 0
        self._item_min = 0

    @typeassert(filename=str, sep=str)
    def load_data(self, filename: str, sep: str=",", columns: str=None):
        """Load data

        Args:
            filename (str): The path of dataset.
            sep (str): The separator/delimiter of columns.
            columns (str): One of 'UI', 'UIR', 'UIT' and 'UIRT'.

        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"There is no file named '{filename}'.")
        if columns not in self._column_dict:
            key_str = ", ".join(self._column_dict.keys())
            raise ValueError(f"'columns' must be one of '{key_str}'.")
        self._config["columns"] = columns

        self._column_name = self._column_dict[columns]

        print("loading data...")
        self._config["filename"] = filename
        self._config["sep"] = sep

        self.all_data = pd.read_csv(filename, sep=sep, header=None, names=self._column_name)
        self.all_data.dropna(inplace=True)

        self._data_name = os.path.basename(filename).split(".")[0]
        self._dir_path = os.path.dirname(filename)

    def drop_duplicates(self, keep: str="last"):
        """Drop duplicate user-item interactions.

        Args:
            keep (str): 'first' or 'last', default 'first'.
                Drop duplicates except for the first or last occurrence.

        Returns:
            An object of pd.DataFrame without duplicates.

        Raises:
            ValueError: If 'keep' is not 'first' or 'last'.
        """

        if keep not in {'first', 'last'}:
            raise ValueError(f"'keep' must be 'first' or 'last', but '{keep}'")
        print("dropping duplicate interactions...")

        if self._TIME in self._column_name:
            sort_key = [self._USER, self._TIME]
        else:
            sort_key = [self._USER, self._ITEM]

        self.all_data.sort_values(by=sort_key, inplace=True)

        self.all_data.drop_duplicates(subset=[self._USER, self._ITEM], keep=keep, inplace=True)

    @typeassert(user_min=int, item_min=int)
    def filter_data(self, user_min: int=0, item_min: int=0):
        """Filter users and items with a few interactions.

        Args:
            user_min (int): The users with less interactions than 'user_min' will be filtered.
            item_min (int): The items with less interactions than 'item_min' will be filtered.
        """
        i = 0
        while True:
            i += 1
            old_len_data = len(self.all_data)
            print(f"{i} filtering items...")
            self.filter_item(item_min)
            print(f"{i} filtering users...")
            self.filter_user(user_min)
            new_len_data = len(self.all_data)
            if old_len_data == new_len_data:  # return if data does not change
                break

    @typeassert(user_min=int)
    def filter_user(self, user_min: int=0):
        """Filter users with a few interactions.

        Args:
            user_min (int): The users with less interactions than 'user_min' will be filtered.
        """
        self._config["user_min"] = str(user_min)
        self._user_min = user_min
        if user_min > 0:
            user_count = self.all_data[self._USER].value_counts(sort=False)
            if user_count.min() < user_min:
                filtered_idx = self.all_data[self._USER].map(lambda x: user_count[x] >= user_min)
                self.all_data = self.all_data[filtered_idx]

    @typeassert(item_min=int)
    def filter_item(self, item_min: int=0):
        """Filter items with a few interactions.

        Args:
            item_min (int): The items with less interactions than 'item_min' will be filtered.
        """
        self._config["item_min"] = str(item_min)
        self._item_min = item_min
        if item_min > 0:
            item_count = self.all_data[self._ITEM].value_counts(sort=False)
            if item_count.min() < item_min:
                filtered_idx = self.all_data[self._ITEM].map(lambda x: item_count[x] >= item_min)
                self.all_data = self.all_data[filtered_idx]

    def remap_data_id(self):
        """Convert user and item IDs to integers, start from 0.

        """
        self.remap_user_id()
        self.remap_item_id()

    def remap_user_id(self):
        """Convert user IDs to integers, start from 0.

        """
        print("remapping user IDs...")
        self._config["remap_user_id"] = "True"
        unique_user = self.all_data[self._USER].unique()
        self.user2id = pd.Series(data=range(len(unique_user)), index=unique_user)

        self.all_data[self._USER] = self.all_data[self._USER].map(self.user2id)

    def remap_item_id(self):
        """Convert item IDs to integers, start from 0.

        """
        print("remapping item IDs...")
        self._config["remap_item_id"] = "True"
        unique_item = self.all_data[self._ITEM].unique()
        self.item2id = pd.Series(data=range(len(unique_item)), index=unique_item)

        self.all_data[self._ITEM] = self.all_data[self._ITEM].map(self.item2id)

    @typeassert(train=float, valid=float, test=float)
    def split_data_by_ratio(self, train: float=0.7, valid: float=0.1, test: float=0.2, by_time: bool=True):
        """Split dataset by the given ratios.

        The dataset will be split by each user.

        Args:
            train (float): The proportion of training data.
            valid (float): The proportion of validation data.
                '0.0' means no validation set.
            test (float): The proportion of testing data.
            by_time (bool): Splitting data randomly or by time.
        """
        if train <= 0.0:
            raise ValueError("'train' must be a positive value.")
        if train + valid + test != 1.0:
            raise ValueError("The sum of 'train', 'valid' and 'test' must be equal to 1.0.")
        print("splitting data by ratio...")

        self._config["split_by"] = "ratio"
        self._config["train"] = str(train)
        self._config["valid"] = str(valid)
        self._config["test"] = str(test)
        self._config["by_time"] = str(by_time)

        if by_time is False or self._TIME not in self._column_name:
            sort_key = [self._USER, self._ITEM]
        else:
            sort_key = [self._USER, self._TIME]

        self.all_data.sort_values(by=sort_key, inplace=True)

        _by_t = "by_time" if by_time is True else "by_random"
        self._split_manner = "ratio_" + _by_t
        train_data = []
        valid_data = []
        test_data = []

        user_grouped = self.all_data.groupby(by=[self._USER])
        for user, u_data in user_grouped:
            u_data_len = len(u_data)
            if not by_time:
                u_data = u_data.sample(frac=1)
            train_end = math.ceil(train * u_data_len)
            train_data.append(u_data.iloc[:train_end])
            if valid != 0:
                test_begin = train_end + math.ceil(valid * u_data_len)
                valid_data.append(u_data.iloc[train_end:test_begin])
            else:
                test_begin = train_end
            test_data.append(u_data.iloc[test_begin:])

        self.train_data = pd.concat(train_data, ignore_index=True)
        if valid != 0:
            self.valid_data = pd.concat(valid_data, ignore_index=True)
        self.test_data = pd.concat(test_data, ignore_index=True)

    @typeassert(valid=int, test=int)
    def split_data_by_leave_out(self, valid: int=1, test: int=1, by_time: bool=True):
        """Split dataset by leave out certain number items.

        The dataset will be split by each user.

        Args:
            valid (int): The number of items of validation set for each user.
                Default to 1 and means leave one out.
            test (int): The number of items of test set for each user.
                Default to 1 and means leave one out.
            by_time (bool): Splitting data randomly or by time.
        """

        self._config["split_by"] = "leave_out"
        self._config["valid"] = str(valid)
        self._config["test"] = str(test)
        self._config["by_time"] = str(by_time)

        if by_time is False or self._TIME not in self._column_name:
            sort_key = [self._USER, self._ITEM]
        else:
            sort_key = [self._USER, self._TIME]
        print("splitting data by leave out...")

        self.all_data.sort_values(by=sort_key, inplace=True)

        _by_t = "by_time" if by_time is True else "by_random"
        self._split_manner = "leave_" + _by_t
        train_data = []
        valid_data = []
        test_data = []

        user_grouped = self.all_data.groupby(by=[self._USER])
        for user, u_data in user_grouped:
            if not by_time:
                u_data = u_data.sample(frac=1)
            train_end = -(valid+test)
            train_data.append(u_data.iloc[:train_end])
            if valid != 0:
                test_begin = train_end + valid
                valid_data.append(u_data.iloc[train_end:test_begin])
            else:
                test_begin = train_end
            test_data.append(u_data.iloc[test_begin:])

        self.train_data = pd.concat(train_data, ignore_index=True)
        if valid != 0:
            self.valid_data = pd.concat(valid_data, ignore_index=True)
        self.test_data = pd.concat(test_data, ignore_index=True)

    def save_data(self, save_dir=None):
        """Save data to disk.

        Args:
            save_dir (str): The directory to save the dataset and information.

        """
        print("saving data to disk...")
        dir_path = save_dir if save_dir is not None else self._dir_path
        filename = f"{self._data_name}_{self._split_manner}_u{self._user_min}_i{self._item_min}"
        dir_path = os.path.join(dir_path, filename)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save data
        filename = os.path.join(dir_path, filename)
        sep = "\t"  # self._config["sep"]
        if self.all_data is not None:
            self.all_data.to_csv(filename+".all", header=False, index=False, sep=sep)
        if self.train_data is not None:
            self.train_data.to_csv(filename + ".train", header=False, index=False, sep=sep)
        if self.valid_data is not None:
            self.valid_data.to_csv(filename + ".valid", header=False, index=False, sep=sep)
        if self.test_data is not None:
            self.test_data.to_csv(filename + ".test", header=False, index=False, sep=sep)
        if self.user2id is not None:
            self.user2id.to_csv(filename + ".user2id", header=False, index=True, sep=sep)
        if self.item2id is not None:
            self.item2id.to_csv(filename + ".item2id", header=False, index=True, sep=sep)

        # calculate statistics
        user_num = len(self.all_data[self._USER].unique())
        item_num = len(self.all_data[self._ITEM].unique())
        rating_num = len(self.all_data)
        sparsity = 1-1.0*rating_num/(user_num*item_num)

        # write log file
        logger = Logger(filename+".info")
        data_info = "\n".join([f"{key} = {value}" for key, value in self._config.items()])
        logger.info("\n"+data_info)
        logger.info("Dataset statistic information:")
        logger.info(f"The number of users: {user_num}")
        logger.info(f"The number of items: {item_num}")
        logger.info(f"The number of ratings: {rating_num}")
        logger.info(f"Average actions of users: {(1.0*rating_num/user_num):.2f}")
        logger.info(f"Average actions of items: {(1.0*rating_num/item_num):.2f}")
        logger.info(f"The sparsity of the dataset: {(sparsity*100)}%%")

        print(f"\nThe processed data has been saved in '{dir_path}'")
        return dir_path
