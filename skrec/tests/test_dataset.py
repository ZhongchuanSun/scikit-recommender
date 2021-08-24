__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = []


import unittest
import numpy as np
import scipy.sparse as sp
from skrec.io import Dataset


def _load_data():
    return Dataset("../../dataset/ml-100k_leave_u5_i5", '\t', "UIRT")


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = _load_data()
        train_data = dataset.train_data
        test_data = dataset.test_data
        self.assertTrue(train_data.num_items == test_data.num_items)
        self.assertTrue(train_data.num_users == test_data.num_users)
        self.assertIsInstance(train_data.to_user_dict_by_time(), dict)
        self.assertIsInstance(train_data.to_user_dict(), dict)
        self.assertIsInstance(train_data.to_item_dict(), dict)
        self.assertIsInstance(train_data.to_truncated_seq_dict(max_len=20), dict)
        self.assertIsInstance(train_data.to_user_item_pairs(), np.ndarray)
        self.assertIsInstance(train_data.to_csr_matrix(), sp.csr_matrix)
        self.assertIsInstance(train_data.to_csc_matrix(), sp.csc_matrix)
        self.assertIsInstance(train_data.to_coo_matrix(), sp.coo_matrix)
        self.assertIsInstance(train_data.to_dok_matrix(), sp.dok_matrix)


if __name__ == '__main__':
    unittest.main()
