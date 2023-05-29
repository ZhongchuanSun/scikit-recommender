__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = []


import unittest
from skrec.utils.py import BatchIterator


class TestBatchIterator(unittest.TestCase):
    def test_all_cases(self):
        bat_size = 4
        users = list(range(10))
        items = list(range(10, 20))
        labels = list(range(20, 30))

        data_iter = BatchIterator(users, items, labels, batch_size=bat_size, shuffle=False)
        for bat_user, bat_item, bat_label in data_iter:
            self.assertTrue(len(bat_user) == len(bat_item))
            self.assertTrue(len(bat_user) == len(bat_label))
            break

        data_iter = BatchIterator(users, items, batch_size=bat_size, shuffle=True, drop_last=True)
        for bat_user, bat_item in data_iter:
            self.assertTrue(len(bat_user) == len(bat_item))
            break


if __name__ == "__main__":
    unittest.main()
