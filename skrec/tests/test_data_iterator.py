__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = []


import unittest
from skrec.io import (
    PointwiseIterator, PairwiseIterator,
    SequentialPointwiseIterator, SequentialPairwiseIterator,
    UserVecIterator, ItemVecIterator
)
from skrec.io import Dataset
from skrec.io import MovieLens100k, Preprocessor


def _load_data():
    data_path = MovieLens100k.download_and_extract("./tmp")

    # pre-process and save dataset
    processor = Preprocessor()
    processor.load_data(data_path, sep="\t", columns="UIRT")
    processor.drop_duplicates(keep="last")
    processor.filter_data(user_min=5, item_min=5)
    processor.remap_data_id()
    # data.split_data_by_leave_out(valid=1, test=1)
    processor.split_data_by_ratio(train=0.7, valid=0.0, test=0.3, by_time=True)
    data_dir = processor.save_data()

    return Dataset(data_dir, '\t', "UIRT")


class TestPointwiseIterator(unittest.TestCase):
    def test_num_neg(self):
        dataset = _load_data()
        for num_neg in [1, 3, 5]:
            bat_size = 1024
            data_iter = PointwiseIterator(dataset.train_data,
                                          num_neg=num_neg, batch_size=bat_size,
                                          shuffle=True, drop_last=False)
            for bat_users, bat_items, bat_labels in data_iter:
                self.assertTrue(bat_users.shape == bat_items.shape)
                self.assertTrue(bat_users.shape == bat_labels.shape)
                break


class TestPairwiseIterator(unittest.TestCase):
    def test_iter(self):
        dataset = _load_data()
        bat_size = 1024
        data_iter = PairwiseIterator(dataset.train_data,
                                     batch_size=bat_size,
                                     shuffle=True, drop_last=False)
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            self.assertTrue(bat_users.shape == bat_pos_items.shape)
            self.assertTrue(bat_neg_items.shape == (bat_size,))
            break


class TestSequentialPointwiseIterator(unittest.TestCase):
    def test_all_cases(self):
        dataset = _load_data()
        bat_size = 256
        for num_neg in [1, 3]:
            for num_previous in [1, 3, 100]:
                for num_next in [1, 3]:
                    for pad in [0, None]:
                        data_iter = SequentialPointwiseIterator(dataset.train_data,
                                                                num_previous=num_previous, num_next=num_next,
                                                                pad=pad, num_neg=num_neg, batch_size=bat_size,
                                                                shuffle=True, drop_last=False)
                        # print(num_neg, num_previous, num_next, pad)
                        for bat_users, bat_seqs, bat_next_items, bat_labels in data_iter:
                            self.assertTrue(bat_users.shape == (bat_size, ))
                            if num_previous == 1:
                                self.assertTrue(bat_seqs.shape == (bat_size,))
                            else:
                                self.assertTrue(bat_seqs.shape == (bat_size, num_previous))
                            if num_next == 1:
                                self.assertTrue(bat_next_items.shape == (bat_size,))
                            else:
                                self.assertTrue(bat_next_items.shape == (bat_size, num_next))

                            self.assertTrue(bat_labels.shape == bat_next_items.shape)
                            break


class TestSequentialPairwiseIterator(unittest.TestCase):
    def test_all_cases(self):
        dataset = _load_data()
        bat_size = 256
        for num_previous in [1, 3, 100]:
            for num_next in [1, 3]:
                for pad in [0, None]:
                    data_iter = SequentialPairwiseIterator(dataset.train_data,
                                                           num_previous=num_previous, num_next=num_next,
                                                           pad=pad, batch_size=bat_size,
                                                           shuffle=True, drop_last=False)
                    # print(num_previous, num_next, pad)
                    for bat_users, bat_seqs, bat_pos_items, bat_neg_items in data_iter:
                        self.assertTrue(bat_users.shape == (bat_size,))
                        if num_previous == 1:
                            self.assertTrue(bat_seqs.shape == (bat_size,))
                        else:
                            self.assertTrue(bat_seqs.shape == (bat_size, num_previous))
                        if num_next == 1:
                            self.assertTrue(bat_pos_items.shape == (bat_size,))
                        else:
                            self.assertTrue(bat_pos_items.shape == (bat_size, num_next))

                        self.assertTrue(bat_pos_items.shape == bat_neg_items.shape)
                        break


class TestUserVecIterator(unittest.TestCase):
    def test_num_neg(self):
        dataset = _load_data()
        num_items = dataset.train_data.num_items
        bat_size = 256
        data_iter = UserVecIterator(dataset.train_data, batch_size=bat_size,
                                    shuffle=True, drop_last=False)
        for batch_user_vec in data_iter:
            self.assertTrue(batch_user_vec.shape == (bat_size, num_items))
            break


class TestItemVecIterator(unittest.TestCase):
    def test_num_neg(self):
        dataset = _load_data()
        num_users = dataset.train_data.num_users
        bat_size = 256
        data_iter = ItemVecIterator(dataset.train_data, batch_size=bat_size,
                                    shuffle=True, drop_last=False)
        for batch_item_vec in data_iter:
            self.assertTrue(batch_item_vec.shape == (bat_size, num_users))
            break


if __name__ == '__main__':
    unittest.main()
