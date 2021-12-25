__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = []


import unittest
from skrec.io import Preprocessor


class TestPreprocessor(unittest.TestCase):
    def test_preprocessor(self):
        processor = Preprocessor()
        processor.load_data("./tmp/ml-100k.rating", sep="\t", columns="UIRT")
        processor.drop_duplicates()
        processor.filter_data(user_min=5, item_min=5)
        processor.remap_data_id()
        # processor.split_data_by_leave_out(valid=1, test=1)
        processor.split_data_by_ratio(train=0.7, valid=0.0, test=0.3, by_time=True)
        processor.save_data()


if __name__ == "__main__":
    unittest.main()
