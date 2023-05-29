__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = []


import unittest
from skrec.io import MovieLens100k


class TestMovieLens(unittest.TestCase):
    def test_parse_from_file(self):
        save_dir = "./tmp"
        rating_path = MovieLens100k.download_and_extract(save_dir)
        print(rating_path)


if __name__ == "__main__":
    unittest.main()
