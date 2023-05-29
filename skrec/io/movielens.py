__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["MovieLens100k"]

import os
import shutil
from urllib import request
from zipfile import ZipFile


class MovieLens100k(object):
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

    @classmethod
    def download(cls, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filepath = os.path.join(data_dir, cls.url.split("/")[-1])
        if not os.path.exists(filepath):
            print("downloading ml-100k.zip ...")
            request.urlretrieve(cls.url, filepath)
        else:
            print(f"File '{filepath}' already downloaded.")

        return filepath

    @classmethod
    def extract(cls, zip_path):
        filename = "ml-100k.rating"
        _rating_path = os.path.join(os.path.dirname(zip_path), filename)
        if not os.path.exists(_rating_path):
            with ZipFile(zip_path, "r") as z:
                with z.open("ml-100k/u.data") as zf, open(_rating_path, "wb") as f:
                    print("extracting...")
                    shutil.copyfileobj(zf, f)
        else:
            print(f"File '{_rating_path}' already existed.")
        return _rating_path

    @classmethod
    def download_and_extract(cls, data_dir):
        _zip_path = MovieLens100k.download(data_dir)
        _rating_path = MovieLens100k.extract(_zip_path)
        return _rating_path
