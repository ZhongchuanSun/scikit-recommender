from typing import Union, Tuple, List
from .utils.py import Config

__all__ = ["RunConfig"]


class RunConfig(Config):
    def __init__(self,
                 recommender="BPRMF",
                 data_dir="dataset/Beauty_loo_u5_i5",
                 file_column="UIRT",
                 sep='\t',
                 gpu_id=0,
                 metric=("Precision", "Recall", "MAP", "NDCG", "MRR"),
                 top_k=(10, 20, 30, 40, 50, 100),
                 test_batch_size=64,
                 test_thread=4,
                 seed=2021,
                 **kwargs):
        super(RunConfig, self).__init__()
        self.recommender: str = recommender
        self.data_dir: str = data_dir  # the directory of training and test sets
        self.file_column: str = file_column  # UI, UIR, UIT, UIRT
        self.sep: str = sep
        self.gpu_id = gpu_id
        self.metric: Union[None, str, Tuple[str], List[str]] = metric  # ("Precision", "Recall", "MAP", "NDCG", "MRR")
        self.top_k: Union[int, List[int], Tuple[int]] = top_k
        # large test_batch_size might cause GPU memory-consuming, especially dataset is large
        self.test_batch_size: int = test_batch_size
        self.test_thread: int = test_thread
        self.seed = seed

    def _validate(self):
        assert isinstance(self.recommender, str) and not self.recommender
        assert isinstance(self.data_dir, str) and not self.data_dir
        assert isinstance(self.file_column, str) and not self.file_column
        assert isinstance(self.sep, str)
        assert isinstance(self.test_batch_size, int) and self.test_batch_size > 0
        assert isinstance(self.test_thread, int) and self.test_thread > 0
        assert isinstance(self.seed, int) and self.seed >= 0
