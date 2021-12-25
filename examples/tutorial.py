__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = []

from typing import Union, Tuple, List
from skrec import RankingEvaluator
from skrec import Dataset
from skrec import Config
from skrec.recommender.BPRMF import BPRMF, BPRMFConfig
from skrec.io import MovieLens100k
from skrec.io import Preprocessor


class RunConfig(Config):
    def __init__(self,
                 recommender="BPRMF",
                 data_dir=None,
                 file_column="UIRT", column_sep='\t',
                 metric=("Precision", "Recall", "MAP", "NDCG", "MRR"),
                 top_k=(10, 20, 30, 40, 50, 100),
                 test_batch_size=64,
                 test_thread=4,
                 **kwargs):
        super(RunConfig, self).__init__(**kwargs)
        self.Recommender: str = recommender
        self.data_dir: str = data_dir
        self.file_column: str = file_column
        self.sep: str = column_sep
        self.metric: Union[None, str, Tuple[str], List[str]] = metric
        self.top_k: Union[int, List[int], Tuple[int]] = top_k
        self.test_batch_size: int = test_batch_size
        self.test_thread: int = test_thread
        if self.data_dir is None:
            raise ValueError("'data_dir' cannot be None!")


if __name__ == '__main__':
    # download and extract MovieLens100k
    data_path = MovieLens100k.download_and_extract("./dataset")

    # pre-process and save dataset
    processor = Preprocessor()
    processor.load_data(data_path, sep="\t", columns="UIRT")
    processor.drop_duplicates(keep="last")
    processor.filter_data(user_min=5, item_min=5)
    processor.remap_data_id()
    # data.split_data_by_leave_out(valid=1, test=1)
    processor.split_data_by_ratio(train=0.7, valid=0.0, test=0.3, by_time=True)
    data_dir = processor.save_data()

    # read config and load dataset
    run_config = RunConfig(data_dir=data_dir)
    # run_config.parse_args_from_ini("./skrec.ini")
    run_config.parse_args_from_cmd()
    dataset = Dataset(run_config.data_dir, run_config.sep, run_config.file_column)

    bpr_config = BPRMFConfig(epochs=5)
    # bpr_config.parse_args_from_ini("./skrec.ini")  # , "bpr"

    # configure evaluator
    evaluator = RankingEvaluator(dataset.train_data.to_user_dict(),
                                 dataset.test_data.to_user_dict(),
                                 metric=run_config.metric, top_k=run_config.top_k,
                                 batch_size=run_config.test_batch_size,
                                 num_thread=run_config.test_thread)

    # train model
    bpr_model = BPRMF(dataset, bpr_config, evaluator)
    bpr_model.fit()
