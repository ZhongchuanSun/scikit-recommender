__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = []

from skrec.io import Config


class BPRConfig(Config):
    def __init__(self,
                 n_dim=64,
                 lr=1e-3,
                 reg=1e-3,
                 batch_size=1024,
                 epochs=1000,
                 **kwargs):
        super(BPRConfig, self).__init__(**kwargs)
        self.n_dim: int = n_dim
        self.lr: float = lr
        self.reg: float = reg
        self.batch_size: int = batch_size
        self.epochs: int = epochs

    def _validate(self):
        assert isinstance(self.n_dim, int)
        assert isinstance(self.lf, float)
        assert isinstance(self.reg, float)
        assert isinstance(self.batch_size, int)
        assert isinstance(self.epochs, int)


class BPR(object):
    def __init__(self, dataset, config: BPRConfig):
        pass
