__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = []


import unittest
from skrec.io import Logger


class TestLogger(unittest.TestCase):
    def test_logger(self):
        log = Logger('_tmp_scikit-recommender.log')
        log.debug('debug')
        log.info('info')
        log.warning('warning')
        log.error('error')
        log.critical('critical')

        log = Logger()
        log.debug('debug')
        log.info('info')
        log.warning('warning')
        log.error('error')
        log.critical('critical')


if __name__ == '__main__':
    unittest.main()
