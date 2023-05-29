__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Logger"]

import sys
import os
import re
import logging
from typing import Optional


class RemoveColorFilter(logging.Filter):

    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True


class Logger(object):
    """`Logger` is a simple encapsulation of python logger.

    This class can show a message on standard output and write it into the
    file named `filename` simultaneously. This is convenient for observing
    and saving training results.
    """

    def __init__(self, filename: Optional[str]=None):
        """Initializes a new `Logger` instance.

        Args:
            filename (str): File name to create. The directory component of this
                file will be created automatically if it is not existing.
        """
        logger_name = "scikit-recommender-logger" if filename is None else filename
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # show on console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)  # add to Handler

        if filename is not None:
            dir_name = os.path.dirname(filename)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

            remove_color_filter = RemoveColorFilter()
            # write into file
            fh = logging.FileHandler(filename)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            fh.addFilter(remove_color_filter)
            self.logger.addHandler(fh)  # add to Handler

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()
