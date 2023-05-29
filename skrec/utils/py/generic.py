__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["OrderedDefaultDict", "pad_sequences", "md5sum", "slugify"]

import os
import sys
import re
import hashlib
import unicodedata
from collections import OrderedDict
import numpy as np


class OrderedDefaultDict(OrderedDict):
    """ A defaultdict with OrderedDict as its base class.
    Reference: https://stackoverflow.com/questions/4126348/4127426#4127426
    """

    def __init__(self, default_factory=None, *args, **kwargs):
        if not (default_factory is None or callable(default_factory)):
            raise TypeError('first argument must be callable or None')
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory  # called by __missing__()

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key,)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):  # Optional, for pickle support.
        args = (self.default_factory,) if self.default_factory else tuple()
        return self.__class__, args, None, None, iter(self.items())

    def __repr__(self):  # Optional.
        return '%s(%r, %r)' % (self.__class__.__name__, self.default_factory, self.items())


def pad_sequences(sequences, value=0, max_len=None,
                  padding='post', truncating='post', dtype=int):
    """Pads sequences to the same length.

    Args:
        sequences (list): A list of lists, where each element is a sequence.
        value (int, float): Padding value. Defaults to `0.`.
        max_len (int or None): Maximum length of all sequences.
        padding (str): `"pre"` or `"post"`: pad either before or after each
            sequence. Defaults to `post`.
        truncating (str): `"pre"` or `"post"`: remove values from sequences
            larger than `max_len`, either at the beginning or at the end of
            the sequences. Defaults to `post`.
        dtype: Type of the output sequences. Defaults to `np.int32`.

    Returns:
        np.ndarray: Numpy array with shape `(len(sequences), max_len)`.

    Raises:
        ValueError: If `padding` or `truncating` is not understood.
    """
    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if max_len is None:
        max_len = np.max(lengths)

    x = np.full([len(sequences), max_len], value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def md5sum(*args):
    """Compute and check MD5 message
    Args:
        *args: one or more file paths

    Returns: a list of MD5 message
    """
    for filename in args:
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
    md5_list = []
    for filename in args:
        with open(filename, "rb") as fin:
            readable_hash = hashlib.md5(fin.read()).hexdigest()
            md5_list.append(readable_hash)
    md5_list = md5_list[0] if len(args) == 1 else md5_list
    return md5_list


def slugify(filename, max_length: int=255) -> str:
    """
    url1: https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename
    url2: https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    filename = str(filename)
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    filename = re.sub(r'[^\w\s.+=-]', '', filename)
    filename = re.sub(r'[\s]+', '_', filename).strip('-_')
    if len(filename) > max_length:
        print(f"Warning, filename truncated because it was over {max_length}. "
              f"Filenames may no longer be unique")
    return filename[:max_length]
