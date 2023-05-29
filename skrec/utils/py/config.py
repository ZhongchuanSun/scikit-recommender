__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Config", "merge_config_with_cmd_args"]

import sys
from typing import Dict
from collections import OrderedDict
from argparse import Namespace
import copy


class OrderedNamespace(Namespace):
    def __init__(self):
        self._ordered_key = []
        super(OrderedNamespace, self).__init__()

    def _get_kwargs(self):
        # retrieve (key, value) pairs in the order they were initialized using _keys
        return [(k, self.__dict__[k]) for k in self._ordered_key]

    def __setattr__(self, key, value):
        # store new attribute (key, value) pairs in builtin __dict__
        self.__dict__[key] = value
        # store the keys in self._keys in the order that they are initialized
        # do not store '_keys' itself and don't enter any key more than once
        if key not in ['_ordered_key'] + self._ordered_key:
            self._ordered_key.append(key)

    def items(self):
        for key, value in self._get_kwargs():
            yield (key, value)


class Config(OrderedNamespace):
    def __init__(self):
        super(Config, self).__init__()

    def _validate(self):
        pass

    def to_string(self, sep: str='\n'):
        arg_strings = [f"{key}={value}" for key, value in self.items()]
        return sep.join(arg_strings)


def merge_config_with_cmd_args(config: Dict, inplace: bool = True) -> Dict:
    args = sys.argv[1:]
    if len(args) % 2 != 0:
        raise SyntaxError("The numbers of arguments and its values are not equal.")

    if inplace is False:
        config = copy.deepcopy(config)

    cmd_args = OrderedDict()
    for arg_name, arg_value in zip(args[0::2], args[1::2]):
        if not arg_name.startswith("--"):
            raise SyntaxError("Command arg must start with '--', but '%s' is not!" % arg_name)
        cmd_args[arg_name[2:]] = arg_value

    # cover the arguments from ini files
    for cmd_argn, cmd_argv in cmd_args.items():
        # convert param from str to value, i.e. int, float or list etc.
        try:
            value = eval(cmd_argv)
            if not isinstance(value, (str, int, float, list, tuple, bool, None.__class__)):
                value = cmd_argv
        except (NameError, SyntaxError):
            if cmd_argv.lower() == "true":
                value = True
            elif cmd_argv.lower() == "false":
                value = False
            else:
                value = cmd_argv
        config[cmd_argn] = value

    return config
