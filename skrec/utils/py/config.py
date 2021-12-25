__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["Config"]

import itertools
from argparse import Namespace, ArgumentParser
from configparser import ConfigParser


class OrderedNamespace(Namespace):
    def __init__(self, **kwargs):
        self._ordered_key = []
        super(OrderedNamespace, self).__init__(**kwargs)

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
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)

    def _validate(self):
        pass

    def to_string(self, sep: str='\n'):
        arg_strings = [f"{key}={value}" for key, value in self.items()]
        return sep.join(arg_strings)

    def parse_args_from_ini(self, filename: str, section: str=None):
        ini_parser = ConfigParser()
        ini_parser.optionxform = str  # option is case sensitive
        ini_parser.read(filename, encoding="utf-8")
        sections = ini_parser.sections()

        if len(sections) == 0:
            raise ValueError(f"'{filename}' is empty!")
        elif section is None:
            section = sections[0]
            print(f"section is not given, "
                  f"and parse arguments from the first ('{section}') section.")
        elif section in sections:  # parse from the given section
            pass
        else:  # the given section is not in the file
            raise ValueError(f"'{filename}' does not have '{section}' section.")

        args_list = [(f"--{arg}", value) for arg, value in ini_parser.items(section)]
        args_list = list(itertools.chain(*args_list))
        self._parse_known_args(args=args_list)
        return self

    def parse_args_from_cmd(self):
        self._parse_known_args()
        return self

    def _parse_known_args(self, args=None):
        _parser = ArgumentParser()
        for key, value in self.items():
            _parser.add_argument(f"--{key}", type=str, default=value, required=False)

        namespace = OrderedNamespace()
        _parser.parse_known_args(args=args, namespace=namespace)
        for key, new_value in namespace.items():
            if not hasattr(self, key):
                continue

            # hasattr(self, key) is True
            if isinstance(new_value, str):  # try to convert to a basic build-in type
                try:
                    if isinstance(eval(new_value), (int, float, str, list,
                                                    tuple, bool, dict, None.__class__)):
                        new_value = eval(new_value)
                except (NameError, SyntaxError):  # cannot be converted to a basic build-in type
                    pass

            old_value = getattr(self, key)
            if isinstance(new_value, type(old_value)):  # if types are same
                setattr(self, key, new_value)  # overwrite the old value
            else:
                raise TypeError(f"'{key}' expects a '{type(old_value)}' object, "
                                f"but the got {new_value} is a '{type(new_value)}' object.")

        self._validate()
        return self
