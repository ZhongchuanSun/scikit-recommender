__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["Config"]


import argparse


class OrderedNamespace(argparse.Namespace):
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

    def to_printable(self):
        arg_strings = [f"{key}={value}" for key, value in self.items()]
        return "\n".join(arg_strings)

    def parse_args(self, args=None, fromfile_prefix_chars=None):
        # TODO 将 参数解析分成两个, 一个从命令行, 一个从ini文件.
        parser = argparse.ArgumentParser(fromfile_prefix_chars=fromfile_prefix_chars)
        for key, value in self.items():
            parser.add_argument(f"--{key}", type=type(value), default=value, required=False)

        return parser.parse_args(args=args, namespace=self)

    def parse_args_from_ini(self):
        pass

    def parse_args_from_cmd(self):
        pass
