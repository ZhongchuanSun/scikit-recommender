__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = []


import unittest
from skrec.utils.py import Config


class TestConfig(unittest.TestCase):
    def test_parse_from_file(self):
        cfg_file = "./tmp/_tmp_args.cfg"
        with open(cfg_file, "w") as fout:
            fout.write("[test]\n")
            fout.write("echo=2\n")
            fout.write("square=2\n")
        config = Config(echo=1, square=1, abs=1)
        config.parse_args_from_ini(cfg_file, "test")
        # config.parse_args(args=[f"@{cfg_file}"], fromfile_prefix_chars="@")

        self.assertTrue(config.echo == 2)
        self.assertTrue(config.square == 2)
        self.assertTrue(config.abs == 1)

    def test_parse_from_args(self):
        config = Config(echo=1, square=1, abs=1)
        config.parse_args_from_cmd()
        self.assertTrue(config.echo == 1)
        self.assertTrue(config.square == 1)
        self.assertTrue(config.abs == 1)


if __name__ == "__main__":
    unittest.main()
