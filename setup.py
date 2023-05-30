import os
import shutil
from pathlib import Path
import setuptools
from setuptools import setup
from setuptools.extension import Extension
from functools import wraps
from Cython.Build import cythonize
import numpy as np


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Information Analysis
Topic :: Software Development :: Libraries
Topic :: Software Development :: Libraries :: Python Modules
"""

MAJOR = 0
MINOR = 0
MICRO = 3
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def get_include_dirs(workspace):
    include_dirs = [np.get_include()]
    for root, dirs, files in os.walk(workspace):
        for file in files:
            if file.endswith(".h") or file.endswith(".hpp"):
                include_dirs.append(root)
                break

    return list(set(include_dirs))


def get_extensions(workspace):
    extensions = []

    for root, dirs, files in os.walk(workspace):
        for file in files:
            if file.endswith(".pyx"):
                pyx_file = os.path.join(root, file)
                pyx_path = pyx_file[:-4].split(os.sep)
                # pyx_path = pyx_path[1:] if pyx_path[0] == '.' else pyx_path  # TODO 这句有问题?
                name = ".".join(pyx_path)

                extension = Extension(name, [pyx_file],
                                      extra_compile_args=["-std=c++11"])
                extensions.append(extension)
    return extensions


def clean(func):
    """clean intermediate file
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        def _listdir(workspace="."):
            list_dirs = set()
            list_files = set()
            for _root, _dirs, _files in os.walk(workspace):
                list_dirs.update([os.path.join(_root, d) for d in _dirs])
                list_files.update([os.path.join(_root, f) for f in _files])
            return list_dirs, list_files

        def _clean_file():
            print("clean file...")
            new_dirs, new_files = _listdir(".")
            for _dir in new_dirs:
                if os.path.exists(_dir) \
                        and _dir not in old_dirs\
                        and "dist" not in _dir.split(os.path.sep):
                    shutil.rmtree(_dir)
                    print(f"removing {_dir}")

            for _file in new_files:
                if os.path.exists(_file) \
                        and _file not in old_files \
                        and "dist" not in _file.split(os.path.sep) \
                        and not _file.endswith(".so") \
                        and not _file.endswith(".pyd"):
                    os.remove(_file)
                    print(f"removing {_file}")

        old_dirs, old_files = _listdir(".")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            _clean_file()
            raise e

        _clean_file()
        return result

    return wrapper


@clean
def setup_package():
    extensions = get_extensions("skrec")
    include_dirs = get_include_dirs("skrec")
    module_list = cythonize(extensions, annotate=False)

    install_requires = ["numpy>=1.17", "scipy", "pandas", "colorama"]
    metadata = dict(
        name="scikit-recommender",  # Replace with your own username
        version=VERSION,
        author="ZhongchuanSun",
        author_email="zhongchuansun@gmail.com",
        description="A science toolkit for recommender systems",
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        url="https://github.com/ZhongchuanSun/scikit-recommender",
        packages=setuptools.find_packages(),
        include_dirs=include_dirs,
        ext_modules=module_list,
        platforms=["Windows", "Linux", "Mac OS"],
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        python_requires='>=3.6',
        install_requires=install_requires
    )
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
