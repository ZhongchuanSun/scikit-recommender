<!-- Add banner here -->

# Scikit-Recommender

<!-- Describe your project in brief -->
Scikit-Recommender is an open source library for researchers of recommender systems.

## Highlighted Features

- Various recommendation models
- Parse arguments from command line and ini-style files
- Diverse data preprocessing
- Fast negative sampling
- Fast model evaluating
- Convenient records logging
- Flexible batch data iterator

## Install Scikit-Recommender

You have three ways to use Scikit-Recommender:
1. Install from PyPI
2. Build and install from sources
3. Run without installation

### Install from PyPI

Binary installers are available at the [Python package index](https://pypi.org/project/scikit-recommender/).
```sh
# PyPI
pip install scikit-recommender
```

### Build and install from sources

#### First, build wheel from sources

To build scikit-recommender from source you need Cython:
```sh
pip install cython
```
Then, in the scikit-recommender directory, execute:
```sh
python setup.py bdist_wheel
```
Now, you can find a `scikit-recommender*.whl` file in `./dist/`

#### Second, install the wheel

Then, install it:
```sh
pip install ./dist/scikit-recommender*.whl
```

### Run without installation

Alternatively, You can directly compile the sources in the current directory without installation.
In the scikit-recommender directory, execute:
```sh
python setup.py build_ext --inplace
```
Then, you can use `skrec` in current directory.

## Usage

You can find examples in [examples/tutorial.py](./examples/tutorial.py).
