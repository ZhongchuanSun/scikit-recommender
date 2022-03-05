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

You can find examples in [tutorial.ipynb](https://github.com/ZhongchuanSun/scikit-recommender/blob/master/tutorial.ipynb) and [run_skrec.py](https://github.com/ZhongchuanSun/scikit-recommender/blob/master/run_skrec.py).

## Models

| Recommender | Implementation    | Paper|
|---|:-:|---|
| BPRMF       | PyTorch           | [Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009.](https://dl.acm.org/doi/10.5555/1795114.1795167) |
| AOBPR       | C/Cython          | [Steffen Rendle et al., Improving Pairwise Learning for Item Recommendation from Implicit Feedback, WSDM 2014.](https://dl.acm.org/doi/10.1145/2556195.2556248) |
| BERT4Rec    | TensorFlow (1.14) | [Fei Sun et al., BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, CIKM 2019](https://dl.acm.org/doi/abs/10.1145/3357384.3357895) |
| LightGCN    | PyTorch           | [Xiangnan He et al., LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, SIGIR 2020.](https://dl.acm.org/doi/10.1145/3397271.3401063)|
| SASRec      | TensorFlow (1.14) | [Wangcheng Kang et al., Self-Attentive Sequential Recommendation, ICDM 2018.](https://ieeexplore.ieee.org/abstract/document/8594844) |
| HGN         |  PyTorch          | [Chen Ma et al., Hierarchical Gating Networks for Sequential Recommendation, KDD 2019](https://dl.acm.org/doi/10.1145/3292500.3330984) |
| TransRec    | PyTorch           | [Ruining He et al., Translation-based Recommendation, RecSys 2017](https://dl.acm.org/doi/10.1145/3109859.3109882) |
