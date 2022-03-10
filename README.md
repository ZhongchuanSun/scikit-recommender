<!-- Add banner here -->

# Scikit-Recommender

<!-- Describe your project in brief -->
Scikit-Recommender is an open source library for researchers of recommender systems.

## Highlighted Features

- Various recommendation models
- Parse arguments from command line and ini-style files
- Diverse data preprocessing
- Fast negative sampling
- Fast model evaluation
- Convenient record logging
- Flexible batch data iterator

## Installation
<!-- ## Install Scikit-Recommender -->

You have three ways to use Scikit-Recommender:

1. Install from PyPI
2. Install from Source
3. Run without Installation

### Install from PyPI

Binary installers are available at the [Python package index](https://pypi.org/project/scikit-recommender/) and you can install the package from pip.

```sh
# PyPI
pip install scikit-recommender
```

### Install from Source

Installing from source requires Cython and the current code works well with the version 0.29.20.

To build scikit-recommender from source you need Cython:

```sh
pip install cython==0.29.20
```

Then, the scikit-recommender can be installed by executing:

```sh
git clone https://github.com/ZhongchuanSun/scikit-recommender.git
cd scikit-recommender
python setup.py install
```

### Run without Installation

Alternatively, You can also run the sources without installation.
Please compile the cython codes before running:

```sh
git clone https://github.com/ZhongchuanSun/scikit-recommender.git
cd scikit-recommender
python setup.py build_ext --inplace
```

## Usage

After installing or compiling this package, now you can run the [run_skrec.py]([./run_skrec.py](https://github.com/ZhongchuanSun/scikit-recommender/blob/master/run_skrec.py)):

```sh
python run_skrec.py
```

You can also find examples in [tutorial.ipynb](https://github.com/ZhongchuanSun/scikit-recommender/blob/master/tutorial.ipynb).

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
| SRGNN       | TensorFlow (1.14) | [Shu Wu et al., Session-Based Recommendation with Graph Neural Networks, AAAI 2019.](https://ojs.aaai.org/index.php/AAAI/article/view/3804) |
