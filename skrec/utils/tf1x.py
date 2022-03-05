__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["inner_product"]

import tensorflow as tf


def inner_product(a, b, name="inner_product"):
    with tf.name_scope(name=name):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1)
